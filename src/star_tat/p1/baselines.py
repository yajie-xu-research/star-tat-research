"""Baseline suite.

Five reproducible comparisons, all fit exclusively on the training block
(the unstratified interval additionally uses the calibration block for its
quantile, mirroring the main method):

  1. median_history        - median remaining hours for similar completed
                             cases (priority x stage reached), with an
                             explicit fallback chain;
  2. stratified_quantiles  - empirical 5th/95th percentile of remaining
                             hours per site/version/stage (an interval);
  3. raw_timestamp_point   - linear regression on raw epoch timestamps and
                             elapsed hours (a point forecast);
  4. tree_point            - gradient boosting point forecast (pooled
                             across strata);
  5. unstratified_interval - the pooled tree point plus a pooled absolute
                             residual quantile from the calibration block
                             (no version stratification, the naive
                             contrast).

None of these read holdout outcomes. The contrast methods deliberately pool
strata; the primary stratified method never does.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from star_tat.p1.common_baseline_helpers import (
    FallbackMedianStats,
    StratifiedQuantileStats,
    build_fallback_median_stats,
    build_stratified_quantile_stats,
)
from star_tat.p1.features import FeatureEncoder
from star_tat.p1.snapshots import Snapshot

BASELINE_NAMES = (
    "median_history",
    "stratified_quantiles",
    "raw_timestamp_point",
    "tree_point",
    "unstratified_interval",
)


class BaselineBundle:
    def __init__(self):
        self.median_stats: FallbackMedianStats | None = None
        self.quantile_stats: StratifiedQuantileStats | None = None
        self.linear: LinearRegression | None = None
        self.linear_scaler: StandardScaler | None = None
        self.tree: GradientBoostingRegressor | None = None
        self.pooled_encoder: FeatureEncoder | None = None
        self.pooled_scaler: StandardScaler | None = None
        self.pooled_q: float | None = None
        self.pooled_conformal_q: float | None = None

    def fit(self, train_snapshots: list[Snapshot], train_targets: list[float],
            calib_snapshots: list[Snapshot], calib_targets: list[float],
            policy: dict) -> "BaselineBundle":
        model_cfg = policy["model"]
        self.median_stats = build_fallback_median_stats(
            train_snapshots, train_targets)
        self.quantile_stats = build_stratified_quantile_stats(
            train_snapshots, train_targets)

        # Raw-timestamp linear regression on epoch seconds.
        self.linear_scaler = StandardScaler()
        x_raw = self.linear_scaler.fit_transform(
            np.vstack([
                [s.received_at_utc.timestamp(),
                 s.as_of_utc.timestamp(),
                 s.elapsed_hours,
                 s.stages_completed,
                 s.hours_in_stage]
                for s in train_snapshots
            ]))
        self.linear = LinearRegression().fit(x_raw, np.asarray(train_targets))

        # Pooled tree model and pooled calibration quantile.
        self.pooled_encoder = FeatureEncoder(
            ["priority_class", "test_family"], include_pooled=True
        ).fit(train_snapshots)
        self.pooled_scaler = StandardScaler()
        x_train = self.pooled_scaler.fit_transform(
            np.vstack([self.pooled_encoder.transform(s) for s in train_snapshots]))
        self.tree = GradientBoostingRegressor(
            n_estimators=int(model_cfg["n_estimators"]),
            learning_rate=float(model_cfg["learning_rate"]),
            max_depth=int(model_cfg["max_depth"]),
            random_state=int(model_cfg["random_state"]),
        ).fit(x_train, np.asarray(train_targets))
        if calib_snapshots:
            x_calib = self.pooled_scaler.transform(
                np.vstack([self.pooled_encoder.transform(s)
                           for s in calib_snapshots]))
            residuals = np.abs(np.asarray(calib_targets)
                               - self.tree.predict(x_calib))
            self.pooled_q = float(np.quantile(
                residuals, float(policy["coverage_target"])))
            self.pooled_conformal_q = float(np.quantile(
                residuals, 1.0 - float(policy["interval"]["conformal_alpha"])))
        return self

    def predict(self, snapshot: Snapshot) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        if self.median_stats is not None:
            out["median_history"] = self.median_stats.lookup(snapshot)
        if self.quantile_stats is not None:
            lo, hi = self.quantile_stats.lookup(snapshot)
            out["stratified_quantile_lower"] = lo
            out["stratified_quantile_upper"] = hi
        if self.linear is not None and self.linear_scaler is not None:
            x = self.linear_scaler.transform(np.asarray([[
                snapshot.received_at_utc.timestamp(),
                snapshot.as_of_utc.timestamp(),
                snapshot.elapsed_hours,
                snapshot.stages_completed,
                snapshot.hours_in_stage,
            ]], dtype=float))
            out["raw_timestamp_point"] = float(self.linear.predict(x)[0])
        if self.tree is not None and self.pooled_encoder is not None:
            x = self.pooled_encoder.transform(snapshot)
            x = self.pooled_scaler.transform(x.reshape(1, -1))
            point = float(self.tree.predict(x)[0])
            out["tree_point"] = point
            if self.pooled_q is not None:
                out["unstratified_lower"] = max(0.0, point - self.pooled_q)
                out["unstratified_upper"] = point + self.pooled_q
        return out
