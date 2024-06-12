"""Stratified interval calibration.

Per site x workflow_version stratum:
  * the point model (gradient boosting on stage/load features) is fit on the
    training block only;
  * the calibration block supplies absolute residuals;
  * q is the empirical quantile of those residuals at the coverage target
    (a policy parameter, not a success gate);
  * a split-conformal control computes its quantile as
    ceil((n+1)*(1-alpha)) / n over the same residuals;
  * the interval for a request is [max(0, pred - q), pred + q] hours added
    to as_of_utc.

Stratum calibration blocks below the policy floor make the whole stratum
abstain with INSUFFICIENT_CALIBRATION.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from star_tat.p1.constants import (
    BLOCK_CALIBRATION,
    BLOCK_TRAIN,
    REASON_INSUFFICIENT_CALIBRATION,
)
from star_tat.p1.features import FeatureEncoder
from star_tat.p1.snapshots import Snapshot


@dataclass
class StratumModel:
    site_key: str
    workflow_version: str
    encoder: FeatureEncoder
    scaler: StandardScaler
    gbm: GradientBoostingRegressor
    q: float
    conformal_q: float
    n_train: int
    n_calibration: int
    coverage_target: float
    train_ranges: dict[str, tuple[float, float]]
    insufficient: bool = False

    def predict(self, snapshot: Snapshot) -> tuple[float, float, float]:
        """Return (point_hours, lower_hours, upper_hours)."""
        x = self.encoder.transform(snapshot)
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        point = float(self.gbm.predict(x_scaled)[0])
        lower = max(0.0, point - self.q)
        upper = point + self.q
        return point, lower, upper

    def conformal_interval(self, snapshot: Snapshot) -> tuple[float, float]:
        point, _, _ = self.predict(snapshot)
        lower = max(0.0, point - self.conformal_q)
        return lower, point + self.conformal_q


def _absolute_residual_quantile(residuals: np.ndarray,
                                coverage_target: float) -> float:
    return float(np.quantile(residuals, coverage_target))


def _split_conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    if n == 0:
        return float("inf")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher"))


def _tukey_fence(values: np.ndarray) -> tuple[float, float]:
    """Robust validated range: q25 - 1.5*IQR .. q75 + 1.5*IQR.

    A raw min/max over the training sample flags ordinary tail noise of
    continuous load features as drift; the fence keeps the check sensitive
    to genuine regime shifts (multiples of the usual range) while ignoring
    sampling noise.
    """
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = q75 - q25
    return (float(q25 - 1.5 * iqr), float(q75 + 1.5 * iqr))


def fit_stratum(site_key: str, workflow_version: str,
                train_snapshots: list[Snapshot], train_targets: list[float],
                calib_snapshots: list[Snapshot], calib_targets: list[float],
                policy: dict) -> StratumModel:
    model_cfg = policy["model"]
    coverage_target = float(policy["coverage_target"])
    conformal_alpha = float(policy["interval"]["conformal_alpha"])
    min_train = int(policy["min_train_samples"])
    min_calib = int(policy["min_calibration_samples"])

    encoder = FeatureEncoder(["priority_class", "test_family"],
                             include_pooled=False).fit(train_snapshots)
    scaler = StandardScaler()
    gbm = GradientBoostingRegressor(
        n_estimators=int(model_cfg["n_estimators"]),
        learning_rate=float(model_cfg["learning_rate"]),
        max_depth=int(model_cfg["max_depth"]),
        random_state=int(model_cfg["random_state"]),
    )
    insufficient = (len(train_snapshots) < min_train
                    or len(calib_snapshots) < min_calib)
    if insufficient:
        # Dummy fit so the object stays uniform; predictions are never used
        # for an insufficient stratum.
        if train_snapshots:
            scaler.fit(np.vstack([encoder.transform(s) for s in train_snapshots]))
        model = StratumModel(
            site_key=site_key, workflow_version=workflow_version,
            encoder=encoder, scaler=scaler, gbm=gbm, q=float("inf"),
            conformal_q=float("inf"), n_train=len(train_snapshots),
            n_calibration=len(calib_snapshots), coverage_target=coverage_target,
            train_ranges={}, insufficient=True,
        )
        return model

    x_train = np.vstack([encoder.transform(s) for s in train_snapshots])
    x_train = scaler.fit_transform(x_train)
    gbm.fit(x_train, np.asarray(train_targets))

    x_calib = scaler.transform(
        np.vstack([encoder.transform(s) for s in calib_snapshots]))
    calib_pred = gbm.predict(x_calib)
    residuals = np.abs(np.asarray(calib_targets) - calib_pred)

    q = _absolute_residual_quantile(residuals, coverage_target)
    conformal_q = _split_conformal_quantile(residuals, conformal_alpha)

    ranges = {}
    for name in ("queue_size_log1p", "batch_load", "instrument_util"):
        values = np.asarray([encoder.transform(s)[encoder.numeric.index(name)]
                             for s in train_snapshots], dtype=float)
        ranges[name] = _tukey_fence(values)

    return StratumModel(
        site_key=site_key, workflow_version=workflow_version,
        encoder=encoder, scaler=scaler, gbm=gbm, q=q,
        conformal_q=conformal_q, n_train=len(train_snapshots),
        n_calibration=len(calib_snapshots), coverage_target=coverage_target,
        train_ranges=ranges, insufficient=False,
    )
