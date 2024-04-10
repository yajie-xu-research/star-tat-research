"""Baseline suite tests: all five comparisons produce outputs from the same
snapshot, and every statistic is built exclusively from the training block
(the unstratified interval adds the calibration block for its quantile)."""

from __future__ import annotations

import pandas as pd
import pytest

from star_tat.p1.baselines import BASELINE_NAMES, BaselineBundle
from star_tat.p1.snapshots import Snapshot
from star_tat.p1.constants import BLOCK_CALIBRATION, BLOCK_TRAIN


def _snaps(n, block=BLOCK_TRAIN):
    out = []
    for i in range(n):
        s = Snapshot(
            request_id=f"REQ_{i:05d}", case_key=f"C{i:05d}",
            as_of_utc=pd.Timestamp("2024-02-03T00:00:00Z"),
            site_key="SITE_A" if i % 2 == 0 else "SITE_B",
            workflow_version="WF_V1",
            test_family="ONCO_PANEL" if i % 3 else "GERMLINE",
            priority_class="STAT" if i % 7 == 0 else "ROUTINE",
            purpose="OPERATIONAL_ETA",
            received_at_utc=pd.Timestamp("2024-02-01T00:00:00Z"),
            block=block, decision="OUTPUT",
        )
        s.stage_reached = "SEQUENCING_STARTED" if i % 2 else "ANALYSIS_COMPLETE"
        s.stages_completed = 4
        s.chain_length = 7
        s.elapsed_hours = 40.0 + i
        s.hours_in_stage = 5.0
        s.load_features = {"queue_size": 200.0 + i, "batch_load": 0.6,
                           "instrument_util": 0.7}
        out.append(s)
    return out


def _policy():
    return {
        "model": {"n_estimators": 60, "learning_rate": 0.05, "max_depth": 2,
                  "random_state": 20240115},
        "coverage_target": 0.90,
        "interval": {"conformal_alpha": 0.10},
        "min_train_samples": 50,
        "min_calibration_samples": 30,
    }


def test_five_baseline_names():
    assert len(BASELINE_NAMES) == 5


def test_all_five_baselines_produce_values():
    policy = _policy()
    train = _snaps(120)
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    bundle = BaselineBundle().fit(train, [50.0 + i % 13 for i in range(120)],
                                  calib, [50.0 + i % 13 for i in range(40)],
                                  policy)
    snap = train[3]
    pred = bundle.predict(snap)
    assert pred["median_history"] is not None
    assert pred["stratified_quantile_lower"] is not None
    assert pred["stratified_quantile_upper"] is not None
    assert pred["raw_timestamp_point"] is not None
    assert pred["tree_point"] is not None
    assert pred["unstratified_lower"] is not None
    assert pred["unstratified_upper"] is not None
    assert pred["unstratified_lower"] <= pred["unstratified_upper"]


def test_median_fallback_chain_always_resolves():
    policy = _policy()
    train = _snaps(120)
    bundle = BaselineBundle().fit(train, [50.0 + i % 13 for i in range(120)],
                                  [], [], policy)
    # A snapshot from a site/version/stage never seen in training falls back
    # to the global median rather than failing.
    snap = _snaps(1)[0]
    snap.site_key = "SITE_C"
    snap.workflow_version = "WF_V3"
    snap.stage_reached = "REVIEWED"
    assert bundle.median_stats.lookup(snap) is not None


def test_stratified_quantile_interval_ordered():
    policy = _policy()
    train = _snaps(120)
    bundle = BaselineBundle().fit(train, [50.0 + i % 13 for i in range(120)],
                                  [], [], policy)
    for snap in train[:10]:
        lo, hi = bundle.quantile_stats.lookup(snap)
        assert lo is not None and hi is not None
        assert lo <= hi


def test_unstratified_q_requires_calibration_block():
    policy = _policy()
    train = _snaps(120)
    bundle = BaselineBundle().fit(train, [50.0 + i % 13 for i in range(120)],
                                  [], [], policy)
    assert bundle.pooled_q is None  # no calibration rows supplied
    pred = bundle.predict(train[0])
    assert pred.get("unstratified_lower") is None
