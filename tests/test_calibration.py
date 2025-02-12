"""Calibration and abstention tests: quantile on calibration block only,
split-conformal control, the interval formula, and every rejection path
including a constructed small stratum for INSUFFICIENT_CALIBRATION
(guide counter-example row 4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from star_tat.p1.calibration import (_split_conformal_quantile, fit_stratum)
from star_tat.p1.constants import (BLOCK_CALIBRATION, BLOCK_TRAIN,
                                   REASON_INSUFFICIENT_CALIBRATION,
                                   REASON_SHIFT_DETECTED)
from star_tat.p1.drift import apply_shift_checks, check_shift
from star_tat.p1.snapshots import Snapshot


def _snaps(n, site="SITE_A", version="WF_V1", block=BLOCK_TRAIN,
           load=(200.0, 0.6, 0.7)):
    out = []
    for i in range(n):
        s = Snapshot(
            request_id=f"REQ_{i:05d}", case_key=f"C{i:05d}",
            as_of_utc=pd.Timestamp("2024-02-03T00:00:00Z"),
            site_key=site, workflow_version=version,
            test_family="ONCO_PANEL", priority_class="ROUTINE",
            purpose="OPERATIONAL_ETA",
            received_at_utc=pd.Timestamp("2024-02-01T00:00:00Z"),
            block=block, decision="OUTPUT",
        )
        s.stage_reached = "SEQUENCING_STARTED"
        s.stage_index = 3
        s.stages_completed = 4
        s.chain_length = 7
        s.elapsed_hours = 48.0 + i
        s.hours_in_stage = 6.0
        s.load_features = {"queue_size": load[0] + i,
                           "batch_load": 0.3 + (i % 7) * 0.08,
                           "instrument_util": 0.4 + (i % 9) * 0.05}
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


def test_q_computed_from_calibration_block_only():
    policy = _policy()
    train = _snaps(80)
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    targets = [50.0 + (i % 17) for i in range(80)]
    calib_targets = [50.0 + (i % 17) for i in range(40)]
    model = fit_stratum("SITE_A", "WF_V1", train, targets, calib,
                        calib_targets, policy)
    assert not model.insufficient
    assert model.q > 0
    assert model.conformal_q >= model.q  # higher-order quantile


def test_interval_formula_bounds(policy_cfg):
    policy = _policy()
    train = _snaps(80)
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    model = fit_stratum("SITE_A", "WF_V1", train,
                        [50.0 + i % 17 for i in range(80)],
                        calib, [50.0 + i % 17 for i in range(40)], policy)
    snap = _snaps(1)[0]
    point, lower, upper = model.predict(snap)
    assert lower == max(0.0, point - model.q)
    assert upper == point + model.q
    assert lower <= point <= upper


def test_conformal_interval_matches_formula(policy_cfg):
    policy = _policy()
    train = _snaps(80)
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    model = fit_stratum("SITE_A", "WF_V1", train,
                        [50.0 + i % 17 for i in range(80)],
                        calib, [50.0 + i % 17 for i in range(40)], policy)
    snap = _snaps(1)[0]
    lo, hi = model.conformal_interval(snap)
    point, _, _ = model.predict(snap)
    assert lo == max(0.0, point - model.conformal_q)
    assert hi == point + model.conformal_q


def test_split_conformal_quantile_formula():
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = _split_conformal_quantile(residuals, 0.10)
    level = min(1.0, math.ceil((5 + 1) * 0.90) / 5)
    assert q == np.quantile(residuals, level, method="higher")
    assert q == 5.0


def test_small_calibration_stratum_is_insufficient():
    policy = _policy()
    train = _snaps(60)
    calib = _snaps(5, block=BLOCK_CALIBRATION)  # below the 30 floor
    model = fit_stratum("SITE_A", "WF_V1", train,
                        [50.0 + i % 7 for i in range(60)],
                        calib, [50.0 + i % 7 for i in range(5)], policy)
    assert model.insufficient
    assert model.q == float("inf")


def test_small_train_stratum_is_insufficient():
    policy = _policy()
    train = _snaps(10)
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    model = fit_stratum("SITE_A", "WF_V1", train,
                        [50.0 + i % 7 for i in range(10)],
                        calib, [50.0 + i % 7 for i in range(40)], policy)
    assert model.insufficient


def test_shift_detected_for_out_of_range_load(policy_cfg):
    policy = _policy()
    train = _snaps(80, load=(200.0, 0.6, 0.7))
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    model = fit_stratum("SITE_A", "WF_V1", train,
                        [50.0 + i % 17 for i in range(80)],
                        calib, [50.0 + i % 17 for i in range(40)], policy)
    surge = _snaps(1, load=(1200.0, 0.99, 0.99))[0]
    assert check_shift(surge, model)


def test_normal_load_does_not_shift(policy_cfg):
    policy = _policy()
    train = _snaps(80, load=(200.0, 0.6, 0.7))
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    model = fit_stratum("SITE_A", "WF_V1", train,
                        [50.0 + i % 17 for i in range(80)],
                        calib, [50.0 + i % 17 for i in range(40)], policy)
    normal = _snaps(1, load=(210.0, 0.65, 0.72))[0]
    assert not check_shift(normal, model)


def test_apply_shift_checks_marks_abstain(policy_cfg):
    policy = _policy()
    train = _snaps(80, load=(200.0, 0.6, 0.7))
    calib = _snaps(40, block=BLOCK_CALIBRATION)
    model = fit_stratum("SITE_A", "WF_V1", train,
                        [50.0 + i % 17 for i in range(80)],
                        calib, [50.0 + i % 17 for i in range(40)], policy)
    surge = _snaps(1, load=(1200.0, 0.99, 0.99))[0]
    flagged = apply_shift_checks([surge], {("SITE_A", "WF_V1"): model})
    assert flagged == 1
    assert surge.decision == "ABSTAIN"
    assert surge.reason_code == REASON_SHIFT_DETECTED
