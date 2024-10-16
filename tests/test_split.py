"""Split tests: stratification, temporal ordering, censored-to-train, and
case-key grouping invariants (guide counter-example row 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from star_tat.p1.constants import (BLOCK_CALIBRATION, BLOCK_HOLDOUT,
                                   BLOCK_TRAIN)
from star_tat.p1.split import (SplitError, assign_blocks,
                               check_case_overlap, verify_block_consistency)


def _cases(policy_cfg, n_per_stratum=150, censored=20):
    rows = []
    for site in ("SITE_A", "SITE_B"):
        for version in ("WF_V1", "WF_V2"):
            for i in range(n_per_stratum):
                rows.append({
                    "case_key": f"C_{site}_{version}_{i:04d}",
                    "site_key": site, "workflow_version": version,
                    "test_family": "ONCO_PANEL",
                    "priority_class": "ROUTINE",
                    "received_at_utc": pd.Timestamp("2024-02-01", tz="UTC")
                    + pd.Timedelta(hours=i),
                    "released_at_utc": pd.Timestamp("2024-02-05", tz="UTC"),
                    "is_completed": "1",
                })
            for i in range(censored):
                rows.append({
                    "case_key": f"C_{site}_{version}_U{i:04d}",
                    "site_key": site, "workflow_version": version,
                    "test_family": "ONCO_PANEL",
                    "priority_class": "ROUTINE",
                    "received_at_utc": pd.Timestamp("2024-02-01", tz="UTC")
                    + pd.Timedelta(hours=i),
                    "released_at_utc": "",
                    "is_completed": "0",
                })
    return pd.DataFrame(rows)


def test_fractions_sum_to_one(policy_cfg):
    cases = _cases(policy_cfg)
    blocks = assign_blocks(cases, policy_cfg)
    completed = cases[cases["is_completed"] == "1"]
    n = len(completed)
    train = sum(1 for k in completed["case_key"]
                if blocks[k] == BLOCK_TRAIN)
    calib = sum(1 for k in completed["case_key"]
                if blocks[k] == BLOCK_CALIBRATION)
    holdout = sum(1 for k in completed["case_key"]
                  if blocks[k] == BLOCK_HOLDOUT)
    assert train + calib + holdout == n
    assert abs(train / n - 0.60) < 0.02
    assert abs(calib / n - 0.20) < 0.02
    assert abs(holdout / n - 0.20) < 0.02


def test_censored_cases_all_in_train(policy_cfg):
    cases = _cases(policy_cfg)
    blocks = assign_blocks(cases, policy_cfg)
    for case_key in cases[cases["is_completed"] == "0"]["case_key"]:
        assert blocks[case_key] == BLOCK_TRAIN


def test_split_is_temporal_within_stratum(policy_cfg):
    cases = _cases(policy_cfg)
    blocks = assign_blocks(cases, policy_cfg)
    for _, stratum in cases.groupby(["site_key", "workflow_version"]):
        completed = stratum[stratum["is_completed"] == "1"].sort_values(
            "received_at_utc")
        ordered_blocks = [blocks[k] for k in completed["case_key"]]
        # Temporal split means blocks appear in train/calib/holdout order.
        first_calib = ordered_blocks.index(BLOCK_CALIBRATION) \
            if BLOCK_CALIBRATION in ordered_blocks else len(ordered_blocks)
        first_holdout = ordered_blocks.index(BLOCK_HOLDOUT) \
            if BLOCK_HOLDOUT in ordered_blocks else len(ordered_blocks)
        assert first_calib <= first_holdout
        assert all(b == BLOCK_TRAIN for b in ordered_blocks[:first_calib])


def test_fraction_sum_mismatch_raises(policy_cfg):
    cases = _cases(policy_cfg)
    bad_policy = dict(policy_cfg)
    bad_policy["split"] = {"train_fraction": 0.7,
                           "calibration_fraction": 0.3,
                           "holdout_fraction": 0.3}
    with pytest.raises(SplitError):
        assign_blocks(cases, bad_policy)


def test_requests_inherit_case_block(policy_cfg):
    cases = _cases(policy_cfg, n_per_stratum=60)
    blocks = assign_blocks(cases, policy_cfg)
    requests = pd.DataFrame([
        {"case_key": f"C_SITE_A_WF_V1_{i:04d}", "as_of_utc": "2024-02-02T00:00:00Z",
         "request_id": f"R{i:05d}", "purpose": "OPERATIONAL_ETA"}
        for i in range(30)
    ])
    verify_block_consistency(blocks, requests)  # must not raise


def test_missing_case_in_block_map_raises(policy_cfg):
    cases = _cases(policy_cfg, n_per_stratum=60)
    blocks = assign_blocks(cases, policy_cfg)
    requests = pd.DataFrame([
        {"case_key": "C_NOT_PRESENT", "as_of_utc": "2024-02-02T00:00:00Z",
         "request_id": "R99999", "purpose": "OPERATIONAL_ETA"}
    ])
    with pytest.raises(SplitError):
        verify_block_consistency(blocks, requests)


def test_case_key_cannot_leak_across_blocks(policy_cfg):
    cases = _cases(policy_cfg, n_per_stratum=60)
    blocks = assign_blocks(cases, policy_cfg)
    train = {k for k, v in blocks.items() if v == BLOCK_TRAIN}
    calib = {k for k, v in blocks.items() if v == BLOCK_CALIBRATION}
    holdout = {k for k, v in blocks.items() if v == BLOCK_HOLDOUT}
    check_case_overlap({"train": train, "calibration": calib,
                        "holdout": holdout})  # must not raise
    # A fabricated overlap must raise.
    with pytest.raises(SplitError):
        check_case_overlap({"train": {"A", "B"}, "calibration": {"B", "C"},
                            "holdout": {"D"}})


def test_duplicate_request_same_case_same_block(policy_cfg):
    cases = _cases(policy_cfg, n_per_stratum=60)
    blocks = assign_blocks(cases, policy_cfg)
    key = "C_SITE_A_WF_V1_0001"
    requests = pd.DataFrame([
        {"case_key": key, "as_of_utc": "2024-02-02T00:00:00Z",
         "request_id": "R00001", "purpose": "OPERATIONAL_ETA"},
        {"case_key": key, "as_of_utc": "2024-02-03T00:00:00Z",
         "request_id": "R00002", "purpose": "RETRO_AUDIT"},
    ])
    verify_block_consistency(blocks, requests)
    assert blocks[key] in (BLOCK_TRAIN, BLOCK_CALIBRATION, BLOCK_HOLDOUT)
