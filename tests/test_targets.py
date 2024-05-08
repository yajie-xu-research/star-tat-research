"""Target tests: censored cases are never zeroed (guide row 3)."""

from __future__ import annotations

import pandas as pd
import pytest

from star_tat.p1.constants import BLOCK_CALIBRATION, BLOCK_TRAIN, \
    EXCLUDE_AFTER_RELEASE
from star_tat.p1.snapshots import Snapshot
from star_tat.p1.targets import (build_training_targets,
                                 target_for_snapshot)


def _snap(case_key, as_of, block=BLOCK_TRAIN):
    return Snapshot(
        request_id=f"REQ_{case_key}", case_key=case_key,
        as_of_utc=as_of, site_key="SITE_A", workflow_version="WF_V1",
        test_family="ONCO_PANEL", priority_class="ROUTINE",
        purpose="OPERATIONAL_ETA",
        received_at_utc=pd.Timestamp("2024-02-01T00:00:00Z"),
        block=block, decision="OUTPUT",
    )


@pytest.fixture
def cases():
    return pd.DataFrame([
        {"case_key": "C1", "is_completed": "1",
         "released_at_utc": pd.Timestamp("2024-02-05T00:00:00Z", tz="UTC")},
        {"case_key": "C2", "is_completed": "0",
         "released_at_utc": pd.NaT},
        {"case_key": "C3", "is_completed": "1",
         "released_at_utc": pd.Timestamp("2024-02-03T00:00:00Z", tz="UTC")},
    ]).set_index("case_key")


def test_completed_target_is_remaining_hours(cases):
    snap = _snap("C1", pd.Timestamp("2024-02-03T00:00:00Z"))
    target, status = target_for_snapshot(snap, cases)
    assert status == "completed"
    assert target == pytest.approx(48.0)


def test_censored_target_is_none_not_zero(cases):
    snap = _snap("C2", pd.Timestamp("2024-02-03T00:00:00Z"))
    target, status = target_for_snapshot(snap, cases)
    assert status == "censored"
    assert target is None


def test_request_at_or_after_release_is_excluded(cases):
    snap = _snap("C3", pd.Timestamp("2024-02-04T00:00:00Z"))
    target, status = target_for_snapshot(snap, cases)
    assert status == EXCLUDE_AFTER_RELEASE
    assert target is None


def test_fit_rows_exclude_censored_with_reason(cases):
    snaps = [
        _snap("C1", pd.Timestamp("2024-02-03T00:00:00Z")),
        _snap("C2", pd.Timestamp("2024-02-03T00:00:00Z")),
        _snap("C3", pd.Timestamp("2024-02-04T00:00:00Z")),
    ]
    rows, targets, excluded = build_training_targets(
        snaps, cases, (BLOCK_TRAIN, BLOCK_CALIBRATION))
    assert [r.case_key for r in rows] == ["C1"]
    assert targets == [pytest.approx(48.0)]
    reasons = {e["reason"] for e in excluded}
    assert "CENSORED" in reasons
    assert EXCLUDE_AFTER_RELEASE in reasons
    # The censored row is reported, never silently dropped and never zeroed.
    censored = [e for e in excluded if e["reason"] == "CENSORED"]
    assert censored[0]["case_key"] == "C2"
    assert all(t != 0.0 for t in targets) or targets == []
