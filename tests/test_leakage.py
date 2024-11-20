"""Leakage tests: outcomes never enter features, the isolation guard blocks
ground-truth access, and deleting the ground-truth directory leaves replay
predictions byte-identical."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from star_tat.common.isolation import (IsolationViolation, isolation_context)
from star_tat.p1.features import FeatureEncoder, assert_feature_names_clean
from star_tat.p1.replay import run_replay, utc_now
from star_tat.p1.snapshots import Snapshot


def _snap():
    s = Snapshot(
        request_id="REQ_00001", case_key="C00001",
        as_of_utc=pd.Timestamp("2024-02-03T00:00:00Z"),
        site_key="SITE_A", workflow_version="WF_V1",
        test_family="ONCO_PANEL", priority_class="ROUTINE",
        purpose="OPERATIONAL_ETA",
        received_at_utc=pd.Timestamp("2024-02-01T00:00:00Z"),
        decision="OUTPUT",
    )
    s.stage_reached = "SEQUENCING_STARTED"
    s.stages_completed = 4
    s.chain_length = 7
    s.elapsed_hours = 48.0
    s.hours_in_stage = 6.0
    s.load_features = {"queue_size": 200.0, "batch_load": 0.6,
                       "instrument_util": 0.7}
    return s


def test_feature_names_never_mention_outcomes():
    encoder = FeatureEncoder(["priority_class", "test_family"],
                             include_pooled=True).fit([_snap()])
    names = encoder.all_feature_names()
    assert names
    for name in names:
        assert "released" not in name.lower()
        assert "outcome" not in name.lower()


def test_feature_guard_rejects_forbidden_names():
    with pytest.raises(ValueError):
        assert_feature_names_clean(["elapsed_hours", "released_at_hours"])


def test_isolation_guard_blocks_ground_truth_open(tmp_path):
    gt_dir = tmp_path / "_ground_truth"
    gt_dir.mkdir()
    (gt_dir / "true_outcomes.csv").write_text("case_key\nC1\n")
    with pytest.raises(IsolationViolation):
        with isolation_context():
            open(gt_dir / "true_outcomes.csv")
    # Outside the context the file is readable again.
    assert (gt_dir / "true_outcomes.csv").read_text().startswith("case_key")


def test_isolation_guard_allows_normal_paths(tmp_path):
    ok = tmp_path / "results.csv"
    ok.write_text("x\n")
    with isolation_context():
        assert open(ok).read() == "x\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replay_predictions_identical_without_ground_truth(
        tmp_path, tiny_data_dir):
    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"
    stages = str(Path(__file__).parents[1] / "configs" / "p1_stages.yaml")
    policy = str(Path(__file__).parents[1] / "configs" / "p1_policy.yaml")
    run_replay(input_dir=tiny_data_dir,
               as_of_file=tiny_data_dir / "predict_requests.csv",
               out_dir=out1, stages_config=stages, policy_config=policy,
               seed=20240115, utc_now=utc_now(), command="test")
    # Delete the ground truth and rerun: results must be byte-identical.
    import shutil
    shutil.rmtree(out1 / "data" / "_ground_truth")
    run_replay(input_dir=tiny_data_dir,
               as_of_file=tiny_data_dir / "predict_requests.csv",
               out_dir=out2, stages_config=stages, policy_config=policy,
               seed=20240115, utc_now=utc_now(), command="test")
    assert _sha(out1 / "results.csv") == _sha(out2 / "results.csv")
    assert _sha(out1 / "calibration_meta.json") == \
        _sha(out2 / "calibration_meta.json")


def test_replay_writes_ground_truth_separately(tiny_replay_dir):
    gt = tiny_replay_dir / "data" / "_ground_truth" / "true_outcomes.csv"
    assert gt.exists()
    content = gt.read_text()
    assert "released_at_utc" in content
    assert "case_key" in content
