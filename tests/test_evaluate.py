"""Evaluation tests: receipt structure and the INSUFFICIENT_CALIBRATION
rejection path exercised end to end with a constructed small stratum."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from star_tat.p1.constants import REASON_INSUFFICIENT_CALIBRATION
from star_tat.p1.replay import run_replay, utc_now
from star_tat.p1.synthetic import generate, write_tables

REPO = Path(__file__).resolve().parents[1]
STAGES = REPO / "configs" / "p1_stages.yaml"
POLICY = REPO / "configs" / "p1_policy.yaml"


def test_receipt_has_required_sections(tiny_replay_dir, tmp_path):
    from star_tat.p1.evaluate import run_evaluate
    out = tmp_path / "eval"
    result = run_evaluate(run_dir=tiny_replay_dir, out_dir=out,
                          utc_now=utc_now(), command="test")
    receipt = result["receipt"]
    for key in ("holdout", "coverage", "point_errors", "long_tat",
                "per_site", "per_version", "per_stratum",
                "rejection_rates", "excluded_rows", "calibration"):
        assert key in receipt
    for method in ("stratified_interval", "conformal",
                   "baseline_stratified_quantiles",
                   "baseline_unstratified_interval"):
        assert method in receipt["coverage"]
    assert (out / "evaluation_report.md").exists()
    assert (out / "evaluation_details.csv").exists()
    assert (out / "manifest.json").exists()


def test_coverage_between_zero_and_one(tiny_replay_dir, tmp_path):
    from star_tat.p1.evaluate import run_evaluate
    out = tmp_path / "eval"
    receipt = run_evaluate(run_dir=tiny_replay_dir, out_dir=out,
                           utc_now=utc_now(), command="test")["receipt"]
    for entry in receipt["coverage"].values():
        if entry["n"]:
            assert 0.0 <= entry["coverage"] <= 1.0
            assert entry["width_hours_mean"] > 0


def test_evaluated_plus_rejected_plus_censored_equals_holdout(
        tiny_replay_dir, tmp_path):
    from star_tat.p1.evaluate import run_evaluate
    out = tmp_path / "eval"
    receipt = run_evaluate(run_dir=tiny_replay_dir, out_dir=out,
                           utc_now=utc_now(), command="test")["receipt"]
    h = receipt["holdout"]
    assert h["evaluated"] + h["rejected_total"] + h["censored"] \
        == h["requests"]


def test_insufficient_calibration_stratum_abstains(tmp_path, stage_map):
    """A constructed small stratum (calibration below the policy floor)
    must abstain with INSUFFICIENT_CALIBRATION, never emit an interval."""
    frames = generate(20240115, stage_map, sites=("SITE_A", "SITE_B"),
                      versions=("WF_V1", "WF_V2"), scale=0.2,
                      include_unknown_version=False)
    data = tmp_path / "small"
    write_tables(data, frames, 20240115)
    out = tmp_path / "run"
    run_replay(input_dir=data,
               as_of_file=data / "predict_requests.csv",
               out_dir=out, stages_config=STAGES, policy_config=POLICY,
               seed=20240115, utc_now=utc_now(), command="test")
    results = pd.read_csv(out / "results.csv", dtype=str,
                          keep_default_na=False)
    meta = json.loads((out / "calibration_meta.json").read_text())
    assert any(s["insufficient"] for s in meta["strata"].values())
    reason_counts = results["reason_code"].value_counts().to_dict()
    assert REASON_INSUFFICIENT_CALIBRATION in reason_counts
    # Rows abstained for calibration reasons carry no interval.
    flagged = results[results["reason_code"]
                      == REASON_INSUFFICIENT_CALIBRATION]
    assert len(flagged) > 0
    assert (flagged["predicted_release_upper_utc"] == "").all()


def test_normal_run_strata_are_sufficient(tiny_replay_dir):
    meta = json.loads(
        (tiny_replay_dir / "calibration_meta.json").read_text())
    assert meta["strata"]
    assert all(not s["insufficient"] for s in meta["strata"].values())


def test_evaluate_reads_only_holdout_for_metrics(tiny_replay_dir, tmp_path):
    from star_tat.p1.evaluate import run_evaluate
    out = tmp_path / "eval"
    receipt = run_evaluate(run_dir=tiny_replay_dir, out_dir=out,
                           utc_now=utc_now(), command="test")["receipt"]
    detail = pd.read_csv(out / "evaluation_details.csv")
    assert (detail["remaining_hours"].notna()).all()
    # Every evaluated row has an interval from the stratified method.
    assert (detail["predicted_release_upper_utc"].notna()).all()
