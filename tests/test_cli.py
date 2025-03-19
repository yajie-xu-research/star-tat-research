"""CLI contract tests: five commands, output artifacts, and honest non-zero
exit codes for broken inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / "venv" / "bin" / "python3")


def _cli(args, cwd=REPO):
    return subprocess.run(
        [PY, "-m", "star_tat.cli", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=600)


def test_generate_and_validate_and_replay_and_evaluate_roundtrip(tmp_path):
    data = tmp_path / "synthetic_data"
    out_run = tmp_path / "runs" / "p1_replay"
    out_eval = tmp_path / "runs" / "p1_evaluation"
    gen = _cli(["generate-synthetic", "--output", str(data),
                "--seed", "20240115"])
    assert gen.returncode == 0, gen.stderr
    for name in ("cases.csv", "events.csv", "load_snapshots.csv",
                 "stage_map.csv", "predict_requests.csv"):
        assert (data / name).exists()

    val = _cli(["validate", "--input", str(data),
                "--config", str(REPO / "configs" / "p1_stages.yaml")])
    assert val.returncode == 0, val.stdout + val.stderr
    assert (data / "data_quality_report.json").exists()
    assert "validation outcome: PASS" in val.stdout

    rep = _cli(["p1", "replay", "--input", str(data),
                "--as-of-file", str(data / "predict_requests.csv"),
                "--out", str(out_run), "--seed", "20240115"])
    assert rep.returncode == 0, rep.stdout + rep.stderr
    assert (out_run / "manifest.json").exists()
    assert (out_run / "results.csv").exists()
    assert (out_run / "excluded_rows.csv").exists()
    assert (out_run / "calibration_meta.json").exists()
    assert (out_run / "feature_availability.csv").exists()
    assert (out_run / "data_quality_report.json").exists()
    assert (out_run / "run.log").exists()
    assert (out_run / "data" / "_ground_truth" / "true_outcomes.csv").exists()
    manifest = json.loads((out_run / "manifest.json").read_text())
    assert manifest["status"] == "INTERNAL_RESEARCH"
    assert manifest["project"] == "P1_STAR_TAT"

    ev = _cli(["p1", "evaluate", "--run", str(out_run),
               "--out", str(out_eval)])
    assert ev.returncode == 0, ev.stdout + ev.stderr
    assert (out_eval / "run_receipt.json").exists()
    assert (out_eval / "evaluation_report.md").exists()
    assert (out_eval / "evaluation_details.csv").exists()
    receipt = json.loads((out_eval / "run_receipt.json").read_text())
    assert receipt["holdout"]["evaluated"] > 0
    assert 0.0 <= receipt["coverage"]["stratified_interval"]["coverage"] <= 1.0

    insp = _cli(["manifest", "inspect", "--run", str(out_run)])
    assert insp.returncode == 0
    inspected = json.loads(insp.stdout)
    assert inspected["run_id"] == manifest["run_id"]


def test_generate_is_deterministic(tmp_path):
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    assert _cli(["generate-synthetic", "--output", str(d1),
                 "--seed", "7"]).returncode == 0
    assert _cli(["generate-synthetic", "--output", str(d2),
                 "--seed", "7"]).returncode == 0
    for name in ("cases.csv", "events.csv", "predict_requests.csv"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes()


def test_validate_missing_file_exits_nonzero(tmp_path):
    r = _cli(["validate", "--input", str(tmp_path),
              "--config", str(REPO / "configs" / "p1_stages.yaml")])
    assert r.returncode == 1


def test_validate_broken_table_exits_nonzero(tmp_path):
    data = tmp_path / "bad"
    _cli(["generate-synthetic", "--output", str(data), "--seed", "3"])
    cases = data / "cases.csv"
    lines = cases.read_text().splitlines()
    lines.insert(1, lines[1])  # duplicate a row -> duplicate case_key
    cases.write_text("\n".join(lines) + "\n")
    r = _cli(["validate", "--input", str(data),
              "--config", str(REPO / "configs" / "p1_stages.yaml")])
    assert r.returncode == 1
    assert "FAIL" in r.stdout


def test_replay_missing_input_exits_nonzero(tmp_path):
    r = _cli(["p1", "replay", "--input", str(tmp_path / "nope"),
              "--as-of-file", str(tmp_path / "nope.csv"),
              "--out", str(tmp_path / "out"), "--seed", "1"])
    assert r.returncode == 2


def test_replay_broken_data_exits_nonzero(tmp_path):
    data = tmp_path / "bad"
    _cli(["generate-synthetic", "--output", str(data), "--seed", "5"])
    cases = data / "cases.csv"
    lines = cases.read_text().splitlines()
    lines.insert(1, lines[1])
    cases.write_text("\n".join(lines) + "\n")
    r = _cli(["p1", "replay", "--input", str(data),
              "--as-of-file", str(data / "predict_requests.csv"),
              "--out", str(tmp_path / "out"), "--seed", "5"])
    assert r.returncode == 1
    assert "validation failed" in r.stderr


def test_evaluate_missing_run_exits_nonzero(tmp_path):
    r = _cli(["p1", "evaluate", "--run", str(tmp_path / "nope"),
              "--out", str(tmp_path / "out")])
    assert r.returncode == 2


def test_manifest_inspect_missing_run_exits_nonzero(tmp_path):
    r = _cli(["manifest", "inspect", "--run", str(tmp_path / "nope")])
    assert r.returncode == 1


def test_cli_help_lists_five_commands():
    r = _cli(["--help"])
    assert r.returncode == 0
    for token in ("generate-synthetic", "validate", "p1", "manifest"):
        assert token in r.stdout
    r2 = _cli(["p1", "--help"])
    assert r2.returncode == 0
    for token in ("replay", "evaluate"):
        assert token in r2.stdout
    r3 = _cli(["manifest", "--help"])
    assert r3.returncode == 0
    assert "inspect" in r3.stdout
