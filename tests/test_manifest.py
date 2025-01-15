"""Manifest structure tests: required fields, status values, and the
deterministic result-hash subset."""

from __future__ import annotations

import json
from pathlib import Path

from star_tat.common.manifest import (REQUIRED_MANIFEST_FIELDS,
                                      build_manifest,
                                      validate_manifest_fields)
from star_tat.p1.constants import (RUN_STATUS_INTERNAL,
                                   RUN_STATUS_EXTERNAL)


def _manifest(utc="2026-06-30T12:00:00Z", result_files=None):
    return build_manifest(
        run_id="P1R_test", utc=utc, project="P1_STAR_TAT",
        status=RUN_STATUS_INTERNAL, commit="n/a", config_hash="c" * 64,
        data_hash="d" * 64, permissions_pointer="governance:internal_research:v1",
        operator="internal-research-operator", input_rows=10,
        excluded_rows=2, model_or_rule_version="v1.0/v1.0/tat-gbm-1.4/x",
        command="star-tat p1 replay", environment={"python": "3.12"},
        seed=20240115, known_issues=[],
        result_files=result_files or {})


def test_required_fields_present(tmp_path):
    m = _manifest()
    assert not validate_manifest_fields(m)
    assert set(REQUIRED_MANIFEST_FIELDS) <= set(m)


def test_missing_field_detected():
    m = _manifest()
    del m["run_id"]
    assert "run_id" in validate_manifest_fields(m)


def test_result_hash_is_stable_under_wall_clock_change(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("a,b\n1,2\n")
    m1 = _manifest(utc="2026-06-01T00:00:00Z", result_files={"r.csv": f})
    m2 = _manifest(utc="2026-06-30T23:59:59Z", result_files={"r.csv": f})
    assert m1["result_hash"] == m2["result_hash"]
    assert m1["run_id"] == m2["run_id"]


def test_result_hash_changes_with_result_files(tmp_path):
    f1 = tmp_path / "a.csv"
    f1.write_text("1\n")
    m1 = _manifest(result_files={"a.csv": f1})
    f1.write_text("2\n")
    m2 = _manifest(result_files={"a.csv": f1})
    assert m1["result_hash"] != m2["result_hash"]


def test_status_enum_values():
    assert RUN_STATUS_INTERNAL == "INTERNAL_RESEARCH"
    assert RUN_STATUS_EXTERNAL == "EXTERNAL_VALIDATION"


def test_manifest_written_with_trailing_newline(tmp_path):
    from star_tat.common.manifest import write_manifest
    m = _manifest()
    write_manifest(tmp_path, m)
    raw = (tmp_path / "manifest.json").read_bytes()
    assert raw.endswith(b"\n")
    parsed = json.loads(raw)
    assert parsed["run_id"] == m["run_id"]
