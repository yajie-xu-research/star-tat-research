"""Five-table contract validation tests (guide P1-1 first checkpoint)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from star_tat.p1.io import load_all
from star_tat.p1.synthetic import generate, write_tables
from star_tat.p1.validation import validate_frames, validate_suite

STAGES = Path(__file__).parents[1] / "configs" / "p1_stages.yaml"


def _frames(stage_map, tmp_path, sites=("SITE_A",), versions=("WF_V1",),
            scale=0.4):
    """Generated frames round-tripped through CSV + strict parsing, so the
    validator sees the same shapes as the CLI does."""
    frames = generate(20240115, stage_map, sites=sites, versions=versions,
                      scale=scale, include_unknown_version=False)
    write_tables(tmp_path, frames, 20240115)
    loaded, _ = load_all(tmp_path)
    return loaded


def test_missing_column_is_fatal(stage_map, tmp_path):
    frames = generate(20240115, stage_map, sites=("SITE_A",),
                      versions=("WF_V1",), scale=0.4,
                      include_unknown_version=False)
    write_tables(tmp_path, frames, 20240115)
    cases = tmp_path / "cases.csv"
    lines = cases.read_text().splitlines()
    cols = lines[0].split(",")
    idx = cols.index("site_key")
    body = [line.split(",") for line in lines[1:]]
    rebuilt = ",".join(c for c in cols if c != "site_key") + "\n" + "\n".join(
        ",".join(v for j, v in enumerate(row) if j != idx) for row in body
    ) + "\n"
    cases.write_text(rebuilt)
    report = validate_suite(tmp_path, STAGES)
    assert report["validation_outcome"] == "FAIL"


def test_duplicate_case_key_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    dup = frames["cases"].iloc[[0]].copy()
    frames["cases"] = pd.concat([frames["cases"], dup], ignore_index=True)
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "DUPLICATE_KEY" for f in report["fatal_issues"])


def test_duplicate_request_id_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    dup = frames["predict_requests"].iloc[[0]].copy()
    frames["predict_requests"] = pd.concat(
        [frames["predict_requests"], dup], ignore_index=True)
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "DUPLICATE_KEY" for f in report["fatal_issues"])


def test_duplicate_event_primary_key_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    dup = frames["events"].iloc[[0]].copy()
    frames["events"] = pd.concat([frames["events"], dup], ignore_index=True)
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "DUPLICATE_KEY" for f in report["fatal_issues"])


def test_inverted_release_time_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    cases = frames["cases"].copy()
    first = cases.index[cases["is_completed"].astype(str) == "1"][0]
    received = cases.at[first, "received_at_utc"]
    cases.at[first, "released_at_utc"] = received - pd.Timedelta(hours=1)
    frames["cases"] = cases
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "INVERTED_TIME" for f in report["fatal_issues"])


def test_event_known_before_occurred_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    events = frames["events"].copy()
    row = events.index[0]
    events.at[row, "known_at_utc"] = \
        events.at[row, "occurred_at_utc"] - pd.Timedelta(hours=5)
    frames["events"] = events
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "INVERTED_TIME" for f in report["fatal_issues"])


def test_orphan_event_case_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    events = frames["events"].copy()
    events.at[events.index[0], "case_key"] = "CASE_99999999"
    frames["events"] = events
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "ORPHAN_CASE" for f in report["fatal_issues"])


def test_orphan_request_case_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    requests = frames["predict_requests"].copy()
    requests.at[requests.index[0], "case_key"] = "CASE_99999998"
    frames["predict_requests"] = requests
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "ORPHAN_CASE" for f in report["fatal_issues"])


def test_completion_flag_mismatch_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    cases = frames["cases"].copy()
    first = cases.index[cases["is_completed"].astype(str) == "1"][0]
    cases.at[first, "released_at_utc"] = pd.NaT
    frames["cases"] = cases
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "COMPLETION_MISMATCH"
               for f in report["fatal_issues"])


def test_negative_queue_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    snaps = frames["load_snapshots"].copy()
    snaps.at[snaps.index[0], "queue_size"] = "-5"
    frames["load_snapshots"] = snaps
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "INVALID_VALUE" for f in report["fatal_issues"])


def test_util_outside_unit_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    snaps = frames["load_snapshots"].copy()
    snaps.at[snaps.index[0], "instrument_util"] = "1.4"
    frames["load_snapshots"] = snaps
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "INVALID_VALUE" for f in report["fatal_issues"])


def test_unknown_canonical_stage_in_map_is_fatal(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    smap = frames["stage_map"].copy()
    smap.at[smap.index[0], "canonical_stage"] = "TELEPORTED"
    frames["stage_map"] = smap
    report = validate_frames(frames, stage_map.cfg)
    assert any(f["issue"] == "UNKNOWN_CANONICAL_STAGE"
               for f in report["fatal_issues"])


def test_malformed_timestamp_is_fatal(stage_map, tmp_path):
    frames = generate(20240115, stage_map, sites=("SITE_A",),
                      versions=("WF_V1",), scale=0.4,
                      include_unknown_version=False)
    write_tables(tmp_path, frames, 20240115)
    events = tmp_path / "events.csv"
    lines = events.read_text().splitlines()
    parts = lines[1].split(",")
    parts[2] = "2024-99-99T00:00:00Z"  # occurred_at_utc is column 2
    lines[1] = ",".join(parts)
    events.write_text("\n".join(lines) + "\n")
    report = validate_suite(tmp_path, STAGES)
    assert report["validation_outcome"] == "FAIL"
    assert any(f["issue"] == "MALFORMED_TIME"
               for f in report["fatal_issues"])


def test_unknown_stage_code_is_warning(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    events = frames["events"].copy()
    row = events.index[0]
    events.at[row, "event_code_raw"] = "XADM"
    frames["events"] = events
    report = validate_frames(frames, stage_map.cfg)
    assert any(w["issue"] == "UNMAPPED_STAGE_CODE"
               for w in report["warnings"])
    assert report["validation_outcome"] == "PASS"


def test_unknown_workflow_version_is_warning(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    cases = frames["cases"].copy()
    cases.at[cases.index[0], "workflow_version"] = "WF_V9"
    frames["cases"] = cases
    report = validate_frames(frames, stage_map.cfg)
    assert any(w["issue"] == "UNKNOWN_WORKFLOW_VERSION"
               for w in report["warnings"])
    assert report["validation_outcome"] == "PASS"


def test_not_comparable_is_reported(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path, sites=("SITE_A", "SITE_C"))
    report = validate_frames(frames, stage_map.cfg)
    assert any(w["issue"] == "NOT_COMPARABLE" for w in report["warnings"])


def test_delayed_entry_is_warning(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    events = frames["events"].copy()
    row = events.index[1]
    events.at[row, "known_at_utc"] = \
        events.at[row, "occurred_at_utc"] + pd.Timedelta(days=90)
    frames["events"] = events
    report = validate_frames(frames, stage_map.cfg)
    assert any(w["issue"] == "DELAYED_ENTRY" for w in report["warnings"])


def test_clean_generated_data_passes(stage_map, tmp_path):
    frames = _frames(stage_map, tmp_path)
    report = validate_frames(frames, stage_map.cfg)
    assert report["validation_outcome"] == "PASS"
    assert report["fatal_count"] == 0
