"""Five-table validation producing data_quality_report.json.

Fatal issues (missing columns, duplicate keys, inverted times, orphan
references, malformed values) make the command exit non-zero. Expected data
profiles (unknown stage codes, unknown workflow versions, non-comparable
combinations, delayed event entry) are reported as warnings; the replay
engine handles those rows by abstention or exclusion, never by silent
deletion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from star_tat.common.hashing import sha256_file
from star_tat.p1 import io
from star_tat.p1.constants import CANONICAL_STAGES, STAGE_UNMAPPED


def _dup_rows(frame: pd.DataFrame, keys: list[str]) -> list[int]:
    if not keys:
        return []
    return frame.index[frame.duplicated(subset=keys, keep=False)].tolist()


def _check_key_uniqueness(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    fatal: list[dict[str, Any]] = []
    for table in ("cases", "predict_requests"):
        frame = frames[table]
        for col in io.unique_columns(table):
            dups = _dup_rows(frame, [col])
            for idx in dups:
                fatal.append({
                    "table": table, "row": int(idx), "severity": "fatal",
                    "issue": "DUPLICATE_KEY",
                    "detail": f"column {col} value {frame[col].iloc[idx]!r} repeated",
                })
    for table in ("events", "load_snapshots", "stage_map"):
        frame = frames[table]
        keys = io.primary_keys(table)
        if not keys:
            continue
        for idx in _dup_rows(frame, keys):
            fatal.append({
                "table": table, "row": int(idx), "severity": "fatal",
                "issue": "DUPLICATE_KEY",
                "detail": f"primary key {keys} repeated",
            })
    return fatal


def _check_cases(frames: dict[str, pd.DataFrame]) -> tuple[list, list]:
    fatal: list[dict[str, Any]] = []
    warn: list[dict[str, Any]] = []
    cases = frames["cases"]
    for idx, row in cases.iterrows():
        received = row["received_at_utc"]
        released = row["released_at_utc"]
        completed = str(row["is_completed"]).strip()
        if completed not in ("0", "1"):
            fatal.append({"table": "cases", "row": int(idx), "severity": "fatal",
                          "issue": "INVALID_FLAG",
                          "detail": f"is_completed={completed!r}"})
            continue
        has_release = not pd.isna(released)
        if completed == "1" and not has_release:
            fatal.append({"table": "cases", "row": int(idx), "severity": "fatal",
                          "issue": "COMPLETION_MISMATCH",
                          "detail": "is_completed=1 but released_at_utc empty"})
        if completed == "0" and has_release:
            fatal.append({"table": "cases", "row": int(idx), "severity": "fatal",
                          "issue": "COMPLETION_MISMATCH",
                          "detail": "is_completed=0 but released_at_utc present"})
        if has_release and released < received:
            fatal.append({"table": "cases", "row": int(idx), "severity": "fatal",
                          "issue": "INVERTED_TIME",
                          "detail": "released_at_utc earlier than received_at_utc"})
    return fatal, warn


def _first_row_lookup(frame: pd.DataFrame, column: str) -> dict:
    """Index frame by column, keeping only the first occurrence, so a
    duplicated key (itself a fatal issue) cannot make lookups ambiguous."""
    dedup = frame.drop_duplicates(subset=[column], keep="first")
    return dedup.set_index(column, drop=False)


def _check_events(frames: dict[str, pd.DataFrame]) -> tuple[list, list]:
    fatal: list[dict[str, Any]] = []
    warn: list[dict[str, Any]] = []
    events = frames["events"]
    cases = _first_row_lookup(frames["cases"], "case_key")
    for idx, row in events.iterrows():
        case_key = row["case_key"]
        if case_key not in cases.index:
            fatal.append({"table": "events", "row": int(idx), "severity": "fatal",
                          "issue": "ORPHAN_CASE",
                          "detail": f"case_key {case_key!r} not in cases"})
            continue
        occurred = row["occurred_at_utc"]
        known = row["known_at_utc"]
        if pd.isna(occurred) or pd.isna(known):
            fatal.append({"table": "events", "row": int(idx), "severity": "fatal",
                          "issue": "MALFORMED_TIME",
                          "detail": "occurred_at_utc or known_at_utc missing"})
            continue
        if known < occurred:
            fatal.append({"table": "events", "row": int(idx), "severity": "fatal",
                          "issue": "INVERTED_TIME",
                          "detail": "known_at_utc earlier than occurred_at_utc"})
        case_row = cases.loc[case_key]
        if occurred < case_row["received_at_utc"] - pd.Timedelta(hours=1):
            fatal.append({"table": "events", "row": int(idx), "severity": "fatal",
                          "issue": "INVERTED_TIME",
                          "detail": "occurred_at_utc before case received_at_utc"})
        if (known - occurred) > pd.Timedelta(hours=48):
            warn.append({"table": "events", "row": int(idx), "severity": "warning",
                         "issue": "DELAYED_ENTRY",
                         "detail": f"event entered the system {(known - occurred).days} days after it occurred"})
    return fatal, warn


def _check_requests(frames: dict[str, pd.DataFrame]) -> tuple[list, list]:
    fatal: list[dict[str, Any]] = []
    warn: list[dict[str, Any]] = []
    requests = frames["predict_requests"]
    cases = _first_row_lookup(frames["cases"], "case_key")
    for idx, row in requests.iterrows():
        case_key = row["case_key"]
        if case_key not in cases.index:
            fatal.append({"table": "predict_requests", "row": int(idx),
                          "severity": "fatal", "issue": "ORPHAN_CASE",
                          "detail": f"case_key {case_key!r} not in cases"})
            continue
        as_of = row["as_of_utc"]
        if pd.isna(as_of):
            fatal.append({"table": "predict_requests", "row": int(idx),
                          "severity": "fatal", "issue": "MALFORMED_TIME",
                          "detail": "as_of_utc missing"})
            continue
        case_row = cases.loc[case_key]
        if as_of < case_row["received_at_utc"]:
            fatal.append({"table": "predict_requests", "row": int(idx),
                          "severity": "fatal", "issue": "INVERTED_TIME",
                          "detail": "as_of_utc before case received_at_utc"})
        released = case_row["released_at_utc"]
        if not pd.isna(released) and as_of >= released:
            warn.append({"table": "predict_requests", "row": int(idx),
                         "severity": "warning", "issue": "AS_OF_AFTER_RELEASE",
                         "detail": "request placed at or after the release timestamp"})
    return fatal, warn


def _check_snapshots(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    fatal: list[dict[str, Any]] = []
    snaps = frames["load_snapshots"]
    for idx, row in snaps.iterrows():
        if pd.isna(row["snapshot_at_utc"]):
            fatal.append({"table": "load_snapshots", "row": int(idx),
                          "severity": "fatal", "issue": "MALFORMED_TIME",
                          "detail": "snapshot_at_utc missing"})
            continue
        try:
            queue = float(row["queue_size"])
            batch = float(row["batch_load"])
            util = float(row["instrument_util"])
        except ValueError:
            fatal.append({"table": "load_snapshots", "row": int(idx),
                          "severity": "fatal", "issue": "INVALID_VALUE",
                          "detail": "non-numeric load value"})
            continue
        if queue < 0:
            fatal.append({"table": "load_snapshots", "row": int(idx),
                          "severity": "fatal", "issue": "INVALID_VALUE",
                          "detail": "negative queue_size"})
        if not 0.0 <= batch <= 1.0:
            fatal.append({"table": "load_snapshots", "row": int(idx),
                          "severity": "fatal", "issue": "INVALID_VALUE",
                          "detail": "batch_load outside [0,1]"})
        if not 0.0 <= util <= 1.0:
            fatal.append({"table": "load_snapshots", "row": int(idx),
                          "severity": "fatal", "issue": "INVALID_VALUE",
                          "detail": "instrument_util outside [0,1]"})
    return fatal


def _check_stage_map(frames: dict[str, pd.DataFrame],
                     stages_cfg: dict) -> tuple[list, list]:
    fatal: list[dict[str, Any]] = []
    warn: list[dict[str, Any]] = []
    stage_map = frames["stage_map"]
    valid_stages = set(CANONICAL_STAGES) | {STAGE_UNMAPPED}
    for idx, row in stage_map.iterrows():
        stage = row["canonical_stage"]
        if stage not in valid_stages:
            fatal.append({"table": "stage_map", "row": int(idx),
                          "severity": "fatal", "issue": "UNKNOWN_CANONICAL_STAGE",
                          "detail": f"canonical_stage {stage!r} not in frozen enum"})
        valid_from = row["valid_from"]
        valid_to = row["valid_to"]
        if (not pd.isna(valid_from)) and (not pd.isna(valid_to)) and valid_to < valid_from:
            fatal.append({"table": "stage_map", "row": int(idx),
                          "severity": "fatal", "issue": "INVERTED_TIME",
                          "detail": "valid_to earlier than valid_from"})
    # Unknown stage codes present in events (warning: replay abstains).
    mapped = set(zip(stage_map["site_key"], stage_map["workflow_version"],
                     stage_map["event_code_raw"]))
    events = frames["events"]
    cases = _first_row_lookup(frames["cases"], "case_key")
    seen: set = set()
    for idx, row in events.iterrows():
        case_key = row["case_key"]
        if case_key not in cases.index:
            continue
        case_row = cases.loc[case_key]
        key = (case_row["site_key"], case_row["workflow_version"],
               row["event_code_raw"])
        if key in seen:
            continue
        seen.add(key)
        if key not in mapped:
            warn.append({"table": "events", "row": int(idx),
                         "severity": "warning", "issue": "UNMAPPED_STAGE_CODE",
                         "detail": f"code {key[2]!r} for {key[0]}/{key[1]} has no stage_map row"})
    # Non-comparable combinations declared in config.
    for entry in stages_cfg.get("non_comparable", []):
        warn.append({"table": "stage_map", "row": -1, "severity": "warning",
                     "issue": "NOT_COMPARABLE",
                     "detail": f"{entry['site_key']} lacks {entry['canonical_stage']}: {entry.get('reason', '')}"})
    # Version validity against workflow windows.
    windows = stages_cfg.get("workflow_versions", {})
    known_versions = set(windows)
    for pos, (_, row) in enumerate(cases.iterrows()):
        version = row["workflow_version"]
        if version not in known_versions:
            warn.append({"table": "cases", "row": pos, "severity": "warning",
                         "issue": "UNKNOWN_WORKFLOW_VERSION",
                         "detail": f"workflow_version {version!r} not in dictionary"})
            continue
        window = windows[version]
        lo = pd.Timestamp(window["valid_from"], tz="UTC")
        hi = pd.Timestamp(window["valid_to"], tz="UTC")
        received = row["received_at_utc"]
        if not pd.isna(received) and not (lo <= received < hi):
            warn.append({"table": "cases", "row": pos, "severity": "warning",
                         "issue": "VERSION_WINDOW_MISMATCH",
                         "detail": f"case received outside {version} window [{window['valid_from']}, {window['valid_to']})"})
    return fatal, warn


def _table_summaries(frames: dict[str, pd.DataFrame]) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for table, frame in frames.items():
        summaries[table] = {"rows": int(len(frame)), "columns": list(frame.columns)}
    cases = frames["cases"]
    summaries["cases"]["censored"] = int((cases["is_completed"].astype(str) == "0").sum())
    summaries["cases"]["completed"] = int((cases["is_completed"].astype(str) == "1").sum())
    return summaries


def validate_frames(frames: dict[str, pd.DataFrame],
                    stages_cfg: dict) -> dict[str, Any]:
    fatal: list[dict[str, Any]] = []
    warn: list[dict[str, Any]] = []
    fatal.extend(_check_key_uniqueness(frames))
    f, w = _check_cases(frames)
    fatal.extend(f); warn.extend(w)
    f, w = _check_events(frames)
    fatal.extend(f); warn.extend(w)
    f, w = _check_requests(frames)
    fatal.extend(f); warn.extend(w)
    fatal.extend(_check_snapshots(frames))
    f, w = _check_stage_map(frames, stages_cfg)
    fatal.extend(f); warn.extend(w)
    report = {
        "tables": _table_summaries(frames),
        "fatal_issues": fatal,
        "warnings": warn,
        "fatal_count": len(fatal),
        "warning_count": len(warn),
        "validation_outcome": "FAIL" if fatal else "PASS",
    }
    return report


def validate_suite(input_dir: str | Path, stages_config_path: str | Path,
                   report_path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    try:
        frames, load_errors = io.load_all(input_dir)
    except (KeyError, FileNotFoundError) as exc:
        # Missing files or missing required columns are hard contract
        # violations; report them as fatal rather than crashing.
        report = {
            "input_dir": str(input_dir),
            "stages_config": str(stages_config_path),
            "tables": {},
            "fatal_issues": [{
                "table": "-", "row": -1, "severity": "fatal",
                "issue": "CONTRACT_VIOLATION", "detail": str(exc),
            }],
            "warnings": [],
            "load_errors": [],
            "fatal_count": 1,
            "warning_count": 0,
            "validation_outcome": "FAIL",
        }
        if report_path is not None:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
                fh.write("\n")
        return report
    with Path(stages_config_path).open(encoding="utf-8") as fh:
        stages_cfg = yaml.safe_load(fh)
    report = validate_frames(frames, stages_cfg)
    report["input_dir"] = str(input_dir)
    report["stages_config"] = str(stages_config_path)
    report["stages_config_hash"] = sha256_file(stages_config_path)
    report["load_errors"] = load_errors
    for err in load_errors:
        report["fatal_issues"].append({
            "table": "-", "row": -1, "severity": "fatal",
            "issue": "MALFORMED_TIME", "detail": err,
        })
    report["fatal_count"] = len(report["fatal_issues"])
    report["validation_outcome"] = "FAIL" if report["fatal_issues"] else "PASS"
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return report
