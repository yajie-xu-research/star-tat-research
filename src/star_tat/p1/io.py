"""Loading and strict parsing of the five contract tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from star_tat.common.timeutil import is_iso8601, parse_utc

TABLE_FILES = {
    "cases": "cases.csv",
    "events": "events.csv",
    "load_snapshots": "load_snapshots.csv",
    "stage_map": "stage_map.csv",
    "predict_requests": "predict_requests.csv",
}

_TIME_COLUMNS = {
    "cases": ["received_at_utc", "released_at_utc"],
    "events": ["occurred_at_utc", "known_at_utc"],
    "load_snapshots": ["snapshot_at_utc"],
    "stage_map": ["valid_from", "valid_to"],
    "predict_requests": ["as_of_utc"],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def schema_dir() -> Path:
    return repo_root() / "schemas"


def load_schema(table: str) -> dict:
    with (schema_dir() / f"{table}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def schema_columns(table: str) -> list[str]:
    return [c["name"] for c in load_schema(table)["columns"]]


def required_columns(table: str) -> list[str]:
    return [c["name"] for c in load_schema(table)["columns"] if c.get("required")]


def unique_columns(table: str) -> list[str]:
    return [c["name"] for c in load_schema(table)["columns"] if c.get("unique")]


def primary_keys(table: str) -> list[str]:
    return load_schema(table).get("primary_key", [])


def load_table_raw(input_dir: str | Path, table: str) -> pd.DataFrame:
    path = Path(input_dir) / TABLE_FILES[table]
    if not path.exists():
        raise FileNotFoundError(f"missing table file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _parse_time_columns(frame: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse time columns strictly; collect rows with malformed timestamps.

    Date-only values (YYYY-MM-DD) are accepted for date-typed columns
    (stage_map validity windows) and interpreted as midnight UTC.
    """
    import re as _re

    _DATE_ONLY = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    errors: list[str] = []
    parsed = {}
    for col in _TIME_COLUMNS.get(table, []):
        if col not in frame.columns:
            continue
        out = []
        for idx, value in enumerate(frame[col]):
            if value == "":
                out.append(pd.NaT)
                continue
            if _DATE_ONLY.match(value):
                value = value + "T00:00:00Z"
            if not is_iso8601(value):
                errors.append(
                    f"{table}:{col} row {idx} malformed timestamp {value!r}"
                )
                out.append(pd.NaT)
                continue
            out.append(parse_utc(value))
        parsed[col] = out
    if parsed:
        for col, values in parsed.items():
            frame = frame.copy()
            frame[col] = values
    return frame, errors


def load_table(input_dir: str | Path, table: str) -> tuple[pd.DataFrame, list[str]]:
    frame = load_table_raw(input_dir, table)
    missing = [c for c in required_columns(table) if c not in frame.columns]
    if missing:
        raise KeyError(f"{table}: missing required columns {missing}")
    return _parse_time_columns(frame, table)


def load_all(input_dir: str | Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for table in TABLE_FILES:
        frame, errs = load_table(input_dir, table)
        frames[table] = frame
        errors.extend(errs)
    return frames, errors
