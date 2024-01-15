"""ISO-8601 UTC helpers. All timestamps are stored and parsed in UTC."""

from __future__ import annotations

import re

import pandas as pd

_ISO_RE = re.compile(
    r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])T"
    r"([01]\d|2[0-3]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$"
)


def is_iso8601(value: str) -> bool:
    return bool(_ISO_RE.match(str(value)))


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, format="ISO8601", utc=True)
    if ts.tz is None:
        raise ValueError(f"timestamp is not timezone-aware: {value!r}")
    return ts.tz_convert("UTC")


def format_utc(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_seconds(value: pd.Timestamp) -> float:
    return value.timestamp()
