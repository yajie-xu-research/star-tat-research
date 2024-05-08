"""Targets.

The learning target for completed cases is
``remaining_hours = released_at_utc - as_of_utc``. Cases that are not
released are censored: their target is None and they are never assigned a
zero duration. Requests at or after the release timestamp are excluded.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from star_tat.p1.constants import (
    EXCLUDE_AFTER_RELEASE,
    EXCLUDE_NO_TARGET,
)
from star_tat.p1.snapshots import Snapshot


def target_for_snapshot(snap: Snapshot, cases_df: pd.DataFrame) -> tuple[float | None, str]:
    """Return (remaining_hours or None, status) for a snapshot."""
    case_row = cases_df.loc[snap.case_key]
    if str(case_row["is_completed"]).strip() != "1":
        return None, "censored"
    released = case_row["released_at_utc"]
    if pd.isna(released):
        return None, "censored"
    if snap.as_of_utc >= released:
        return None, EXCLUDE_AFTER_RELEASE
    return (released - snap.as_of_utc).total_seconds() / 3600.0, "completed"


def build_training_targets(snapshots: list[Snapshot], cases_df: pd.DataFrame,
                           allowed_blocks: tuple[str, ...]) -> tuple[list[Snapshot], list[float], list[dict[str, Any]]]:
    """Rows usable as fit inputs: completed targets in the allowed blocks.

    Censored rows and post-release requests are reported, never zeroed.
    """
    fit_rows: list[Snapshot] = []
    targets: list[float] = []
    excluded: list[dict[str, Any]] = []
    for snap in snapshots:
        if snap.block not in allowed_blocks:
            continue
        if snap.decision != "OUTPUT" and snap.reason_code:
            continue
        target, status = target_for_snapshot(snap, cases_df)
        if status == "completed" and target is not None:
            fit_rows.append(snap)
            targets.append(target)
        elif status == "censored":
            excluded.append({
                "row_type": "request", "request_id": snap.request_id,
                "case_key": snap.case_key, "reason": "CENSORED",
                "detail": "case not released; awaiting outcome, target kept unset",
            })
        elif status == EXCLUDE_AFTER_RELEASE:
            excluded.append({
                "row_type": "request", "request_id": snap.request_id,
                "case_key": snap.case_key, "reason": EXCLUDE_AFTER_RELEASE,
                "detail": "request placed at or after the release timestamp",
            })
    return fit_rows, targets, excluded
