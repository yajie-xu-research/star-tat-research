"""As-of snapshot construction.

A snapshot is everything the system may legitimately know about a case at
``as_of_utc``: events whose ``known_at_utc <= as_of_utc`` and load snapshots
whose ``snapshot_at_utc <= as_of_utc``. Late-known events and future load
snapshots are excluded with an explicit reason. The outcome timestamp is
never present in this module's inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from star_tat.p1.constants import (
    EXCLUDE_FUTURE_SNAPSHOT,
    EXCLUDE_LATE_KNOWN_EVENT,
    EXCLUDE_MISSING_CASE_ROW,
    REASON_FEATURE_UNAVAILABLE,
    REASON_UNKNOWN_WORKFLOW_VERSION,
    REASON_UNMAPPED_STAGE,
    STAGE_UNMAPPED,
)
from star_tat.p1.stage_map import StageMap


@dataclass
class Snapshot:
    request_id: str
    case_key: str
    as_of_utc: pd.Timestamp
    site_key: str
    workflow_version: str
    test_family: str
    priority_class: str
    purpose: str
    received_at_utc: pd.Timestamp
    block: str = ""
    decision: str = ""
    reason_code: str = ""
    known_events: list[dict[str, Any]] = field(default_factory=list)
    excluded_events: list[dict[str, Any]] = field(default_factory=list)
    stage_reached: str = ""
    stage_index: int = -1
    stages_completed: int = 0
    chain_length: int = 0
    elapsed_hours: float = 0.0
    hours_in_stage: float = 0.0
    load_features: dict[str, float] | None = None
    load_snapshot_at: pd.Timestamp | None = None
    non_comparable_stages: list[str] = field(default_factory=list)


def _latest_known_load(site: str, version: str, as_of: pd.Timestamp,
                       load_df: pd.DataFrame) -> tuple[dict[str, float] | None,
                                                       pd.Timestamp | None,
                                                       list[dict[str, Any]]]:
    excluded: list[dict[str, Any]] = []
    mask = (load_df["site_key"] == site) & (load_df["workflow_version"] == version)
    rows = load_df[mask]
    for _, row in rows.iterrows():
        if row["snapshot_at_utc"] > as_of:
            excluded.append({
                "row_type": "load_snapshot",
                "key": f"{site}/{version}@{row['snapshot_at_utc'].strftime('%Y-%m-%dT%H:%M:%SZ')}",
                "reason": EXCLUDE_FUTURE_SNAPSHOT,
                "detail": ("snapshot taken after as_of_utc; not knowable at "
                           "prediction time"),
            })
    past = rows[rows["snapshot_at_utc"] <= as_of]
    if past.empty:
        return None, None, excluded
    latest = past.loc[past["snapshot_at_utc"].idxmax()]
    features = {
        "queue_size": float(latest["queue_size"]),
        "batch_load": float(latest["batch_load"]),
        "instrument_util": float(latest["instrument_util"]),
    }
    return features, latest["snapshot_at_utc"], excluded


def build_snapshot(request_row: pd.Series, case_row: pd.Series,
                   events_df: pd.DataFrame, load_df: pd.DataFrame,
                   stage_map: StageMap, require_load_features: bool) -> Snapshot:
    """Build the as-of snapshot for one prediction request.

    ``case_row`` must carry site/workflow/test-family/priority/received
    fields only; the outcome column is dropped by the caller before this
    function sees the row.
    """
    case_key = request_row["case_key"]
    as_of = request_row["as_of_utc"]
    snap = Snapshot(
        request_id=request_row["request_id"],
        case_key=case_key,
        as_of_utc=as_of,
        site_key=case_row["site_key"],
        workflow_version=case_row["workflow_version"],
        test_family=case_row["test_family"],
        priority_class=case_row["priority_class"],
        purpose=request_row["purpose"],
        received_at_utc=case_row["received_at_utc"],
    )
    if pd.isna(snap.received_at_utc):
        snap.decision = "ABSTAIN"
        snap.reason_code = REASON_FEATURE_UNAVAILABLE
        return snap

    if not stage_map.has_site_version(snap.site_key, snap.workflow_version):
        snap.decision = "ABSTAIN"
        snap.reason_code = REASON_UNKNOWN_WORKFLOW_VERSION
        return snap

    snap.chain_length = len(stage_map.chain(snap.site_key, snap.workflow_version))
    snap.non_comparable_stages = stage_map.non_comparable_stages(snap.site_key)

    events = events_df[events_df["case_key"] == case_key].sort_values(
        "occurred_at_utc")
    mapped: dict[str, pd.Timestamp] = {}
    unmapped_codes: list[str] = []
    for _, ev in events.iterrows():
        if ev["known_at_utc"] <= as_of:
            canonical = stage_map.lookup(snap.site_key, snap.workflow_version,
                                         ev["event_code_raw"])
            snap.known_events.append({
                "event_code_raw": ev["event_code_raw"],
                "occurred_at_utc": ev["occurred_at_utc"],
                "known_at_utc": ev["known_at_utc"],
                "canonical_stage": canonical,
            })
            if canonical == STAGE_UNMAPPED:
                unmapped_codes.append(ev["event_code_raw"])
            else:
                mapped.setdefault(canonical, ev["occurred_at_utc"])
        elif ev["occurred_at_utc"] <= as_of < ev["known_at_utc"]:
            # The event happened before as_of_utc but was not yet queryable:
            # a late-known event. Excluded with an explicit reason.
            snap.excluded_events.append({
                "row_type": "event",
                "case_key": case_key,
                "event_code_raw": ev["event_code_raw"],
                "occurred_at_utc": ev["occurred_at_utc"].strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "known_at_utc": ev["known_at_utc"].strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "reason": EXCLUDE_LATE_KNOWN_EVENT,
                "detail": ("event not known at as_of_utc; using it would leak "
                           "future information"),
            })
        # Events that have not occurred yet at as_of_utc are not part of the
        # snapshot; they are normal and carry no exclusion row.

    if unmapped_codes:
        snap.decision = "ABSTAIN"
        snap.reason_code = REASON_UNMAPPED_STAGE
        snap.unmapped_codes = sorted(set(unmapped_codes))
        return snap

    chain = stage_map.chain(snap.site_key, snap.workflow_version)
    reached = [s for s in chain if s in mapped]
    if reached:
        snap.stage_reached = reached[-1]
        snap.stage_index = chain.index(snap.stage_reached)
        snap.stages_completed = len(reached)
    snap.elapsed_hours = (as_of - snap.received_at_utc).total_seconds() / 3600.0
    if reached:
        last_occurred = mapped[snap.stage_reached]
        snap.hours_in_stage = (as_of - last_occurred).total_seconds() / 3600.0
    else:
        snap.hours_in_stage = snap.elapsed_hours

    snap.load_features, snap.load_snapshot_at, _ = _latest_known_load(
        snap.site_key, snap.workflow_version, as_of, load_df)

    if require_load_features and snap.load_features is None:
        snap.decision = "ABSTAIN"
        snap.reason_code = REASON_FEATURE_UNAVAILABLE
    return snap


def build_all_snapshots(requests_df: pd.DataFrame, cases_df: pd.DataFrame,
                        events_df: pd.DataFrame, load_df: pd.DataFrame,
                        stage_map: StageMap,
                        require_load_features: bool,
                        block_map: dict[str, str] | None = None) -> tuple[list[Snapshot], list[dict[str, Any]]]:
    """Build snapshots for every request; returns snapshots and excluded rows.

    The cases frame passed here must not carry the outcome column; the caller
    drops it. Missing case rows are recorded as exclusions rather than
    guessed.
    """
    snapshots: list[Snapshot] = []
    excluded: list[dict[str, Any]] = []
    case_index = cases_df.set_index("case_key", drop=False)
    for _, req in requests_df.iterrows():
        case_key = req["case_key"]
        if case_key not in case_index.index:
            excluded.append({
                "row_type": "request",
                "case_key": case_key,
                "request_id": req["request_id"],
                "reason": EXCLUDE_MISSING_CASE_ROW,
                "detail": "no matching case row",
            })
            continue
        snap = build_snapshot(req, case_index.loc[case_key], events_df,
                              load_df, stage_map, require_load_features)
        if block_map is not None:
            snap.block = block_map[case_key]
        snapshots.append(snap)
        for exc in snap.excluded_events:
            exc["request_id"] = snap.request_id
            exc["as_of_utc"] = snap.as_of_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            excluded.append(exc)
    return snapshots, excluded
