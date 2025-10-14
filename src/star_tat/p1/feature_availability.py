"""Feature availability audit.

One row per model input stating where it comes from, when it is generated,
and whether it can be queried at as_of_utc. Outcome fields are listed with
outcome_only=1 and never enter any feature matrix.
"""

from __future__ import annotations

import pandas as pd

from star_tat.p1.features import NUMERIC_FEATURES

_REVIEWER = "workflow-ops-review-01"


def _rows() -> list[dict]:
    return [
        {
            "feature_name": "elapsed_hours",
            "source_table": "cases+events",
            "business_definition": "hours between case received_at_utc and as_of_utc",
            "generated_at": "at snapshot construction",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "always (received_at_utc is known at receipt)",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "hours_in_stage",
            "source_table": "events",
            "business_definition": "hours since the latest known stage event occurred",
            "generated_at": "at snapshot construction",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "requires at least one event with known_at_utc <= as_of_utc",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "stages_completed",
            "source_table": "events+stage_map",
            "business_definition": "number of canonical stages reached by known events",
            "generated_at": "at snapshot construction",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "requires dictionary row for site/workflow_version",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "stage_progress",
            "source_table": "events+stage_map",
            "business_definition": "stages_completed divided by the site's stage chain length",
            "generated_at": "at snapshot construction",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "requires dictionary row for site/workflow_version",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "queue_size_log1p",
            "source_table": "load_snapshots",
            "business_definition": "log1p of queue size from the latest load snapshot known at as_of_utc",
            "generated_at": "at snapshot time",
            "queryable_at": "snapshot_at_utc",
            "available_at_as_of": "only when a snapshot with snapshot_at_utc <= as_of_utc exists for the site/workflow_version",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "batch_load",
            "source_table": "load_snapshots",
            "business_definition": "batch load from the latest load snapshot known at as_of_utc",
            "generated_at": "at snapshot time",
            "queryable_at": "snapshot_at_utc",
            "available_at_as_of": "same as queue_size_log1p",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "instrument_util",
            "source_table": "load_snapshots",
            "business_definition": "instrument utilization from the latest load snapshot known at as_of_utc",
            "generated_at": "at snapshot time",
            "queryable_at": "snapshot_at_utc",
            "available_at_as_of": "same as queue_size_log1p",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "priority_class",
            "source_table": "cases",
            "business_definition": "case priority at receipt (STAT or ROUTINE)",
            "generated_at": "at case creation",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "always",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "test_family",
            "source_table": "cases",
            "business_definition": "test family assigned at receipt",
            "generated_at": "at case creation",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "always",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "site_key",
            "source_table": "cases",
            "business_definition": "originating site (used by pooled comparison models only)",
            "generated_at": "at case creation",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "always",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "workflow_version",
            "source_table": "cases",
            "business_definition": "workflow version in force at receipt (used by pooled comparison models only)",
            "generated_at": "at case creation",
            "queryable_at": "as_of_utc",
            "available_at_as_of": "requires dictionary row for the version",
            "outcome_only": 0,
            "reviewed_by": _REVIEWER,
        },
        {
            "feature_name": "released_at_utc",
            "source_table": "cases",
            "business_definition": "release timestamp; the prediction outcome",
            "generated_at": "at release",
            "queryable_at": "never as a feature",
            "available_at_as_of": "never; outcome only",
            "outcome_only": 1,
            "reviewed_by": _REVIEWER,
        },
    ]


def build_feature_availability() -> pd.DataFrame:
    rows = _rows()
    defined = {r["feature_name"] for r in rows}
    missing = set(NUMERIC_FEATURES) - defined
    if missing:
        raise ValueError(f"feature audit missing rows for {sorted(missing)}")
    return pd.DataFrame(rows).sort_values("feature_name")
