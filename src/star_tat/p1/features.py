"""Feature construction.

Every feature is computable from information knowable at as_of_utc. The
outcome (released_at_utc) never appears here; a name guard rejects any
feature whose name mentions it, and tests assert the full feature list.

Preprocessing (scaling, one-hot frame) is fit on the training block only and
applied to the other blocks to avoid leakage.
"""

from __future__ import annotations

import numpy as np

from star_tat.p1.snapshots import Snapshot

# Feature families. Load-regime features come from the latest load snapshot
# known at as_of_utc; stage features from known events only.
NUMERIC_FEATURES = (
    "elapsed_hours",
    "hours_in_stage",
    "stages_completed",
    "stage_progress",
    "queue_size_log1p",
    "batch_load",
    "instrument_util",
)

CATEGORICAL_FEATURES = ("priority_class", "test_family")
POOLED_CATEGORICAL = ("site_key", "workflow_version")

_FORBIDDEN_NAME_FRAGMENTS = ("released", "release", "outcome", "truth")


def assert_feature_names_clean(names: list[str]) -> None:
    for name in names:
        lowered = name.lower()
        for fragment in _FORBIDDEN_NAME_FRAGMENTS:
            if fragment in lowered:
                raise ValueError(
                    f"feature name {name!r} contains forbidden fragment "
                    f"{fragment!r}; outcomes must not enter features"
                )


def _numeric_vector(snap: Snapshot, expected: tuple[str, ...]) -> list[float]:
    load = snap.load_features or {}
    values: dict[str, float] = {
        "elapsed_hours": snap.elapsed_hours,
        "hours_in_stage": snap.hours_in_stage,
        "stages_completed": float(snap.stages_completed),
        "stage_progress": (snap.stages_completed / snap.chain_length
                           if snap.chain_length else 0.0),
        "queue_size_log1p": np.log1p(load.get("queue_size", 0.0)),
        "batch_load": load.get("batch_load", 0.0),
        "instrument_util": load.get("instrument_util", 0.0),
    }
    return [values[name] for name in expected]


class FeatureEncoder:
    """One-hot encoder for categorical columns, fit once on train data."""

    def __init__(self, columns: list[str], include_pooled: bool):
        self.columns = list(columns)
        if include_pooled:
            self.columns = self.columns + list(POOLED_CATEGORICAL)
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []
        self.numeric = list(NUMERIC_FEATURES)

    @property
    def total_features(self) -> int:
        return len(self.numeric) + len(self.feature_names)

    def fit(self, snapshots: list[Snapshot]) -> "FeatureEncoder":
        for col in self.columns:
            cats = sorted({getattr(s, col) for s in snapshots})
            self.categories[col] = cats
            self.feature_names.extend(
                f"{col}={c}" for c in cats if c
            )
        return self

    def transform(self, snapshot: Snapshot) -> np.ndarray:
        row: list[float] = _numeric_vector(snapshot, self.numeric)
        for col in self.columns:
            row.extend(1.0 if getattr(snapshot, col) == c else 0.0
                       for c in self.categories[col] if c)
        return np.asarray(row, dtype=float)

    def all_feature_names(self) -> list[str]:
        names = list(self.numeric) + list(self.feature_names)
        assert_feature_names_clean(names)
        return names
