"""Fallback-chain median and stratified quantile statistics for baselines."""

from __future__ import annotations

import numpy as np

from star_tat.p1.snapshots import Snapshot

# Priority classes considered when building "similar case" groups.
_SIMILAR_PRIORITIES = ("priority_class",)


class FallbackMedianStats:
    """Median remaining hours with an explicit fallback chain:

    (site, version, priority, stage) -> (site, version, stage) ->
    (site, version) -> (version) -> global. Each step is recorded when
    used so reports can state which support a prediction relied on.
    """

    def __init__(self, groups: dict[tuple, float], fallback_order: tuple):
        self.groups = groups
        self.fallback_order = fallback_order

    def lookup(self, snap: Snapshot) -> float | None:
        stage = snap.stage_reached or "NONE"
        candidates = (
            (snap.site_key, snap.workflow_version, snap.priority_class, stage),
            (snap.site_key, snap.workflow_version, stage),
            (snap.site_key, snap.workflow_version),
            (snap.workflow_version,),
            (),
        )
        for key in candidates:
            if key in self.groups:
                return self.groups[key]
        return None


def build_fallback_median_stats(
    snapshots: list[Snapshot], targets: list[float]
) -> FallbackMedianStats:
    groups: dict[tuple, list[float]] = {}
    for snap, target in zip(snapshots, targets):
        stage = snap.stage_reached or "NONE"
        for key in (
            (snap.site_key, snap.workflow_version, snap.priority_class, stage),
            (snap.site_key, snap.workflow_version, stage),
            (snap.site_key, snap.workflow_version),
            (snap.workflow_version,),
            (),
        ):
            groups.setdefault(key, []).append(target)
    medians = {key: float(np.median(vals)) for key, vals in groups.items()}
    return FallbackMedianStats(medians, fallback_order=("priority+stage", "stage", "site+version", "version", "global"))


class StratifiedQuantileStats:
    """Empirical 5th/95th percentiles of remaining hours per
    (site, version, stage)."""

    def __init__(self, lower: dict[tuple, float], upper: dict[tuple, float]):
        self.lower = lower
        self.upper = upper

    def lookup(self, snap: Snapshot) -> tuple[float | None, float | None]:
        stage = snap.stage_reached or "NONE"
        for key in (
            (snap.site_key, snap.workflow_version, stage),
            (snap.site_key, snap.workflow_version),
        ):
            if key in self.lower:
                return self.lower[key], self.upper[key]
        return None, None


def build_stratified_quantile_stats(
    snapshots: list[Snapshot], targets: list[float]
) -> StratifiedQuantileStats:
    grouped: dict[tuple, list[float]] = {}
    for snap, target in zip(snapshots, targets):
        stage = snap.stage_reached or "NONE"
        for key in (
            (snap.site_key, snap.workflow_version, stage),
            (snap.site_key, snap.workflow_version),
        ):
            grouped.setdefault(key, []).append(target)
    lower = {key: float(np.quantile(vals, 0.05)) for key, vals in grouped.items()}
    upper = {key: float(np.quantile(vals, 0.95)) for key, vals in grouped.items()}
    return StratifiedQuantileStats(lower, upper)
