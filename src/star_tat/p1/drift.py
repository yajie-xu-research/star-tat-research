"""Pre-result drift checks.

Before outcomes exist, only the load regime and stage distribution can be
watched. A request whose load features fall outside the validated range of
its stratum's training block abstains with SHIFT_DETECTED. No future
information is consulted.
"""

from __future__ import annotations

from star_tat.p1.calibration import StratumModel
from star_tat.p1.constants import REASON_SHIFT_DETECTED
from star_tat.p1.snapshots import Snapshot


def check_shift(snapshot: Snapshot, model: StratumModel) -> bool:
    if model.insufficient:
        return False
    if snapshot.load_features is None:
        return False
    encoder = model.encoder
    names = ("queue_size_log1p", "batch_load", "instrument_util")
    idx = {name: encoder.numeric.index(name) for name in names}
    values = encoder.transform(snapshot)
    for name in names:
        lo, hi = model.train_ranges.get(name, (float("-inf"), float("inf")))
        value = values[idx[name]]
        if value < lo or value > hi:
            return True
    return False


def apply_shift_checks(snapshots: list[Snapshot],
                       models: dict[tuple[str, str], StratumModel]) -> int:
    """Mark out-of-range snapshots ABSTAIN/SHIFT_DETECTED; return count."""
    flagged = 0
    for snap in snapshots:
        if snap.decision == "ABSTAIN":
            continue
        model = models.get((snap.site_key, snap.workflow_version))
        if model is None:
            continue
        if check_shift(snap, model):
            snap.decision = "ABSTAIN"
            snap.reason_code = REASON_SHIFT_DETECTED
            flagged += 1
    return flagged
