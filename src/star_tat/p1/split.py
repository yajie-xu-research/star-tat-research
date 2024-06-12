"""Data split.

Splits are stratified by site x workflow_version and are temporal within
each stratum (cases ordered by received_at_utc, then by case_key for
stability). Every request of a case_key inherits the case's block, so one
case can never appear in two blocks. Censored cases (not yet released) are
assigned to the training block where they serve as awaiting-outcome
examples; they never receive a fabricated target.
"""

from __future__ import annotations

import pandas as pd

from star_tat.p1.constants import BLOCK_CALIBRATION, BLOCK_HOLDOUT, BLOCK_TRAIN


class SplitError(ValueError):
    """Raised when a split invariant is violated."""


def assign_blocks(cases_df: pd.DataFrame, policy: dict) -> dict[str, str]:
    split_cfg = policy["split"]
    train_f = float(split_cfg["train_fraction"])
    calib_f = float(split_cfg["calibration_fraction"])
    holdout_f = float(split_cfg["holdout_fraction"])
    if abs(train_f + calib_f + holdout_f - 1.0) > 1e-9:
        raise SplitError("split fractions do not sum to 1")
    block_map: dict[str, str] = {}
    strata = cases_df.groupby(["site_key", "workflow_version"],
                              sort=False, dropna=False)
    for _, stratum in strata:
        stratum = stratum.sort_values(["received_at_utc", "case_key"],
                                      kind="stable")
        completed = stratum[stratum["is_completed"].astype(str) == "1"]
        censored = stratum[stratum["is_completed"].astype(str) != "1"]
        for case_key in censored["case_key"]:
            block_map[case_key] = BLOCK_TRAIN
        n = len(completed)
        if n == 0:
            continue
        n_train = max(1, round(n * train_f))
        n_calib = max(1, round(n * calib_f))
        n_train = min(n_train, n - 2) if n >= 3 else n
        n_calib = min(n_calib, n - n_train - (1 if n - n_train > 0 else 0))
        ordered = list(completed["case_key"])
        for case_key in ordered[:n_train]:
            block_map[case_key] = BLOCK_TRAIN
        for case_key in ordered[n_train:n_train + n_calib]:
            block_map[case_key] = BLOCK_CALIBRATION
        for case_key in ordered[n_train + n_calib:]:
            block_map[case_key] = BLOCK_HOLDOUT
    return block_map


def verify_block_consistency(block_map: dict[str, str],
                             requests_df: pd.DataFrame) -> None:
    """Every request must inherit its case's block; a case split across
    blocks is a contract violation."""
    for _, req in requests_df.iterrows():
        case_key = req["case_key"]
        if case_key not in block_map:
            raise SplitError(f"case_key {case_key!r} has no assigned block")
        # Block assignment is per case_key; there is no per-request column
        # that could disagree. The invariant that matters: two requests of
        # one case always resolve to the identical block.
    grouped = requests_df.groupby("case_key")["case_key"].count()
    for case_key in grouped.index:
        if case_key not in block_map:
            raise SplitError(f"case_key {case_key!r} missing from block map")


def check_case_overlap(blocks: dict[str, set[str]]) -> None:
    """Raise if any case_key appears in more than one block."""
    pairs = [set(v) for v in blocks.values()]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if pairs[i] & pairs[j]:
                raise SplitError(
                    f"case_key leakage between blocks: {sorted(pairs[i] & pairs[j])[:5]}"
                )


def block_counts(block_map: dict[str, str]) -> dict[str, int]:
    counts = {BLOCK_TRAIN: 0, BLOCK_CALIBRATION: 0, BLOCK_HOLDOUT: 0}
    for block in block_map.values():
        counts[block] = counts.get(block, 0) + 1
    return counts
