"""Stable run identifiers."""

from __future__ import annotations

from star_tat.common.hashing import combine_hashes


def make_run_id(project_prefix: str, data_hash: str, config_hash: str,
                seed: int, rule_version: str) -> str:
    """Deterministic run id: identical inputs yield the identical id."""
    digest = combine_hashes(data_hash, config_hash, str(seed), rule_version)
    return f"{project_prefix}_{digest[:12]}"
