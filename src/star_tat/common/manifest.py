"""Run manifest construction and inspection."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from star_tat.common.hashing import combine_hashes, json_digest, sha256_file

MANIFEST_FILENAME = "manifest.json"


def current_commit() -> str:
    """Return the repository HEAD hash, or "unknown" outside a git tree."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"

# Fields that identify the deterministic computation. The result hash covers
# config + data + seed + rule/model versions + result file contents. Wall
# clock (utc), operator, and command are bookkeeping and are excluded from
# the result hash, so a rerun on identical inputs produces a byte-identical
# result hash.
RESULT_HASH_FIELDS = (
    "data_hash",
    "config_hash",
    "seed",
    "model_version",
    "rule_version",
    "result_files_hash",
)

REQUIRED_MANIFEST_FIELDS = (
    "run_id",
    "utc",
    "project",
    "status",
    "commit",
    "config_hash",
    "data_hash",
    "permissions_pointer",
    "operator",
    "input_rows",
    "excluded_rows",
    "model_or_rule_version",
    "command",
    "environment",
    "result_hash",
    "seed",
    "known_issues",
)


def build_manifest(*, run_id: str, utc: str, project: str, status: str,
                   commit: str, config_hash: str, data_hash: str,
                   permissions_pointer: str, operator: str,
                   input_rows: int, excluded_rows: int,
                   model_or_rule_version: str, command: str,
                   environment: dict, seed: int, known_issues: list,
                   result_files: dict[str, Path]) -> dict:
    """Build a manifest dict.

    ``result_files`` maps the path relative to the run directory (including
    subdirectories such as ``data/_ground_truth/true_outcomes.csv``) to the
    absolute file path. The result hash covers these files' contents plus
    the deterministic inputs; wall-clock fields do not enter it.
    """
    result_files_hash = combine_hashes(
        *[sha256_file(p) for _, p in sorted(result_files.items())]
    )
    manifest = {
        "run_id": run_id,
        "utc": utc,
        "project": project,
        "status": status,
        "commit": commit,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "permissions_pointer": permissions_pointer,
        "operator": operator,
        "input_rows": input_rows,
        "excluded_rows": excluded_rows,
        "model_or_rule_version": model_or_rule_version,
        "command": command,
        "environment": environment,
        "result_hash": combine_hashes(
            data_hash,
            config_hash,
            str(seed),
            model_or_rule_version,
            result_files_hash,
        ),
        "seed": seed,
        "known_issues": known_issues,
    }
    manifest["_result_files"] = sorted(result_files)
    return manifest


def write_manifest(path: str | Path, manifest: dict) -> str:
    """Write manifest.json and return the result hash."""
    p = Path(path) / MANIFEST_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest["result_hash"]


def load_manifest(run_dir: str | Path) -> dict:
    p = Path(run_dir) / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"no manifest at {p}")
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_manifest_fields(manifest: dict) -> list[str]:
    missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in manifest]
    return missing


def manifest_digest(manifest: dict) -> str:
    subset = {k: manifest.get(k) for k in RESULT_HASH_FIELDS}
    return json_digest(subset)
