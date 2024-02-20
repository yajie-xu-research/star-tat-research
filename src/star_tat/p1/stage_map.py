"""Stage dictionary lookup.

The mapping from raw event codes to canonical stages is read from
configs/p1_stages.yaml (the frozen dictionary). Lookups are qualified by
site, workflow version, and event time; a code without a mapping row is
UNMAPPED and never guessed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from star_tat.p1.constants import CANONICAL_STAGES, STAGE_UNMAPPED


class StageMap:
    def __init__(self, config_path: str | Path):
        with Path(config_path).open(encoding="utf-8") as fh:
            self.cfg = yaml.safe_load(fh)
        self.canonical_stages: tuple[str, ...] = tuple(self.cfg["canonical_stages"])
        self.site_codes: dict[str, dict[str, str]] = self.cfg["site_event_codes"]
        self.version_windows: dict[str, dict[str, str]] = {
            k: dict(v) for k, v in self.cfg.get("workflow_versions", {}).items()
        }
        self.non_comparable: list[dict[str, Any]] = list(
            self.cfg.get("non_comparable", [])
        )
        self.reviewed_by: str = self.cfg.get("reviewed_by", "")
        # (site, version, code) -> canonical stage
        self._table: dict[tuple[str, str, str], str] = {}
        for site, code_map in self.site_codes.items():
            for version in self.version_windows:
                for canonical, code in code_map.items():
                    self._table[(site, version, code)] = canonical
        # canonical stage chain per (site, version): SITE_C lacks the
        # sequencing step (declared non-comparable), so its chain is shorter.
        self._chains: dict[tuple[str, str], tuple[str, ...]] = {}
        for site in self.site_codes:
            for version in self.version_windows:
                stages = [s for s in self.canonical_stages if s in self.site_codes[site]]
                self._chains[(site, version)] = tuple(stages)

    def known_versions(self) -> set[str]:
        return set(self.version_windows)

    def version_window(self, version: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        window = self.version_windows[version]
        return (pd.Timestamp(window["valid_from"], tz="UTC"),
                pd.Timestamp(window["valid_to"], tz="UTC"))

    def has_site_version(self, site: str, version: str) -> bool:
        return site in self.site_codes and version in self.version_windows

    def chain(self, site: str, version: str) -> tuple[str, ...]:
        return self._chains[(site, version)]

    def lookup(self, site: str, version: str, code: str) -> str:
        return self._table.get((site, version, code), STAGE_UNMAPPED)

    def non_comparable_stages(self, site: str) -> list[str]:
        return [e["canonical_stage"] for e in self.non_comparable
                if e["site_key"] == site]

    def stage_index(self, site: str, version: str, stage: str) -> int | None:
        chain = self.chain(site, version)
        return chain.index(stage) if stage in chain else None
