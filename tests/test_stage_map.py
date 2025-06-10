"""Stage dictionary tests: three site vocabularies, the frozen seven-stage
chain, UNMAPPED lookups, and the NOT_COMPARABLE declaration."""

from __future__ import annotations

import pandas as pd

from star_tat.p1.constants import CANONICAL_STAGES, STAGE_UNMAPPED


def test_seven_canonical_stages_frozen(stage_map):
    assert stage_map.canonical_stages == CANONICAL_STAGES
    assert len(CANONICAL_STAGES) == 7
    assert CANONICAL_STAGES[0] == "SAMPLE_RECEIVED"
    assert CANONICAL_STAGES[-1] == "RELEASED"


def test_site_a_codes_map_to_all_seven(stage_map):
    for stage in CANONICAL_STAGES:
        assert stage_map.lookup("SITE_A", "WF_V1",
                                stage_map.site_codes["SITE_A"][stage]) == stage


def test_site_b_codes_map_to_all_seven(stage_map):
    expected = {"REC": "SAMPLE_RECEIVED", "ACC": "ACCESSIONED",
                "QC": "LIBRARY_PREPARED", "SEQ": "SEQUENCING_STARTED",
                "ANA": "ANALYSIS_COMPLETE", "CHK": "REVIEWED",
                "OUT": "RELEASED"}
    for code, stage in expected.items():
        assert stage_map.lookup("SITE_B", "WF_V2", code) == stage


def test_site_c_codes_map_without_sequencing(stage_map):
    expected = {"IN": "SAMPLE_RECEIVED", "VER": "ACCESSIONED",
                "PREP": "LIBRARY_PREPARED", "DONE": "ANALYSIS_COMPLETE",
                "SIGN": "REVIEWED", "ISSUE": "RELEASED"}
    for code, stage in expected.items():
        assert stage_map.lookup("SITE_C", "WF_V3", code) == stage


def test_site_c_chain_lacks_sequencing_started(stage_map):
    chain = stage_map.chain("SITE_C", "WF_V1")
    assert "SEQUENCING_STARTED" not in chain
    assert len(chain) == 6


def test_site_a_chain_has_seven_stages(stage_map):
    assert len(stage_map.chain("SITE_A", "WF_V1")) == 7


def test_unknown_code_is_unmapped(stage_map):
    assert stage_map.lookup("SITE_A", "WF_V1", "XADM") == STAGE_UNMAPPED


def test_unknown_version_not_known(stage_map):
    assert not stage_map.has_site_version("SITE_A", "WF_V4")
    assert "WF_V4" not in stage_map.known_versions()


def test_non_comparable_declares_site_c_sequencing(stage_map):
    assert "SEQUENCING_STARTED" in stage_map.non_comparable_stages("SITE_C")
    assert stage_map.non_comparable_stages("SITE_A") == []


def test_version_windows_staggered(stage_map):
    lo1, hi1 = stage_map.version_window("WF_V1")
    lo2, hi2 = stage_map.version_window("WF_V2")
    lo3, hi3 = stage_map.version_window("WF_V3")
    assert lo1 < hi1 <= lo2 < hi2 <= lo3 < hi3
    assert hi1 == pd.Timestamp("2024-09-01", tz="UTC")
    assert hi2 == pd.Timestamp("2025-06-01", tz="UTC")
    assert hi3 == pd.Timestamp("2026-06-30", tz="UTC")


def test_stage_index_positions(stage_map):
    assert stage_map.stage_index("SITE_A", "WF_V1", "SAMPLE_RECEIVED") == 0
    assert stage_map.stage_index("SITE_A", "WF_V1", "RELEASED") == 6
    assert stage_map.stage_index("SITE_C", "WF_V1",
                                 "SEQUENCING_STARTED") is None
