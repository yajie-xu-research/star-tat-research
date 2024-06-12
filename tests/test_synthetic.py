"""Synthetic data profile tests: volumes, injected anomalies, windows."""

from __future__ import annotations

import pandas as pd
import pytest

from star_tat.p1.synthetic import SITE_ORDER, VERSION_ORDER, generate


@pytest.fixture(scope="module")
def frames(stage_map):
    return generate(20240115, stage_map)


def test_three_sites_three_versions(frames):
    assert set(frames["cases"]["site_key"]) == set(SITE_ORDER)
    versions = set(frames["cases"]["workflow_version"]) - {"WF_V4"}
    assert versions == set(VERSION_ORDER)


def test_signed_volume_per_combo_in_range(frames):
    signed = frames["cases"][frames["cases"]["is_completed"] == 1]
    counts = signed.groupby(["site_key", "workflow_version"]).size()
    for combo in counts.index:
        if combo[1] == "WF_V4":
            continue
        assert 400 <= counts[combo] <= 700, combo


def test_total_signed_about_5000(frames):
    signed = (frames["cases"]["is_completed"] == 1).sum()
    assert 4500 <= signed <= 5500


def test_total_unsigned_about_600(frames):
    unsigned = (frames["cases"]["is_completed"] == 0).sum()
    assert 500 <= unsigned <= 700


def test_events_per_case_between_5_and_8(frames):
    counts = frames["events"].groupby("case_key").size()
    assert counts.min() >= 5
    assert counts.max() <= 8


def test_late_events_about_2_percent(frames):
    ev = frames["events"]
    delayed = (pd.to_datetime(ev["known_at_utc"])
               - pd.to_datetime(ev["occurred_at_utc"]))
    frac = float((delayed > pd.Timedelta(days=40)).mean())
    assert 0.01 <= frac <= 0.035


def test_unmapped_codes_about_1_percent_of_cases(frames):
    ev = frames["events"]
    known_codes = set(frames["stage_map"]["event_code_raw"])
    unmapped = ev[~ev["event_code_raw"].isin(known_codes)]
    affected_cases = unmapped["case_key"].nunique()
    total_cases = frames["cases"]["case_key"].nunique()
    frac = affected_cases / total_cases
    assert 0.002 <= frac <= 0.02
    assert len(unmapped) > 0


def test_site_c_stage_map_has_no_sequencing_row(frames):
    smap = frames["stage_map"]
    c = smap[smap["site_key"] == "SITE_C"]
    assert not (c["canonical_stage"] == "SEQUENCING_STARTED").any()


def test_version_windows_are_respected(frames, stage_map):
    cases = frames["cases"]
    for _, row in cases.iterrows():
        if row["workflow_version"] == "WF_V4":
            continue
        lo, hi = stage_map.version_window(row["workflow_version"])
        received = pd.Timestamp(row["received_at_utc"])
        assert lo <= received < hi


def test_unknown_version_cases_present(frames):
    assert (frames["cases"]["workflow_version"] == "WF_V4").sum() > 0


def test_load_snapshots_per_combo_present(frames):
    snaps = frames["load_snapshots"]
    combos = snaps.groupby(["site_key", "workflow_version"]).size()
    assert set(combos.index) == {(s, v) for s in SITE_ORDER
                                 for v in VERSION_ORDER}


def test_surge_window_injected(frames):
    snaps = frames["load_snapshots"]
    sat = pd.to_datetime(snaps["snapshot_at_utc"])
    surge = snaps[(snaps["site_key"] == "SITE_A")
                  & (snaps["workflow_version"] == "WF_V3")
                  & (sat >= "2026-03-10") & (sat <= "2026-03-30")]
    normal = snaps[(snaps["site_key"] == "SITE_A")
                   & (snaps["workflow_version"] == "WF_V3")
                   & (sat < "2026-03-01")]
    assert len(surge) > 0
    assert surge["queue_size"].median() > 3 * normal["queue_size"].median()


def test_release_after_receive_and_within_era(frames):
    cases = frames["cases"]
    completed = cases[cases["is_completed"] == 1]
    received = pd.to_datetime(completed["received_at_utc"])
    released = pd.to_datetime(completed["released_at_utc"])
    assert (released > received).all()
    assert released.max() <= pd.Timestamp("2026-06-30", tz="UTC")


def test_unsigned_cases_have_no_release(frames):
    cases = frames["cases"]
    unsigned = cases[cases["is_completed"] == 0]
    assert (unsigned["released_at_utc"] == "").all()


def test_every_case_has_at_least_one_request(frames):
    cases = set(frames["cases"]["case_key"])
    requests = set(frames["predict_requests"]["case_key"])
    assert cases == requests


def test_request_as_of_before_release(frames):
    cases = frames["cases"].set_index("case_key", drop=False)
    for _, req in frames["predict_requests"].iterrows():
        row = cases.loc[req["case_key"]]
        if str(row["is_completed"]) == "1":
            assert pd.Timestamp(req["as_of_utc"]) < \
                pd.Timestamp(row["released_at_utc"])


def test_all_timestamps_utc_iso(frames):
    import re
    iso = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    for col in ("received_at_utc", "released_at_utc"):
        for v in frames["cases"][col]:
            assert v == "" or iso.match(v)
    for v in frames["events"]["occurred_at_utc"]:
        assert iso.match(v)
