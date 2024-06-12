"""Synthetic data generator for STAR-TAT.

Deterministic under a fixed seed. Produces the five contract tables plus the
injected data profiles required by the research plan:

  * 3 sites (SITE_A/B/C) x 3 workflow versions (WF_V1/V2/V3) with staggered
    validity windows (WF_V1 through 2024-09, WF_V2 through 2025-06, WF_V3
    from 2025-06);
  * about 5000 released cases (400-700 per site/version) and about 600
    not-yet-released cases (training-block censored examples);
  * 5-8 events per case;
  * about 2% of events known to the system only long after they occurred
    (late-known events; the replay excludes them at snapshot time);
  * about 1% of events carrying codes absent from the stage dictionary
    (UNMAPPED; affected requests abstain);
  * one site (SITE_C) whose local event stream lacks the sequencing step
    (declared NOT_COMPARABLE);
  * a load surge in a holdout window (SHIFT_DETECTED path) and a snapshot
    lag at each version rollout (FEATURE_UNAVAILABLE path);
  * a small set of cases on a workflow version absent from the dictionary
    (UNKNOWN_WORKFLOW_VERSION path).

All timestamps are UTC and bounded by 2026-06-30.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from star_tat.p1.constants import CANONICAL_STAGES
from star_tat.p1.stage_map import StageMap

SITE_ORDER = ("SITE_A", "SITE_B", "SITE_C")
VERSION_ORDER = ("WF_V1", "WF_V2", "WF_V3")

# Data window close: every timestamp in every table is bounded by this
# instant (all date concepts in this repository end at 2026-06-30).
HORIZON_END = pd.Timestamp("2026-06-30T00:00:00Z")

TEST_FAMILIES = ("ONCO_PANEL", "GERMLINE", "RARE_DISEASE", "PHARMA")
FAMILY_WEIGHTS = (0.40, 0.25, 0.25, 0.10)
FAMILY_FACTORS = {
    "ONCO_PANEL": 1.0,
    "GERMLINE": 1.3,
    "RARE_DISEASE": 1.2,
    "PHARMA": 0.85,
}
PRIORITY_CLASSES = ("ROUTINE", "STAT")
PRIORITY_FACTORS = {"ROUTINE": 1.0, "STAT": 0.45}
VERSION_FACTORS = {"WF_V1": 1.0, "WF_V2": 0.8, "WF_V3": 0.68, "WF_V4": 0.7}

# Base stage-duration gaps (hours) per site. SITE_C has no sequencing step
# in its local event stream; its PREP->DONE gap folds in the offsite run.
GAP_BASES = {
    "SITE_A": [5.0, 20.0, 28.0, 75.0, 9.0, 4.0],
    "SITE_B": [6.0, 22.0, 30.0, 80.0, 10.0, 4.0],
    "SITE_C": [5.0, 18.0, 110.0, 10.0, 4.0],
}

# Skip candidates: indices into the site code chain that may be absent from
# an individual case's event stream (codes skipped, durations kept).
SKIP_CANDIDATES = {"SITE_A": (2, 4, 5), "SITE_B": (2, 4, 5), "SITE_C": (2, 4)}

UNMAPPED_CODES = ("XADM", "XERR", "XMISC", "XQCC")

BASE_QUEUE = {"SITE_A": 260.0, "SITE_B": 190.0, "SITE_C": 320.0}

SURGE = {
    "site": "SITE_A",
    "version": "WF_V3",
    "window": ("2026-03-10", "2026-03-30"),
    "queue_mult": (3.6, 5.2),
    "util": (0.95, 1.0),
    "batch": (0.90, 1.0),
}

UNKNOWN_VERSION_CASES = {"per_site": 8, "version": "WF_V4",
                         "received_window": ("2026-01-01", "2026-06-10")}


def _site_chain(stage_map: StageMap, site: str) -> list[str]:
    return [stage_map.site_codes[site][stage]
            for stage in CANONICAL_STAGES
            if stage in stage_map.site_codes[site]]


def _hours(days: float) -> int:
    return int(round(days * 24))


class _Generator:
    def __init__(self, seed: int, stage_map: StageMap, *,
                 sites: tuple[str, ...] = SITE_ORDER,
                 versions: tuple[str, ...] = VERSION_ORDER,
                 scale: float = 1.0,
                 include_unknown_version: bool = True):
        self.rng = np.random.RandomState(seed)
        self.stage_map = stage_map
        self.sites = sites
        self.versions = versions
        self.scale = scale
        self.include_unknown_version = include_unknown_version
        self.cases: list[dict] = []
        self.events: list[dict] = []
        self.requests: list[dict] = []
        self.snapshots: list[dict] = []
        self._request_counter = 0
        self._case_counter = 0

    # -- helpers ---------------------------------------------------------

    def _fmt(self, ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fmt_date(self, ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d")

    def _make_case_key(self) -> str:
        self._case_counter += 1
        return f"CASE_{self._case_counter:08d}"

    def _make_request_id(self) -> str:
        self._request_counter += 1
        return f"REQ_{self._request_counter:07d}"

    # -- case + event generation -----------------------------------------

    def _emit_case(self, site: str, version: str, received: pd.Timestamp,
                   unsigned: bool) -> None:
        case_key = self._make_case_key()
        family = str(self.rng.choice(TEST_FAMILIES, p=FAMILY_WEIGHTS))
        priority = "STAT" if self.rng.rand() < 0.15 else "ROUTINE"
        chain = _site_chain(self.stage_map, site)
        if unsigned:
            chain = chain[:-1]
        gaps = list(GAP_BASES[site])
        if unsigned:
            gaps = gaps[:-1]

        skip_pool = [i for i in SKIP_CANDIDATES[site] if i < len(chain)]
        if unsigned:
            n_skip = int(self.rng.choice([0, 0, 0, 1]))
            # Unsigned SITE_C chains are already five codes; a skip would
            # drop them below the five-event floor.
            if site == "SITE_C":
                n_skip = 0
        else:
            n_skip = int(self.rng.choice([0, 0, 0, 1, 1, 2]))
            # SITE_C chains carry six codes; two skips would drop a case
            # below the five-event floor.
            if site == "SITE_C":
                n_skip = min(n_skip, 1)
        n_skip = min(n_skip, len(skip_pool))
        skips = set(self.rng.choice(skip_pool, size=n_skip, replace=False)) \
            if n_skip else set()

        factor = (FAMILY_FACTORS[family] * PRIORITY_FACTORS[priority]
                  * VERSION_FACTORS[version])
        durations = []
        for i, base in enumerate(gaps):
            dur = base * factor * float(self.rng.lognormal(0.0, 0.16))
            durations.append(max(1.0, dur))
        anomaly = (not unsigned) and self.rng.rand() < 0.02
        if anomaly:
            idx = int(self.rng.choice([i for i in (3, 4) if i < len(durations)]))
            durations[idx] *= float(self.rng.uniform(5.0, 14.0))

        rework_delta = 0.0
        review_idx = len(chain) - 2  # the review stage, when present
        rework = (not unsigned and review_idx >= 0
                  and review_idx not in skips
                  and self.rng.rand() < 0.25)
        if rework:
            rework_delta = float(self.rng.uniform(4.0, 18.0))

        total_hours = sum(durations) + rework_delta
        released = received + pd.Timedelta(hours=total_hours)

        # Event rows: (code, hours_from_received, record_version)
        rows: list[tuple[str, float, int]] = []
        t = 0.0
        for i, code in enumerate(chain):
            if i == 0:
                rows.append((code, 0.0, 1))
                continue
            t += durations[i - 1]
            if i in skips:
                continue
            rows.append((code, t, 1))
        if rework and review_idx < len(chain):
            rev_code = chain[review_idx]
            t_rev = sum(durations[:review_idx]) if review_idx > 0 else 0.0
            # Shift events strictly after the review point, then insert the
            # re-review event.
            rows = [(c, h + (rework_delta if h > t_rev + 1e-9 else 0.0), v)
                    for (c, h, v) in rows]
            rows.append((rev_code, t_rev + rework_delta, 2))
            rows.sort(key=lambda r: (r[1], r[2]))

        has_unmapped = self.rng.rand() < 0.01 and len(rows) < 8
        if has_unmapped:
            code = str(self.rng.choice(UNMAPPED_CODES))
            rows.append((code, float(self.rng.uniform(0.25, 0.65)) * total_hours, 1))

        self.cases.append({
            "case_key": case_key,
            "site_key": site,
            "workflow_version": version,
            "test_family": family,
            "priority_class": priority,
            "received_at_utc": self._fmt(received),
            "released_at_utc": self._fmt(released) if not unsigned else "",
            "is_completed": 0 if unsigned else 1,
        })

        for pos, (code, hours_off, rec_version) in enumerate(rows):
            occurred = received + pd.Timedelta(hours=hours_off)
            is_final = ((not unsigned) and code == chain[-1]
                        and rec_version == 1)
            is_unmapped = code in UNMAPPED_CODES
            late = (pos > 0 and not is_final and not is_unmapped
                    and self.rng.rand() < 0.028)
            if late:
                known = occurred + pd.Timedelta(
                    days=float(self.rng.uniform(60, 240)))
            else:
                known = occurred + pd.Timedelta(hours=float(self.rng.uniform(0, 6)))
            # The data window closes at HORIZON_END; a "late" entry delay may
            # not push knowledge past it.
            if known > HORIZON_END:
                known = HORIZON_END
            source = f"LIS-{site}"
            seq_code = chain[3] if len(chain) > 3 else None
            if seq_code is not None and code == seq_code:
                source = f"WB-{site}"
            self.events.append({
                "case_key": case_key,
                "event_code_raw": code,
                "occurred_at_utc": self._fmt(occurred),
                "known_at_utc": self._fmt(known),
                "source_system": source,
                "record_version": rec_version,
            })

        # prediction requests
        if unsigned:
            last_hours = rows[-1][1] if rows else 24.0
            as_of = received + pd.Timedelta(
                hours=last_hours + float(self.rng.uniform(2, 8)))
            self.requests.append({
                "case_key": case_key,
                "as_of_utc": self._fmt(as_of),
                "request_id": self._make_request_id(),
                "purpose": self._purpose(),
            })
        else:
            f1 = float(self.rng.uniform(0.40, 0.72))
            as_of1 = received + pd.Timedelta(hours=f1 * total_hours)
            self.requests.append({
                "case_key": case_key,
                "as_of_utc": self._fmt(as_of1),
                "request_id": self._make_request_id(),
                "purpose": self._purpose(),
            })
            if self.rng.rand() < 0.25:
                f2 = float(self.rng.uniform(0.18, 0.38))
                as_of2 = received + pd.Timedelta(hours=f2 * total_hours)
                self.requests.append({
                    "case_key": case_key,
                    "as_of_utc": self._fmt(as_of2),
                    "request_id": self._make_request_id(),
                    "purpose": self._purpose(),
                })

    def _purpose(self) -> str:
        return str(self.rng.choice(
            ["OPERATIONAL_ETA"] * 8 + ["PILOT_CHECK"] + ["RETRO_AUDIT"]))

    # -- load snapshots ----------------------------------------------------

    def _emit_snapshots(self, site: str, version: str) -> None:
        start, end = self.stage_map.version_window(version)
        first = start + pd.Timedelta(days=float(self.rng.uniform(6, 14)))
        t = first
        base = BASE_QUEUE[site] * VERSION_FACTORS[version]
        surge = (SURGE["site"] == site and SURGE["version"] == version)
        surge_lo, surge_hi = (pd.Timestamp(SURGE["window"][0], tz="UTC"),
                              pd.Timestamp(SURGE["window"][1], tz="UTC"))
        record_version = 0
        while t < end:
            record_version += 1
            queue = base * float(self.rng.lognormal(0.0, 0.15))
            batch = float(self.rng.uniform(0.3, 0.9))
            util = float(self.rng.uniform(0.4, 0.95))
            if surge and surge_lo <= t <= surge_hi:
                queue *= float(self.rng.uniform(*SURGE["queue_mult"]))
                util = float(self.rng.uniform(*SURGE["util"]))
                batch = float(self.rng.uniform(*SURGE["batch"]))
            self.snapshots.append({
                "site_key": site,
                "workflow_version": version,
                "snapshot_at_utc": self._fmt(t),
                "queue_size": int(round(queue)),
                "batch_load": round(batch, 4),
                "instrument_util": round(util, 4),
                "source_system": f"OPS-{site}",
                "record_version": record_version,
            })
            t += pd.Timedelta(days=float(self.rng.uniform(2, 4)))

    # -- stage map ---------------------------------------------------------

    def _stage_map_rows(self) -> list[dict]:
        rows: list[dict] = []
        for site in SITE_ORDER:
            for version in VERSION_ORDER:
                lo, hi = self.stage_map.version_window(version)
                for stage in CANONICAL_STAGES:
                    if stage not in self.stage_map.site_codes[site]:
                        continue
                    code = self.stage_map.site_codes[site][stage]
                    rows.append({
                        "site_key": site,
                        "workflow_version": version,
                        "event_code_raw": code,
                        "canonical_stage": stage,
                        "valid_from": self._fmt_date(lo),
                        "valid_to": self._fmt_date(hi),
                        "reviewed_by": self.stage_map.reviewed_by,
                    })
        return rows

    # -- main ----------------------------------------------------------------

    def run(self) -> dict[str, pd.DataFrame]:
        # Released cases per site/version.
        for site in SITE_ORDER:
            if site not in self.sites:
                continue
            for version in VERSION_ORDER:
                if version not in self.versions:
                    continue
                lo, hi = self.stage_map.version_window(version)
                n_signed = max(30, int(self.rng.randint(400, 701)
                                       * self.scale))
                for _ in range(n_signed):
                    received = lo + pd.Timedelta(
                        hours=float(self.rng.uniform(
                            0, (hi - lo).total_seconds() / 3600.0 - 720.0)))
                    self._emit_case(site, version, received, unsigned=False)
                # Not-yet-released cases: none for WF_V1, a small backlog for
                # WF_V2, the bulk for WF_V3.
                if version == "WF_V2":
                    n_unsigned = max(5, int(self.rng.randint(40, 51)
                                            * self.scale))
                elif version == "WF_V3":
                    n_unsigned = max(10, int(self.rng.randint(150, 161)
                                             * self.scale))
                else:
                    n_unsigned = 0
                for _ in range(n_unsigned):
                    received = lo + pd.Timedelta(
                        hours=float(self.rng.uniform(
                            0, (hi - lo).total_seconds() / 3600.0 - 720.0)))
                    self._emit_case(site, version, received, unsigned=True)
            # A small set on a workflow version absent from the dictionary.
            if not self.include_unknown_version:
                continue
            per_site = max(2, int(UNKNOWN_VERSION_CASES["per_site"]
                                  * self.scale))
            lo_u, hi_u = (pd.Timestamp(UNKNOWN_VERSION_CASES["received_window"][0], tz="UTC"),
                          pd.Timestamp(UNKNOWN_VERSION_CASES["received_window"][1], tz="UTC"))
            for _ in range(per_site):
                received = lo_u + pd.Timedelta(
                    hours=float(self.rng.uniform(0, (hi_u - lo_u).total_seconds() / 3600.0)))
                self._emit_case(site, UNKNOWN_VERSION_CASES["version"],
                                received, unsigned=False)

        for site in SITE_ORDER:
            if site not in self.sites:
                continue
            for version in VERSION_ORDER:
                if version not in self.versions:
                    continue
                self._emit_snapshots(site, version)

        cases = pd.DataFrame(self.cases).sort_values("case_key")
        events = pd.DataFrame(self.events).sort_values(
            ["case_key", "occurred_at_utc", "event_code_raw",
             "record_version"])
        requests = pd.DataFrame(self.requests).sort_values("request_id")
        snapshots = pd.DataFrame(self.snapshots).sort_values(
            ["site_key", "workflow_version", "snapshot_at_utc"])
        stage_map = pd.DataFrame(self._stage_map_rows()).sort_values(
            ["site_key", "workflow_version", "event_code_raw"])
        return {
            "cases": cases,
            "events": events,
            "load_snapshots": snapshots,
            "stage_map": stage_map,
            "predict_requests": requests,
        }


def generate(seed: int, stage_map: StageMap, *,
             sites: tuple[str, ...] = SITE_ORDER,
             versions: tuple[str, ...] = VERSION_ORDER,
             scale: float = 1.0,
             include_unknown_version: bool = True) -> dict[str, pd.DataFrame]:
    return _Generator(seed, stage_map, sites=sites, versions=versions,
                      scale=scale,
                      include_unknown_version=include_unknown_version).run()


def write_tables(output_dir: str | Path, frames: dict[str, pd.DataFrame],
                 seed: int) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    filenames = {
        "cases": "cases.csv",
        "events": "events.csv",
        "load_snapshots": "load_snapshots.csv",
        "stage_map": "stage_map.csv",
        "predict_requests": "predict_requests.csv",
    }
    for table, filename in filenames.items():
        path = out / filename
        frames[table].to_csv(path, index=False)
        paths[table] = path
    return paths
