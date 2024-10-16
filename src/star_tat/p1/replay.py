"""Replay orchestration: from the five tables to per-request intervals.

Pipeline:
  1. validate the input tables (fatal issues abort);
  2. split cases into train / calibration / holdout blocks (stratified by
     site x workflow_version, temporal within stratum, censored cases to
     the training block);
  3. build one as-of snapshot per prediction request, excluding late-known
     events and future load snapshots with explicit reasons (the outcome
     column is dropped before snapshots are built, and the whole step runs
     under the isolation guard);
  4. fit per-stratum point models on the training block and calibration
     quantiles on the calibration block; fit the five baselines;
  5. run pre-result shift checks and abstention rules;
  6. emit intervals and baseline predictions for every request;
  7. write results, excluded rows, calibration metadata, the feature audit,
     the data quality report, the ground-truth outcomes (physically
     separated), a run log, and the manifest.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from star_tat.common.environment import collect_environment
from star_tat.common.hashing import (combine_hashes, hash_directory,
                                     sha256_file)
from star_tat.common.isolation import isolation_context
from star_tat.common.manifest import build_manifest, current_commit, write_manifest
from star_tat.common.run_id import make_run_id
from star_tat.p1 import io
from star_tat.p1.baselines import BASELINE_NAMES, BaselineBundle
from star_tat.p1.calibration import StratumModel, fit_stratum
from star_tat.p1.constants import (
    BLOCK_CALIBRATION,
    BLOCK_HOLDOUT,
    BLOCK_TRAIN,
    DECISION_ABSTAIN,
    DECISION_OUTPUT,
    REASON_INSUFFICIENT_CALIBRATION,
    RUN_STATUS_INTERNAL,
)
from star_tat.p1.drift import apply_shift_checks
from star_tat.p1.feature_availability import build_feature_availability
from star_tat.p1.snapshots import Snapshot, build_all_snapshots
from star_tat.p1.split import assign_blocks, verify_block_consistency
from star_tat.p1.stage_map import StageMap
from star_tat.p1.targets import build_training_targets
from star_tat.p1.validation import validate_frames

GROUND_TRUTH_DIR = "_ground_truth"
GROUND_TRUTH_FILE = "true_outcomes.csv"


class ReplayError(RuntimeError):
    pass


def _load_policy(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _stratum_key(site: str, version: str) -> tuple[str, str]:
    return (site, version)


def _round_hours(value: float) -> float:
    return round(value, 4)


def run_replay(*, input_dir: str | Path, as_of_file: str | Path,
               out_dir: str | Path, stages_config: str | Path,
               policy_config: str | Path, seed: int, utc_now: str,
               command: str) -> dict:
    log: list[str] = []

    def log_line(msg: str) -> None:
        log.append(f"{msg}")

    stage_map = StageMap(stages_config)
    policy = _load_policy(policy_config)

    frames, _ = io.load_all(input_dir)
    report = validate_frames(frames, stage_map.cfg)
    if report["validation_outcome"] == "FAIL":
        raise ReplayError(
            f"input validation failed with {report['fatal_count']} fatal issues"
        )

    cases = frames["cases"].sort_values("case_key").reset_index(drop=True)
    events = frames["events"]
    load = frames["load_snapshots"]
    requests = pd.read_csv(as_of_file, dtype=str, keep_default_na=False)
    for col in ("as_of_utc",):
        requests[col] = requests[col].apply(
            lambda v: pd.to_datetime(v, format="ISO8601", utc=True))

    data_hash = combine_hashes(
        *[sha256_file(Path(input_dir) / f) for f in io.TABLE_FILES.values()],
        sha256_file(as_of_file),
    )
    config_hash = combine_hashes(sha256_file(stages_config),
                                 sha256_file(policy_config))
    rule_version = (
        f"{policy['event_dictionary_version']}/"
        f"{policy['stage_mapping_version']}/"
        f"{policy['model_version']}/"
        f"{policy['calibration_version']}"
    )
    run_id = make_run_id("P1R", data_hash, config_hash, seed, rule_version)
    log_line(f"run_id={run_id}")
    log_line(f"seed={seed} data_hash={data_hash[:16]} "
             f"config_hash={config_hash[:16]}")
    log_line(f"cases={len(cases)} events={len(events)} "
             f"load_snapshots={len(load)} requests={len(requests)}")

    # Split. Censored cases land in the training block.
    block_map = assign_blocks(cases, policy)
    verify_block_consistency(block_map, requests)
    from star_tat.p1.split import block_counts
    counts = block_counts(block_map)
    log_line(f"block_counts={counts}")

    # Outcome column is dropped before snapshot construction; the snapshot
    # builder never sees released_at_utc.
    cases_for_snapshots = cases.drop(columns=["released_at_utc"])

    with isolation_context():
        snapshots, excluded_rows = build_all_snapshots(
            requests, cases_for_snapshots, events, load, stage_map,
            bool(policy["require_load_features"]), block_map)

    # Fit inputs: training/calibration rows with completed targets.
    cases_indexed = cases.set_index("case_key", drop=False)
    fit_rows, targets, fit_excluded = build_training_targets(
        snapshots, cases_indexed, (BLOCK_TRAIN, BLOCK_CALIBRATION))
    excluded_rows.extend(fit_excluded)

    train_rows = [s for s, t in zip(fit_rows, targets) if s.block == BLOCK_TRAIN]
    train_targets = [t for s, t in zip(fit_rows, targets) if s.block == BLOCK_TRAIN]
    calib_rows = [s for s, t in zip(fit_rows, targets)
                  if s.block == BLOCK_CALIBRATION]
    calib_targets = [t for s, t in zip(fit_rows, targets)
                     if s.block == BLOCK_CALIBRATION]
    log_line(f"fit_rows train={len(train_rows)} calib={len(calib_rows)}")

    # Per-stratum models.
    models: dict[tuple[str, str], StratumModel] = {}
    strata = {}
    for key in sorted({(s.site_key, s.workflow_version)
                       for s in train_rows + calib_rows}):
        site, version = key
        t_rows = [s for s in train_rows
                  if (s.site_key, s.workflow_version) == key]
        t_tgt = [t for s, t in zip(train_rows, train_targets)
                 if (s.site_key, s.workflow_version) == key]
        c_rows = [s for s in calib_rows
                  if (s.site_key, s.workflow_version) == key]
        c_tgt = [t for s, t in zip(calib_rows, calib_targets)
                 if (s.site_key, s.workflow_version) == key]
        model = fit_stratum(site, version, t_rows, t_tgt, c_rows, c_tgt, policy)
        models[key] = model
        strata[f"{site}/{version}"] = {
            "site_key": site, "workflow_version": version,
            "n_train": model.n_train, "n_calibration": model.n_calibration,
            "q": _round_hours(model.q) if not model.insufficient else None,
            "conformal_q": (_round_hours(model.conformal_q)
                            if not model.insufficient else None),
            "insufficient": model.insufficient,
            "coverage_target": model.coverage_target,
            "train_ranges": {k: [round(a, 4), round(b, 4)]
                             for k, (a, b) in model.train_ranges.items()},
        }
        log_line(f"stratum {site}/{version}: n_train={model.n_train} "
                 f"n_calib={model.n_calibration} "
                 f"insufficient={model.insufficient}")

    baselines = BaselineBundle().fit(train_rows, train_targets,
                                     calib_rows, calib_targets, policy)
    log_line("baselines fit: " + ", ".join(BASELINE_NAMES))

    shift_flags = apply_shift_checks(snapshots, models)
    log_line(f"shift_flags={shift_flags}")

    # Predictions for every snapshot.
    result_rows: list[dict] = []
    for snap in sorted(snapshots, key=lambda s: s.request_id):
        row = {
            "request_id": snap.request_id,
            "case_key": snap.case_key,
            "as_of_utc": snap.as_of_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "block": snap.block,
            "site_key": snap.site_key,
            "workflow_version": snap.workflow_version,
            "test_family": snap.test_family,
            "priority_class": snap.priority_class,
            "purpose": snap.purpose,
            "stage_reached": snap.stage_reached or "",
            "stages_completed": snap.stages_completed,
            "elapsed_hours": _round_hours(snap.elapsed_hours),
            "decision": snap.decision or DECISION_OUTPUT,
            "reason_code": snap.reason_code or "",
        }
        if snap.decision == DECISION_ABSTAIN:
            row.update({
                "predicted_remaining_hours": "",
                "predicted_release_lower_utc": "",
                "predicted_release_upper_utc": "",
                "conformal_lower_utc": "",
                "conformal_upper_utc": "",
            })
            for b in BASELINE_NAMES:
                row[f"baseline_{b}"] = ""
            result_rows.append(row)
            continue

        model = models.get((snap.site_key, snap.workflow_version))
        if model is None or model.insufficient:
            row["decision"] = DECISION_ABSTAIN
            row["reason_code"] = REASON_INSUFFICIENT_CALIBRATION
            row.update({
                "predicted_remaining_hours": "",
                "predicted_release_lower_utc": "",
                "predicted_release_upper_utc": "",
                "conformal_lower_utc": "",
                "conformal_upper_utc": "",
            })
            for b in BASELINE_NAMES:
                row[f"baseline_{b}"] = ""
            result_rows.append(row)
            continue

        point, lower, upper = model.predict(snap)
        conf_lower, conf_upper = model.conformal_interval(snap)
        base = snap.as_of_utc
        row.update({
            "decision": DECISION_OUTPUT,
            "predicted_remaining_hours": _round_hours(point),
            "predicted_release_lower_utc": (
                base + pd.Timedelta(hours=lower)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "predicted_release_upper_utc": (
                base + pd.Timedelta(hours=upper)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conformal_lower_utc": (
                base + pd.Timedelta(hours=conf_lower)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conformal_upper_utc": (
                base + pd.Timedelta(hours=conf_upper)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        bpred = baselines.predict(snap)
        row["baseline_median_history"] = (
            _round_hours(bpred["median_history"])
            if bpred.get("median_history") is not None else "")
        if bpred.get("stratified_quantile_lower") is not None:
            row["baseline_stratified_quantiles"] = (
                f"{_round_hours(bpred['stratified_quantile_lower'])}|"
                f"{_round_hours(bpred['stratified_quantile_upper'])}")
        else:
            row["baseline_stratified_quantiles"] = ""
        row["baseline_raw_timestamp_point"] = (
            _round_hours(bpred["raw_timestamp_point"])
            if bpred.get("raw_timestamp_point") is not None else "")
        row["baseline_tree_point"] = (
            _round_hours(bpred["tree_point"])
            if bpred.get("tree_point") is not None else "")
        if bpred.get("unstratified_lower") is not None:
            row["baseline_unstratified_interval"] = (
                f"{_round_hours(bpred['unstratified_lower'])}|"
                f"{_round_hours(bpred['unstratified_upper'])}")
        else:
            row["baseline_unstratified_interval"] = ""
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    excluded_df = pd.DataFrame(excluded_rows)
    if excluded_df.empty:
        excluded_df = pd.DataFrame(
            columns=["row_type", "request_id", "case_key", "reason", "detail"])

    # Ground truth outcomes, physically separated.
    out_dir = Path(out_dir)
    gt_dir = out_dir / "data" / GROUND_TRUTH_DIR
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt = cases[["case_key", "site_key", "workflow_version",
                "received_at_utc", "released_at_utc", "is_completed"]].copy()
    gt["received_at_utc"] = gt["received_at_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    gt["released_at_utc"] = gt["released_at_utc"].apply(
        lambda v: v.strftime("%Y-%m-%dT%H:%M:%SZ") if not pd.isna(v) else "")
    gt.to_csv(gt_dir / GROUND_TRUTH_FILE, index=False)

    results.to_csv(out_dir / "results.csv", index=False)
    excluded_df.to_csv(out_dir / "excluded_rows.csv", index=False)
    with (out_dir / "calibration_meta.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "strata": strata,
            "policy": {
                "coverage_target": policy["coverage_target"],
                "min_calibration_samples": policy["min_calibration_samples"],
                "min_train_samples": policy["min_train_samples"],
                "split": policy["split"],
                "interval": policy["interval"],
            },
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")
    build_feature_availability().to_csv(
        out_dir / "feature_availability.csv", index=False)
    with (out_dir / "data_quality_report.json").open("w",
                                                     encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    log_line(f"result_rows={len(results)} "
             f"excluded_rows={len(excluded_df)}")
    log_line("outputs: results.csv excluded_rows.csv calibration_meta.json "
             "feature_availability.csv data_quality_report.json "
             f"data/{GROUND_TRUTH_DIR}/{GROUND_TRUTH_FILE}")
    (out_dir / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")

    abstain_counts = results[results["decision"] == DECISION_ABSTAIN][
        "reason_code"].value_counts().to_dict()
    manifest = build_manifest(
        run_id=run_id,
        utc=utc_now,
        project="P1_STAR_TAT",
        status=RUN_STATUS_INTERNAL,
        commit=current_commit(),
        config_hash=config_hash,
        data_hash=data_hash,
        permissions_pointer=policy["run"]["permissions_pointer"],
        operator=policy["run"]["operator"],
        input_rows=int(len(requests)),
        excluded_rows=int(len(excluded_df)),
        model_or_rule_version=rule_version,
        command=command,
        environment=collect_environment(),
        seed=seed,
        known_issues=list(policy["run"].get("known_issues", [])),
        result_files={
            "results.csv": out_dir / "results.csv",
            "excluded_rows.csv": out_dir / "excluded_rows.csv",
            "calibration_meta.json": out_dir / "calibration_meta.json",
            "feature_availability.csv": out_dir / "feature_availability.csv",
            "data_quality_report.json": out_dir / "data_quality_report.json",
            f"data/{GROUND_TRUTH_DIR}/{GROUND_TRUTH_FILE}":
                gt_dir / GROUND_TRUTH_FILE,
        },
    )
    result_hash = write_manifest(out_dir, manifest)
    manifest["result_hash"] = result_hash
    manifest["_abstain_summary"] = {k: int(v) for k, v in abstain_counts.items()}
    return {
        "out_dir": str(out_dir),
        "run_id": run_id,
        "result_hash": result_hash,
        "manifest": manifest,
        "n_requests": int(len(requests)),
        "n_excluded_rows": int(len(excluded_df)),
        "abstain_summary": abstain_counts,
        "block_counts": counts,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
