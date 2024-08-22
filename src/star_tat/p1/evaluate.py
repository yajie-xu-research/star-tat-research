"""Holdout evaluation.

Reads a replay run (results, calibration metadata, manifest) plus the
ground-truth outcomes and computes the metrics that become the single
source of truth: per-method empirical coverage with a Wilson confidence
interval, interval widths, point-error statistics, long-turnaround lead
time, rejection and exclusion counts, and per-site/version/stratum
breakdowns. All cited numbers trace back to run_receipt.json written here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from star_tat.common.environment import collect_environment
from star_tat.common.hashing import combine_hashes, sha256_file
from star_tat.common.manifest import (build_manifest, current_commit,
                                      load_manifest, write_manifest)
from star_tat.common.run_id import make_run_id
from star_tat.p1.constants import (
    BLOCK_HOLDOUT,
    DECISION_ABSTAIN,
    RUN_STATUS_INTERNAL,
)
from star_tat.p1.replay import GROUND_TRUTH_DIR, GROUND_TRUTH_FILE

INTERVAL_METHODS = (
    "stratified_interval",
    "conformal",
    "baseline_stratified_quantiles",
    "baseline_unstratified_interval",
)
POINT_METHODS = (
    "baseline_median_history",
    "baseline_raw_timestamp_point",
    "baseline_tree_point",
    "stratified_point",
)


class EvaluateError(RuntimeError):
    pass


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def _parse_hours(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def run_evaluate(*, run_dir: str | Path, out_dir: str | Path,
                 utc_now: str, command: str) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    replay_manifest = load_manifest(run_dir)
    results = pd.read_csv(run_dir / "results.csv", dtype=str,
                          keep_default_na=False)
    gt = pd.read_csv(run_dir / "data" / GROUND_TRUTH_DIR / GROUND_TRUTH_FILE,
                     dtype=str, keep_default_na=False)
    with (run_dir / "calibration_meta.json").open(encoding="utf-8") as fh:
        calibration_meta = json.load(fh)

    # Holdout rows only; outcomes joined from the ground-truth file.
    holdout = results[results["block"] == BLOCK_HOLDOUT].copy()
    gt_index = gt.set_index("case_key", drop=False)
    truth: list = []
    for _, row in holdout.iterrows():
        case_key = row["case_key"]
        g = gt_index.loc[case_key] if case_key in gt_index.index else None
        if g is None:
            truth.append({"completed": False, "censored": True})
            continue
        if str(g["is_completed"]).strip() != "1" or not g["released_at_utc"]:
            truth.append({"completed": False, "censored": True})
            continue
        truth.append({
            "completed": True,
            "censored": False,
            "released_at_utc": pd.Timestamp(g["released_at_utc"]),
        })
    holdout["released_at_utc"] = [
        t["released_at_utc"] if t.get("completed") else pd.NaT
        for t in truth]
    holdout["censored"] = [t["censored"] for t in truth]
    holdout["as_of_utc_ts"] = pd.to_datetime(
        holdout["as_of_utc"], format="ISO8601", utc=True)
    holdout["remaining_hours"] = [
        (r["released_at_utc"] - r["as_of_utc_ts"]).total_seconds() / 3600.0
        if not r["censored"] else None
        for _, r in holdout.iterrows()]

    rejected = holdout[holdout["decision"] == DECISION_ABSTAIN]
    rejected_by_reason = rejected["reason_code"].value_counts().to_dict()
    censored = holdout[holdout["censored"]]
    evaluated = holdout[(holdout["censored"] == False)  # noqa: E712
                        & (holdout["decision"] != DECISION_ABSTAIN)].copy()
    evaluated = evaluated.reset_index(drop=True)

    def _hours_col(col: str) -> pd.Series:
        return pd.to_numeric(evaluated[col], errors="coerce")

    # Coverage for interval methods.
    coverage: dict = {}
    for method in INTERVAL_METHODS:
        if method in ("stratified_interval", "conformal"):
            if method == "stratified_interval":
                lower = pd.to_datetime(evaluated["predicted_release_lower_utc"],
                                       format="ISO8601", utc=True,
                                       errors="coerce")
                upper = pd.to_datetime(evaluated["predicted_release_upper_utc"],
                                       format="ISO8601", utc=True,
                                       errors="coerce")
            else:
                lower = pd.to_datetime(evaluated["conformal_lower_utc"],
                                       format="ISO8601", utc=True,
                                       errors="coerce")
                upper = pd.to_datetime(evaluated["conformal_upper_utc"],
                                       format="ISO8601", utc=True,
                                       errors="coerce")
        else:
            col = f"baseline_{method.removeprefix('baseline_')}"
            split_vals = []
            for v in evaluated[col]:
                if v:
                    lo, hi = v.split("|")
                    split_vals.append((float(lo), float(hi)))
                else:
                    split_vals.append((None, None))
            lower_h = pd.Series([a for a, _ in split_vals], dtype=float)
            upper_h = pd.Series([b for _, b in split_vals], dtype=float)
            lower = evaluated["as_of_utc_ts"] + pd.to_timedelta(lower_h, unit="h")
            upper = evaluated["as_of_utc_ts"] + pd.to_timedelta(upper_h, unit="h")
        valid = lower.notna() & upper.notna()
        n = int(valid.sum())
        covered = int(((lower[valid] <= evaluated["released_at_utc"][valid])
                       & (evaluated["released_at_utc"][valid]
                          <= upper[valid])).sum())
        frac = covered / n if n else float("nan")
        width = (upper[valid] - lower[valid]).dt.total_seconds() / 3600.0
        lo_ci, hi_ci = _wilson(frac, n) if n else (0.0, 0.0)
        coverage[method] = {
            "n": n,
            "covered": covered,
            "coverage": round(frac, 4) if n else None,
            "coverage_ci95": [round(lo_ci, 4), round(hi_ci, 4)] if n else None,
            "width_hours_mean": round(float(width.mean()), 2) if n else None,
            "width_hours_median": round(float(width.median()), 2) if n else None,
        }

    # Point errors.
    point_errors: dict = {}
    remaining = pd.to_numeric(evaluated["remaining_hours"], errors="coerce")
    for method, col in (
        ("baseline_median_history", "baseline_median_history"),
        ("baseline_raw_timestamp_point", "baseline_raw_timestamp_point"),
        ("baseline_tree_point", "baseline_tree_point"),
        ("stratified_point", "predicted_remaining_hours"),
    ):
        pred = _hours_col(col)
        valid = pred.notna() & remaining.notna()
        n = int(valid.sum())
        if n:
            err = (pred[valid] - remaining[valid])
            point_errors[method] = {
                "n": n,
                "mae_hours": round(float(err.abs().mean()), 2),
                "rmse_hours": round(float((err ** 2).mean() ** 0.5), 2),
                "bias_hours": round(float(err.mean()), 2),
            }
        else:
            point_errors[method] = {"n": 0, "mae_hours": None,
                                    "rmse_hours": None, "bias_hours": None}

    # Long-turnaround cases: remaining >= P90 of holdout remaining.
    rem = remaining.dropna()
    threshold = float(rem.quantile(0.90)) if len(rem) else float("nan")
    long_mask = remaining >= threshold
    long_rows = evaluated[long_mask].reset_index(drop=True)
    long_tat: dict = {"threshold_hours": round(threshold, 2),
                      "long_count": int(long_mask.sum()),
                      "methods": {}}
    for method in INTERVAL_METHODS:
        if method == "stratified_interval":
            upper = pd.to_datetime(
                long_rows["predicted_release_upper_utc"],
                format="ISO8601", utc=True, errors="coerce")
        elif method == "conformal":
            upper = pd.to_datetime(long_rows["conformal_upper_utc"],
                                   format="ISO8601", utc=True, errors="coerce")
        else:
            col = f"baseline_{method.removeprefix('baseline_')}"
            upper_h = pd.Series([
                float(v.split("|")[1]) if v else float("nan")
                for v in long_rows[col]], dtype=float)
            upper = long_rows["as_of_utc_ts"] + pd.to_timedelta(upper_h, unit="h")
        lead = (long_rows["released_at_utc"] - upper).dt.total_seconds() / 3600.0
        valid = lead.notna()
        late_missed = int((lead[valid] < 0).sum())
        n = int(valid.sum())
        long_tat["methods"][method] = {
            "n": n,
            "lead_hours_median": round(float(lead[valid].median()), 2) if n else None,
            "late_missed": late_missed,
            "late_missed_rate": round(late_missed / n, 4) if n else None,
        }

    # Breakdowns.
    def _breakdown(group_col: str) -> list[dict]:
        rows = []
        for name, grp in evaluated.groupby(group_col):
            n = len(grp)
            lower = pd.to_datetime(grp["predicted_release_lower_utc"],
                                   format="ISO8601", utc=True, errors="coerce")
            upper = pd.to_datetime(grp["predicted_release_upper_utc"],
                                   format="ISO8601", utc=True, errors="coerce")
            valid = lower.notna() & upper.notna()
            covered = int(((lower[valid] <= grp["released_at_utc"][valid])
                           & (grp["released_at_utc"][valid]
                              <= upper[valid])).sum())
            n_valid = int(valid.sum())
            frac = covered / n_valid if n_valid else None
            lo_ci, hi_ci = _wilson(frac, n_valid) if n_valid else (0.0, 0.0)
            rows.append({
                "group": group_col,
                "value": str(name),
                "n_evaluated": n,
                "coverage": round(frac, 4) if frac is not None else None,
                "coverage_ci95": [round(lo_ci, 4), round(hi_ci, 4)]
                if frac is not None else None,
            })
        return rows

    per_site = _breakdown("site_key")
    per_version = _breakdown("workflow_version")

    evaluated = evaluated.copy()
    evaluated["stratum"] = (evaluated["site_key"] + "/"
                            + evaluated["workflow_version"])
    per_stratum = _breakdown("stratum")

    # Rejection rates.
    holdout_n = len(holdout)
    overall_n = len(results)
    rejection_rates = {
        "holdout_rejection_rate": round(
            len(rejected) / holdout_n, 4) if holdout_n else None,
        "overall_rejection_rate": round(
            int((results["decision"] == DECISION_ABSTAIN).sum()) / overall_n, 4)
        if overall_n else None,
    }

    # Excluded rows.
    excluded_path = run_dir / "excluded_rows.csv"
    excluded_counts = {"total": 0, "by_reason": {}}
    if excluded_path.exists():
        excluded_df = pd.read_csv(excluded_path, dtype=str,
                                  keep_default_na=False)
        excluded_counts = {
            "total": int(len(excluded_df)),
            "by_reason": excluded_df["reason"].value_counts().to_dict(),
        }

    receipt = {
        "project": "P1_STAR_TAT",
        "status": RUN_STATUS_INTERNAL,
        "evaluated_run_id": replay_manifest["run_id"],
        "evaluated_result_hash": replay_manifest["result_hash"],
        "holdout": {
            "requests": int(len(holdout)),
            "completed_with_truth": int((~holdout["censored"]).sum()),
            "censored": int(len(censored)),
            "rejected_total": int(len(rejected)),
            "rejected_by_reason": {k: int(v) for k, v in
                                   rejected_by_reason.items()},
            "evaluated": int(len(evaluated)),
        },
        "coverage": coverage,
        "point_errors": point_errors,
        "long_tat": long_tat,
        "per_site": per_site,
        "per_version": per_version,
        "per_stratum": per_stratum,
        "rejection_rates": rejection_rates,
        "excluded_rows": excluded_counts,
        "calibration": {
            "coverage_target": calibration_meta["policy"]["coverage_target"],
            "min_calibration_samples":
                calibration_meta["policy"]["min_calibration_samples"],
            "strata": calibration_meta["strata"],
        },
    }
    with (out_dir / "run_receipt.json").open("w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # Machine-readable detail rows + human report.
    detail_cols = [
        "request_id", "case_key", "as_of_utc", "site_key",
        "workflow_version", "stage_reached", "decision", "reason_code",
        "remaining_hours", "predicted_remaining_hours",
        "predicted_release_lower_utc", "predicted_release_upper_utc",
        "conformal_lower_utc", "conformal_upper_utc",
        "baseline_median_history", "baseline_stratified_quantiles",
        "baseline_raw_timestamp_point", "baseline_tree_point",
        "baseline_unstratified_interval",
    ]
    detail = evaluated[["released_at_utc", *detail_cols]].copy()
    detail["released_at_utc"] = detail["released_at_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    detail.to_csv(out_dir / "evaluation_details.csv", index=False)

    lines = _render_report(receipt, replay_manifest)
    (out_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    log.append(f"evaluated_run={replay_manifest['run_id']}")
    log.append(f"holdout_requests={len(holdout)} "
               f"evaluated={len(evaluated)} rejected={len(rejected)} "
               f"censored={len(censored)}")
    for method, entry in coverage.items():
        log.append(f"coverage[{method}]={entry}")
    (out_dir / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")

    result_files = {
        "run_receipt.json": out_dir / "run_receipt.json",
        "evaluation_details.csv": out_dir / "evaluation_details.csv",
        "evaluation_report.md": out_dir / "evaluation_report.md",
    }
    rule_version = replay_manifest["model_or_rule_version"]
    eval_run_id = make_run_id("P1E", replay_manifest["result_hash"],
                              combine_hashes(
                                  sha256_file(out_dir / "run_receipt.json")),
                              int(replay_manifest["seed"]), rule_version)
    manifest = build_manifest(
        run_id=eval_run_id,
        utc=utc_now,
        project="P1_STAR_TAT",
        status=RUN_STATUS_INTERNAL,
        commit=current_commit(),
        config_hash=replay_manifest["config_hash"],
        data_hash=replay_manifest["result_hash"],
        permissions_pointer=replay_manifest["permissions_pointer"],
        operator=replay_manifest["operator"],
        input_rows=int(len(results)),
        excluded_rows=int(len(rejected)),
        model_or_rule_version=rule_version,
        command=command,
        environment=collect_environment(),
        seed=int(replay_manifest["seed"]),
        known_issues=list(replay_manifest.get("known_issues", [])),
        result_files=result_files,
    )
    write_manifest(out_dir, manifest)
    return {"out_dir": str(out_dir), "receipt": receipt,
            "manifest": manifest, "evaluated_run_id": replay_manifest["run_id"]}


def _render_report(receipt: dict, replay_manifest: dict) -> list[str]:
    h = receipt["holdout"]
    c = receipt["coverage"]
    lines = [
        "# P1 STAR-TAT holdout evaluation report",
        "",
        f"- evaluated run: {receipt['evaluated_run_id']}",
        f"- holdout requests: {h['requests']} (completed with truth: "
        f"{h['completed_with_truth']}, censored: {h['censored']})",
        f"- rejected (abstained): {h['rejected_total']} "
        f"{h['rejected_by_reason']}",
        f"- evaluated rows: {h['evaluated']}",
        "",
        "## Interval coverage (holdout, completed, non-abstained)",
        "",
        "| method | n | coverage | 95% CI | width median (h) |",
        "|---|---|---|---|---|",
    ]
    for method, entry in c.items():
        ci = entry["coverage_ci95"] or ["-", "-"]
        lines.append(
            f"| {method} | {entry['n']} | {entry['coverage']} | "
            f"{ci[0]} - {ci[1]} | {entry['width_hours_median']} |")
    lines += [
        "",
        "## Point errors (hours)",
        "",
        "| method | n | MAE | RMSE | bias |",
        "|---|---|---|---|---|",
    ]
    for method, entry in receipt["point_errors"].items():
        lines.append(
            f"| {method} | {entry['n']} | {entry['mae_hours']} | "
            f"{entry['rmse_hours']} | {entry['bias_hours']} |")
    lt = receipt["long_tat"]
    lines += [
        "",
        f"## Long-turnaround cases (remaining >= P90 = "
        f"{lt['threshold_hours']} h, n = {lt['long_count']})",
        "",
        "| method | n | lead median (h) | late missed | late missed rate |",
        "|---|---|---|---|---|",
    ]
    for method, entry in lt["methods"].items():
        lines.append(
            f"| {method} | {entry['n']} | {entry['lead_hours_median']} | "
            f"{entry['late_missed']} | {entry['late_missed_rate']} |")
    lines += [
        "",
        "Coverage figures carry the sampling variance of 80-140-row strata",
        "(on the order of +/-5-7 percentage points); they are not precision",
        "calibration claims.",
    ]
    return lines
