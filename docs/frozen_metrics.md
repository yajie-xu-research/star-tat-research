# STAR-TAT frozen metrics

These metrics describe the locked evaluation run
(`outputs/runs/p1_replay` + `outputs/runs/p1_evaluation`) on the shipped
data package (`synthetic_data/`, seed `20240115`, generated 2026-06-30).
They are the single source of truth for any document citing STAR-TAT
numbers. A code change that alters these numbers must land in a new version
with new metrics; the numbers here are not restated or recalculated.

## Run manifest (constituents)

| Field | Value |
| --- | --- |
| replay run_id | `P1R_c08ba354f3c8` (utc `2026-06-30T09:00:00Z`) |
| evaluation run_id | `P1E_f9a0e9f4741b` (utc `2026-06-30T09:15:00Z`) |
| seed | `20240115` |
| model/rule version | `v1.0/v1.0/tat-gbm-1.4/stratified-absres-1.4` |
| config hash | `d875de556a67883ae442347d18c396c2257cdf358eed9502b883c2f8df27c69b` |
| data hash | `0a55b4ac664dc747dfd951582438e094e71f23bf09548b14752ed1d5d1f31cd6` |
| replay result hash | `d226c4495ba79fc3f40bb269475cacfb001be4650b565e8d5cbebd29275782bc` |
| evaluation result hash | `80a6a0108233516aa1e10653808a91bbf325223e12288ae531d958215f1f6c4d` |

Code version: the release recorded in `CHANGELOG.md` 1.4.0; the run
manifests record the commit at which each run was produced. Reproduction:
the documented five-command pipeline (generate → validate → replay →
evaluate → manifest inspect) reproduces every artifact byte-for-byte; see
`outputs/runs/reproducibility_check.log`.

## Request accounting

| Item | Count |
| --- | --- |
| prediction requests in the data package | 6,887 |
| excluded rows (audit ledger `excluded_rows.csv`) | 1,438 = 585 `CENSORED` + 853 `LATE_KNOWN_EVENT` |
| predictable requests | 5,449 = 6,887 − 1,438 |
| predicted | 5,019 |
| rejected (abstained) | 430 |

Rejection reasons (whole run): `FEATURE_UNAVAILABLE` 185,
`SHIFT_DETECTED` 172, `UNMAPPED_STAGE` 40, `UNKNOWN_WORKFLOW_VERSION` 33.
Overall rejection rate **0.0624** = 430 / 6,887.

Identity note (not a coincidence): 5,019 predicted =
5,449 predictable − 430 rejected, and 5,604 released cases − 585 censored
cases = 5,019 released-with-prediction cases. Both computations count the
same population from different directions; the equality is an arithmetic
identity of the accounting, stated explicitly here.

`LATE_KNOWN_EVENT` is an event-level exclusion: an event whose `known_at`
is after the request's `as_of` is dropped from that request's as-of
snapshot, and each such exclusion is one row in the ledger (853 rows for
the shipped run).

`CENSORED` is a request-level exclusion applied to ground-truth rows only:
585 requests target cases that never completed. Those cases keep their
`released_at_utc` unset and contribute no target; they are removed from the
auditable denominator as non-informative rows. (`is_completed = false` is
never imputed as a load or time value.)

## Holdout accounting (three related denominators)

| Denominator | Value | Meaning |
| --- | --- | --- |
| holdout requests | 999 | rows in `predict_requests.csv` with `split = holdout` |
| holdout result rows | 1,249 | holdout rows after replay (including post-hoc requests targeting holdout cases) |
| holdout evaluated | 1,161 | result rows with a target and no rejection |

1,249 = 1,161 evaluated + 88 rejected. Holdout rejection rate **0.0705** =
88 / 1,249 (reasons: `SHIFT_DETECTED` 81, `UNMAPPED_STAGE` 4,
`UNKNOWN_WORKFLOW_VERSION` 3). The holdout is larger than the
holdout-request count because the holdout split is assigned per case and
post-hoc requests to those cases also carry the holdout flag; every holdout
case remains unseen by training and calibration regardless of how many
requests reference it.

## Coverage (1,161 holdout evaluated rows)

| Method | Coverage (95% CI) | Interval width |
| --- | --- | --- |
| **stratified interval (primary)** | **0.9087** [0.8908, 0.9239] (1055/1161) | median 46.4 h (mean 49.3 h) |
| split-conformal comparison | 0.9268 [0.9104, 0.9404] (1076/1161) | median 50.1 h (mean 53.3 h) |
| unstratified interval baseline | 0.9208 [0.9038, 0.9349] (1069/1161) | median 46.8 h (mean 46.6 h) |
| stratified-quantiles baseline | 0.8699 [0.8494, 0.8881] (1010/1161) | median 74.9 h (mean 76.8 h) |

The primary method is the stratified interval calibrated at the 0.90
target; the conformal comparison and the two baselines are reported at the
same holdout to show the coverage–width tradeoff, not to be selected.

## Point errors (1,161 holdout evaluated rows)

| Method | Bias | MAE | RMSE |
| --- | --- | --- | --- |
| stratified point (primary) | −0.12 h | 11.97 h | 19.01 h |
| tree point baseline | +0.15 h | 11.32 h | 20.44 h |
| median-history baseline | −2.93 h | 15.54 h | 25.36 h |
| raw-timestamp baseline | −2.13 h | 21.99 h | 29.24 h |

## Group breakdowns (from the same evaluation receipt)

By site:

| Group | Evaluated | Coverage |
| --- | --- | --- |
| SITE_A | 361 | 0.9114 |
| SITE_B | 399 | 0.8972 |
| SITE_C | 401 | 0.9177 |

By workflow version:

| Group | Evaluated | Coverage |
| --- | --- | --- |
| WF_V1 | 364 | 0.8791 |
| WF_V2 | 401 | 0.9177 |
| WF_V3 | 396 | 0.9268 |

SITE_C and WF_V2 both show 401 evaluated rows at coverage 0.9177. That
pair of identical values is a real feature of the data (each group
evaluated 401 rows; the two coverages round to the same value), verified
against the evaluation receipt — not a transcription artifact.

By stratum (nine site × workflow-version strata): evaluated 105–152 rows
per stratum, per-stratum coverage 0.8571–0.9528 (full table in
`outputs/runs/p1_evaluation/evaluation_details.csv`).

## Calibration parameters

- Training/calibration/holdout split: 60 / 20 / 20, stratified by
  site × workflow-version (holdout assigned per case).
- Interval: training-block point model (gradient-boosted trees) +
  calibration-block absolute-residual quantile per stratum (q range
  19.9–34.0 h across the nine strata).
- Sample-size floor: a stratum with fewer than 30 calibration rows
  abstains (`INSUFFICIENT_CALIBRATION`); the shipped run has 101–158
  calibration rows per stratum and never triggers it (the rejection path
  is exercised by dedicated tests).
- Coverage target 0.90 is a program parameter used to select q, not a
  pass/fail gate (see `configs/p1_policy.yaml`).

## Drift screening

The replay applies a two-sample KS test on the feature distribution of
each stratum's requests against its training block; a flagged stratum
rejects its requests with `SHIFT_DETECTED`. The shipped run flags 172
requests (6 strata); 81 of those fall in the holdout.

## Long-turnaround report (holdout rows with remaining time ≥ holdout P90)

Threshold: 101.3 h. Cases in report: **117**.

| Method | Late-missed | Late-miss rate | Lead time (median) |
| --- | --- | --- | --- |
| stratified interval (primary) | 81 / 117 | 0.692 | −10.2 h |
| split-conformal comparison | 87 / 117 | 0.744 | −12.3 h |
| unstratified interval baseline | 80 / 117 | 0.684 | −5.7 h |
| stratified-quantiles baseline | 75 / 117 | 0.641 | −5.4 h |

Long-turnaround rows are the subpopulation the intervals under-cover most
(about 31% coverage for the primary method against 90.9% overall); lead
times are negative on average, meaning point estimates run ahead of the
true release times on these cases. This is the documented limitation the
method's rejection paths are designed around.

## What these metrics do NOT claim

- They describe behavior on generated data with controlled anomalies, not
  on real patient records.
- A covered interval is a calibrated claim about the prediction, not a
  causal model of the laboratory.
- External validation has not been obtained (`external/status.json`:
  NOT_OBTAINED); these are internal research metrics.
