# Method Card — STAR-TAT (P1)

Stage-normalized turnaround-time interval prediction for multisite
laboratory workflows.

## 1. Problem statement

A laboratory receives prediction requests ("when will this case be
released?") at arbitrary points during a case's lifecycle. Different sites
use different event vocabularies and workflow versions have staggered
validity windows, so raw event streams are not directly comparable. The
system must answer each request using only information available at the
request's as-of time, and must decline to answer when it cannot do so
honestly.

## 2. Stage normalization

Every site's raw event codes map onto a frozen seven-stage canonical chain:

```
SAMPLE_RECEIVED -> ACCESSIONED -> LIBRARY_PREPARED -> SEQUENCING_STARTED
-> ANALYSIS_COMPLETE -> REVIEWED -> RELEASED
```

Three sites use three vocabularies (SITE_A: `RCPT/ACC/LIBP/SEQ/ANL/REV/REL`;
SITE_B: `REC/ACC/QC/SEQ/ANA/CHK/OUT`; SITE_C: `IN/VER/PREP/RUN/DONE/SIGN/
ISSUE`). SITE_C has no mapping for `SEQUENCING_STARTED`: its local chain
runs the sequencing step offsite, so the dictionary carries six stages for
SITE_C and the six-stage chain is used for SITE_C predictions. Only the
missing-stage comparison is declared not comparable (the dictionary records
the SITE_C/SEQUENCING_STARTED combination as NOT_COMPARABLE); the method
never fabricates a sequencing event for SITE_C.

## 3. As-of snapshot construction

For each request `(case_key, as_of_utc)`:

1. Collect the case's events with `known_at_utc <= as_of_utc`. Events that
   occurred earlier but were only known later (`known_at_utc > as_of_utc`)
   are excluded and reported as `LATE_KNOWN_EVENT`.
2. Take the latest load snapshot with `snapshot_at_utc <= as_of_utc`; a
   snapshot strictly after `as_of_utc` is `FUTURE_LOAD_SNAPSHOT`.
3. Map the visible event codes through the stage dictionary; a case whose
   visible prefix contains an unmapped code abstains with `UNMAPPED_STAGE`.
4. Derive features: stage reached, stage index, stages completed, chain
   length, elapsed hours since receipt, hours in current stage, and the
   latest load triple (queue size, batch load, instrument utilization).
5. Requests after the case's release are excluded (`AS_OF_AFTER_RELEASE`);
   requests whose case is not yet in any mapped stage carry no target and
   are excluded (`NO_TARGET`).

A feature availability audit file lists, for every feature, the fields it
needs and whether those fields were available at `as_of_utc`.

## 4. Outcome variable

`remaining_hours = released_at_utc - as_of_utc`, only for completed cases.
Censored cases never produce a target and are confined to the training
block. The release timestamp is a target-side field: it is excluded from
snapshot frames before feature extraction, and the feature-name guard
refuses any feature whose name references it. Evaluation reads outcomes
only from `data/_ground_truth/true_outcomes.csv`, written by replay and
blocked from feature code by an isolation guard.

## 5. Block assignment

Requests are assigned to blocks temporally and stratified by site x
workflow version: 60% training, 20% calibration, 20% holdout, split on
`received_at_utc` within stratum. All requests sharing a case_key fall in
the same block (case-level leakage prevention). Censored cases go to the
training block.

## 6. Models

- **Point model (per stratum):** gradient boosting regressor on stage and
  load features, fitted on the training block only.
- **Prediction interval:** `[max(0, pred - q), pred + q]`, where `q` is the
  calibration-block quantile of absolute residuals chosen so the empirical
  coverage reaches the policy target (0.90 by default). The target is a
  program parameter used to select `q`.
- **Split-conformal comparison:** quantile level
  `ceil((n_cal + 1)(1 - alpha)) / n_cal` on calibration residuals.
- **Stratum with fewer than `min_calibration_samples` calibration rows:**
  the whole stratum abstains with `INSUFFICIENT_CALIBRATION`.

## 7. Five baselines (same as-of snapshots, fitted on train only)

| # | Baseline | Interval |
|---|----------|----------|
| 1 | Historical median | symmetric median residual band |
| 2 | Stage/version stratified empirical quantiles | `[q05, q95]` of stratified history |
| 3 | Raw-timestamp linear point prediction | symmetric residual band |
| 4 | Tree point prediction | symmetric residual band |
| 5 | Unstratified interval | calibration quantile over pooled residuals |

## 8. Drift screening and rejection

After block assignment, the training block establishes robust (Tukey
fence) ranges for continuous load features. Any request whose load
features fall outside the trained ranges is rejected with `SHIFT_DETECTED`.
Requests with unknown workflow versions reject with
`UNKNOWN_WORKFLOW_VERSION`; missing load snapshots reject with
`FEATURE_UNAVAILABLE`.

## 9. Evaluation

The holdout block is scored against ground truth: coverage and interval
width per method (with Wilson 95% intervals), point-error bias/MAE/RMSE,
per-stratum / per-site / per-version breakdowns, rejection rates by reason,
and a long-turnaround report (cases with remaining time at or above the
holdout P90): late-miss rate and lead time per method.

Intermediate evaluations of earlier method versions ran on internal sample
data from the seeded generator; the metrics reported in
`docs/frozen_metrics.md` are those of the final locked run on the shipped
seed-20240115 data package.

## 10. Reproducibility

`run_id = make_run_id(prefix, result_hash, ...)` where the result hash
combines data hash, config hash, seed, model/rule version, and hashes of
result files. Wall-clock timestamps and the operator name never enter the
hash, so re-running on identical inputs yields the same run id and result
hash. The stage dictionary is frozen; changes require a new version and a
new config hash.
