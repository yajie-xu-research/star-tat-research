# P1 STAR-TAT holdout evaluation report

- evaluated run: P1R_c08ba354f3c8
- holdout requests: 1249 (completed with truth: 1249, censored: 0)
- rejected (abstained): 88 {'SHIFT_DETECTED': 81, 'UNMAPPED_STAGE': 4, 'UNKNOWN_WORKFLOW_VERSION': 3}
- evaluated rows: 1161

## Interval coverage (holdout, completed, non-abstained)

| method | n | coverage | 95% CI | width median (h) |
|---|---|---|---|---|
| stratified_interval | 1161 | 0.9087 | 0.8908 - 0.9239 | 46.43 |
| conformal | 1161 | 0.9268 | 0.9104 - 0.9404 | 50.08 |
| baseline_stratified_quantiles | 1161 | 0.8699 | 0.8494 - 0.8881 | 74.92 |
| baseline_unstratified_interval | 1161 | 0.9208 | 0.9038 - 0.9349 | 46.78 |

## Point errors (hours)

| method | n | MAE | RMSE | bias |
|---|---|---|---|---|
| baseline_median_history | 1161 | 15.54 | 25.36 | -2.93 |
| baseline_raw_timestamp_point | 1161 | 21.99 | 29.24 | -2.13 |
| baseline_tree_point | 1161 | 11.32 | 20.44 | 0.15 |
| stratified_point | 1161 | 11.97 | 19.01 | -0.12 |

## Long-turnaround cases (remaining >= P90 = 101.34 h, n = 117)

| method | n | lead median (h) | late missed | late missed rate |
|---|---|---|---|---|
| stratified_interval | 117 | -10.18 | 81 | 0.6923 |
| conformal | 117 | -12.32 | 87 | 0.7436 |
| baseline_stratified_quantiles | 117 | -5.44 | 75 | 0.641 |
| baseline_unstratified_interval | 117 | -5.74 | 80 | 0.6838 |

Coverage figures carry the sampling variance of 80-140-row strata
(on the order of +/-5-7 percentage points); they are not precision
calibration claims.
