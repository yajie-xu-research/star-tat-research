# D12 — Honest abstention over forced answers

- **Question:** What should the system output when it cannot answer a
  request soundly?
- **Alternatives:** (a) always predict (best effort); (b) abstain with a
  machine-readable reason for each failure mode; (c) raise and stop the
  whole run.
- **Decision:** (b). Five abstention reasons cover the failure modes:
  `UNKNOWN_WORKFLOW_VERSION`, `FEATURE_UNAVAILABLE`, `UNMAPPED_STAGE`,
  `INSUFFICIENT_CALIBRATION`, `SHIFT_DETECTED`. Each abstention is a row
  in results with an empty interval; rejection rates are reported by
  reason in the evaluation receipt. Prediction quality is measured only
  on rows the system actually answered.
- **Code/tests:** `src/star_tat/p1/constants.py`,
  `src/star_tat/p1/replay.py`, `tests/test_calibration.py`,
  `tests/test_evaluate.py`, `tests/test_snapshots.py`.
