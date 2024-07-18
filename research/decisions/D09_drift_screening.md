# D09 — Drift screening with robust ranges, reject on exceedance

- **Question:** How to detect that a request's load conditions fall outside
  the training distribution?
- **Alternatives:** (a) min/max training ranges; (b) Tukey-fence (IQR)
  robust ranges on training features, reject requests outside them;
  (c) distributional distance tests.
- **Decision:** (b). Min/max ranges are degenerate for continuous features
  (every new extreme is a "shift"); Tukey fences are robust to the
  training tail while still catching genuine regime changes. Out-of-range
  requests are rejected with `SHIFT_DETECTED` rather than silently
  predicted.
- **Code/tests:** `src/star_tat/p1/drift.py`,
  `src/star_tat/p1/calibration.py`, `tests/test_calibration.py`.
