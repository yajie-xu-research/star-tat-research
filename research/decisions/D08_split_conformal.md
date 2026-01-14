# D08 — Split-conformal comparison alongside the quantile interval

- **Question:** Which second interval construction validates the primary
  calibration-quantile interval?
- **Alternatives:** (a) no second interval; (b) bootstrap intervals;
  (c) split-conformal quantile on calibration residuals.
- **Decision:** (c). Split conformal is distribution-free, requires no
  extra fitting, and its coverage guarantee is interpretable. It is
  reported as a comparison, not a replacement: the primary interval uses
  the policy coverage target; conformal uses the finite-sample level
  `ceil((n_cal + 1)(1 - alpha)) / n_cal`.
- **Code/tests:** `src/star_tat/p1/calibration.py`,
  `tests/test_calibration.py`.
