# D07 — Coverage target is a parameter, not a pass gate

- **Question:** How should the 0.90 coverage target be used?
- **Alternatives:** (a) hard success gate that fails the run when missed;
  (b) program parameter that selects the calibration quantile, reported
  but never used to fail the run; (c) ignored.
- **Decision:** (b). The target picks `q` (the calibration-block absolute
  residual quantile achieving it). Whether a dataset achieves it is an
  empirical finding reported in the evaluation receipt, never a pass/fail
  gate. This keeps evaluation honest: the metric measures the method, not
  the method's success.
- **Code/tests:** `configs/p1_policy.yaml`,
  `src/star_tat/p1/calibration.py`, `tests/test_calibration.py`.
