# D02 — One model per site x workflow version stratum

- **Question:** Should the point model be pooled across sites and workflow
  versions?
- **Alternatives:** (a) one pooled model with site/version dummies;
  (b) one model per stratum; (c) site-level models pooled across versions.
- **Decision:** (b). Workflow versions change stage definitions and site
  conditions differ materially; a stratum (site x version) is the finest
  unit whose calibration we can trust. Pooling would force a single
  residual distribution across incomparable workflows.
- **Code/tests:** `src/star_tat/p1/calibration.py`,
  `src/star_tat/p1/split.py`, `tests/test_calibration.py`.
