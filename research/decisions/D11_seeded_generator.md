# D11 — Seeded data generator with injected anomalies

- **Question:** How to obtain a reusable input dataset for method work?
- **Alternatives:** (a) fetch live production exports; (b) hand-write a few
  fixture rows; (c) a deterministic seeded generator that emits full
  five-table datasets with controlled anomaly profiles.
- **Decision:** (c). The generator (seed `20240115`) produces synthetic
  case records for method demonstration and reproducibility with a fixed
  anomaly profile: about 2% late-known events, about 1% unmapped event
  codes, one site whose local chain has no sequencing step (its
  missing-stage comparison is declared not comparable), rework, skipped
  stages, and a queue-surge window that exercises drift screening.
- **Code/tests:** `src/star_tat/p1/synthetic.py`,
  `tests/test_synthetic.py`, `tests/test_validation.py`.
