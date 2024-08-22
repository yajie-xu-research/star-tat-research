# D05 — Temporal block assignment, stratified, case-grouped

- **Question:** How to divide requests into training / calibration /
  holdout blocks without leakage?
- **Alternatives:** (a) random split; (b) plain temporal split; (c) temporal
  split stratified by site x workflow version with all requests of one
  case forced into the same block.
- **Decision:** (c). Random splits leak future information into training;
  plain temporal splits ignore that versions have staggered validity
  windows. Case-grouping prevents the same case's requests from appearing
  in both training and evaluation.
- **Code/tests:** `src/star_tat/p1/split.py`, `tests/test_split.py`.
