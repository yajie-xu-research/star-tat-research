# D04 — Release timestamp is outcome-only

- **Question:** Can the release timestamp appear anywhere on the feature
  side?
- **Alternatives:** (a) allowed with a flag; (b) allowed only as elapsed
  time; (c) forbidden entirely from features, kept only in ground truth.
- **Decision:** (c). `released_at_utc` defines the outcome. It is removed
  from snapshot frames before feature extraction and a name guard rejects
  any feature referencing it. Ground truth lives under
  `data/_ground_truth/` with an isolation guard preventing feature code
  from opening it.
- **Code/tests:** `src/star_tat/p1/features.py`,
  `src/star_tat/common/isolation.py`, `tests/test_leakage.py`.
