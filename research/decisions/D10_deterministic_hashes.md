# D10 — Deterministic result hashes exclude wall-clock fields

- **Question:** What identifies a run and proves two runs are the same?
- **Alternatives:** (a) timestamps and operator identity; (b) run id from
  data hash + config hash + seed + model/rule version + result files;
  (c) manual version strings.
- **Decision:** (b). The result hash covers every input that affects
  outputs; wall-clock fields (`utc`, operator) are recorded for provenance
  but excluded from the hash, so identical inputs always reproduce the
  same run id and result hash. This is what makes the locked historical
  run checkable.
- **Code/tests:** `src/star_tat/common/hashing.py`,
  `src/star_tat/common/manifest.py`, `tests/test_manifest.py`,
  `tests/test_reproducibility.py`.
