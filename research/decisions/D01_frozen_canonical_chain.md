# D01 — Frozen seven-stage canonical chain

- **Question:** Which stage vocabulary normalizes three sites with three
  different raw event vocabularies?
- **Alternatives:** (a) per-site stage names used directly; (b) one frozen
  canonical chain of seven stages with a per-site/per-version mapping
  table; (c) learned stage embeddings.
- **Decision:** (b). One chain — `SAMPLE_RECEIVED`, `ACCESSIONED`,
  `LIBRARY_PREPARED`, `SEQUENCING_STARTED`, `ANALYSIS_COMPLETE`,
  `REVIEWED`, `RELEASED` — with an explicit stage dictionary mapping each
  site/version's raw codes onto it. The dictionary is frozen: mapping
  changes require a new dictionary version and a new config hash.
- **Code/tests:** `configs/p1_stages.yaml`, `src/star_tat/p1/stage_map.py`,
  `tests/test_stage_map.py`.
