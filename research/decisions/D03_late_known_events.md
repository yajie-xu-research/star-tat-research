# D03 — Late-known events excluded, not corrected

- **Question:** What to do with events whose `known_at_utc` is after a
  request's `as_of_utc` (the system did not know them at request time)?
- **Alternatives:** (a) include them anyway (leaks the future);
  (b) exclude them from that request's snapshot and record the reason;
  (c) impute a corrected known time.
- **Decision:** (b). As-of honesty is the core promise: the snapshot must
  contain only what the system could know. Excluded rows carry reason
  `LATE_KNOWN_EVENT` so the exclusion is auditable.
- **Code/tests:** `src/star_tat/p1/snapshots.py`,
  `tests/test_snapshots.py`, `tests/test_validation.py`.
