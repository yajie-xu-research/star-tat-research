# D06 — Censored cases: training only, never a target

- **Question:** What role do not-yet-completed (censored) cases play?
- **Alternatives:** (a) drop them; (b) use them with a zeroed target;
  (c) keep them in the training block's feature context, no target.
- **Decision:** (c). Censored cases carry legitimate as-of snapshots for
  feature context, but any target derived from them would be fabricated.
  A zeroed remaining-time target is explicitly wrong (it would train the
  model to predict zero).
- **Code/tests:** `src/star_tat/p1/split.py`,
  `src/star_tat/p1/targets.py`, `tests/test_targets.py`,
  `tests/test_split.py`.
