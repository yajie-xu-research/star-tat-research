# External Validation Protocol — Draft

Status: **NOT_OBTAINED** (see `external/status.json`).

This draft describes how the method in this repository would be checked on
data it has never seen, should such a dataset become available. No external
organization has been engaged, and no claims about external results are
made anywhere in this repository.

## 1. Purpose

Demonstrate that the stage-normalized TAT interval predictor transfers
beyond the historical research dataset: calibration behavior, rejection
behavior, and coverage against held-out ground truth.

## 2. Input requirements

- The five tables conforming to `schemas/` (cases, events, load
  snapshots, stage map, prediction requests).
- A stage dictionary for the external site(s), or an explicit mapping onto
  the frozen canonical chain; sites without a comparable chain would be
  reported as not comparable, per the method card.
- Ground truth withheld from all model-fitting code until evaluation.

## 3. Procedure

1. Run `validate`; resolve fatal issues before proceeding.
2. Run `p1 replay` with run status `EXTERNAL_VALIDATION` and the external
   dictionary hash recorded in the manifest.
3. Run `p1 evaluate`; record the receipt.
4. Compare coverage, width, rejection rates, and long-turnaround behavior
   against `docs/frozen_metrics.md` (the historical locked run).
5. File the receipt, manifest, and a written comparison under
   `outputs/runs/` as a curated run.

## 4. Honesty rules

- If the external stage dictionary cannot express the canonical chain,
  the finding is "not comparable", not a failure or success of the model.
- Rejection rates are reported by reason, never collapsed into a single
  headline number.
- The coverage target remains a program parameter; an external dataset
  that yields different coverage is a finding, not a violation.

## 5. Governance

- Run status `EXTERNAL_VALIDATION` is reserved for runs against datasets
  covered by a data-use agreement.
- This protocol stays in draft state until a dataset is actually
  available; `status.json` is updated at that point.
