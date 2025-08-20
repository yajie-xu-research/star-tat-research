# Pilot usage log — weeks 1-5

Contemporaneous weekly notes from the reporting team for the first five
weeks of the internal pilot (2025-07-16 through 2025-08-19). The
consolidated biweekly summaries filed with the record keeper reconcile to
these notes.

## Week 1 (2025-07-16 to 2025-07-22)

- Requests processed: 97
- Outputs: prediction intervals; abstentions
- Abstentions: FEATURE_UNAVAILABLE 2, SHIFT_DETECTED 3,
  UNKNOWN_WORKFLOW_VERSION 1
- Notes: initial ramp. Requests carrying unknown workflow-version codes
  from a legacy export were set aside; the reporting team asked how
  abstention reasons were logged and reviewed the reason-code list.

## Week 2 (2025-07-23 to 2025-07-29)

- Requests processed: 98
- Outputs: prediction intervals; abstentions
- Abstentions: FEATURE_UNAVAILABLE 3, SHIFT_DETECTED 2, UNMAPPED_STAGE 1
- Notes: ramp continued. No changes to the frozen stage dictionary.

## Week 3 (2025-07-30 to 2025-08-05)

- Requests processed: 106
- Outputs: prediction intervals; abstentions
- Abstentions: FEATURE_UNAVAILABLE 3, SHIFT_DETECTED 2, UNMAPPED_STAGE 1
- Notes: site vocabularies were walked through with data operations;
  unmapped stage codes traced to legacy codes outside the frozen
  dictionary. Team direction: abstain rather than guess mappings.

## Week 4 (2025-08-06 to 2025-08-12)

- Requests processed: 106
- Outputs: prediction intervals; abstentions
- Abstentions: FEATURE_UNAVAILABLE 3, SHIFT_DETECTED 3,
  UNKNOWN_WORKFLOW_VERSION 1
- Notes: late-known events excluded from snapshots were reviewed; the
  exclusion notes were confirmed against the event log. Feedback:
  request-to-case traceability was useful.

## Week 5 (2025-08-13 to 2025-08-19)

- Requests processed: 114
- Outputs: prediction intervals; abstentions
- Abstentions: FEATURE_UNAVAILABLE 3, SHIFT_DETECTED 3, UNMAPPED_STAGE 1
- Notes: load snapshots around an instrument maintenance window produced
  drift alerts; the screening was reviewed with the team and left as-is.

## Reconciliation

- Weeks 1-2 (first biweekly reporting period): 195 requests,
  FEATURE_UNAVAILABLE 5, SHIFT_DETECTED 5, UNMAPPED_STAGE 1,
  UNKNOWN_WORKFLOW_VERSION 1.
- Weeks 3-4 (second reporting period): 212 requests,
  FEATURE_UNAVAILABLE 6, SHIFT_DETECTED 5, UNMAPPED_STAGE 1,
  UNKNOWN_WORKFLOW_VERSION 1.
- Week 5 falls inside the third reporting period (228 requests over weeks
  5-6) and accounts for 114 of those requests.

All figures are pilot-internal.
