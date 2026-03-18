# Changelog

All notable changes to this research codebase are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.3.0] - 2026-03-18

### Added

- External validation protocol v2 and the site readiness checklist
  (`external/`); status remains not obtained.

## [1.2.0] - 2025-11-19

### Added

- Temporal block assignment (train 60% / calibration 20% / holdout 20%)
  stratified by site x workflow version, with all requests of one case
  forced into the same block to prevent case-level leakage.
- Deterministic result hashing: run id and result hash recompute from data,
  config, seed, model/rule version, and result files; wall-clock fields are
  excluded.

## [1.1.0] - 2025-07-15

### Added

- External validation protocol v1 (draft) and the pilot deployment plan
  (`external/`); status recorded as not obtained.

## [1.0.0] - 2025-06-10

### Added

- Seeded synthetic data generator (seed 20240115) emitting cases, events,
  load snapshots, stage map, and prediction requests across three sites and
  three workflow versions with staggered validity windows.
- Frozen seven-stage canonical chain: `SAMPLE_RECEIVED`, `ACCESSIONED`,
  `LIBRARY_PREPARED`, `SEQUENCING_STARTED`, `ANALYSIS_COMPLETE`,
  `REVIEWED`, `RELEASED`.
- `p1 replay` and the first three baselines (historical median, stratified
  empirical quantiles, raw-timestamp linear point prediction).
- Manifest writer with `INTERNAL_RESEARCH` / `EXTERNAL_VALIDATION` run
  statuses; historical runs in this repository use the former.

## [0.8.0] - 2025-03-19

### Added

- Release candidate: calibration stratified by workflow version, the
  internal replay report, and the internal analysis plan freeze.

## [0.5.0] - 2024-11-20

### Added

- Frozen validation run: abstention rules and drift screening, temporal and
  workflow-version holdouts, and the site holdout.

## [0.2.0] - 2024-06-12

### Added

- Calibration prototype: snapshot construction with known-at audit,
  stratified baselines, and per-stratum calibration with temporal blocks.

### Changed

- Research design and data contract frozen, including the seven-stage
  canonical chain and the event dictionary with site-version mapping.
