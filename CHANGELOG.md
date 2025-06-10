# Changelog

All notable changes to this research codebase are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
