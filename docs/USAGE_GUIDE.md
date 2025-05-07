# Usage Guide — STAR-TAT (P1)

## Environment

```bash
uv venv venv --python 3.12
uv pip install --python venv/bin/python3 -e .
```

Python 3.12 with numpy / pandas / scikit-learn / pytest / pyyaml (see
`environment.json` for the dual snapshot).

## Command reference

All commands are `venv/bin/star-tat <command>`. The five commands:

### 1. generate-synthetic

```bash
venv/bin/star-tat generate-synthetic --output synthetic_data --seed 20240115
```

Writes the five tables plus `data_quality_report.json` and
`predict_requests.csv`. The generator is deterministic: the same seed and
dictionary reproduce the same files byte for byte.

Options: `--output DIR`, `--seed INT`, `--config PATH` (stage dictionary).

### 2. validate

```bash
venv/bin/star-tat validate --input synthetic_data \
  --config configs/p1_stages.yaml
```

Runs the five-table contract checks and writes
`synthetic_data/data_quality_report.json`. Exit code 0 means PASS; any
fatal issue exits non-zero with the report listing every fatal issue.

### 3. p1 replay

```bash
venv/bin/star-tat p1 replay --input synthetic_data \
  --as-of-file synthetic_data/predict_requests.csv \
  --out outputs/runs/p1_replay --seed 20240115
```

Builds as-of snapshots, assigns blocks, fits per-stratum models, produces
predictions. Outputs:

| File | Contents |
|---|---|
| `results.csv` | One row per request: decision, reason, point prediction, lower/upper, block, stratum |
| `excluded_rows.csv` | Dropped rows with exclusion reasons |
| `calibration_meta.json` | Per-stratum q, conformal q, training ranges, sample counts |
| `feature_availability.csv` | Per-feature availability audit at each as-of time |
| `data_quality_report.json` | Validation report copy |
| `data/_ground_truth/true_outcomes.csv` | Ground truth written by replay; feature code never reads this directory |
| `manifest.json` | Provenance: run id, hashes, status, known issues |
| `run.log` | Human-readable summary |

### 4. p1 evaluate

```bash
venv/bin/star-tat p1 evaluate --run outputs/runs/p1_replay \
  --out outputs/runs/p1_evaluation
```

Scores the replay's holdout block against ground truth. Outputs:
`run_receipt.json` (machine-readable metrics), `evaluation_details.csv`,
`evaluation_report.md`, `manifest.json`, `run.log`.

### 5. manifest inspect

```bash
venv/bin/star-tat manifest inspect --run outputs/runs/p1_replay
```

Prints the run's manifest: run id, result hash, config/data hashes,
status, model/rule version, known issues.

## Reproducing a locked run

The historical run shipped in `outputs/runs/p1_replay` was produced with
seed `20240115` on the tracked `synthetic_data/`. Re-run:

```bash
rm -rf outputs/runs/p1_replay
venv/bin/star-tat p1 replay --input synthetic_data \
  --as-of-file synthetic_data/predict_requests.csv \
  --out outputs/runs/p1_replay --seed 20240115
```

The run id and result hash in the manifest must match the committed run.
Deleting `outputs/runs/p1_replay/data/_ground_truth/` before the replay
must leave `results.csv` byte-identical (isolation check, also covered by
the test suite).

## Ground-truth isolation rules

- Only replay writes `data/_ground_truth/true_outcomes.csv`; only evaluate
  reads it.
- Feature extraction frames are built from cases/events/load snapshots with
  the release column removed; the feature-name guard refuses any feature
  referencing it.
- The isolation guard (`src/star_tat/common/isolation.py`) raises if
  feature or prediction code opens anything under `data/_ground_truth/`.

## Tests

```bash
venv/bin/python3 -m pytest
```

114 cases covering the guide's counter-example table, table contracts,
calibration blocking, leakage, reproducibility, and the constructed
small-stratum rejection path.
