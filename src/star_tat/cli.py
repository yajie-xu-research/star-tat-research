"""Command-line entry point (star-tat).

Commands:
  star-tat generate-synthetic --output synthetic_data/ --seed 20240115
  star-tat validate --input synthetic_data/ --config configs/p1_stages.yaml
  star-tat p1 replay --input synthetic_data/ \
      --as-of-file synthetic_data/predict_requests.csv --out runs/p1_replay/
  star-tat p1 evaluate --run runs/p1_replay/ --out runs/p1_evaluation/
  star-tat manifest inspect --run runs/p1_replay/

Every command exits non-zero on missing inputs, structural validation
failure, or missing run artifacts; success is never faked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from star_tat.common.manifest import load_manifest
from star_tat.p1.evaluate import EvaluateError, run_evaluate
from star_tat.p1.io import repo_root
from star_tat.p1.replay import ReplayError, run_replay, utc_now
from star_tat.p1.stage_map import StageMap
from star_tat.p1.synthetic import generate, write_tables
from star_tat.p1.validation import validate_suite

DEFAULT_SEED = 20240115


def _default_stages() -> str:
    return str(repo_root() / "configs" / "p1_stages.yaml")


def _default_policy() -> str:
    return str(repo_root() / "configs" / "p1_policy.yaml")


def cmd_generate(args: argparse.Namespace) -> int:
    stages_config = args.stages_config or _default_stages()
    if not Path(stages_config).exists():
        print(f"error: stages config not found: {stages_config}",
              file=sys.stderr)
        return 2
    stage_map = StageMap(stages_config)
    frames = generate(args.seed, stage_map)
    write_tables(args.output, frames, args.seed)
    for table, frame in frames.items():
        print(f"wrote {table}: {len(frame)} rows")
    print(f"output directory: {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if not Path(args.input).exists():
        print(f"error: input directory not found: {args.input}",
              file=sys.stderr)
        return 2
    if not Path(args.config).exists():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    report_path = Path(args.input) / "data_quality_report.json"
    report = validate_suite(args.input, args.config, report_path)
    print(f"validation outcome: {report['validation_outcome']}")
    print(f"tables: " + ", ".join(
        f"{k}={v['rows']}" for k, v in report["tables"].items()))
    print(f"fatal issues: {report['fatal_count']}, "
          f"warnings: {report['warning_count']}")
    print(f"report written: {report_path}")
    if report["warning_count"]:
        kinds = sorted({w["issue"] for w in report["warnings"]})
        print("warning kinds: " + ", ".join(kinds))
    return 1 if report["validation_outcome"] == "FAIL" else 0


def cmd_replay(args: argparse.Namespace) -> int:
    stages_config = args.stages_config or _default_stages()
    policy_config = args.policy_config or _default_policy()
    for path, label in ((args.input, "input"), (args.as_of_file, "as-of-file"),
                        (stages_config, "stages config"),
                        (policy_config, "policy config")):
        if not Path(path).exists():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2
    command = " ".join(["star-tat", "p1", "replay",
                        f"--input {args.input}",
                        f"--as-of-file {args.as_of_file}",
                        f"--out {args.out}",
                        f"--stages-config {args.stages_config or 'configs/p1_stages.yaml'}",
                        f"--policy-config {args.policy_config or 'configs/p1_policy.yaml'}",
                        f"--seed {args.seed}"])
    try:
        summary = run_replay(
            input_dir=args.input,
            as_of_file=args.as_of_file,
            out_dir=args.out,
            stages_config=stages_config,
            policy_config=policy_config,
            seed=args.seed,
            utc_now=args.utc or utc_now(),
            command=command,
        )
    except ReplayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"run_id: {summary['run_id']}")
    print(f"result_hash: {summary['result_hash']}")
    print(f"requests: {summary['n_requests']}, "
          f"excluded rows: {summary['n_excluded_rows']}")
    print(f"block counts: {summary['block_counts']}")
    print(f"abstain summary: {summary['abstain_summary']}")
    print(f"outputs: {summary['out_dir']}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not Path(args.run).exists():
        print(f"error: run directory not found: {args.run}", file=sys.stderr)
        return 2
    command = " ".join(["star-tat", "p1", "evaluate",
                        f"--run {args.run}", f"--out {args.out}"]
                       + (["--utc", args.utc] if args.utc else []))
    try:
        result = run_evaluate(run_dir=args.run, out_dir=args.out,
                              utc_now=args.utc or utc_now(), command=command)
    except EvaluateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"evaluated run: {result['evaluated_run_id']}")
    receipt = result["receipt"]
    print(f"holdout: {receipt['holdout']}")
    print("coverage:")
    for method, entry in receipt["coverage"].items():
        print(f"  {method}: n={entry['n']} coverage={entry['coverage']} "
              f"width_median={entry['width_hours_median']}h")
    print(f"outputs: {result['out_dir']}")
    return 0


def cmd_manifest_inspect(args: argparse.Namespace) -> int:
    try:
        manifest = load_manifest(args.run)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="star-tat")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-synthetic")
    gen.add_argument("--output", required=True)
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--stages-config", default=None)
    gen.set_defaults(func=cmd_generate)

    val = sub.add_parser("validate")
    val.add_argument("--input", required=True)
    val.add_argument("--config", required=True)
    val.set_defaults(func=cmd_validate)

    p1 = sub.add_parser("p1")
    p1_sub = p1.add_subparsers(dest="p1_command", required=True)
    rep = p1_sub.add_parser("replay")
    rep.add_argument("--input", required=True)
    rep.add_argument("--as-of-file", required=True)
    rep.add_argument("--out", required=True)
    rep.add_argument("--stages-config", default=None)
    rep.add_argument("--policy-config", default=None)
    rep.add_argument("--seed", type=int, default=DEFAULT_SEED)
    rep.add_argument("--utc", default=None,
                     help="wall-clock timestamp for the manifest "
                          "(ISO-8601 UTC); defaults to the current time. "
                          "Historical runs pass a fixed past timestamp.")
    rep.set_defaults(func=cmd_replay)
    ev = p1_sub.add_parser("evaluate", help="evaluate a replay run on holdout")
    ev.add_argument("--run", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--utc", default=None,
                    help="wall-clock timestamp for the manifest "
                         "(ISO-8601 UTC); defaults to the current time.")
    ev.set_defaults(func=cmd_evaluate)

    mf = sub.add_parser("manifest")
    mf_sub = mf.add_subparsers(dest="manifest_command", required=True)
    inspect = mf_sub.add_parser("inspect")
    inspect.add_argument("--run", required=True)
    inspect.set_defaults(func=cmd_manifest_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
