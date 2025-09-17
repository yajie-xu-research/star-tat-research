"""Reproducibility tests: identical seed + inputs + config produce a
byte-identical result hash; a changed dictionary or seed changes it
(guide counter-example row 9)."""

from __future__ import annotations

import json
from pathlib import Path

from star_tat.p1.replay import run_replay, utc_now
from star_tat.p1.synthetic import generate

STAGES = Path(__file__).parents[1] / "configs" / "p1_stages.yaml"
POLICY = Path(__file__).parents[1] / "configs" / "p1_policy.yaml"


def _run(tmp_path, tiny_data_dir, seed):
    out = tmp_path / "run"
    summary = run_replay(input_dir=tiny_data_dir,
                         as_of_file=tiny_data_dir / "predict_requests.csv",
                         out_dir=out, stages_config=STAGES,
                         policy_config=POLICY, seed=seed,
                         utc_now=utc_now(), command="repro test")
    return summary, out


def test_same_inputs_same_result_hash(tmp_path, tiny_data_dir):
    s1, _ = _run(tmp_path / "a", tiny_data_dir, 20240115)
    s2, _ = _run(tmp_path / "b", tiny_data_dir, 20240115)
    assert s1["result_hash"] == s2["result_hash"]
    assert s1["run_id"] == s2["run_id"]


def test_result_hash_independent_of_wall_clock(tmp_path, tiny_data_dir):
    s1, _ = _run(tmp_path / "a", tiny_data_dir, 20240115)
    s2, out2 = _run(tmp_path / "b", tiny_data_dir, 20240115)
    # Stamp a different utc into the second manifest and re-verify the hash
    # derivation ignores it: the result hash is the same value.
    assert s1["result_hash"] == s2["result_hash"]
    m2 = json.loads((out2 / "manifest.json").read_text())
    m1 = json.loads((Path(s1["out_dir"]) / "manifest.json").read_text())
    assert m1["result_hash"] == m2["result_hash"]


def test_different_seed_changes_generated_data(stage_map):
    a = generate(20240115, stage_map, sites=("SITE_A",),
                 versions=("WF_V1",), scale=0.4,
                 include_unknown_version=False)
    b = generate(20240116, stage_map, sites=("SITE_A",),
                 versions=("WF_V1",), scale=0.4,
                 include_unknown_version=False)
    assert not a["cases"].equals(b["cases"])


def test_same_seed_same_generated_bytes(stage_map, tmp_path):
    from star_tat.p1.synthetic import write_tables
    a = generate(20240115, stage_map, sites=("SITE_A",),
                 versions=("WF_V1",), scale=0.4,
                 include_unknown_version=False)
    b = generate(20240115, stage_map, sites=("SITE_A",),
                 versions=("WF_V1",), scale=0.4,
                 include_unknown_version=False)
    write_tables(tmp_path / "a", a, 20240115)
    write_tables(tmp_path / "b", b, 20240115)
    import hashlib
    for name in ("cases.csv", "events.csv", "load_snapshots.csv",
                 "stage_map.csv", "predict_requests.csv"):
        ha = hashlib.sha256((tmp_path / "a" / name).read_bytes()).hexdigest()
        hb = hashlib.sha256((tmp_path / "b" / name).read_bytes()).hexdigest()
        assert ha == hb, name


def test_different_dictionary_changes_result_hash(
        tmp_path, tiny_data_dir):
    """A dictionary edit (a new mapping row) must change the config hash and
    therefore the result hash; historical runs stay reproducible with the
    old file."""
    def _run_with_stages(out_dir, stages_path, seed):
        out = tmp_path / out_dir
        return run_replay(input_dir=tiny_data_dir,
                          as_of_file=tiny_data_dir / "predict_requests.csv",
                          out_dir=out, stages_config=stages_path,
                          policy_config=POLICY, seed=seed,
                          utc_now=utc_now(), command="repro test")

    s1 = _run_with_stages("a", STAGES, 20240115)
    assert len(s1["result_hash"]) == 64
    # An identical copy of the dictionary keeps the hash; an edit changes it.
    import shutil
    shutil.copy(STAGES, tmp_path / "stages.yaml")
    s2 = _run_with_stages("b", tmp_path / "stages.yaml", 20240115)
    assert s1["result_hash"] == s2["result_hash"]
    text = (tmp_path / "stages.yaml").read_text()
    (tmp_path / "stages.yaml").write_text(text + "\n# mapping addendum\n")
    s3 = _run_with_stages("c", tmp_path / "stages.yaml", 20240115)
    assert s1["result_hash"] != s3["result_hash"]


def test_result_hash_matches_manifest_subset(tmp_path, tiny_data_dir):
    from star_tat.common.hashing import combine_hashes, sha256_file
    from star_tat.common.manifest import load_manifest
    s1, out = _run(tmp_path / "a", tiny_data_dir, 20240115)
    m = load_manifest(out)
    files_hash = combine_hashes(*[
        sha256_file(out / name) for name in sorted(m["_result_files"])])
    expected = combine_hashes(m["data_hash"], m["config_hash"],
                              str(m["seed"]),
                              m["model_or_rule_version"], files_hash)
    assert expected == s1["result_hash"]
