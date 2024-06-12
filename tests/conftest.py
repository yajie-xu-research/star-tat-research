"""Shared fixtures for the star-tat test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from star_tat.p1 import io  # noqa: E402
from star_tat.p1.stage_map import StageMap  # noqa: E402
from star_tat.p1.synthetic import generate, write_tables  # noqa: E402

CONFIGS = REPO / "configs"
STAGES = CONFIGS / "p1_stages.yaml"
POLICY = CONFIGS / "p1_policy.yaml"


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def stage_map() -> StageMap:
    return StageMap(STAGES)


@pytest.fixture(scope="session")
def stages_cfg() -> dict:
    with STAGES.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def policy_cfg() -> dict:
    with POLICY.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def tiny_frames(stage_map) -> dict:
    """One-stratum data at reduced scale: calib block stays above the
    policy floor so the replay exercises the full OUTPUT path fast."""
    return generate(20240115, stage_map, sites=("SITE_A",),
                    versions=("WF_V1",), scale=0.6,
                    include_unknown_version=False)


@pytest.fixture(scope="session")
def tiny_data_dir(tmp_path_factory, stage_map) -> Path:
    frames = generate(20240115, stage_map, sites=("SITE_A",),
                      versions=("WF_V1",), scale=0.6,
                      include_unknown_version=False)
    out = tmp_path_factory.mktemp("tiny_data")
    write_tables(out, frames, 20240115)
    return out


@pytest.fixture(scope="session")
def tiny_replay_dir(tmp_path_factory, tiny_data_dir, policy_cfg) -> Path:
    from star_tat.p1.replay import run_replay, utc_now
    out = tmp_path_factory.mktemp("tiny_replay")
    run_replay(
        input_dir=tiny_data_dir,
        as_of_file=tiny_data_dir / "predict_requests.csv",
        out_dir=out,
        stages_config=STAGES,
        policy_config=POLICY,
        seed=20240115,
        utc_now=utc_now(),
        command="pytest fixture replay",
    )
    return out
