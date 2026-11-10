"""Registry contract tests, config consistency, and end-to-end reruns."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nuisauc import experiments
from nuisauc.cli import main as cli_main
from nuisauc.io import compare_tables, read_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"


@pytest.mark.parametrize("name", experiments.NAMES)
def test_contract(name):
    m = experiments.get(name)
    assert m.NAME == name
    assert isinstance(m.DESCRIPTION, str) and m.DESCRIPTION
    assert isinstance(m.PAPER, str) and m.PAPER
    assert m.REQUIRES in {"none", "cache", "raw"}
    assert m.OUTPUTS and all(isinstance(o, str) for o in m.OUTPUTS)
    assert isinstance(m.default_config(), dict)
    assert callable(m.run)


@pytest.mark.parametrize("name", experiments.NAMES)
def test_config_file_matches_defaults(name):
    p = CONFIGS / f"{name}.yaml"
    assert p.exists(), f"missing {p}"
    assert yaml.safe_load(p.read_text(encoding="utf-8")) == experiments.get(name).default_config()


@pytest.mark.parametrize("name", experiments.NAMES)
def test_reference_tables_present(name):
    m = experiments.get(name)
    missing = [o for o in m.OUTPUTS if not (RESULTS / f"{o}.csv").exists()]
    assert not missing, missing


FAST = ["theory_law", "control_bins"]


@pytest.mark.slow
@pytest.mark.parametrize("name", FAST)
def test_rerun_matches_reference(name, tmp_path, monkeypatch):
    monkeypatch.setenv("NUISAUC_REFERENCE", str(RESULTS))
    assert cli_main(["run", name, "--out", str(tmp_path)]) == 0
    for t in experiments.get(name).OUTPUTS:
        c = compare_tables(read_table(tmp_path, t), read_table(RESULTS, t), t)
        assert c.equal, str(c)
