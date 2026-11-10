"""Experiment registry.

Each experiment is a module in this package that defines:

    NAME: str                 registry name (equals the module name)
    DESCRIPTION: str          one line
    PAPER: str                where the results appear in the paper
    REQUIRES: str             "none" | "cache" (image-derived inputs in NUISAUC_CACHE)
                              | "raw" (raw images in NUISAUC_RAW)
    OUTPUTS: list[str]        names of the tables it writes
    def default_config() -> dict
    def run(cfg: dict, out_dir: Path) -> dict[str, pandas.DataFrame]
    def summarize(tables: dict) -> str      (optional)

Modules are imported lazily so that listing experiments does not import the
image stack.
"""
from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

# Order follows the paper.
NAMES: list[str] = [
    # Section 3 and Appendix A-B: the regime rule
    "theory_law", "theory_montecarlo", "law_grid", "control_bins",
    # Section 4: what the checks miss
    "family_i", "families_ii_iii", "cutoff_sweep", "overlap_medmnist",
    # Sections 3-4 on real descriptors with synthetic labels
    "semisynthetic_checks", "semisynthetic_regimes", "semisynthetic_sensitivity",
    "oracle_bins",
    # Section 5: real labels
    "realdata_screen", "realdata_enrichment", "realdata_panel", "realdata_learners",
    "realdata_uncertainty", "bins_balance", "ties", "colour_shift",
    # Section 5 and Appendix E: label-fit selection
    "selection_validation", "selection_cluster", "selection_replication",
    "selection_linear",
    # Section 5 and Appendix D: held-out tests
    "heldout_statistic", "heldout_specificity", "heldout_hob", "heldout_lod",
    "heldout_null", "heldout_attribution",
]


def get(name: str) -> ModuleType:
    if name not in NAMES:
        raise KeyError(f"unknown experiment {name!r}; see `nuisauc list`")
    return importlib.import_module(f"nuisauc.experiments.{name}")


def available() -> list[str]:
    """Registered experiments whose module exists."""
    return [n for n in NAMES
            if importlib.util.find_spec(f"nuisauc.experiments.{n}") is not None]
