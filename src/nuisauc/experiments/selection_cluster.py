"""Cluster-aware inference for the real-data label-fit class selection.

The paired per-tile log-loss z treats tiles as independent. Here the
cross-fitted per-tile log-loss difference (linear member minus tree member)
on each training fold is bootstrapped over dependence units (Kaggle slides;
DINOv2 near-duplicate clusters at PRIMARY_TAU elsewhere), B = 2000:

  S1  the five original outer folds of the real-data selection;
  S2  five StratifiedGroupKFold(4) splits with the unit as group
      (random_state 100 + r), so no unit straddles train and test.

The tree class is significant on a fold when the 95% percentile CI of the
mean difference lies above 0; each check needs >= 4/5 folds per corpus.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from nuisauc.data.cache import corpus_cache, nuisance_models
from nuisauc.data.corpora import PANEL_A
from nuisauc.data.groups import cluster_units_for
from nuisauc.inference.bootstrap import CLUSTER_N_BOOT, cluster_ci
from nuisauc.inference.selection import crossfit_scores, family_dl
from nuisauc.io import write_table
from nuisauc.seeds import record_run

NAME = "selection_cluster"
DESCRIPTION = ("Cluster-bootstrap inference for the real-data class selection "
               "on original and unit-grouped folds")
PAPER = "App E"
REQUIRES = "cache"
OUTPUTS = ["adjudication_cluster"]

N_SPLIT = 5

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"cap": 4000, "n_boot": CLUSTER_N_BOOT}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for corpus in PANEL_A:
        y, _C, _pk = corpus_cache(corpus, cfg["cap"])
        T16 = [m for m in nuisance_models(corpus, cfg["cap"])
               if m[0] == "C1"][0][1]
        units, unit_kind = cluster_units_for(corpus, y, cfg["cap"])
        n_units = len(np.unique(units))
        big = int(pd.Series(units).value_counts().iloc[0])
        log.info("%s: unit=%s, %d units, largest %d tiles",
                 corpus, unit_kind, n_units, big)

        for mode, splitter in (
            ("original", [next(iter(StratifiedKFold(
                4, shuffle=True, random_state=r)
                .split(np.zeros(len(y)), y))) for r in range(N_SPLIT)]),
            ("grouped", [next(iter(StratifiedGroupKFold(
                4, shuffle=True, random_state=100 + r)
                .split(np.zeros(len(y)), y, groups=units)))
                for r in range(N_SPLIT)]),
        ):
            for rep, (tr, te) in enumerate(splitter):
                seed = rep if mode == "original" else 100 + rep
                scores = crossfit_scores(T16, y, tr, seed)
                dl, lin, tre = family_dl(scores)
                lo, hi, zc = cluster_ci(dl, units[tr], 7000 + seed,
                                        n_boot=cfg["n_boot"])
                sig_tree = lo > 0
                rows.append(dict(
                    corpus=corpus, mode=mode, rep=rep,
                    unit=unit_kind, n_units_tr=len(np.unique(units[tr])),
                    mean_dl=float(dl.mean()), ci_lo=lo, ci_hi=hi,
                    z_cluster=zc, member_lin=lin, member_tree=tre,
                    sig_tree=bool(sig_tree)))
                log.info("%s %s fold %d: mean dl %+.4f CI [%+.4f,%+.4f] "
                         "z_cl %5.1f %s", corpus, mode, rep, dl.mean(), lo, hi,
                         zc, "tree*" if sig_tree else "NOT SIG")
        gc.collect()
    d = pd.DataFrame(rows)
    write_table(d, out_dir, "adjudication_cluster")
    record_run(out_dir, NAME, cfg)
    return {"adjudication_cluster": d}


def summarize(tables: dict) -> str:
    d = tables["adjudication_cluster"]
    out = ["=" * 78, "  Cluster-aware inference for the class selection", "=" * 78]
    ok_all = True
    for corpus, g in d.groupby("corpus"):
        for gate, mode in (("S1", "original"), ("S2", "grouped")):
            gg = g[g["mode"] == mode]
            k = int(gg.sig_tree.sum())
            ok = k >= 4
            ok_all &= ok
            out.append(f"  {gate} {corpus:<22} {mode:<9} tree significant "
                       f"{k}/{len(gg)} folds  z_cl "
                       f"{gg.z_cluster.min():.1f}-{gg.z_cluster.max():.1f}  "
                       f"--> {'PASS' if ok else 'FAIL'}")
    out.append(f"  overall: {'ALL PASS' if ok_all else 'AT LEAST ONE FAIL'}")
    return "\n".join(out)
