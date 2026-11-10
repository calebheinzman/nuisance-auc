"""Ties and unequal bin masses in the real-data C1/C2 comparison.

For each corpus (zero-shot BiomedCLIP detector, the five screen splits, q = 20):
  * bin-mass imbalance of each model's strata: max/min bin size, coefficient
    of variation of bin sizes, and the share of test tiles in bins whose size
    differs from n/q by more than 25%;
  * the adjusted AUC recomputed after breaking propensity ties at random
    (tiny jitter, 10 draws), which forces equal-count bins, and with bins
    weighted by the equal-mass counterpart of the pair weight
    (pi_k (1 - pi_k) instead of n1_k n0_k).
The default (ties kept, pair weights) must reproduce the panel_folds means
(checked against the reference table when it is present).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import N_SPLIT
from nuisauc.data.cache import CORPORA, corpus_cache, nuisance_models
from nuisauc.detectors.vlm import embed_cache, prompt_embeddings, score
from nuisauc.estimand import strata
from nuisauc.io import write_table
from nuisauc.paths import reference_dir
from nuisauc.propensity import propensity
from nuisauc.seeds import record_run

NAME = "ties"
DESCRIPTION = "Propensity ties, bin-mass imbalance and tie-broken / equal-mass adjusted AUCs"
PAPER = "App F"
REQUIRES = "cache"
OUTPUTS = ["ties"]

CAP = 4000
Q = 20
N_JIT = 10

log = logging.getLogger(f"nuisauc.{NAME}")


def binned_auc(y: np.ndarray, s: np.ndarray, b: np.ndarray, q: int,
               weight: str = "pairs") -> float:
    """Adjusted AUC over given bins; weight "pairs" = n1 n0, else pi (1 - pi)."""
    num = den = 0.0
    for k in range(q):
        m = b == k
        yk = y[m]
        n1, n0 = int(yk.sum()), int(len(yk) - yk.sum())
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0 if weight == "pairs" else (n1 / len(yk)) * (n0 / len(yk))
        num += w * roc_auc_score(yk, s[m])
        den += w
    return num / den


def default_config() -> dict:
    return {"corpora": list(CORPORA)}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Writes `ties`, or `ties_rerun` for a subset of corpora."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpora = cfg["corpora"]
    rows = []
    for corpus in corpora:
        y, _C, _pk = corpus_cache(corpus, CAP)
        T16 = [m for m in nuisance_models(corpus, CAP) if m[0] == "C1"][0][1]
        E, _ = embed_cache("biomedclip", corpus, CAP, dev)
        T = prompt_embeddings("biomedclip", corpus, dev)
        s = score(E, T[f"{corpus}__pos"], T[f"{corpus}__neg"],
                  np.arange(len(T[f"{corpus}__pos"])),
                  np.arange(len(T[f"{corpus}__neg"])))
        for rep in range(N_SPLIT):
            tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                               .split(np.zeros(len(y)), y)))
            yy, ss = y[te], s[te]
            row = dict(corpus=corpus, rep=rep)
            for tag, kind in (("C1", "linear"), ("C2", "gbm")):
                e = propensity(kind, T16, tr, y, rep).predict_proba(T16[te])[:, 1]
                b = strata(e, Q)
                sizes = np.bincount(b, minlength=Q)
                target = len(te) / Q
                row[f"{tag}_n_distinct"] = int(len(np.unique(e)))
                row[f"{tag}_size_ratio"] = float(sizes.max() / max(sizes.min(), 1))
                row[f"{tag}_size_cv"] = float(sizes.std() / sizes.mean())
                row[f"{tag}_share_off25"] = float(
                    sizes[np.abs(sizes - target) > 0.25 * target].sum() / len(te))
                row[f"{tag}_default"] = binned_auc(yy, ss, b, Q)
                row[f"{tag}_eqmass_w"] = binned_auc(yy, ss, b, Q, weight="prop")
                rng = np.random.default_rng([rep, 7])
                jit = []
                for _ in range(N_JIT):
                    ej = e + rng.uniform(-1e-9, 1e-9, len(e))
                    jit.append(binned_auc(yy, ss, strata(ej, Q), Q))
                row[f"{tag}_tiebreak"] = float(np.mean(jit))
            rows.append(row)
        log.info("%s done", corpus)
    d = pd.DataFrame(rows)
    name = "ties" if list(corpora) == CORPORA else "ties_rerun"
    write_table(d, out_dir, name)
    ref_p = reference_dir() / "panel_folds.csv"
    if ref_p.exists():
        ref = pd.read_csv(ref_p)
        ref = ref[ref.detector == "zero_shot"].groupby("corpus")[["C1", "C2"]].mean()
        g = d.groupby("corpus")[["C1_default", "C2_default"]].mean()
        g.columns = ["C1", "C2"]
        assert (g - ref.loc[g.index]).abs().max().max() < 5e-4
        log.info("default reproduces panel_folds: ok")
    else:
        log.warning("no reference panel_folds table at %s; check skipped", ref_p)
    record_run(out_dir, NAME, cfg)
    return {name: d}


def summarize(tables: dict) -> str:
    d = next(iter(tables.values())).copy()
    for v in ("default", "eqmass_w", "tiebreak"):
        d[f"gap_{v}"] = d[f"C1_{v}"] - d[f"C2_{v}"]
    return (d.groupby("corpus").mean(numeric_only=True).drop(columns="rep")
            .round(4).T.to_string())
