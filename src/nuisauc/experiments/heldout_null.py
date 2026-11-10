"""False-positive rate of the split-family test on real descriptors when the
nuisance model's inputs are correct.

Per corpus, the real 62-feature descriptor A and the real held-out family B
(frequency or morphology). Synthetic labels depend on A only:
    logit P(y=1) = alpha core(A) + u,  u ~ N(0,1) independent of (A, B)
    core "linear"    = standardised A @ w, w a fixed random direction
    core "nonlinear" = std(p1 p2 + 0.7 (p3^2 - 1)) on A's first three PCs
alpha in {1, 2}. Per (corpus, family, core, alpha): 5 outer folds (fresh
label draw per fold) x 6 random halvings = 30 EXC values, computed as on the
real corpora. Reported against the archived threshold 0.0167.
"""
from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from nuisauc.data.cache import CORPORA, descriptor62
from nuisauc.data.descriptors import frequency16, morph16
from nuisauc.diagnostics.heldout import exc_statistic
from nuisauc.estimand import n_strata
from nuisauc.io import write_table
from nuisauc.seeds import record_run

NAME = "heldout_null"
DESCRIPTION = "Split-family test's false-firing rate on real descriptors with correctly specified synthetic labels"
PAPER = "App D"
REQUIRES = "cache"
OUTPUTS = ["sff_null"]

SEED = 61006
THRESHOLD = 0.0167
N_FOLD, N_HALF = 5, 6
ALPHAS = [1.0, 2.0]

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"cap": 4000, "corpora": list(CORPORA)}


def cores(A: np.ndarray) -> dict[str, np.ndarray]:
    """The linear and non-linear label cores built from A."""
    As = StandardScaler().fit_transform(A)
    w = np.random.default_rng(SEED).standard_normal(As.shape[1])
    lin = As @ w
    p = PCA(n_components=3, random_state=0).fit_transform(As)
    p = p / p.std(0)
    non = p[:, 0] * p[:, 1] + 0.7 * (p[:, 2] ** 2 - 1)
    return {"linear": lin / lin.std(), "nonlinear": non / non.std()}


def _crc16(s: str) -> int:
    return zlib.crc32(s.encode()) % 2 ** 16


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    cap = cfg["cap"]
    rows = []
    for corpus in cfg["corpora"]:
        A = descriptor62(corpus, cap)
        fams = (("frequency", frequency16(corpus, cap)),
                ("morphology", morph16(corpus, cap)))
        for cname, core in cores(A).items():
            for alpha in ALPHAS:
                for fam, B in fams:
                    for rep in range(N_FOLD):
                        rng = np.random.default_rng(
                            [SEED, _crc16(corpus), _crc16(cname),
                             int(alpha * 10), rep])
                        u = rng.standard_normal(len(core))
                        logit = alpha * core + u
                        y = (rng.random(len(core))
                             < 1 / (1 + np.exp(-logit))).astype(int)
                        tr, te = next(iter(StratifiedKFold(
                            4, shuffle=True, random_state=rep)
                            .split(np.zeros(len(y)), y)))
                        for hs in range(N_HALF):
                            hr = np.random.default_rng(
                                [SEED, 9, _crc16(fam), rep, hs])
                            exc, floor = exc_statistic(
                                A, B, y, tr, te, n_strata(len(te)), rep, hr)
                            rows.append(dict(corpus=corpus, core=cname,
                                             alpha=alpha, family=fam,
                                             rep=rep, half=hs, exc=exc,
                                             floor=floor))
                    g = pd.DataFrame(rows)
                    g = g[(g.corpus == corpus) & (g.core == cname)
                          & (g.alpha == alpha) & (g.family == fam)]
                    log.info("%s %s a=%s %s median EXC %+.4f fire %.2f",
                             corpus, cname, alpha, fam, g.exc.median(),
                             (g.exc > THRESHOLD).mean())
    d = pd.DataFrame(rows)
    write_table(d, out_dir, OUTPUTS[0])
    record_run(out_dir, NAME, cfg)
    return {OUTPUTS[0]: d}


def summarize(tables: dict) -> str:
    d = tables[OUTPUTS[0]]
    med = d.groupby(["corpus", "core", "alpha", "family"]).exc.median()
    out = [f"per-halving null fire rate: {(d.exc > THRESHOLD).mean():.3f}",
           f"conditions with median EXC > {THRESHOLD}: "
           f"{int((med > THRESHOLD).sum())}/{len(med)}",
           "per-corpus fire rate:",
           d.groupby("corpus").exc.apply(lambda v: (v > THRESHOLD).mean()).to_string()]
    return "\n".join(out)
