"""Label-fit class selection when the linear class is materially better.

Same machinery as the semi-synthetic part of `selection_validation` (real
trivial-16 geometry, synthetic label and score, fold-matched truth with 100
bins of the true core, cross-fitted log-loss selection with |z| >= 2
abstention), with label mechanisms on which the tree class pays an
approximation cost:

    dense   core = std(T16s @ 1)     all 16 standardised features, equal weights
    random  core = std(T16s @ w1)    the fixed random dense direction

score s = u + std(T16s @ w2) + eps; alpha in {2, 3}; audit size n in
{1000, 2000}; 12 replicates; five corpora (40 conditions, 480 cells).
"""
from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.data.corpora import CORPORA
from nuisauc.estimand import conditional_auc, n_strata, pair_weighted, qbins
from nuisauc.inference.selection import crossfit_scores, select_family
from nuisauc.io import write_table
from nuisauc.propensity import fit_any
from nuisauc.seeds import record_run
from nuisauc.simulate.gaussian import Q_FINE
from nuisauc.simulate.semisynthetic import ADJ_SEED, adj_load_geometry, adj_synth

NAME = "selection_linear"
DESCRIPTION = ("Class selection versus always-trees on regimes where the "
               "linear class is materially better")
PAPER = "App E"
REQUIRES = "cache"
OUTPUTS = ["adjudication_linear"]

ALPHAS = [2.0, 3.0]
NS = [1000, 2000]
N_REP = 12
DRAW_SEED = 61008

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"alphas": list(ALPHAS), "ns": list(NS), "n_rep": N_REP,
            "cap": 4000, "corpora": list(CORPORA)}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for corpus in cfg["corpora"]:
        T16, T16s, *_ = adj_load_geometry(corpus, cfg["cap"])
        std = lambda v: v / v.std()  # noqa: E731
        rd = np.random.default_rng(ADJ_SEED)
        w1 = rd.standard_normal(16)
        w2 = rd.standard_normal(16)
        g_f = std(T16s @ w2)
        for regime, core_f in (("dense", std(T16s.sum(1))),
                               ("random", std(T16s @ w1))):
            for alpha in cfg["alphas"]:
                for n in cfg["ns"]:
                    for rep in range(cfg["n_rep"]):
                        rng = np.random.default_rng(
                            [DRAW_SEED, zlib.crc32(corpus.encode()) % 2 ** 16,
                             zlib.crc32(regime.encode()) % 2 ** 16,
                             int(alpha * 10), n, rep])
                        keep = rng.choice(len(core_f), n, replace=False)
                        core, g, X = core_f[keep], g_f[keep], T16[keep]
                        y, u = adj_synth(core, alpha, rng)
                        s = u + g + rng.standard_normal(n)
                        tr, te = next(iter(StratifiedKFold(
                            4, shuffle=True, random_state=rep)
                            .split(np.zeros(n), y)))
                        q = n_strata(len(te))
                        tB = pair_weighted(y[te], s[te], qbins(core[te], Q_FINE))[0]
                        sc = crossfit_scores(X, y, tr, rep)
                        verdict, _m, fam, z = select_family(sc, 0)
                        errs = {}
                        for f_, (tag, _ll) in fam.items():
                            e = fit_any(tag, X, y, tr, rep).predict_proba(X[te])[:, 1]
                            errs[f_] = abs(float(conditional_auc(y[te], s[te], e, q)[0]) - tB)
                        rows.append(dict(corpus=corpus, regime=regime,
                                         alpha=alpha, n=n, rep=rep,
                                         theta_B=float(tB), verdict=verdict,
                                         z=z, err_linear=errs["linear"],
                                         err_tree=errs["tree"],
                                         member_lin=fam["linear"][0],
                                         member_tree=fam["tree"][0]))
                    g = pd.DataFrame(rows)
                    g = g[(g.corpus == corpus) & (g.regime == regime)
                          & (g.alpha == alpha) & (g.n == n)]
                    log.info("%-20s %-6s a=%s n=%s: err lin %.4f tree %.4f  "
                             "verdicts %s", corpus, regime, alpha, n,
                             g.err_linear.mean(), g.err_tree.mean(),
                             g.verdict.value_counts().to_dict())
    d = pd.DataFrame(rows)
    write_table(d, out_dir, "adjudication_linear")
    record_run(out_dir, NAME, cfg)
    return {"adjudication_linear": d}


def summarize(tables: dict) -> str:
    d = tables["adjudication_linear"].copy()
    # Abstentions are scored as the always-trees fallback.
    d["err_rule"] = np.where(d.verdict == "linear", d.err_linear, d.err_tree)
    return (f"per-cell mean |err|: rule {d.err_rule.mean():.4f}  "
            f"always-tree {d.err_tree.mean():.4f}  always-linear "
            f"{d.err_linear.mean():.4f}")
