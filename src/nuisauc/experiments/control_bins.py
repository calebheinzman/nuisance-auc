"""How the self-stratified control behaves with the number of strata, in Family I.

Family I (alpha in {0.5, 1, 2}, lambda in {0, 1, 2}), n = 20,000, 75/25 split,
10 replicates; the logistic model omitting Z2 and the correct logistic model on
(Z1, Z2). For q in {4, 10, 20, 50}: each model's control value, the screen
verdict (degenerate strata <= 20% and control in [0.45, 0.55]) and the error of
its report against the correct model's report at the same q. With few strata
the control can leave the window, so the pass is near-certain, not guaranteed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from nuisauc.estimand import strat_auc
from nuisauc.io import write_table
from nuisauc.seeds import record_run

NAME = "control_bins"
DESCRIPTION = "Self-stratified control and screen verdict vs the number of strata, Family I"
PAPER = "App B"
REQUIRES = "none"
OUTPUTS = ["control_bins"]

SEED = 61011
ALPHAS = (0.5, 1.0, 2.0)
LAMBDAS = (0.0, 1.0, 2.0)
QS = (4, 10, 20, 50)
MATERIAL_ERR = 0.05

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n": 20_000, "reps": 10}


def degen(y, e, q):
    """Share of the q quantile strata of e that hold a single class."""
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.digitize(e, edges[1:-1]), 0, q - 1)
    bad = sum(1 for k in range(q) if len(set(y[b == k])) < 2)
    return bad / q


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    n = cfg["n"]
    rows = []
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            for rep in range(cfg["reps"]):
                rng = np.random.default_rng([SEED, int(alpha * 10), int(lam * 10), rep])
                Z = rng.standard_normal((n, 2)); u, eps = rng.standard_normal((2, n))
                y = (rng.random(n) < expit(alpha * Z.sum(1) + u)).astype(int)
                s = u + lam * (Z[:, 0] - Z[:, 1]) + eps
                tr = rng.random(n) < 0.75; te = ~tr
                e_ok = LogisticRegression().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
                e_bad = LogisticRegression().fit(Z[tr][:, :1], y[tr]).predict_proba(Z[te][:, :1])[:, 1]
                for q in QS:
                    truth = strat_auc(y[te], s[te], e_ok, q)
                    for name, e in (("correct", e_ok), ("omits_Z2", e_bad)):
                        ctl = strat_auc(y[te], e, e, q)
                        dg = degen(y[te], e, q)
                        rows.append(dict(alpha=alpha, lam=lam, rep=rep, q=q,
                                         model=name, control=ctl, degen=dg,
                                         passes=bool(dg <= 0.2 and 0.45 <= ctl <= 0.55),
                                         err=strat_auc(y[te], s[te], e, q) - truth))
            log.info("alpha=%s lam=%s done", alpha, lam)
    d = pd.DataFrame(rows)
    write_table(d, out_dir, "control_bins")
    record_run(out_dir, NAME, cfg)
    return {"control_bins": d}


def summarize(tables: dict) -> str:
    d = tables["control_bins"]
    lines = []
    for (q, m), g in d.groupby(["q", "model"]):
        bad = g[g.err.abs() > MATERIAL_ERR]
        lines.append(f"q={q:<3} {m:<9} control mean {g.control.mean():.3f} "
                     f"range [{g.control.min():.3f},{g.control.max():.3f}] pass "
                     f"{g.passes.mean():.2f}; materially wrong {len(bad)} pass "
                     f"{bad.passes.sum()}")
    return "\n".join(lines)
