"""Families II and III: the screen misses misspecification beyond one Gaussian family.

Family II, "nonlinear" (a functional-form omission):
  Z1, Z2, U, eps ~ N(0,1); core = (Z1 + 0.7 (Z2^2 - 1)) / sd;
  logit P(y=1) = alpha core + U;  S = U + lam (Z2^2 - 1)/sqrt(2) + eps.
  Correct: boosted trees on (Z1, Z2). Misspecified: a LINEAR model on (Z1, Z2),
  which cannot represent Z2^2.
Family III, "correlated" (a high-dimensional, partially absorbed omission):
  Z in R^12, AR(1) correlation 0.6; core = (1'Z) / sd; logit = alpha core + U;
  S = U + lam mean(Z_7..Z_12) / sd + eps.
  Correct: boosted trees on all twelve columns. Misspecified: boosted trees on
  the first six.
theta_B stratifies on the true label core; the screen is degen <= 0.20 and the
self-control in [0.45, 0.55].

Gates (fixed in advance):
  L1  correct-fit mean |est - theta_B| <= 0.03 per family
  L2  per family, >= 20 misspecified cells with err > 0.05, >= 80% of which
      pass the screen
  L3  worst-cell error per family (reported)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import DEGEN_MAX, NEG_WINDOW
from nuisauc.estimand import conditional_auc, n_strata, pair_weighted, qbins
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple
from nuisauc.seeds import record_run
from nuisauc.simulate.gaussian import ALPHAS, LAMBDAS, Q_FINE

NAME = "families_ii_iii"
DESCRIPTION = "Screen vs misspecification in a nonlinear and a correlated 12-D family"
PAPER = "§4, App B"
REQUIRES = "none"
OUTPUTS = ["screen_families"]

SEED = 4300
N = 20_000
BAD_TOL = 0.05
L1_TOL = 0.03
L2_MIN_CELLS = 20
L2_FRAC = 0.80
AR_RHO = 0.6
P_DIM = 12
P_SEEN = 6

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": 10}


def sim_nonlinear(alpha, lam, rng):
    z1, z2, u, eps = rng.standard_normal((4, N))
    core = z1 + 0.7 * (z2 ** 2 - 1.0)
    core = core / np.std(core)
    logit = alpha * core + u
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    s = u + lam * (z2 ** 2 - 1.0) / np.sqrt(2.0) + eps
    return np.column_stack([z1, z2]), y, s, core


def sim_correlated(alpha, lam, rng):
    L = np.linalg.cholesky(AR_RHO ** np.abs(
        np.subtract.outer(np.arange(P_DIM), np.arange(P_DIM))))
    Z = rng.standard_normal((N, P_DIM)) @ L.T
    u, eps = rng.standard_normal((2, N))
    core = Z @ np.ones(P_DIM)
    core = core / np.std(core)
    logit = alpha * core + u
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    hidden = Z[:, P_SEEN:].mean(axis=1)
    s = u + lam * hidden / np.std(hidden) + eps
    return Z, y, s, core


# fit kinds: "gbm" boosted trees, anything else standardised logistic regression
FAMILIES = {
    "nonlinear": dict(sim=sim_nonlinear,
                      correct=("gbm", slice(None)),
                      mis=("linear", slice(None))),
    "correlated": dict(sim=sim_correlated,
                       correct=("gbm", slice(None)),
                       mis=("gbm", slice(0, P_SEEN))),
}


def run_cell(fam, alpha, lam, rep):
    spec = FAMILIES[fam]
    rng = np.random.default_rng([SEED, list(FAMILIES).index(fam),
                                 int(alpha * 10), int(lam * 10), rep])
    Z, y, s, core = spec["sim"](alpha, lam, rng)
    tB = pair_weighted(y, s, qbins(core, Q_FINE))[0]
    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                       .split(np.zeros(len(y)), y)))
    q = n_strata(len(te))
    out = []
    for role in ("correct", "mis"):
        kind, cols = spec[role]
        X = Z[:, cols]
        m = fit_simple(kind, X, y, tr, rep)
        e = m.predict_proba(X[te])[:, 1]
        est, degen = conditional_auc(y[te], s[te], e, q)
        neg, _ = conditional_auc(y[te], e, e, q)
        passes = (degen <= DEGEN_MAX
                  and NEG_WINDOW[0] <= neg <= NEG_WINDOW[1])
        out.append(dict(family=fam, alpha=alpha, lam=lam, rep=rep,
                        fit=role, theta_B=float(tB), est=float(est),
                        err=abs(float(est) - float(tB)),
                        degen=float(degen), neg=float(neg),
                        screen=bool(passes)))
    return out


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for fam in FAMILIES:
        for alpha in ALPHAS:
            for lam in LAMBDAS:
                for rep in range(cfg["n_rep"]):
                    rows.extend(run_cell(fam, alpha, lam, rep))
            log.info("%-12s alpha=%s done", fam, alpha)
    d = pd.DataFrame(rows)
    write_table(d, out_dir, "screen_families")
    record_run(out_dir, NAME, cfg)
    return {"screen_families": d}


def summarize(tables: dict) -> str:
    d = tables["screen_families"]
    out = ["GATES"]
    for fam, g in d.groupby("family"):
        c = g[g.fit == "correct"]
        m = g[g.fit == "mis"]
        bad = m[m.err > BAD_TOL]
        l1 = c.err.mean() <= L1_TOL
        frac = float(bad.screen.mean()) if len(bad) else float("nan")
        l2 = len(bad) >= L2_MIN_CELLS and frac >= L2_FRAC
        out.append(f"\n  {fam}")
        out.append(f"    L1 correct mean err {c.err.mean():.4f} "
                   f"(max {c.err.max():.4f})  --> {'PASS' if l1 else 'FAIL'}")
        out.append(f"    L2 bad mis cells {len(bad)}, screen-pass {frac:.2f}  "
                   f"--> {'PASS' if l2 else 'FAIL'}")
        if len(bad):
            w = bad.loc[bad.err.idxmax()]
            out.append(f"    L3 worst cell: est {w.est:.3f} vs theta_B "
                       f"{w.theta_B:.3f} (err {w.err:.3f}), "
                       f"screen={'pass' if w.screen else 'fail'}")
        out.append(f"    mis mean err {m.err.mean():.4f}   "
                   f"screen pass rate correct={float(c.screen.mean()):.2f} "
                   f"mis={float(m.screen.mean()):.2f}")
    return "\n".join(out)
