"""The regime law on a fine loading grid around the reversal boundary.

Family I   Z1, Z2, U, eps ~ N(0,1); logit = alpha (Z1+Z2) + U;
           S = U + lambda (Z1 - Z2) + eps; wrong stratifier Z1;
           alpha*lambda in {0, .5, .8, .9, .95, 1, 1.05, 1.1, 1.2, 1.5, 2, 4}.
Family III Z ~ N(0, AR1(0.6)) in R^12, a = 1/sd, logit = alpha a'Z + U,
           S = U + lambda h'Z + eps, h = hidden-six mean / sd; the wrong model
           sees the first six columns and is fitted both by logistic regression
           (the ideal index's estimator) and by 200-iteration boosted trees.
Law = Gauss-Hermite quadrature of the analytic within-stratum AUC, pair
weighted, logistic link. Simulation: n = 20,000, 75/25 split, 20 quantile
strata on the test fold, 10 replicates. Before running, the law is checked
against three Family I cells of Table 1.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from nuisauc.estimand import strat_auc
from nuisauc.io import write_table
from nuisauc.seeds import record_run
from nuisauc.theory.law import law

NAME = "law_grid"
DESCRIPTION = "Law vs simulation on a fine loading grid around the reversal boundary"
PAPER = "App B"
REQUIRES = "none"
OUTPUTS = ["law_grid"]

ALPHAS = (0.5, 1.0, 2.0)
KAPPAS = (0, .5, .8, .9, .95, 1, 1.05, 1.1, 1.2, 1.5, 2, 4)   # alpha * lambda, Family I
LAMBDAS_III = (0.0, 0.5, 1.0, 1.5, 2.0)
SEED_I, SEED_III = 61009, 61010
# (alpha, lambda, truth, wrong) Family I cells of Table 1 the law must reproduce
TABLE1_CELLS = ((1, 2, 0.575, 0.415), (2, 2, 0.576, 0.296), (1, 0, 0.670, 0.648))
TABLE1_TOL = 1.5e-3

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n": 20_000, "reps": 10, "q": 20}


def sim(Z, a_logit, s, fits, rng, q):
    y = (rng.random(len(s)) < expit(a_logit)).astype(int)
    tr = rng.random(len(s)) < 0.75
    te = ~tr
    out = {}
    for name, (cols, kind) in fits.items():
        X = Z[:, cols]
        m = (LogisticRegression(max_iter=2000) if kind == "lin"
             else HistGradientBoostingClassifier(max_iter=200, random_state=0))
        m.fit(X[tr], y[tr])
        out[name] = strat_auc(y[te], s[te], m.predict_proba(X[te])[:, 1], q)
    return out


def family_I(rows, n, reps, q):
    S = np.eye(2)
    for alpha in ALPHAS:
        for kap in KAPPAS:
            lam = kap / alpha
            a, b = alpha * np.array([1., 1.]), lam * np.array([1., -1.])
            truth, rt = law(S, a, b, 1.0, np.array([1., 1.]))
            wrong, rw = law(S, a, b, 1.0, np.array([1., 0.]))
            sims = []
            for rep in range(reps):
                rng = np.random.default_rng([SEED_I, int(alpha * 10), int(kap * 100), rep])
                Z = rng.standard_normal((n, 2)); u, e = rng.standard_normal((2, n))
                sims.append(sim(Z, Z @ a + u, u + Z @ b + e,
                                {"truth": ([0, 1], "lin"), "wrong": ([0], "lin")}, rng, q))
            sd = pd.DataFrame(sims)
            rows.append(dict(family="I", alpha=alpha, alpha_lambda=kap, lam=lam,
                             rho_wrong=rw, law_truth=truth, law_wrong=wrong,
                             sim_truth=sd.truth.mean(), sim_wrong=sd.wrong.mean(),
                             sim_wrong_sd=sd.wrong.std(ddof=1),
                             sim_wrong_below_half=int((sd.wrong < 0.5).sum())))
            r = rows[-1]
            log.info("I  a=%s al=%-4s rho=%+.3f law %.3f sim %.3f (truth %.3f/%.3f)",
                     alpha, kap, rw, wrong, r["sim_wrong"], truth, r["sim_truth"])


def family_III(rows, n, reps, q):
    p = 12
    idx = np.arange(p)
    Sig = 0.6 ** np.abs(idx[:, None] - idx[None, :])
    one = np.ones(p)
    a0 = one / np.sqrt(one @ Sig @ one)
    hraw = np.r_[np.zeros(6), np.ones(6)]
    h = hraw / np.sqrt(hraw @ Sig @ hraw)
    Sv, Hd = idx[:6], idx[6:]
    L = np.linalg.cholesky(Sig)
    for alpha in ALPHAS:
        for lam in LAMBDAS_III:
            a, b = alpha * a0, lam * h
            g = a[Sv] + np.linalg.solve(Sig[np.ix_(Sv, Sv)], Sig[np.ix_(Sv, Hd)] @ a[Hd])
            w = np.r_[g, np.zeros(6)]
            truth, _ = law(Sig, a, b, 1.0, a)
            wrong, rw = law(Sig, a, b, 1.0, w)
            sims = []
            for rep in range(reps):
                rng = np.random.default_rng([SEED_III, int(alpha * 10), int(lam * 10), rep])
                Z = rng.standard_normal((n, p)) @ L.T
                u, e = rng.standard_normal((2, n))
                sims.append(sim(Z, Z @ a + u, u + Z @ b + e,
                                {"truth": (list(idx), "lin"),
                                 "wrong_lin": (list(Sv), "lin"),
                                 "wrong_tree": (list(Sv), "tree")}, rng, q))
            sd = pd.DataFrame(sims)
            rows.append(dict(family="III", alpha=alpha, lam=lam, rho_wrong=rw,
                             law_truth=truth, law_wrong=wrong,
                             sim_truth=sd.truth.mean(),
                             sim_wrong=sd.wrong_lin.mean(),
                             sim_wrong_tree=sd.wrong_tree.mean()))
            r = rows[-1]
            log.info("III a=%s l=%s law %.3f sim-lin %.3f sim-tree %.3f (truth %.3f/%.3f)",
                     alpha, lam, wrong, r["sim_wrong"], r["sim_wrong_tree"], truth,
                     r["sim_truth"])


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    S = np.eye(2)
    for (al, lm, t, wv) in TABLE1_CELLS:
        a, b = al * np.array([1., 1.]), lm * np.array([1., -1.])
        tt = law(S, a, b, 1.0, np.array([1., 1.]))[0]
        ww = law(S, a, b, 1.0, np.array([1., 0.]))[0]
        assert abs(tt - t) < TABLE1_TOL and abs(ww - wv) < TABLE1_TOL, (al, lm, tt, ww)
    log.info("law reproduces the Table 1 Family I cells")
    rows: list[dict] = []
    family_I(rows, cfg["n"], cfg["reps"], cfg["q"])
    family_III(rows, cfg["n"], cfg["reps"], cfg["q"])
    out = pd.DataFrame(rows)
    write_table(out, out_dir, "law_grid")
    record_run(out_dir, NAME, cfg)
    return {"law_grid": out}
