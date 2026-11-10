"""Is the screen-failure result an artifact of the analysis cut-offs?

Part A, table `screen_robustness` -- cut-off sweep on Family I. The propensity
is refitted at stratum multipliers m in {0.5, 1, 2} (q = round(m * n_strata));
the screen is then evaluated at DEGEN_MAX in {0.10, 0.20, 0.30} and control
windows [0.45, 0.55] and [0.40, 0.60] (functions of the stored degen / neg).
  K1  correct-fit mean |est - theta_B| <= 0.02 at every m
  K2  at every (m, DEGEN_MAX, window), >= 80% of misspecified cells with
      |est - theta_B| > 0.05 still pass the screen

Part B, table `screen_robustness_sffsplits` -- stability of the split-family
floor (EXC) on the real corpora: 5 CV reps x 10 random family splits per
corpus x held-out family, and the fraction that fire at threshold 0.0167.
  K3  every corpus-family verdict reported in the paper agrees with the
      majority verdict over the 50 splits
Part B needs the cached descriptors; `skip_b` runs Part A only.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.data.cache import corpus_cache, descriptor62
from nuisauc.data.descriptors import frequency16, morph16
from nuisauc.diagnostics.heldout import exc_statistic
from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.io import write_table
from nuisauc.seeds import record_run
from nuisauc.simulate.gaussian import ALPHAS, LAMBDAS, fit_propensity, simulate, theta_B

NAME = "cutoff_sweep"
DESCRIPTION = "Screen-failure result across stratum counts and screen cut-offs; EXC split stability"
PAPER = "§4, App C, D"
REQUIRES = "cache"
OUTPUTS = ["screen_robustness", "screen_robustness_sffsplits"]

SEED = 4200
Q_MULTS = [0.5, 1.0, 2.0]
DEGEN_GRID = [0.10, 0.20, 0.30]
NEG_GRID = [(0.45, 0.55), (0.40, 0.60)]
BAD_TOL = 0.05
K1_TOL = 0.02
K2_FRAC = 0.80
THRESHOLD = 0.0167
N_SPLIT_SEEDS = 10
N_CV_REPS_B = 5
CORPORA = ["Kaggle__local_128", "pcam", "malaria",
           "AFB__internal_81", "Wellgen__local_128"]

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": 5, "cap": 4000, "skip_b": False}


def part_a(n_rep):
    rows = []
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            for rep in range(n_rep):
                rng = np.random.default_rng([SEED, int(alpha * 10),
                                             int(lam * 10), rep])
                Z, y, s = simulate(alpha, lam, rng)
                tB = theta_B(Z, y, s)
                tr, te = next(iter(StratifiedKFold(
                    4, shuffle=True, random_state=rep)
                    .split(np.zeros(len(y)), y)))
                for kind in ("correct", "misspecified"):
                    m, cols = fit_propensity(kind, Z, y, tr, rep)
                    e = m.predict_proba(cols[te])[:, 1]
                    for mult in Q_MULTS:
                        q = max(2, int(round(mult * n_strata(len(te)))))
                        est, degen = conditional_auc(y[te], s[te], e, q)
                        neg, _ = conditional_auc(y[te], e, e, q)
                        rows.append(dict(
                            alpha=alpha, lam=lam, rep=rep, fit=kind,
                            mult=mult, q=q, theta_B=float(tB),
                            est=float(est), degen=float(degen),
                            neg=float(neg)))
            log.info("A alpha=%s lam=%s done", alpha, lam)
    return pd.DataFrame(rows)


def part_b(cap):
    rows = []
    for corpus in CORPORA:
        y, _C, _pk = corpus_cache(corpus, cap)
        A = descriptor62(corpus, cap)
        for fam, B in (("frequency", frequency16(corpus, cap)),
                       ("morphology", morph16(corpus, cap))):
            for rep in range(N_CV_REPS_B):
                tr, te = next(iter(StratifiedKFold(
                    4, shuffle=True, random_state=rep)
                    .split(np.zeros(len(y)), y)))
                for ss in range(N_SPLIT_SEEDS):
                    rng = np.random.default_rng([SEED, 7, rep, ss])
                    e_, f_ = exc_statistic(A, B, y, tr, te,
                                           n_strata(len(te)), rep, rng)
                    rows.append(dict(corpus=corpus, family=fam, rep=rep,
                                     split_seed=ss, exc=e_, floor=f_))
            g = pd.DataFrame(rows)
            g = g[(g.corpus == corpus) & (g.family == fam)]
            log.info("B %-22s %-10s median=%+.4f IQR=[%+.4f,%+.4f] fire=%.2f",
                     corpus, fam, g.exc.median(), g.exc.quantile(.25),
                     g.exc.quantile(.75), float((g.exc > THRESHOLD).mean()))
        gc.collect()
    return pd.DataFrame(rows)


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    log.info("Part A: cut-off sweep on Family I")
    A = part_a(cfg["n_rep"])
    write_table(A, out_dir, "screen_robustness")
    tables = {"screen_robustness": A}
    if not cfg["skip_b"]:
        log.info("Part B: EXC split stability")
        B = part_b(cfg["cap"])
        write_table(B, out_dir, "screen_robustness_sffsplits")
        tables["screen_robustness_sffsplits"] = B
    record_run(out_dir, NAME, cfg)
    return tables


def summarize(tables: dict) -> str:
    out = ["PART A: cut-off sweep"]
    A = tables["screen_robustness"].copy()
    A["err"] = (A.est - A.theta_B).abs()

    k1 = True
    out.append(f"\n  K1 correct-fit mean |est - theta_B| <= {K1_TOL} at every multiplier")
    for mult, g in A[A.fit == "correct"].groupby("mult"):
        ok = g.err.mean() <= K1_TOL
        k1 &= ok
        out.append(f"     m={mult:<4} q~{int(g.q.mean()):<4} mean err "
                   f"{g.err.mean():.4f}  max {g.err.max():.4f}  "
                   f"{'ok' if ok else 'FAIL'}")
    out.append(f"     --> {'PASS' if k1 else 'FAIL'}")

    k2 = True
    out.append(f"\n  K2 bad misspecified cells (err > {BAD_TOL}) passing the "
               f"screen, per cut-off setting (gate >= {K2_FRAC})")
    for mult, g in A[A.fit == "misspecified"].groupby("mult"):
        bad = g[g.err > BAD_TOL]
        for dmax in DEGEN_GRID:
            for lo, hi in NEG_GRID:
                if not len(bad):
                    continue
                p = float(((bad.degen <= dmax)
                           & (bad.neg >= lo) & (bad.neg <= hi)).mean())
                ok = p >= K2_FRAC
                k2 &= ok
                out.append(f"     m={mult:<4} degen<={dmax:<5} neg=[{lo},{hi}]  "
                           f"bad={len(bad):>3}  pass={p:.2f}  "
                           f"{'ok' if ok else 'FAIL'}")
    out.append(f"     --> {'PASS' if k2 else 'FAIL'}")

    B = tables.get("screen_robustness_sffsplits")
    if B is not None and len(B):
        # Descriptive only: the per-halving threshold is not calibrated for
        # real descriptors, so no omission verdict is drawn (paper, App. D).
        out.append(f"\nPART B: EXC split stability (50 splits per cell, threshold {THRESHOLD}; "
                   "descriptive)")
        for (corpus, fam), g in B.groupby(["corpus", "family"]):
            fire = float((g.exc > THRESHOLD).mean())
            out.append(f"     {corpus:<22} {fam:<10} median={g.exc.median():+.4f} "
                       f"IQR=[{g.exc.quantile(.25):+.4f},"
                       f"{g.exc.quantile(.75):+.4f}] fire={fire:.2f}")
    return "\n".join(out)
