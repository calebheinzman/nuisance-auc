"""Family I: does stratifying on a scalar balancing score recover its estimand?

Gaussian nuisance with a direction the balancing score cannot see (Z1 - Z2,
independent of the label). For each (alpha, lambda, replicate) and each
propensity fit (correct, misspecified = Z1 only, flexible = boosted trees) the
table holds the population targets theta_B (strata of the true balancing score)
and theta_Z (full-covariate grid), the estimate, and the screen quantities. A
pure-nuisance score under the same overlap regime checks that the true
balancing score removes label confounding.

Gates (fixed in advance):
  J1  |est - theta_B| <= 0.02 with a correct propensity in >= 90% of
      screen-passing cells
  J2  the pure-nuisance score conditions to 0.5 +- 0.02
  J3  cost of misspecification (reported)
  J4  >= 80% of cells with |est - theta_B| > 0.05 fail the screen (evaluable
      only with >= 20 such cells)
  J5  estimand gap theta_Z - theta_B (reported)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nuisauc.checks import DEGEN_MAX, NEG_WINDOW
from nuisauc.io import write_table
from nuisauc.seeds import record_run
from nuisauc.simulate.gaussian import ALPHAS, FITS, LAMBDAS, estimate, simulate, theta_B, theta_Z

NAME = "family_i"
DESCRIPTION = "Family I: estimator vs its estimand, misspecification cost, and the screen"
PAPER = "§4, Tab 3"
REQUIRES = "none"
OUTPUTS = ["estimand"]

SEED = 2404
J1_TOL = 0.02
J2_TOL = 0.02
J4_TOL = 0.05
J4_MIN_CELLS = 20

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": 10}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            for rep in range(cfg["n_rep"]):
                rng = np.random.default_rng([SEED, int(alpha * 10),
                                             int(lam * 10), rep])
                Z, y, s = simulate(alpha, lam, rng)
                tB, tZ = theta_B(Z, y, s), theta_Z(Z, y, s)
                for kind in FITS:
                    est, degen, neg = estimate(kind, Z, y, s, rep)
                    passes = (degen <= DEGEN_MAX
                              and NEG_WINDOW[0] <= neg <= NEG_WINDOW[1])
                    rows.append(dict(alpha=alpha, lam=lam, rep=rep, fit=kind,
                                     theta_B=tB, theta_Z=tZ, est=est,
                                     degen=degen, neg_ctrl=neg,
                                     screen=bool(passes), pure=False))
                # J2: a pure-nuisance score under the same overlap regime
                Zp, yp, sp = simulate(alpha, lam, rng, pure=True)
                est, degen, neg = estimate("correct", Zp, yp, sp, rep)
                passes = (degen <= DEGEN_MAX
                          and NEG_WINDOW[0] <= neg <= NEG_WINDOW[1])
                rows.append(dict(alpha=alpha, lam=lam, rep=rep, fit="correct",
                                 theta_B=0.5, theta_Z=0.5, est=est,
                                 degen=degen, neg_ctrl=neg,
                                 screen=bool(passes), pure=True))
            log.info("alpha=%s lambda=%s done", alpha, lam)

    d = pd.DataFrame(rows)
    write_table(d, out_dir, "estimand")
    record_run(out_dir, NAME, cfg)
    return {"estimand": d}


def summarize(tables: dict) -> str:
    d = tables["estimand"]
    out: list[str] = []
    p = out.append
    sig = d[~d.pure].copy()
    sig["err_B"] = (sig.est - sig.theta_B).abs()
    sig["gap"] = sig.theta_Z - sig.theta_B

    p("Population targets and the estimator")
    piv = sig[sig.fit == "correct"].pivot_table(
        index="alpha", columns="lam", values=["theta_B", "theta_Z", "est"])
    p(piv.round(4).to_string())

    p("\nGATES")
    c = sig[(sig.fit == "correct") & sig.screen]
    frac1 = float((c.err_B <= J1_TOL).mean()) if len(c) else float("nan")
    j1 = len(c) > 0 and frac1 >= 0.90
    p(f"\n  J1 |est - theta_B| <= {J1_TOL} with a correct propensity, in "
      f">= 90% of screen-passing cells")
    p(f"     screen-passing cells: {len(c)} of {int((sig.fit=='correct').sum())}"
      f"   mean |err| {c.err_B.mean():.4f}   max {c.err_B.max():.4f}"
      if len(c) else "     none passed the screen")
    p(f"     --> {frac1:.0%}  {'PASS' if j1 else 'FAIL'}")

    pu = d[d.pure & d.screen]
    frac2 = float(((pu.est - 0.5).abs() <= J2_TOL).mean()) if len(pu) else np.nan
    j2 = len(pu) > 0 and frac2 >= 0.90
    p(f"\n  J2 pure-nuisance score conditions to 0.5 +- {J2_TOL}")
    if len(pu):
        p(f"     screen-passing: {len(pu)} of {int(d.pure.sum())}   "
          f"mean est {pu.est.mean():.4f}   max |dev| "
          f"{(pu.est - 0.5).abs().max():.4f}")
    p(f"     --> {frac2:.0%}  {'PASS' if j2 else 'FAIL'}")

    p("\n  J3 cost of misspecification (screen-passing cells)")
    for k in FITS:
        sub = sig[(sig.fit == k) & sig.screen]
        if not len(sub):
            p(f"     {k:<16} no screen-passing cells")
            continue
        p(f"     {k:<16} mean |est - theta_B| = {sub.err_B.mean():.4f}   "
          f"max {sub.err_B.max():.4f}   (n={len(sub)})")
    mis = sig[(sig.fit == "misspecified") & sig.screen]
    if len(mis):
        v = float(mis.err_B.mean())
        p(f"     --> reading: capacity ladder is "
          f"{'REQUIRED' if v > 0.05 else 'second-order'} (mean {v:.4f})")

    bad = sig[sig.err_B > J4_TOL]
    p(f"\n  J4 among cells with |est - theta_B| > {J4_TOL}, fraction "
      f"failing the screen >= 0.80")
    if len(bad) < J4_MIN_CELLS:
        p(f"     only {len(bad)} such cells (< {J4_MIN_CELLS}) --> NOT EVALUABLE")
    else:
        f4 = float((~bad.screen).mean())
        p(f"     {len(bad)} cells, {f4:.0%} fail the screen  "
          f"--> {'PASS' if f4 >= 0.80 else 'FAIL'}")

    p("\n  J5 (no gate) estimand gap theta_Z - theta_B, by lambda")
    g = sig[sig.fit == "correct"].groupby("lam").gap.agg(["mean", "min", "max"])
    p(g.round(4).to_string())
    neg_gap = sig[(sig.fit == "correct") & sig.screen & (sig.gap < 0)]
    p(f"     screen-passing cells with theta_B > theta_Z: {len(neg_gap)}")
    p("     --> " + ("theta_B is the conservative estimand" if not len(neg_gap)
                     else "theta_B exceeds theta_Z on some screen-passing cells"))

    p("\n  screen pass rate by alpha (overlap strength):")
    p(sig.pivot_table(index="alpha", columns="fit", values="screen").round(3).to_string())
    return "\n".join(out)
