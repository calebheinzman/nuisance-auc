"""Limit of detection of the split-family test.

The label driver sits spread across the held-out family (the omitted_spread
condition of `heldout_statistic`) with strength alpha swept from 0.05 to 2.
For each cell: whether the test fires at the archived threshold 0.0167, and
the true estimation bias |est - theta_B| of the boosted propensity on the
four analysis columns. The question is how large the bias can be among quiet
cells.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from nuisauc.diagnostics.heldout import SIM_N, exc_statistic
from nuisauc.estimand import conditional_auc, n_strata, pair_weighted, qbins
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple
from nuisauc.seeds import record_run

NAME = "heldout_lod"
DESCRIPTION = "Limit of detection of the split-family test and the bias it can leave unflagged"
PAPER = "App D"
REQUIRES = "none"
OUTPUTS = ["lod"]

SEED = 3900
THRESHOLD = 0.0167
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
LAMBDAS = [0.0, 1.0, 2.0]
N_REP = 10
Q_FINE = 100
L1_RHO = 0.8
L2_BAR = 0.05               # materiality bar for a bad cell
LOD_RATE = 0.80

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {}


def cell(alpha: float, lam: float, rep: int) -> dict:
    rng = np.random.default_rng([SEED, int(alpha * 100), int(lam * 10), rep])
    Z = rng.standard_normal((SIM_N, 8))
    u, eps = rng.standard_normal((2, SIM_N))
    core = Z[:, 4:8].mean(axis=1) * 2.0                 # omitted_spread
    logit = alpha * core + u
    y = (rng.random(SIM_N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    s = u + lam * (Z[:, 0] - Z[:, 1]) + eps

    theta_b = pair_weighted(y, s, qbins(core, Q_FINE))[0]
    drv_auc = float(roc_auc_score(y, core))

    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                       .split(np.zeros(len(y)), y)))
    q = n_strata(len(te))
    m = fit_simple("gbm", Z[:, :4], y, tr, rep)
    e = m.predict_proba(Z[te, :4])[:, 1]
    est, _ = conditional_auc(y[te], s[te], e, q)

    exc, _floor = exc_statistic(Z[:, :4], Z[:, 4:8], y, tr, te, q, rep, rng)
    return dict(alpha=alpha, lam=lam, rep=rep, exc=exc,
                fires=bool(exc > THRESHOLD),
                theta_b=float(theta_b), est=float(est),
                bias=float(abs(est - theta_b)), driver_auc=drv_auc)


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            for rep in range(N_REP):
                rows.append(cell(alpha, lam, rep))
        g = pd.DataFrame([r for r in rows if r["alpha"] == alpha])
        log.info("alpha=%s fire=%.2f bias mean=%.4f driver AUC=%.3f", alpha,
                 g.fires.mean(), g.bias.mean(), g.driver_auc.mean())
    d = pd.DataFrame(rows)
    write_table(d, out_dir, OUTPUTS[0])
    record_run(out_dir, NAME, cfg)
    return {OUTPUTS[0]: d}


def summarize(tables: dict) -> str:
    d = tables[OUTPUTS[0]]
    per = d.groupby("alpha").agg(fire=("fires", "mean"),
                                 bias95=("bias", lambda x: np.percentile(x, 95)),
                                 drv=("driver_auc", "mean"))
    out = [f"  {'alpha':>7}{'fire rate':>11}{'bias p95':>11}{'driver AUC':>12}"]
    for a, r in per.iterrows():
        out.append(f"  {a:>7}{r.fire:>11.2f}{r.bias95:>11.4f}{r.drv:>12.3f}")
    rho = float(spearmanr(per.index.to_numpy(), per.fire.to_numpy())[0])
    out.append(f"fire rate monotone in alpha: Spearman rho = {rho:.3f} "
               f"(need >= {L1_RHO}) -> {'pass' if rho >= L1_RHO else 'fail'}")
    quiet = d[~d.fires]
    if len(quiet):
        b95 = float(np.percentile(quiet.bias, 95))
        out.append(f"quiet cells ({len(quiet)}): bias p95 = {b95:.4f} "
                   f"(need <= {L2_BAR}) -> {'pass' if b95 <= L2_BAR else 'fail'}")
        out.append(f"  quiet-cell bias max = {quiet.bias.max():.4f} at "
                   f"alpha={float(quiet.loc[quiet.bias.idxmax(), 'alpha'])}")
    else:
        out.append("no quiet cells anywhere in the sweep")
    lod = [a for a, r in per.iterrows() if r.fire >= LOD_RATE]
    out.append(f"limit of detection alpha* = "
               f"{min(lod) if lod else '> ' + str(max(ALPHAS))}"
               + (f"  (driver AUC {per.loc[min(lod), 'drv']:.3f})" if lod else ""))
    above = d[d.alpha >= min(lod)] if lod else d.iloc[0:0]
    below = d[d.alpha < min(lod)] if lod else d
    if len(above) and len(below):
        out.append(f"  bias p95 above LOD = {np.percentile(above.bias, 95):.4f}"
                   f"   below LOD = {np.percentile(below.bias, 95):.4f}")
    return "\n".join(out)
