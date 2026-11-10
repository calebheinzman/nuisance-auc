"""Specificity of the split-family test against target-only signal.

Z ~ N(0,1)^(n x 8); A = Z[:, 0:4] and the label mechanism is
logit P(y=1) = alpha (Z0 + Z1) + u, so the propensity fitted on A is correct.
The held-out family B (4 columns) carries only target proxies:
    diffuse        : B_j = rho u + eta_j in all 4 columns
    concentrated_k : rho u in k of 4 columns (k = 1, 2), the rest pure noise
    noise          : B pure noise
The threshold is fixed at the calibration value of `heldout_statistic`
(0.0167); nothing is recalibrated.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nuisauc.diagnostics.heldout import SIM_N, exc_statistic
from nuisauc.estimand import n_strata
from nuisauc.io import write_table
from nuisauc.seeds import record_run

NAME = "heldout_specificity"
DESCRIPTION = "False-firing rate of the split-family test when the held-out family carries only target signal"
PAPER = "App D"
REQUIRES = "none"
OUTPUTS = ["sff_specificity"]

SEED = 4100
THRESHOLD = 0.0167                # archived calibration, not recomputed
ALPHAS = [0.5, 1.0, 2.0]
RHOS = [0.3, 0.6, 1.0, 1.5]
N_REP = 10
REGIMES = ["noise", "diffuse", "concentrated_2", "concentrated_1"]
S_FIRE_MAX = 0.10

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": N_REP}


def sim_target_only(regime: str, alpha: float, rho: float,
                    rng: np.random.Generator):
    """A is correct nuisance; B carries target proxies only (or noise)."""
    Z = rng.standard_normal((SIM_N, 8))
    u, _eps = rng.standard_normal((2, SIM_N))
    B = Z[:, 4:8].copy()
    if regime == "diffuse":
        k = 4
    elif regime.startswith("concentrated"):
        k = int(regime.split("_")[1])
    elif regime == "noise":
        k = 0
    else:
        raise ValueError(regime)
    for j in range(k):
        B[:, j] = rho * u + B[:, j]          # eta_j is the original N(0,1)
    logit = alpha * (Z[:, 0] + Z[:, 1]) + u
    y = (rng.random(SIM_N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return Z[:, :4], B, y


def b_probe_auc(B_te: np.ndarray, y_te: np.ndarray, seed: int) -> float:
    """Cross-fitted raw AUC of a logistic probe on the full family."""
    p = np.empty(len(y_te), float)
    for tr, te in StratifiedKFold(4, shuffle=True,
                                  random_state=seed).split(B_te, y_te):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(B_te[tr], y_te[tr])
        p[te] = m.decision_function(B_te[te])
    return float(roc_auc_score(y_te, p))


def run_cell(regime: str, alpha: float, rho: float, rep: int) -> dict:
    rng = np.random.default_rng([SEED, REGIMES.index(regime),
                                 int(alpha * 10), int(rho * 10), rep])
    A, B, y = sim_target_only(regime, alpha, rho, rng)
    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                       .split(np.zeros(len(y)), y)))
    exc, floor = exc_statistic(A, B, y, tr, te, n_strata(len(te)), rep, rng)
    return dict(regime=regime, alpha=alpha, rho=rho, rep=rep,
                exc=exc, floor=floor,
                b_auc=b_probe_auc(B[te], y[te], rep))


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for regime in REGIMES:
        rhos = [0.0] if regime == "noise" else RHOS
        for alpha in ALPHAS:
            for rho in rhos:
                for rep in range(cfg["n_rep"]):
                    rows.append(run_cell(regime, alpha, rho, rep))
            log.info("%s alpha=%s done", regime, alpha)
    d = pd.DataFrame(rows)
    write_table(d, out_dir, OUTPUTS[0])
    record_run(out_dir, NAME, cfg)
    return {OUTPUTS[0]: d}


def summarize(tables: dict) -> str:
    d = tables[OUTPUTS[0]]
    d = d[np.isfinite(d.exc)]
    fire = d.exc > THRESHOLD
    out = [f"split-family specificity against threshold {THRESHOLD}",
           f"  {'regime':<18}{'cells':>7}{'EXC mean':>11}{'fires':>8}"
           f"{'B-probe AUC':>13}"]
    for reg in REGIMES:
        g = d[d.regime == reg]
        out.append(f"  {reg:<18}{len(g):>7}{g.exc.mean():>11.4f}"
                   f"{float(fire[g.index].mean()):>8.2f}"
                   f"{g.b_auc.mean():>13.3f}")
    diff = d[d.regime == "diffuse"]
    conc = d[d.regime.str.startswith("concentrated")]
    noise = d[d.regime == "noise"]
    s1 = float(fire[diff.index].mean())
    s2 = float(fire[conc.index].mean())
    s4 = float(fire[noise.index].mean())
    out.append(f"diffuse fire rate <= {S_FIRE_MAX}: {s1:.3f} -> "
               f"{'pass' if s1 <= S_FIRE_MAX else 'fail'}")
    out.append(f"concentrated fire rate <= {S_FIRE_MAX}: {s2:.3f} -> "
               f"{'pass' if s2 <= S_FIRE_MAX else 'fail'}")
    out.append(f"noise fire rate <= {S_FIRE_MAX}: {s4:.3f} -> "
               f"{'pass' if s4 <= S_FIRE_MAX else 'fail'}")
    out.append("trend")
    for reg in ["diffuse", "concentrated_2", "concentrated_1"]:
        g = d[d.regime == reg]
        rs, p = spearmanr(g.rho, g.exc)
        out.append(f"  {reg:<18} Spearman(EXC, rho) = {rs:+.3f} (p={p:.3g})")
    out.append("fire rate by achieved held-out-probe raw AUC")
    tgt = d[d.regime != "noise"].copy()
    tgt["bucket"] = pd.cut(tgt.b_auc, [0.5, 0.55, 0.6, 0.65, 0.7, 1.0])
    for b, g in tgt.groupby("bucket", observed=True):
        if len(g):
            out.append(f"  {str(b):<16} n={len(g):>4}  "
                       f"fires={float((g.exc > THRESHOLD).mean()):.2f}")
    out.append("per-regime, per-rho fire rates")
    for (reg, rho), g in d[d.regime != "noise"].groupby(["regime", "rho"]):
        out.append(f"  {reg:<18} rho={rho:<4} EXC={g.exc.mean():+.4f}  "
                   f"fires={float((g.exc > THRESHOLD).mean()):.2f}  "
                   f"B-AUC={g.b_auc.mean():.3f}")
    return "\n".join(out)
