"""Are the real-geometry conclusions specific to one construction?

Part "atten" (attenuation floor) under three label directions:
  a_lo   the top-10 PCA component least explained by trivial-16
  a_lo2  the second least explained component
  a_rand a random unit combination of components 2..10 (fixed seed)
  correct condition, oracle vs boosted-tree 62-feature propensity.
Part "blind" (residual blind spot) under four omitted directions:
  b1p_gbm   boosted-tree residual of B's top axis on A
  b1p_ridge ridge residual of the same axis
  b2p_gbm   boosted-tree residual of B's second axis
  brand_gbm boosted-tree residual of a random unit combination of B
  delta in {0, 0.3, 1.0}, boosted-tree 62-feature propensity, split-family
  statistic against the within-run real-feature null (95th percentile of
  the oracle cells' EXC).
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from nuisauc.data.cache import descriptor62
from nuisauc.data.descriptors import frequency16
from nuisauc.diagnostics.heldout import exc_statistic
from nuisauc.estimand import n_strata, qbins
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple
from nuisauc.seeds import array_hash, corpus_seed, record_run
from nuisauc.simulate.semisynthetic import CORPORA, estimate_with, synth

NAME = "semisynthetic_sensitivity"
DESCRIPTION = "Attenuation floor and residual blind spot under alternative label and driver directions"
PAPER = "App H"
REQUIRES = "cache"
OUTPUTS = ["semisynthetic_sens_stable"]

SEED = 4600
DELTAS = [0.0, 0.3, 1.0]
LAM = 1.0

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": 5, "cap": 4000, "corpora": list(CORPORA), "seed": SEED,
            "lam": LAM, "deltas": list(DELTAS),
            "table": "semisynthetic_sens_stable"}


def oof_resid(target: np.ndarray, X: np.ndarray, kind: str = "gbm"):
    """Standardised out-of-fold residual of target on X, and its R^2."""
    pred = np.empty_like(target)
    for tr, te in KFold(4, shuffle=True, random_state=0).split(X):
        if kind == "gbm":
            m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        else:
            m = Ridge(alpha=1.0)
        m.fit(X[tr], target[tr])
        pred[te] = m.predict(X[te])
    resid = target - pred
    return resid / resid.std(), 1.0 - resid.var() / target.var()


def build(corpus: str, cap: int, seed: int = SEED):
    """(A, B, a1, label directions, omitted-driver directions)."""
    A = np.nan_to_num(descriptor62(corpus, cap), nan=0.0,
                      posinf=0.0, neginf=0.0)
    B = np.nan_to_num(frequency16(corpus, cap), nan=0.0,
                      posinf=0.0, neginf=0.0)
    As = StandardScaler().fit_transform(A)
    Bs = StandardScaler().fit_transform(B)
    pa = PCA(n_components=10, random_state=0).fit_transform(As)
    pa = pa / pa.std(axis=0)
    T16 = A[:, :16]
    r2s = np.array([oof_resid(pa[:, j], T16)[1] for j in range(1, 10)])
    order = 1 + np.argsort(r2s)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(9)
    a_rand = pa[:, 1:10] @ (w / np.linalg.norm(w))
    a_dirs = {"a_lo": pa[:, order[0]], "a_lo2": pa[:, order[1]],
              "a_rand": a_rand / a_rand.std()}

    pb = PCA(n_components=2, random_state=0).fit_transform(Bs)
    b1 = pb[:, 0] / pb[:, 0].std()
    b2 = pb[:, 1] / pb[:, 1].std()
    wb = rng.standard_normal(B.shape[1])
    brand = Bs @ (wb / np.linalg.norm(wb))
    brand = brand / brand.std()
    b_dirs = {"b1p_gbm": oof_resid(b1, A, "gbm")[0],
              "b1p_ridge": oof_resid(b1, A, "ridge")[0],
              "b2p_gbm": oof_resid(b2, A, "gbm")[0],
              "brand_gbm": oof_resid(brand, A, "gbm")[0]}
    return A, B, pa[:, 0], a_dirs, b_dirs


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    seed, lam = cfg["seed"], cfg["lam"]
    rows = []
    inputs = {}
    for corpus in cfg["corpora"]:
        A, B, a1, a_dirs, b_dirs = build(corpus, cfg["cap"], seed)
        inputs[corpus] = {k: array_hash(v) for k, v in
                          dict(A=A, B=B, a1=a1, **a_dirs, **b_dirs).items()}
        for rep in range(cfg["n_rep"]):
            rng = np.random.default_rng([seed, corpus_seed(corpus), rep])
            eps = rng.standard_normal(len(a1))

            for aname, ad in a_dirs.items():
                core = a1 + ad
                y, u = synth(core, rng)
                s = u + lam * (a1 - ad) + eps
                tr, te = next(iter(StratifiedKFold(
                    4, shuffle=True, random_state=rep)
                    .split(np.zeros(len(y)), y)))
                q = n_strata(len(te))
                for tag, e_te in (
                    ("oracle", qbins(core[te], q).astype(float)),
                    ("gbm62", fit_simple("gbm", A, y, tr, rep)
                     .predict_proba(A[te])[:, 1]),
                ):
                    tB, est, err, degen, neg, ok = estimate_with(
                        e_te, y, s, core, te, q)
                    exc = np.nan
                    if tag == "oracle":
                        exc, _floor = exc_statistic(A, B, y, tr, te, q,
                                                    rep, rng)
                    rows.append(dict(corpus=corpus, part="atten",
                                     variant=aname, fit=tag, delta=np.nan,
                                     rep=rep, theta_B=tB, est=est, err=err,
                                     screen=ok, exc=exc))

            for bname, bd in b_dirs.items():
                for delta in cfg["deltas"]:
                    core = a1 + delta * bd
                    y, u = synth(core, rng)
                    s = u + lam * bd + eps
                    tr, te = next(iter(StratifiedKFold(
                        4, shuffle=True, random_state=rep)
                        .split(np.zeros(len(y)), y)))
                    q = n_strata(len(te))
                    m = fit_simple("gbm", A, y, tr, rep)
                    e_te = m.predict_proba(A[te])[:, 1]
                    tB, est, err, degen, neg, ok = estimate_with(
                        e_te, y, s, core, te, q)
                    exc, _floor = exc_statistic(A, B, y, tr, te, q, rep, rng)
                    rows.append(dict(corpus=corpus, part="blind",
                                     variant=bname, fit="gbm62",
                                     delta=delta, rep=rep, theta_B=tB,
                                     est=est, err=err, screen=ok, exc=exc))
        log.info("%s done", corpus)
        gc.collect()

    d = pd.DataFrame(rows)
    write_table(d, out_dir, cfg["table"])
    record_run(out_dir, NAME, cfg, inputs)
    return {cfg["table"]: d}


def summarize(tables: dict) -> str:
    d = next(iter(tables.values()))
    out = []
    at = d[d.part == "atten"]
    n1 = n2 = True
    out.append("attenuation floor under every label direction "
               "(oracle err <= 0.02; boosted-tree err > oracle err)")
    for v, g in at.groupby("variant"):
        o = g[g.fit == "oracle"].err.mean()
        f = g[g.fit == "gbm62"].err.mean()
        ok1, ok2 = o <= 0.02, f > o
        n1 &= ok1
        n2 &= ok2
        out.append(f"  {v:<8} oracle {o:.4f}  gbm62 {f:.4f}  "
                   f"{'ok' if ok1 and ok2 else 'fail'}")
    out.append(f"  oracle recovery -> {'pass' if n1 else 'fail'}   "
               f"floor exists -> {'pass' if n2 else 'fail'}")

    t = float(np.nanpercentile(at[at.fit == "oracle"].exc, 95))
    bl = d[d.part == "blind"].copy()
    bl["fires"] = bl.exc > t
    out.append(f"blind spot under every residual direction (threshold {t:.4f})")
    n3_count = 0
    for v, g in bl.groupby("variant"):
        hi = g[g.delta == 1.0]
        fr, mb = float(hi.fires.mean()), float(hi.err.mean())
        ok = fr <= 0.20 and mb >= 0.03
        n3_count += ok
        out.append(f"  {v:<10} delta=1: fire {fr:.2f}  bias {mb:.4f}  "
                   f"{'ok' if ok else 'no'}")
        for delta, gg in g.groupby("delta"):
            out.append(f"     delta={delta:<4} fire {float(gg.fires.mean()):.2f}"
                       f"  bias {gg.err.mean():.4f}")
    out.append(f"  -> {'pass' if n3_count >= 3 else 'fail'} "
               f"({n3_count}/4 variants)")
    return "\n".join(out)
