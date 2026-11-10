"""The misspecification failure and its detection on real descriptor geometry.

Every covariate is a real measured feature of a real corpus image (A: the
62-feature descriptor, whose first 16 columns are trivial-16; B: the frequency
family); only the label mechanism and the target latent are synthetic, so the
fold-matched truth theta_B is exact.

Conditions per corpus and replicate:
  correct       core = a1 + a_lo, s = u + lam (a1 - a_lo) + eps; three
                propensities on the same cell: the oracle (bins of the core),
                boosted trees on all 62 features, logistic on trivial-16.
  omitted_meas  core = a1 + delta b1, s = u + lam b1 + eps (a measurable
                omitted driver: the frequency family's top axis).
  omitted_res   core = a1 + delta b1_perp, s = u + lam b1_perp + eps (the part
                of that axis the descriptor cannot absorb).
The split-family statistic EXC is recorded on the oracle row of the correct
condition (a real-feature null) and on every omitted-driver cell.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.diagnostics.heldout import exc_statistic
from nuisauc.estimand import n_strata, qbins
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple
from nuisauc.seeds import array_hash, corpus_seed, record_run
from nuisauc.simulate.semisynthetic import (CORPORA, estimate_with, load_features,
                                          synth)

NAME = "semisynthetic_checks"
DESCRIPTION = "Misspecification and split-family detection on real descriptors with synthetic labels"
PAPER = "§4, App H"
REQUIRES = "cache"
OUTPUTS = ["semisynthetic_stable"]

SEED = 4400
LAM = 1.0
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
BAD_TOL = 0.05

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": 10, "cap": 4000, "corpora": list(CORPORA),
            "seed": SEED, "lam": LAM, "deltas": list(DELTAS),
            "table": "semisynthetic_stable"}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    seed, lam = cfg["seed"], cfg["lam"]
    rows = []
    inputs = {}
    for corpus in cfg["corpora"]:
        A, B, a1, a_lo, b1, b1p, info = load_features(corpus, cfg["cap"])
        inputs[corpus] = {k: array_hash(v) for k, v in [
            ("A", A), ("B", B), ("a1", a1), ("a_lo", a_lo), ("b1", b1), ("b1p", b1p)]}
        T16 = A[:, :16]
        log.info("%s: a_lo=PC%d R2(a_lo|T16)=%.3f R2(b1|A)=%.3f R2(b1_perp|B)=%.3f",
                 corpus, info["j_lo"], info["r2_alo_T16"], info["r2_b1_A"],
                 info["r2_b1p_B"])
        for rep in range(cfg["n_rep"]):
            rng = np.random.default_rng([seed, corpus_seed(corpus), rep])
            eps = rng.standard_normal(len(a1))

            # correct condition, three propensities on the same cell
            core = a1 + a_lo
            y, u = synth(core, rng)
            s = u + lam * (a1 - a_lo) + eps
            tr, te = next(iter(StratifiedKFold(
                4, shuffle=True, random_state=rep)
                .split(np.zeros(len(y)), y)))
            q = n_strata(len(te))
            e_or = qbins(core[te], q).astype(float)
            for tag, e_te in (
                ("oracle", e_or),
                ("gbm62", fit_simple("gbm", A, y, tr, rep)
                 .predict_proba(A[te])[:, 1]),
                ("lin16", fit_simple("linear", T16, y, tr, rep)
                 .predict_proba(T16[te])[:, 1]),
            ):
                tB, est, err, degen, neg, ok = estimate_with(
                    e_te, y, s, core, te, q)
                rows.append(dict(corpus=corpus, cond="correct", fit=tag,
                                 delta=np.nan, rep=rep, theta_B=tB,
                                 est=est, err=err, degen=degen, neg=neg,
                                 screen=ok, exc=np.nan, floor=np.nan,
                                 **info))
            exc, floor = exc_statistic(A, B, y, tr, te, q, rep, rng)
            rows[-3]["exc"], rows[-3]["floor"] = exc, floor  # oracle row

            # omitted-driver sweeps on real geometry
            for cond, driver in (("omitted_meas", b1),
                                 ("omitted_res", b1p)):
                for delta in cfg["deltas"]:
                    core = a1 + delta * driver
                    y, u = synth(core, rng)
                    s = u + lam * driver + eps
                    tr, te = next(iter(StratifiedKFold(
                        4, shuffle=True, random_state=rep)
                        .split(np.zeros(len(y)), y)))
                    q = n_strata(len(te))
                    m = fit_simple("gbm", A, y, tr, rep)
                    e_te = m.predict_proba(A[te])[:, 1]
                    tB, est, err, degen, neg, ok = estimate_with(
                        e_te, y, s, core, te, q)
                    exc, floor = exc_statistic(A, B, y, tr, te, q, rep, rng)
                    rows.append(dict(corpus=corpus, cond=cond,
                                     fit="gbm62", delta=delta, rep=rep,
                                     theta_B=tB, est=est, err=err,
                                     degen=degen, neg=neg, screen=ok,
                                     exc=exc, floor=floor, **info))
        log.info("%s done", corpus)
        gc.collect()

    d = pd.DataFrame(rows)
    write_table(d, out_dir, cfg["table"])
    record_run(out_dir, NAME, cfg, inputs)
    return {cfg["table"]: d}


def summarize(tables: dict) -> str:
    d = next(iter(tables.values()))
    out = []
    c = d[d.cond == "correct"]
    orc = c[c.fit == "oracle"]
    out.append(f"oracle recovery (fold-matched): mean err {orc.err.mean():.4f} "
               f"(max {orc.err.max():.4f}) over {len(orc)} cells "
               f"[gate <= 0.02: {'pass' if orc.err.mean() <= 0.02 else 'fail'}]")
    out.append("estimation attenuation of fitted propensities")
    for tag in ("gbm62", "lin16"):
        g = c[c.fit == tag]
        for corpus, gg in g.groupby("corpus"):
            out.append(f"  {tag}  {corpus:<22} mean err {gg.err.mean():.4f}  "
                       f"screen-pass {float(gg.screen.mean()):.2f}")
    lin = c[c.fit == "lin16"]
    bad = lin[lin.err > BAD_TOL]
    frac = float(bad.screen.mean()) if len(bad) else float("nan")
    ok = len(bad) >= 15 and frac >= 0.80
    out.append(f"trivial-16 logistic misspecification: {len(bad)} cells with "
               f"err > {BAD_TOL}, screen-pass {frac:.2f} "
               f"[gate >= 15 cells and >= 0.80: {'pass' if ok else 'fail'}]")
    t = float(np.nanpercentile(orc.exc, 95))
    out.append(f"split-family threshold from real-feature nulls: {t:.4f}")

    meas = d[d.cond == "omitted_meas"].copy()
    meas["fires"] = meas.exc > t
    m3a = True
    out.append("measurable driver at delta >= 0.5: fires >= 0.80 or absorbed "
               "to mean err <= 0.03, per corpus")
    for corpus, g in meas[meas.delta >= 0.5].groupby("corpus"):
        fr, me = float(g.fires.mean()), float(g.err.mean())
        ok = fr >= 0.80 or me <= 0.03
        m3a &= ok
        out.append(f"  {corpus:<22} fire={fr:.2f} err={me:.4f}  "
                   f"{'ok' if ok else 'fail'}")
    out.append(f"  -> {'pass' if m3a else 'fail'}")

    res = d[d.cond == "omitted_res"].copy()
    res["fires"] = res.exc > t
    by_delta = res.groupby("delta").fires.mean()
    m3b = bool((by_delta <= 0.20).all())
    out.append(f"residual driver stays quiet at every delta (predicted blind "
               f"spot): max fire rate {float(by_delta.max()):.2f} -> "
               f"{'as predicted' if m3b else 'not as predicted'}")
    for name, om in (("measurable", meas), ("residual", res)):
        out.append(f"  {name} sweep: delta   fire    mean err   quiet p95/max (n)")
        for delta, g in om.groupby("delta"):
            quiet = g[~g.fires]
            p95 = (float(np.percentile(quiet.err, 95)) if len(quiet) else np.nan)
            mx = float(quiet.err.max()) if len(quiet) else np.nan
            out.append(f"    {delta:<7} {float(g.fires.mean()):<7.2f} "
                       f"{g.err.mean():<10.4f} {p95:.4f}/{mx:.4f} ({len(quiet)})")
        quiet_all = om[~om.fires]
        if len(quiet_all):
            out.append(f"  {name} quiet-cell bias: p95 "
                       f"{float(np.percentile(quiet_all.err, 95)):.4f}  max "
                       f"{float(quiet_all.err.max()):.4f}  (n={len(quiet_all)})")
    return "\n".join(out)
