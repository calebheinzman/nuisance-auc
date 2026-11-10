"""The three regimes of the bias law on real descriptor geometry.

Same construction as the correct condition of `semisynthetic_checks`: real
features A (62), label core = a1 + a_lo, synthetic target latent u,
logit P(y=1) = core/sd + u. Only the detector's loading on the nuisance
changes:

    none : s = u + eps                     (attenuation regime)
    opp  : s = u + lam (a1 - a_lo) + eps   (loads against the label's sign)
    same : s = u + lam (a1 + a_lo) + eps   (loads with the label's sign)

Truth is the oracle stratification on the core (fold-matched). Predicted
signs: none and opp below truth (reversal possible as lam grows), same above.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.estimand import n_strata, qbins
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple
from nuisauc.seeds import array_hash, corpus_seed, record_run
from nuisauc.simulate.semisynthetic import (CORPORA, REGIMES_SEED, VARIANTS,
                                            estimate_with, load_features, synth)

NAME = "semisynthetic_regimes"
DESCRIPTION = "Sign of the adjusted-AUC error in the three regimes on real descriptors"
PAPER = "§3, Tab 2"
REQUIRES = "cache"
OUTPUTS = ["semisynthetic_regimes_stable"]

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": 10, "cap": 4000, "corpora": list(CORPORA),
            "seed": REGIMES_SEED, "table": "semisynthetic_regimes_stable"}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    inputs = {}
    for corpus in cfg["corpora"]:
        A, B, a1, a_lo, b1, b1p, info = load_features(corpus, cfg["cap"])
        inputs[corpus] = {k: array_hash(v) for k, v in [
            ("A", A), ("B", B), ("a1", a1), ("a_lo", a_lo)]}
        T16 = A[:, :16]
        log.info("%s: R2(a_lo|T16)=%.3f", corpus, info["r2_alo_T16"])
        for rep in range(cfg["n_rep"]):
            rng = np.random.default_rng([cfg["seed"], corpus_seed(corpus), rep])
            eps = rng.standard_normal(len(a1))
            core = a1 + a_lo
            y, u = synth(core, rng)
            tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                               .split(np.zeros(len(y)), y)))
            q = n_strata(len(te))
            fits = {
                "oracle": qbins(core[te], q).astype(float),
                "gbm62": fit_simple("gbm", A, y, tr, rep).predict_proba(A[te])[:, 1],
                "lin16": fit_simple("linear", T16, y, tr, rep).predict_proba(T16[te])[:, 1],
            }
            for variant, lam in VARIANTS:
                if variant == "none":
                    s = u + eps
                elif variant == "opp":
                    s = u + lam * (a1 - a_lo) + eps
                else:
                    s = u + lam * (a1 + a_lo) + eps
                for tag, e_te in fits.items():
                    tB, est, err, degen, neg, ok = estimate_with(e_te, y, s, core, te, q)
                    rows.append(dict(corpus=corpus, variant=variant, lam=lam, fit=tag,
                                     rep=rep, theta_B=tB, est=est, signed=est - tB,
                                     degen=degen, neg=neg, screen=ok,
                                     r2_alo_T16=info["r2_alo_T16"]))
        log.info("%s done", corpus)
        gc.collect()

    d = pd.DataFrame(rows)
    write_table(d, out_dir, cfg["table"])
    record_run(out_dir, NAME, cfg, inputs)
    return {cfg["table"]: d}


def summarize(tables: dict) -> str:
    d = next(iter(tables.values()))
    out = []
    with pd.option_context("display.width", 200):
        m = d[d.fit != "oracle"]
        out.append("mean signed error (est - truth) by corpus x fit x variant/lam")
        out.append(m.pivot_table(index=["corpus", "fit"], columns=["variant", "lam"],
                                 values="signed").round(3).to_string())
        out.append("mean truth by corpus x variant/lam (oracle est)")
        out.append(d[d.fit == "oracle"].pivot_table(
            index="corpus", columns=["variant", "lam"],
            values="theta_B").round(3).to_string())
        pred = {"none": -1, "opp": -1, "same": +1}
        m = m.assign(pred=m.variant.map(pred))
        agree = (np.sign(m.signed) == m.pred)
        out.append("sign agreement with the law, by fit and variant (fraction of cells):")
        out.append(agree.groupby([m.fit, m.variant]).mean().round(3).to_string())
        rev = m[(m.est < 0.5) & (m.theta_B > 0.5)]
        out.append(f"reversal cells (est < 0.5, truth > 0.5): {len(rev)}")
        if len(rev):
            out.append(rev.groupby(["corpus", "fit", "variant", "lam"]).size().to_string())
        out.append(f"screen pass rate (non-oracle): {round(float(m.screen.mean()), 3)}")
        out.append(f"mean self-control among non-oracle: {round(float(m.neg.mean()), 4)}")
    return "\n".join(out)
