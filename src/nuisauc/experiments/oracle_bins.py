"""Sensitivity of the three-regime results to the number of oracle bins.

The fold-matched truth bins the true core into Q_FINE = 100 equal-frequency
bins. This recomputes it with 50, 100 and 200 bins on the same draws as
`semisynthetic_regimes` (seed 4500); the 100-bin truth must reproduce that
table.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc import paths
from nuisauc.estimand import conditional_auc, n_strata, pair_weighted, qbins
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple
from nuisauc.seeds import corpus_seed, record_run
from nuisauc.simulate.semisynthetic import (CORPORA, REGIMES_SEED, VARIANTS,
                                            load_features, synth)

NAME = "oracle_bins"
DESCRIPTION = "Three-regime errors with the oracle truth binned at 50, 100 and 200"
PAPER = "§3, App C"
REQUIRES = "cache"
OUTPUTS = ["oracle_bins"]

QF = (50, 100, 200)
REGIMES_TABLE = "semisynthetic_regimes_stable"

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": 10, "cap": 4000, "corpora": list(CORPORA),
            "seed": REGIMES_SEED, "q_fine": list(QF)}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for corpus in cfg["corpora"]:
        A, _B, a1, a_lo, *_ = load_features(corpus, cfg["cap"])
        T16 = A[:, :16]
        for rep in range(cfg["n_rep"]):
            rng = np.random.default_rng([cfg["seed"], corpus_seed(corpus), rep])
            eps = rng.standard_normal(len(a1))
            core = a1 + a_lo
            y, u = synth(core, rng)
            tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                               .split(np.zeros(len(y)), y)))
            q = n_strata(len(te))
            fits = {
                "gbm62": fit_simple("gbm", A, y, tr, rep).predict_proba(A[te])[:, 1],
                "lin16": fit_simple("linear", T16, y, tr, rep).predict_proba(T16[te])[:, 1],
            }
            for variant, lam in VARIANTS:
                s = (u + eps if variant == "none" else
                     u + lam * (a1 - a_lo) + eps if variant == "opp" else
                     u + lam * (a1 + a_lo) + eps)
                truths = {qf: float(pair_weighted(y[te], s[te], qbins(core[te], qf))[0])
                          for qf in cfg["q_fine"]}
                for tag, e in fits.items():
                    est = float(conditional_auc(y[te], s[te], e, q)[0])
                    r = dict(corpus=corpus, variant=variant, lam=lam, fit=tag,
                             rep=rep, est=est)
                    for qf, t in truths.items():
                        r[f"truth{qf}"] = t
                    rows.append(r)
        log.info("%s done", corpus)
    d = pd.DataFrame(rows)
    write_table(d, out_dir, NAME)
    record_run(out_dir, NAME, cfg)
    _check_against_regimes(d, out_dir)
    return {NAME: d}


def _check_against_regimes(d: pd.DataFrame, out_dir: Path) -> None:
    """The 100-bin truth and the estimates must reproduce the regimes table."""
    for directory in (Path(out_dir), paths.reference_dir()):
        f = Path(directory) / f"{REGIMES_TABLE}.csv"
        if f.exists() and "truth100" in d:
            ref = pd.read_csv(f)
            ref = ref[ref.fit != "oracle"]
            m = d.merge(ref, on=["corpus", "variant", "lam", "fit", "rep"],
                        suffixes=("", "_ref"))
            assert np.abs(m.truth100 - m.theta_B).max() < 1e-9
            assert np.abs(m.est - m.est_ref).max() < 1e-9
            log.info("100-bin truth reproduces %s: ok", f)
            return
    log.warning("no %s table found; consistency check skipped", REGIMES_TABLE)


def summarize(tables: dict) -> str:
    d = tables[NAME]
    qfs = [int(c[5:]) for c in d.columns if c.startswith("truth")]
    out = []
    pred = np.where(d.variant == "same", 1, -1)
    for qf in qfs:
        err = d.est - d[f"truth{qf}"]
        agree = int((np.sign(err) == pred).sum())
        cond = d.assign(err=err).groupby(["corpus", "variant", "lam", "fit"]).err.mean()
        cpred = np.where(cond.index.get_level_values("variant") == "same", 1, -1)
        rev = d[(d.fit == "lin16") & (d.variant == "opp") & (d.lam == 3)]
        nrev = int(((rev.est < 0.5) & (rev[f"truth{qf}"] > 0.5)).sum())
        rev2 = d[(d.fit == "lin16") & (d.variant == "opp") & (d.lam == 2)]
        nrev2 = int(((rev2.est < 0.5) & (rev2[f"truth{qf}"] > 0.5)).sum())
        out.append(f"Q_FINE={qf}: direction {agree}/{len(d)}, condition means "
                   f"{int((np.sign(cond.values) == cpred).sum())}/{len(cond)}, "
                   f"reversals lam3 {nrev}/{len(rev)} lam2 {nrev2}/{len(rev2)}, "
                   f"material |err|>0.05: {int((err.abs() > 0.05).sum())}")
    if 100 in qfs:
        for a in (50, 200):
            if a in qfs:
                diff = np.abs(d[f"truth{a}"] - d["truth100"])
                out.append(f"max |truth{a} - truth100| = {diff.max():.4f}, "
                           f"mean {diff.mean():.4f}")
    return "\n".join(out)
