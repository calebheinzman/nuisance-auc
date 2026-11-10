"""The overlap screen: degenerate-stratum share and the self-stratified control.

A nuisance model passes the screen when at most DEGEN_MAX of its strata lack a
comparable pair and its own propensity, scored within its own strata, reads
inside NEG_WINDOW. Passing certifies overlap, not that the model is correct.
"""
from __future__ import annotations

import numpy as np

from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.propensity import propensity

DEGEN_MAX = 0.20
NEG_WINDOW = (0.45, 0.55)
N_SPLIT = 5           # repeated 4-fold splits used for real-data screening


def passes_screen(degen: float, neg: float) -> bool:
    """True when the degenerate share and the self-control are both in range."""
    return bool(degen <= DEGEN_MAX
                and NEG_WINDOW[0] <= neg <= NEG_WINDOW[1])


def self_control(y: np.ndarray, e: np.ndarray, q: int) -> tuple[float, float]:
    """The propensity's own score stratified on itself: (auc, degenerate share)."""
    return conditional_auc(y, e, e, q)


def screen_evaluate(s: np.ndarray, y: np.ndarray, C: np.ndarray, kind: str,
                    splits: list) -> tuple[float, float, float, bool]:
    """Conditional AUC plus the screen quantities for one nuisance model.

    Returns (mean conditional AUC, mean degenerate share, mean self-control,
    passes) over the given (train, test) splits.
    """
    cs, dg, nc = [], [], []
    for rep, (tr, te) in enumerate(splits):
        m = propensity(kind, C, tr, y, rep)
        e = m.predict_proba(C[te])[:, 1]
        q = n_strata(len(te))
        c, d = conditional_auc(y[te], s[te], e, q)
        n, _ = conditional_auc(y[te], e, e, q)
        cs.append(float(c)); dg.append(float(d)); nc.append(float(n))
    dgm, ncm = float(np.mean(dg)), float(np.mean(nc))
    return (float(np.mean(cs)), dgm, ncm,
            bool(dgm <= DEGEN_MAX and NEG_WINDOW[0] <= ncm <= NEG_WINDOW[1]))


def cond_and_screen(s: np.ndarray, y: np.ndarray, C: np.ndarray, kind: str,
                    splits: list) -> tuple[float, bool]:
    """Mean conditional AUC over the splits and whether the model passes."""
    cs, dg, nc = [], [], []
    for rep, (tr, te) in enumerate(splits):
        m = propensity(kind, C, tr, y, rep)
        e = m.predict_proba(C[te])[:, 1]
        q = n_strata(len(te))
        c, d = conditional_auc(y[te], s[te], e, q)
        n, _ = conditional_auc(y[te], e, e, q)
        cs.append(float(c)); dg.append(float(d)); nc.append(float(n))
    dgm, ncm = float(np.mean(dg)), float(np.mean(nc))
    return (float(np.mean(cs)),
            bool(dgm <= DEGEN_MAX and NEG_WINDOW[0] <= ncm <= NEG_WINDOW[1]))
