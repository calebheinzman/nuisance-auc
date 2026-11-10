"""The adjusted (stratified, pair-weighted) AUC and its binning helpers.

`conditional_auc` is the estimator used throughout: equal-frequency strata of a
propensity score, within-stratum AUCs, pooled with weights n_pos * n_neg.
`pair_weighted` is the same pooling over arbitrary integer bins.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

STRAT_AUC_Q = 20      # default number of strata of `strat_auc`


def conditional_auc(y: np.ndarray, s: np.ndarray, e: np.ndarray,
                    q: int) -> tuple[float, float]:
    """Pair-weighted mean of within-stratum AUCs; strata are equal-frequency
    bins of the propensity e. Returns (auc, degenerate_stratum_fraction)."""
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.digitize(e, edges[1:-1]), 0, q - 1)
    num, den, degenerate = 0.0, 0.0, 0
    for k in range(q):
        m = b == k
        yk = y[m]
        npos, nneg = int(yk.sum()), int((1 - yk).sum())
        if npos == 0 or nneg == 0:
            degenerate += 1                 # stratum has no comparable pair
            continue
        w = npos * nneg
        num += w * roc_auc_score(yk, s[m])
        den += w
    # Within-stratum comparison discards every cross-stratum pair by design;
    # the overlap signal is the share of strata with no comparable pair.
    return (num / den if den else np.nan), degenerate / q


def n_strata(n_test: int) -> int:
    """Number of propensity strata for a test fold of `n_test` tiles."""
    return 20 if n_test >= 1000 else (10 if n_test >= 200 else 4)


def pair_weighted(y: np.ndarray, s: np.ndarray,
                  bins: np.ndarray) -> tuple[float, float]:
    """Pair-weighted mean of within-bin AUCs, over arbitrary integer bins."""
    num = den = 0.0
    degen = tot = 0
    for b in np.unique(bins):
        m = bins == b
        yy = y[m]
        npos, nneg = int(yy.sum()), int(len(yy) - yy.sum())
        tot += 1
        if npos == 0 or nneg == 0:
            degen += 1
            continue
        w = npos * nneg
        num += w * roc_auc_score(yy, s[m])
        den += w
    return (num / den if den else np.nan), degen / max(tot, 1)


def qbins(x: np.ndarray, q: int) -> np.ndarray:
    """Equal-frequency integer bins 0..q-1 of x."""
    e = np.quantile(x, np.linspace(0, 1, q + 1))
    e[0], e[-1] = -np.inf, np.inf
    return np.clip(np.digitize(x, e[1:-1]), 0, q - 1)


def strata(e: np.ndarray, q: int) -> np.ndarray:
    """Equal-frequency strata of a propensity (same binning as `qbins`)."""
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.clip(np.digitize(e, edges[1:-1]), 0, q - 1)


def strat_auc(y: np.ndarray, s: np.ndarray, e: np.ndarray,
              q: int = STRAT_AUC_Q) -> float:
    """Rank-based pair-weighted adjusted AUC over q quantile strata of e."""
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.clip(np.digitize(e, edges[1:-1]), 0, q - 1)
    from scipy.stats import rankdata
    num = den = 0.0
    for kk in range(q):
        m = bins == kk
        yk, sk = y[m], s[m]
        n1 = int(yk.sum()); n0 = len(yk) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(sk)
        auc = (r[yk == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        num += n1 * n0 * auc
        den += n1 * n0
    # no empty-denominator guard: division by zero if no stratum has both classes
    return num / den
