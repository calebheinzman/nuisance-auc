"""Fast rank-based AUCs for bootstrap loops, and the cluster bootstrap."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

CLUSTER_N_BOOT = 2000


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (ties half credit); nan when a class is absent."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cauc(y: np.ndarray, s: np.ndarray, e: np.ndarray, q: int) -> float:
    """Same estimator as conditional_auc, rank-based for bootstrap speed."""
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.digitize(e, edges[1:-1]), 0, q - 1)
    num = den = 0.0
    for k in range(q):
        m = b == k
        yk = y[m]
        n1 = int(yk.sum())
        n0 = len(yk) - n1
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0
        num += w * fast_auc(yk, s[m])
        den += w
    return num / den if den else np.nan


def cluster_ci(dl: np.ndarray, units: np.ndarray, seed: int,
               n_boot: int = CLUSTER_N_BOOT) -> tuple[float, float, float]:
    """Percentile bootstrap of mean(dl) resampling units with replacement.

    Returns (2.5th percentile, 97.5th percentile, mean / bootstrap SE).
    """
    uniq = np.unique(units)
    idx_of = {u: np.where(units == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        vals = np.concatenate([dl[idx_of[u]] for u in draw])
        means[b] = vals.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    se = means.std(ddof=1)
    return float(lo), float(hi), float(dl.mean() / se) if se > 0 else 0.0
