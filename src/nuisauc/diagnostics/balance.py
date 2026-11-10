"""Within-stratum covariate balance between the classes."""
from __future__ import annotations

import numpy as np

from nuisauc.estimand import strata


def balance(X: np.ndarray, y: np.ndarray, e: np.ndarray, q: int) -> np.ndarray:
    """Pair-weighted within-stratum SMD per column (pooled-SD scaled)."""
    b = strata(e, q)
    sd = X.std(0) + 1e-12
    num = np.zeros(X.shape[1])
    den = 0.0
    for k in range(q):
        m = b == k
        yk = y[m]
        n1, n0 = int(yk.sum()), int((1 - yk).sum())
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0
        num += w * (X[m][yk == 1].mean(0) - X[m][yk == 0].mean(0))
        den += w
    return num / den / sd
