"""Calibration of a propensity within its own strata."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from nuisauc.estimand import strata


def calibration(y: np.ndarray, e: np.ndarray, q: int) -> tuple[float, float, float, float]:
    """(ECE, Hosmer-Lemeshow statistic, its chi2(q - 2) p-value, calibration slope)."""
    b = strata(e, q)
    ece = hl = 0.0
    for k in range(q):
        m = b == k
        if m.sum() == 0:
            continue
        o, p = y[m].mean(), e[m].mean()
        ece += m.mean() * abs(o - p)
        hl += m.sum() * (o - p) ** 2 / max(p * (1 - p), 1e-9)
    lg = np.log(np.clip(e, 1e-6, 1 - 1e-6) / np.clip(1 - e, 1e-6, 1))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(
        lg[:, None], y).coef_[0, 0]
    return ece, hl, float(chi2.sf(hl, q - 2)), float(slope)
