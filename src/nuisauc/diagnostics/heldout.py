"""Held-out feature-family tests of a fitted propensity.

HOB: within-stratum predictive power of features the propensity never saw.
EXC (split-family floor): fit the propensity on A plus half of the held-out
family B, take HOB(excluded half) - HOB(included half), maximised over both
role assignments. Thresholds come from the statistic's null distribution.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nuisauc.estimand import conditional_auc
from nuisauc.propensity import fit_simple

HOB_SEED = 3500       # default fold seed of the cross-fitted B-probe
SIM_N = 20_000        # tiles per simulated replicate in the EXC validation


def hob(e_te: np.ndarray, y_te: np.ndarray, B_te: np.ndarray, q: int,
        seed: int = HOB_SEED) -> tuple[float, float]:
    """Within-stratum predictive power of the held-out features.

    Cross-fitted so the B-probe is never scored on tiles it was fitted on.
    Returns (|conditional AUC - 0.5|, unconditional probe AUC); (nan, nan) if
    the probe cannot be fitted.
    """
    p = np.empty(len(y_te), float)
    try:
        for tr, te in StratifiedKFold(4, shuffle=True,
                                      random_state=seed).split(B_te, y_te):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000))
            m.fit(B_te[tr], y_te[tr])
            p[te] = m.decision_function(B_te[te])
    except Exception:
        return float("nan"), float("nan")
    c, _d = conditional_auc(y_te, p, e_te, q)
    uncond = float(roc_auc_score(y_te, p))
    return (abs(float(c) - 0.5) if np.isfinite(c) else float("nan")), uncond


def null_threshold(vals, pct: float = 95) -> float:
    """Percentile of the finite null values (nan if there are none)."""
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return float(np.percentile(v, pct)) if len(v) else float("nan")


def exc_statistic(A: np.ndarray, B: np.ndarray, y: np.ndarray, tr: np.ndarray,
                  te: np.ndarray, q: int, rep: int,
                  rng: np.random.Generator) -> tuple[float, float]:
    """EXC = max over both role assignments of HOB(excluded) - HOB(included).

    Returns (EXC, mean floor HOB(included)).
    """
    cols = rng.permutation(B.shape[1])
    half = B.shape[1] // 2
    h1, h2 = cols[:half], cols[half:]
    best = -np.inf
    floors = []
    for inc, exc in ((h1, h2), (h2, h1)):
        X = np.concatenate([A, B[:, inc]], axis=1)
        m = fit_simple("gbm", X, y, tr, rep)
        e = m.predict_proba(X[te])[:, 1]
        h_exc, _ = hob(e, y[te], B[te][:, exc], q)
        h_inc, _ = hob(e, y[te], B[te][:, inc], q)
        if np.isfinite(h_exc) and np.isfinite(h_inc):
            best = max(best, h_exc - h_inc)
            floors.append(h_inc)
    return (best if np.isfinite(best) else float("nan"),
            float(np.mean(floors)) if floors else float("nan"))
