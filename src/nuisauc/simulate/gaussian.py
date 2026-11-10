"""Family I: Gaussian nuisance with a direction the balancing score cannot see.

    Z1, Z2, U, eps ~ N(0, 1);  logit P(Y=1) = alpha (Z1 + Z2) + BETA U
    S = GAMMA U + lam (Z1 - Z2) + eps          (or a pure-nuisance score)

theta_B stratifies on the true balancing score Z1 + Z2; theta_Z on a quantile
grid over (Z1, Z2). The "misspecified" propensity sees only Z1.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nuisauc.estimand import conditional_auc, n_strata, pair_weighted, qbins

N = 20_000
ALPHAS = [0.5, 1.0, 2.0]
LAMBDAS = [0.0, 1.0, 2.0]
BETA = GAMMA = 1.0
Q_FINE = 100          # strata for the population target theta_B
GRID = 10             # GRID x GRID quantile grid on (Z1, Z2) for theta_Z
FITS = ["correct", "misspecified", "flexible"]


def simulate(alpha: float, lam: float, rng: np.random.Generator,
             pure: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw (Z, y, s) from Family I; `pure` makes s a pure-nuisance score."""
    z1, z2, u, eps = rng.standard_normal((4, N))
    logit = alpha * (z1 + z2) + BETA * u
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    if pure:
        # a score built only from the nuisance direction that predicts y
        s = (z1 + z2) + eps
    else:
        s = GAMMA * u + lam * (z1 - z2) + eps
    Z = np.column_stack([z1, z2])
    return Z, y, s


def theta_B(Z: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    """True balancing score: e(Z) is monotone in Z1 + Z2, so bin that."""
    return pair_weighted(y, s, qbins(Z[:, 0] + Z[:, 1], Q_FINE))[0]


def theta_Z(Z: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    """Full-covariate target: a GRID x GRID quantile grid on (Z1, Z2)."""
    b = qbins(Z[:, 0], GRID) * GRID + qbins(Z[:, 1], GRID)
    return pair_weighted(y, s, b)[0]


def fit_propensity(kind: str, Z: np.ndarray, y: np.ndarray, tr: np.ndarray,
                   seed: int):
    """Fit a "correct", "misspecified" (Z1 only) or "flexible" propensity.

    Returns (model, columns the model expects).
    """
    if kind == "flexible":
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        m.fit(Z[tr], y[tr])
        return m, Z
    cols = Z if kind == "correct" else Z[:, :1]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(cols[tr], y[tr])
    return m, cols


def estimate(kind: str, Z: np.ndarray, y: np.ndarray, s: np.ndarray,
             seed: int) -> tuple[float, float, float]:
    """Fit on a fold, stratify the held-out fold: (estimate, degen, self-control)."""
    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=seed)
                       .split(np.zeros(len(y)), y)))
    m, cols = fit_propensity(kind, Z, y, tr, seed)
    e = m.predict_proba(cols[te])[:, 1]
    q = n_strata(len(te))
    est, degen = conditional_auc(y[te], s[te], e, q)
    # capacity-matched negative control: the propensity's own score
    neg, _ = conditional_auc(y[te], e, e, q)
    return float(est), float(degen), float(neg)
