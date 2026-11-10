"""Semi-synthetic cells: real nuisance descriptors, synthetic label mechanism.

`load_features` returns the real 62-feature descriptor A, the frequency family
B and derived directions (a1: first PCA direction of A; a_lo: the top-10 PCA
direction least explained by trivial-16; b1, b1p: frequency directions). The
label is drawn from a logistic model in a chosen core direction, so the
fold-matched target theta_B is known.

Two generators exist: `synth` (core standardised, slope SEMI_ALPHA) and
`adj_synth` (core used as given, slope alpha), used by the label-fit
selection experiments together with `adj_load_geometry` and `adj_regimes`.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from nuisauc.checks import passes_screen
from nuisauc.data.cache import descriptor62
from nuisauc.data.corpora import CORPORA
from nuisauc.data.descriptors import frequency16
from nuisauc.estimand import conditional_auc, pair_weighted, qbins
from nuisauc.simulate.gaussian import Q_FINE

__all__ = ["CORPORA", "SEMI_ALPHA", "REGIMES_SEED", "VARIANTS", "ADJ_SEED",
           "ADJ_ALPHAS", "load_features", "synth", "estimate_with",
           "adj_load_geometry", "adj_regimes", "adj_synth"]

SEMI_ALPHA = 1.0
REGIMES_SEED = 4500
VARIANTS = [("none", 0.0), ("opp", 1.0), ("opp", 2.0), ("opp", 3.0),
            ("same", 1.0), ("same", 2.0)]
ADJ_SEED = 4800
ADJ_ALPHAS = [1.0, 2.0]


def load_features(corpus: str, cap: int):
    """(A, B, a1, a_lo, b1, b1p, info) from the cached descriptors."""
    A = np.nan_to_num(descriptor62(corpus, cap), nan=0.0,
                      posinf=0.0, neginf=0.0)
    B = np.nan_to_num(frequency16(corpus, cap), nan=0.0,
                      posinf=0.0, neginf=0.0)
    As = StandardScaler().fit_transform(A)
    Bs = StandardScaler().fit_transform(B)
    pa = PCA(n_components=10, random_state=0).fit_transform(As)
    pa = pa / pa.std(axis=0)
    a1 = pa[:, 0]
    b1 = PCA(n_components=1, random_state=0).fit_transform(Bs)[:, 0]
    b1 = b1 / b1.std()

    def oof_resid(target, X):
        pred = np.empty_like(target)
        for tr, te in KFold(4, shuffle=True, random_state=0).split(X):
            m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
            m.fit(X[tr], target[tr])
            pred[te] = m.predict(X[te])
        resid = target - pred
        return resid, 1.0 - resid.var() / target.var()

    # a_lo: the top-10 component least explained by the trivial-16 subspace
    T16 = A[:, :16]
    r2s = [oof_resid(pa[:, j], T16)[1] for j in range(1, 10)]
    j_lo = 1 + int(np.argmin(r2s))
    a_lo = pa[:, j_lo]
    # b1_perp: family variation the descriptor cannot absorb
    resid, r2_b1_A = oof_resid(b1, A)
    b1p = resid / resid.std()
    _, r2_b1p_B = oof_resid(b1p, B)
    return (A, B, a1, a_lo, b1, b1p,
            dict(j_lo=j_lo, r2_alo_T16=float(min(r2s)),
                 r2_b1_A=float(r2_b1_A), r2_b1p_B=float(r2_b1p_B)))


def synth(core: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """y ~ Bern(sigmoid(SEMI_ALPHA * core / sd(core) + u)); returns (y, u)."""
    u = rng.standard_normal(len(core))
    logit = SEMI_ALPHA * (core / np.std(core)) + u
    y = (rng.random(len(core)) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return y, u


def estimate_with(e_te: np.ndarray, y: np.ndarray, s: np.ndarray,
                  core: np.ndarray, te: np.ndarray, q: int):
    """Fold-matched truth and one propensity's estimate.

    Returns (theta_B, est, |est - theta_B|, degen, self-control, passes screen).
    """
    tB = pair_weighted(y[te], s[te], qbins(core[te], Q_FINE))[0]
    est, degen = conditional_auc(y[te], s[te], e_te, q)
    neg, _ = conditional_auc(y[te], e_te, e_te, q)
    return (float(tB), float(est), abs(float(est) - float(tB)),
            float(degen), float(neg), passes_screen(degen, neg))


def adj_load_geometry(corpus: str, cap: int):
    """(T16, standardised T16, p1, p2, p3, a_lo, info): trivial-16 geometry with
    its first three unit-variance principal components."""
    A, _B, _a1, a_lo, _b1, _b1p, info = load_features(corpus, cap)
    T16 = A[:, :16]
    T16s = StandardScaler().fit_transform(T16)
    pt = PCA(n_components=3, random_state=0).fit_transform(T16s)
    pt = pt / pt.std(axis=0)
    return T16, T16s, pt[:, 0], pt[:, 1], pt[:, 2], a_lo, info


def adj_regimes(T16s: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                p3: np.ndarray, a_lo: np.ndarray) -> dict:
    """{regime: (core, score contrast)} for the linear, nonlinear and outspan regimes."""
    def std(v):
        return v / v.std()
    rng_dir = np.random.default_rng(ADJ_SEED)
    w1 = rng_dir.standard_normal(T16s.shape[1])
    w2 = rng_dir.standard_normal(T16s.shape[1])
    return {
        "linear": (std(T16s @ w1), std(T16s @ w2)),
        "nonlinear": (std(p1 * p2 + 0.7 * (p3 ** 2 - 1)),
                      std(p1 * p2 - 0.7 * (p3 ** 2 - 1))),
        "outspan": (std(p1 + a_lo), std(p1 - a_lo)),
    }


def adj_synth(core: np.ndarray, alpha: float,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """y ~ Bern(sigmoid(alpha * core + u)); returns (y, u)."""
    u = rng.standard_normal(len(core))
    logit = alpha * core + u
    y = (rng.random(len(core)) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return y, u
