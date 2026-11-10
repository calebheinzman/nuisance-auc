"""Correlation law for the propensity-adjusted AUC, by Gauss-Hermite quadrature.

Within any stratum of a linear index T = w'Z, the stratum-centred label index
and score are bivariate normal with a correlation rho that does not depend on
the stratum:

    rho = (1 + <a_perp, b_perp>) / sqrt((1 + |a_perp|^2)(1 + |b_perp|^2 + sig^2))

The within-stratum AUC is strictly increasing in rho with AUC(0) = 1/2, so
sign(theta - 1/2) = sign(rho). Family I (Z1, Z2; stratifier Z1 + Z2 or Z1)
reverses iff alpha * lam > 1. Family III: Z ~ N(0, AR1(0.6)) in R^12, the
misspecified model sees the first six columns. The pooled estimand weights
strata by pi(t)(1 - pi(t)) f_T(t).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit
from scipy.stats import norm

NODES = 96
AR_RHO, P_DIM, P_SEEN = 0.6, 12, 6
# general law: 80 probabilists' nodes, weights normalised to sum to one
LAW_GH_X, LAW_GH_W = hermegauss(80)
LAW_GH_W = LAW_GH_W / LAW_GH_W.sum()


def gh(n: int = NODES) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite nodes and weights for E[f(g)], g ~ N(0, 1)."""
    x, w = hermegauss(n)
    return x, w / np.sqrt(2 * np.pi)


def sigmoid(x):
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def within_auc(m: float, v_l: float, rho: float) -> tuple[float, float]:
    """Within-stratum AUC and prevalence for L = m + sqrt(v_l) g, Y ~ Bern(sigmoid(L)),
    and a score whose stratum-centred part has correlation rho with L.

    P(S_1 > S_0 | l_1, l_0) = Phi(rho (l_1 - l_0) / (sqrt(2 (1 - rho^2)) sqrt(v_l))).
    """
    x, w = gh()
    l = m + np.sqrt(v_l) * x
    p = sigmoid(l)
    pi = float(np.sum(w * p))
    if abs(rho) >= 1.0:
        k = np.inf * np.sign(rho)
    else:
        k = rho / (np.sqrt(2.0 * (1.0 - rho ** 2)) * np.sqrt(v_l))
    D = np.subtract.outer(l, l)                     # l_i (positive) - l_j (negative)
    P = norm.cdf(k * D) if np.isfinite(k) else (D > 0).astype(float) + 0.5 * (D == 0)
    W = np.outer(w * p, w * (1.0 - p))
    return float(np.sum(W * P) / np.sum(W)), pi


def pooled(slope: float, v_t: float, v_l: float, rho: float) -> float:
    """Pair-weighted pooled conditional AUC over strata t ~ N(0, v_t) with
    stratum mean label index m(t) = slope * t."""
    x, w = gh()
    t = np.sqrt(v_t) * x
    num = den = 0.0
    for ti, wi in zip(t, w):
        a, pi = within_auc(slope * ti, v_l, rho)
        wt = wi * pi * (1.0 - pi)
        num += wt * a
        den += wt
    return num / den


def rho_general(a_perp, b_perp, sig2: float = 1.0) -> float:
    """Within-stratum correlation from the uncontrolled label and score loadings."""
    a_perp, b_perp = np.atleast_1d(a_perp), np.atleast_1d(b_perp)
    return float((1.0 + a_perp @ b_perp)
                 / np.sqrt((1.0 + a_perp @ a_perp) * (1.0 + b_perp @ b_perp + sig2)))


def family_I(alpha: float, lam: float) -> tuple[float, float, float, float]:
    """(rho_true, theta_true, rho_mis, theta_mis) for Family I."""
    rho_t = rho_general(np.zeros(1), np.array([np.sqrt(2) * lam]))
    rho_m = rho_general(np.array([alpha]), np.array([-lam]))
    th_t = pooled(alpha, 2.0, 1.0, rho_t)            # T = Z1+Z2 ~ N(0,2), L~ = U
    th_m = pooled(alpha, 1.0, 1.0 + alpha ** 2, rho_m)  # T = Z1, L~ = alpha Z2 + U
    return rho_t, th_t, rho_m, th_m


def family_III(alpha: float, lam: float):
    """(rho_true, theta_true, rho_mis, theta_mis, extras) for Family III.

    Stratifying on the fitted linear index of the seen block is, up to a
    monotone map, stratifying on m = E[core | Z_S] (coefficient
    Sigma_SS^-1 Sigma_SH a_H).
    """
    idx = np.arange(P_DIM)
    Sig = AR_RHO ** np.abs(np.subtract.outer(idx, idx))
    a = np.ones(P_DIM)
    a = a / np.sqrt(a @ Sig @ a)                       # core = a'Z, Var 1
    h = np.zeros(P_DIM); h[P_SEEN:] = 1.0 / (P_DIM - P_SEEN)
    h = h / np.sqrt(h @ Sig @ h)                       # hidden_n = h'Z, Var 1
    S, H = slice(0, P_SEEN), slice(P_SEEN, P_DIM)
    Sss, Shs, Shh = Sig[S, S], Sig[H, S], Sig[H, H]
    Sss_inv = np.linalg.inv(Sss)
    Sr = Shh - Shs @ Sss_inv @ Shs.T                   # Cov(Z_H | Z_S)
    aH, hH = a[H], h[H]
    v_rc = aH @ Sr @ aH                                # Var(core residual | Z_S)
    v_rh = hH @ Sr @ hH                                # Var(hidden residual | Z_S)
    c_rr = aH @ Sr @ hH                                # Cov of the two residuals
    g = a[S] + Sss_inv @ Shs.T @ aH                    # m = g'Z_S = E[core|Z_S]
    k = Sss_inv @ Shs.T @ hH                           # E[hidden_n|Z_S] = k'Z_S
    v_m = g @ Sss @ g
    v_k_perp = k @ Sss @ k - (k @ Sss @ g) ** 2 / v_m  # seen score part not fixed by m
    # misspecified (index of the seen block)
    v_L = alpha ** 2 * v_rc + 1.0
    v_S = 1.0 + lam ** 2 * (v_rh + v_k_perp) + 1.0
    cov = 1.0 + alpha * lam * c_rr
    rho_m = cov / np.sqrt(v_L * v_S)
    th_m = pooled(alpha, v_m, v_L, rho_m)
    # truth (oracle stratifier on the core itself)
    b_perp2 = 1.0 - (h @ Sig @ a) ** 2
    rho_t = 1.0 / np.sqrt(1.0 + lam ** 2 * b_perp2 + 1.0)
    th_t = pooled(alpha, 1.0, 1.0, rho_t)
    return rho_t, th_t, rho_m, th_m, dict(v_rc=v_rc, v_rh=v_rh, c_rr=c_rr,
                                          absorbed_R2=1.0 - v_rh)


def law(Sigma: np.ndarray, a: np.ndarray, b: np.ndarray, sigma2: float,
        w: np.ndarray) -> tuple[float, float]:
    """Pair-weighted adjusted AUC under exact level sets of w'Z, for
    Z ~ N(0, Sigma), logit = a'Z + U, S = U + b'Z + eps (Var eps = sigma2).

    Returns (theta, rho). Uses 80-node quadrature (LAW_GH_X, LAW_GH_W).
    """
    vT = float(w @ Sigma @ w)
    Sw = Sigma - np.outer(Sigma @ w, Sigma @ w) / vT
    vL = 1 + a @ Sw @ a
    vS = 1 + b @ Sw @ b + sigma2
    rho = (1 + a @ Sw @ b) / np.sqrt(vL * vS)
    beta = (a @ Sigma @ w) / vT
    l = LAW_GH_X * np.sqrt(vL)                # L-tilde nodes
    num = den = 0.0
    k = rho * np.sqrt(vS / vL) / np.sqrt(2 * vS * (1 - rho ** 2))
    for t, wt in zip(LAW_GH_X * np.sqrt(vT), LAW_GH_W):
        m = beta * t
        p1 = expit(m + l) * LAW_GH_W
        p0 = (1 - expit(m + l)) * LAW_GH_W
        pi = p1.sum()
        if pi * (1 - pi) < 1e-12:
            continue
        auc = (p1[:, None] * p0[None, :]
               * norm.cdf(k * (l[:, None] - l[None, :]))).sum() / (pi * (1 - pi))
        num += wt * pi * (1 - pi) * auc
        den += wt * pi * (1 - pi)
    return num / den, rho
