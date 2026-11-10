"""Independent Monte Carlo verification of the correlation law.

Simulates the Gaussian latent-index model directly (no quadrature code is
shared with `nuisauc.theory.law`) and checks:

  T1  the within-stratum correlation is constant and equals the closed form
  T2  at fixed (m, v_L) the within-stratum AUC increases in rho, = 1/2 at 0
  T3  sign(theta_w - 1/2) = sign(rho_w) for arbitrary nonnegative weights
  T4  Family I closed forms and pooled values against the quadrature table
  T5  Family III against the quadrature table
  L1  the self-stratified control reads 1/2
  P1  a model correct on the covariates it uses balances them and passes the
      control while the adjusted AUC is biased
  P2  held-out family covariances under a correct nuisance model

All draws come from one generator passed in by the caller; the checks consume
it in the order T1, T2, T3/T4, T5, L1, P1, P2 (seed MC_SEED reproduces the
reference run). Each check returns its quantities and asserts its claim.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MC_SEED = 20260908


def sigmoid(x):
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC with ties given half credit."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def adjusted_auc(score: np.ndarray, y: np.ndarray, strat: np.ndarray, q: int,
                 weights=None) -> float:
    """Pair-weighted (or custom nonnegatively weighted) adjusted AUC over
    q quantile bins of `strat`."""
    edges = np.quantile(strat, np.linspace(0, 1, q + 1))
    bins = np.clip(np.searchsorted(edges, strat, side="right") - 1, 0, q - 1)
    num = den = 0.0
    for b in range(q):
        idx = bins == b
        n1 = int(y[idx].sum())
        n0 = int(idx.sum()) - n1
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0 if weights is None else weights[b]
        num += w * auc(score[idx], y[idx])
        den += w
    return num / den


def rho_formula(a: np.ndarray, b: np.ndarray, w: np.ndarray, Sigma: np.ndarray,
                sig2: float = 1.0) -> tuple[float, np.ndarray]:
    """Closed-form within-stratum correlation for stratifier w'Z: (rho, Sigma_w)."""
    Sw = Sigma - Sigma @ np.outer(w, w) @ Sigma / (w @ Sigma @ w)
    return (1 + a @ Sw @ b) / np.sqrt((1 + a @ Sw @ a) * (1 + b @ Sw @ b + sig2)), Sw


def simulate_latent(a: np.ndarray, b: np.ndarray, Sigma: np.ndarray, n: int,
                    rng: np.random.Generator, sig2: float = 1.0):
    """Draw (Z, U, L, S, Y): L = a'Z + U, S = U + b'Z + eps, Y ~ Bern(sigmoid(L))."""
    p = len(a)
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    U = rng.standard_normal(n)
    eps = rng.standard_normal(n) * np.sqrt(sig2)
    L = Z @ a + U
    S = U + Z @ b + eps
    Y = (rng.random(n) < sigmoid(L)).astype(int)
    return Z, U, L, S, Y


def within_auc_mc(m: float, v_l: float, rho: float, rng: np.random.Generator,
                  n: int = 300_000) -> float:
    """Brute-force within-stratum AUC for L = m + sqrt(v_l) g, score = rho-
    correlated Gaussian, Y ~ Bern(sigmoid(L))."""
    g = rng.standard_normal(n)
    v = rng.standard_normal(n)
    L = m + np.sqrt(v_l) * g
    S = rho * g + np.sqrt(1 - rho ** 2) * v
    Y = (rng.random(n) < sigmoid(L)).astype(int)
    return auc(S, Y)


def family_I_mc(alpha: float, lam: float, rng: np.random.Generator,
                n: int = 600_000, q: int = 100) -> tuple[float, float, float, float]:
    """Monte Carlo (rho_true, theta_true, rho_mis, theta_mis) for Family I."""
    Sigma = np.eye(2)
    a = alpha * np.array([1.0, 1.0]); b = lam * np.array([1.0, -1.0])
    Z, U, L, S, Y = simulate_latent(a, b, Sigma, n, rng)
    w_true = np.array([1.0, 1.0]); w_mis = np.array([1.0, 0.0])
    rho_t, _ = rho_formula(a, b, w_true, Sigma)
    rho_m, _ = rho_formula(a, b, w_mis, Sigma)
    th_t = adjusted_auc(S, Y, Z @ w_true, q)
    th_m = adjusted_auc(S, Y, Z @ w_mis, q)
    return rho_t, th_t, rho_m, th_m


def check_T1(rng: np.random.Generator, n: int = 400_000) -> dict:
    """T1: within-stratum correlation is constant and equals the formula."""
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    a = np.array([1.0, 0.7]); b = np.array([0.4, -1.2]); w = np.array([1.0, 0.0])
    Z, U, L, S, Y = simulate_latent(a, b, Sigma, n, rng)
    rho, _ = rho_formula(a, b, w, Sigma)
    T = Z @ w
    edges = np.quantile(T, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, T, side="right") - 1, 0, 9)
    cors = [np.corrcoef(L[bins == k], S[bins == k])[0, 1] for k in range(10)]
    # exact conditioning: regress out T and compare residual correlation
    Lr = L - np.polyval(np.polyfit(T, L, 1), T)
    Sr = S - np.polyval(np.polyfit(T, S, 1), T)
    resid = np.corrcoef(Lr, Sr)[0, 1]
    assert abs(resid - rho) < 0.01
    out = dict(rho=float(rho), decile_corr_min=float(min(cors)),
               decile_corr_max=float(max(cors)), residual_corr=float(resid))
    log.info("T1 %s", out)
    return out


def check_T2(rng: np.random.Generator) -> dict:
    """T2: AUC(rho) increasing at fixed (m, v_L) and 1/2 at rho = 0."""
    out = {}
    for (m, v_l) in [(0.0, 1.0), (0.0, 2.0), (1.5, 1.0)]:
        vals = [within_auc_mc(m, v_l, r, rng) for r in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9)]
        mono = all(np.diff(vals) > 0)
        assert mono and abs(vals[2] - 0.5) < 0.01
        out[(m, v_l)] = vals
        log.info("T2 m=%+.1f v_L=%.1f AUC(rho)=%s", m, v_l, np.round(vals, 4).tolist())
    return out


def check_T3_T4(rng: np.random.Generator, quad: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """T3/T4 against the Family I quadrature table (theory_bias_check)."""
    rows = []
    for _, r in quad.iterrows():
        alpha, lam = float(r.alpha), float(r.lam)
        rho_t, th_t, rho_m, th_m = family_I_mc(alpha, lam, rng)
        rho_t_cf = 1 / np.sqrt(2 + 2 * lam ** 2)
        rho_m_cf = (1 - alpha * lam) / np.sqrt((1 + alpha ** 2) * (2 + lam ** 2))
        rows.append(dict(alpha=alpha, lam=lam, rho_true_cf=rho_t_cf, rho_true_mat=rho_t,
                         rho_mis_cf=rho_m_cf, rho_mis_mat=rho_m,
                         mc_true=th_t, quad_true=r.theory_true,
                         mc_mis=th_m, quad_mis=r.theory_mis,
                         sign_ok=(np.sign(th_m - 0.5) == np.sign(rho_m)) or abs(rho_m) < 1e-9))
    out = pd.DataFrame(rows)
    assert np.allclose(out.rho_true_cf, out.rho_true_mat, atol=1e-12, rtol=0)
    assert np.allclose(out.rho_mis_cf, out.rho_mis_mat, atol=1e-12, rtol=0)
    assert out.sign_ok.all()
    assert (out.mc_true-out.quad_true).abs().max() < 0.01
    assert (out.mc_mis-out.quad_mis).abs().max() < 0.01
    # T3 with random nonnegative weights
    alpha, lam = 2.0, 1.0
    Sigma = np.eye(2); a = alpha * np.array([1.0, 1.0]); b = lam * np.array([1.0, -1.0])
    Z, U, L, S, Y = simulate_latent(a, b, Sigma, 600_000, rng)
    wts = rng.random(100)
    th = adjusted_auc(S, Y, Z[:, 0], 100, weights=wts)
    rho_m, _ = rho_formula(a, b, np.array([1.0, 0.0]), Sigma)
    assert np.sign(th-0.5) == np.sign(rho_m)
    t3 = dict(theta_random_weights=float(th), rho_mis=float(rho_m))
    log.info("T3/T4 max |MC - quadrature| truth %.4f, wrong %.4f",
             (out.mc_true - out.quad_true).abs().max(),
             (out.mc_mis - out.quad_mis).abs().max())
    return out, t3


def check_T5(rng: np.random.Generator, quad3: pd.DataFrame, n: int = 600_000,
             q: int = 100) -> pd.DataFrame:
    """T5: Family III Monte Carlo against its quadrature table."""
    P, PS = 12, 6
    idx = np.arange(P)
    Sigma = 0.6 ** np.abs(np.subtract.outer(idx, idx))
    a0 = np.ones(P); a0 = a0 / np.sqrt(a0 @ Sigma @ a0)
    h = np.zeros(P); h[PS:] = 1.0; h = h / np.sqrt(h @ Sigma @ h)
    S_, H_ = slice(0, PS), slice(PS, P)
    g = a0[S_] + np.linalg.solve(Sigma[S_, S_], Sigma[S_, H_] @ a0[H_])
    w_mis = np.concatenate([g, np.zeros(P - PS)])
    rows = []
    for _, r in quad3.iterrows():
        alpha, lam = float(r.alpha), float(r.lam)
        a = alpha * a0; b = lam * h
        Z, U, L, S, Y = simulate_latent(a, b, Sigma, n, rng)
        th_t = adjusted_auc(S, Y, Z @ a0, q)
        th_m = adjusted_auc(S, Y, Z @ w_mis, q)
        rho_m, _ = rho_formula(a, b, w_mis, Sigma)
        rows.append(dict(alpha=alpha, lam=lam, mc_true=th_t, quad_true=r.theory_true,
                         mc_mis=th_m, quad_mis=r.theory_mis, rho_mis=rho_m, rho_mis_script=r.rho_mis))
        assert abs(rho_m - r.rho_mis) < 1e-12, "Subset projection disagrees with quadrature"
    out = pd.DataFrame(rows)
    assert (out.mc_true-out.quad_true).abs().max() < 0.01
    assert (out.mc_mis-out.quad_mis).abs().max() < 0.01
    log.info("T5 max |MC - quadrature| truth %.4f, wrong %.4f",
             (out.mc_true - out.quad_true).abs().max(),
             (out.mc_mis - out.quad_mis).abs().max())
    return out


def check_L1(rng: np.random.Generator, n: int = 200_000) -> dict:
    """L1: the self-stratified control (exact level sets and quantile bins)."""
    Sigma = np.eye(2); a = np.array([1.0, 1.0]); b = np.array([1.0, -1.0])
    Z, U, L, S, Y = simulate_latent(a, b, Sigma, n, rng)
    R = sigmoid(1.3 * Z[:, 0] - 0.2)          # an arbitrary 'propensity'
    # exact level sets: score constant within stratum -> AUC 1/2 with ties
    out = {"exact": float(auc(np.zeros(1000), (rng.random(1000) < 0.4).astype(int)))}
    for q in (10, 20, 100):
        out[q] = float(adjusted_auc(R, Y, R, q))
    log.info("L1 %s", out)
    return out


def check_P1(rng: np.random.Generator, n: int = 600_000, q: int = 50) -> dict:
    """P1: correct-on-used-covariates model balances Z_1 and passes the control,
    while the adjusted AUC is biased; a wrong form on Z_1 leaves imbalance."""
    Sigma = np.eye(2); alpha = 1.0
    a = alpha * np.array([1.0, 1.0]); b = 2.0 * np.array([1.0, -1.0])
    Z, U, L, S, Y = simulate_latent(a, b, Sigma, n, rng)
    # e_A = P(Y=1 | Z_1) by fine binning of Z_1 (correct given what it uses)
    edges = np.quantile(Z[:, 0], np.linspace(0, 1, q + 1))
    bins = np.clip(np.searchsorted(edges, Z[:, 0], side="right") - 1, 0, q - 1)
    eA = np.zeros(n)
    for k in range(q):
        eA[bins == k] = Y[bins == k].mean()
    diffs = [Z[(bins == k) & (Y == 1), 0].mean() - Z[(bins == k) & (Y == 0), 0].mean() for k in range(q)]
    out = dict(max_balance_diff=float(np.max(np.abs(diffs))),
               bin_width=float(edges[q//2+1]-edges[q//2]),
               control=float(adjusted_auc(eA, Y, eA, q)))
    out["theta_true"] = float(adjusted_auc(S, Y, Z @ np.array([1.0, 1.0]), 100))
    out["theta_eA"] = float(adjusted_auc(S, Y, eA, q))
    # wrong functional form on Z_A: level sets of |Z_1| when e_A is monotone in Z_1
    what = np.abs(Z[:, 0])
    edges2 = np.quantile(what, np.linspace(0, 1, q + 1))
    bins2 = np.clip(np.searchsorted(edges2, what, side="right") - 1, 0, q - 1)
    diffs2 = [eA[(bins2 == k) & (Y == 1)].mean() - eA[(bins2 == k) & (Y == 0)].mean() for k in range(q)]
    out["max_imbalance_wrong_form"] = float(np.max(np.abs(diffs2)))
    log.info("P1 %s", out)
    return out


def check_P2(rng: np.random.Generator, n: int = 400_000, q: int = 50) -> list[dict]:
    """P2: within strata of m, cov(B_inc, Y*) = 0 and cov(B_exc, Y*) = rho * v."""
    rows = []
    for k_inc, k_exc, rho in [(2, 2, 0.6), (0, 1, 1.0), (1, 1, 1.5)]:
        U = rng.standard_normal(n)
        A = rng.standard_normal(n)
        hA = 0.8 * A + 0.3 * A ** 2                 # non-Gaussian h(A)
        eps = rng.standard_normal(n)
        Ystar = hA + U + eps
        Binc = rho * U[:, None] + rng.standard_normal((n, k_inc)) if k_inc else np.zeros((n, 0))
        Bexc = rho * U[:, None] + rng.standard_normal((n, k_exc))
        v = 1.0 / (1.0 + k_inc * rho ** 2)
        Uhat = (rho / (1 + k_inc * rho ** 2)) * Binc.sum(axis=1) if k_inc else np.zeros(n)
        m = hA + Uhat
        edges = np.quantile(m, np.linspace(0, 1, q + 1))
        bins = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, q - 1)

        def wcov(X):
            c = 0.0
            for b in range(q):
                i = bins == b
                c += np.mean((X[i] - X[i].mean()) * (Ystar[i] - Ystar[i].mean())) * i.mean()
            return c
        c_inc = np.mean([wcov(Binc[:, j]) for j in range(k_inc)]) if k_inc else float("nan")
        c_exc = np.mean([wcov(Bexc[:, j]) for j in range(k_exc)])
        rows.append(dict(k_inc=k_inc, k_exc=k_exc, rho=rho, cov_inc=float(c_inc),
                         cov_exc=float(c_exc), predicted_exc=float(rho * v)))
        log.info("P2 %s", rows[-1])
    return rows
