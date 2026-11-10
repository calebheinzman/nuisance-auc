"""Sanity properties of the core library (no legacy code, no cached data)."""
from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

from nuisauc import checks, estimand
from nuisauc.diagnostics.heldout import null_threshold
from nuisauc.inference.bootstrap import cauc, fast_auc
from nuisauc.inference.selection import majority_verdict
from nuisauc.simulate import gaussian
from nuisauc.theory import law

CORE = ["estimand", "checks", "propensity", "data.corpora", "data.descriptors",
        "data.cache", "data.groups", "data.medmnist", "detectors.vlm",
        "detectors.selftrain", "detectors.probes", "diagnostics.balance",
        "diagnostics.calibration", "diagnostics.heldout", "inference.bootstrap",
        "inference.selection", "simulate.gaussian", "simulate.semisynthetic",
        "theory.law", "theory.montecarlo"]


def test_core_imports_without_torch():
    code = ("import importlib, sys\n"
            f"for m in {CORE!r}: importlib.import_module('nuisauc.' + m)\n"
            "bad = [m for m in ('torch', 'open_clip', 'transformers', 'timm', 'PIL') if m in sys.modules]\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("q", [3, 10, 37])
def test_qbins_balanced(q):
    x = np.random.default_rng(0).standard_normal(3700)
    counts = np.bincount(estimand.qbins(x, q), minlength=q)
    assert counts.sum() == 3700 and counts.max() - counts.min() <= 1
    assert np.array_equal(estimand.qbins(x, q), estimand.strata(x, q))


def test_self_stratified_score_is_half_under_exact_strata():
    rng = np.random.default_rng(1)
    level = rng.integers(0, 8, 4000)              # 8 exact level sets
    e = level / 7.0
    y = (rng.random(4000) < 0.2 + 0.6 * e).astype(int)
    auc, degen = estimand.pair_weighted(y, e, level)
    assert auc == pytest.approx(0.5, abs=1e-12) and degen == 0.0
    auc2, degen2 = checks.self_control(y, e, 8)
    assert auc2 == pytest.approx(0.5, abs=1e-12) and degen2 == 0.0


def test_conditional_auc_matches_rank_versions():
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.3).astype(int)
    s = rng.standard_normal(2000) + y
    e = rng.random(2000)
    a, _ = estimand.conditional_auc(y, s, e, 10)
    assert a == pytest.approx(cauc(y, s, e, 10), abs=1e-12)
    assert a == pytest.approx(estimand.strat_auc(y, s, e, 10), abs=1e-12)


def test_conditional_auc_degenerate_share():
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    s = np.arange(100.0)
    auc, degen = estimand.conditional_auc(y, s, s, 4)   # strata perfectly separate classes
    assert np.isnan(auc) and degen == 1.0


def test_fast_auc():
    assert fast_auc(np.array([0, 0, 1, 1]), np.array([1., 2., 3., 4.])) == 1.0
    assert fast_auc(np.array([0, 1]), np.array([1., 1.])) == 0.5
    assert np.isnan(fast_auc(np.zeros(3, int), np.arange(3.0)))


@pytest.mark.parametrize("n,q", [(150, 4), (200, 10), (999, 10), (1000, 20)])
def test_n_strata(n, q):
    assert estimand.n_strata(n) == q


def test_screen():
    assert checks.passes_screen(0.2, 0.45) and checks.passes_screen(0.0, 0.55)
    assert not checks.passes_screen(0.21, 0.5) and not checks.passes_screen(0.1, 0.56)


def test_majority_verdict():
    assert majority_verdict(["linear", "linear", "tree", "abstain"]) == "linear"
    assert majority_verdict(["linear", "tree"]) == "abstain"
    assert majority_verdict(["abstain"] * 3 + ["tree"]) == "abstain"


def test_null_threshold_ignores_nan():
    assert null_threshold([1.0, 2.0, np.nan, 3.0], 50) == 2.0
    assert np.isnan(null_threshold([np.nan]))


def test_gaussian_simulator_is_deterministic():
    a = gaussian.simulate(1.0, 2.0, np.random.default_rng(7))
    b = gaussian.simulate(1.0, 2.0, np.random.default_rng(7))
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    Z, y, s = a
    assert Z.shape == (gaussian.N, 2) and set(np.unique(y)) <= {0, 1}


def test_theory_law_sign_and_boundary():
    for alpha, lam in [(1.0, 0.5), (1.0, 2.0), (2.0, 1.0), (0.5, 1.0)]:
        rho_t, th_t, rho_m, th_m = law.family_I(alpha, lam)
        assert rho_t > 0 and th_t > 0.5
        assert np.sign(th_m - 0.5) == np.sign(rho_m) == np.sign(1 - alpha * lam)
    _, _, rho_m, th_m = law.family_I(1.0, 1.0)            # alpha * lam = 1
    assert abs(rho_m) < 1e-12 and th_m == pytest.approx(0.5, abs=1e-9)


def test_theory_law_reproduces_published_cells():
    S = np.eye(2)
    for al, lm, t, w in ((1, 2, 0.575, 0.415), (2, 2, 0.576, 0.296), (1, 0, 0.670, 0.648)):
        a, b = al * np.array([1., 1.]), lm * np.array([1., -1.])
        assert law.law(S, a, b, 1.0, np.array([1., 1.]))[0] == pytest.approx(t, abs=1.5e-3)
        assert law.law(S, a, b, 1.0, np.array([1., 0.]))[0] == pytest.approx(w, abs=1.5e-3)
        assert law.family_I(al, lm)[3] == pytest.approx(w, abs=1.5e-3)


def test_fit_alias():
    propensity = importlib.import_module("nuisauc.propensity")
    assert propensity.fit is propensity.fit_simple
