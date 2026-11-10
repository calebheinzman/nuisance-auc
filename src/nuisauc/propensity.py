"""Propensity (nuisance) model fitters.

Registered pair: C1 = standardised logistic regression, C2 = histogram
gradient boosting. Alternates: C1e = elastic-net logistic regression,
C2r = random forest. CLASSES groups them into the linear and tree classes.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = {"linear": ("C1", "C1e"), "tree": ("C2", "C2r")}


def propensity(kind: str, C: np.ndarray, tr: np.ndarray, y: np.ndarray,
               rep: int):
    """Fit the C1 ("linear") or C2 (anything else: boosted) propensity on rows tr."""
    if kind == "linear":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    else:
        m = HistGradientBoostingClassifier(max_iter=200, random_state=rep)
    m.fit(C[tr], y[tr])
    return m


def fit_learner(tag: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray,
                seed: int):
    """Fit an alternate learner: C1e (elastic-net logistic) or C2r (random forest)."""
    if tag in ("C1", "C2"):
        raise ValueError("use propensity() for the registered pair")
    if tag == "C1e":
        m = make_pipeline(StandardScaler(), LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            max_iter=5000, random_state=seed))
    elif tag == "C2r":
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                   random_state=seed, n_jobs=-1)
    else:
        raise ValueError(tag)
    m.fit(X[tr], y[tr])
    return m


def fit_any(tag: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray,
            seed: int):
    """Fit any of C1, C2, C1e, C2r on rows tr."""
    if tag == "C1":
        return propensity("linear", X, tr, y, seed)
    if tag == "C2":
        return propensity("gbm", X, tr, y, seed)
    return fit_learner(tag, X, y, tr, seed)


def fit_simple(kind: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray,
               seed: int):
    """Boosted ("gbm") or logistic (max_iter=2000) propensity, used in simulations."""
    if kind == "gbm":
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        m.fit(X[tr], y[tr])
        return m
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(X[tr], y[tr])
    return m


fit = fit_simple
