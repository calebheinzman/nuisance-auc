"""Label-free self-training on top of a zero-shot score.

Pseudo-label the top and bottom K_PSEUDO percent of the current score, fit a
cross-fitted logistic student on the image embeddings, repeat N_ITER times.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K_PSEUDO, N_ITER = 20, 2
SELFTRAIN_SEED = 31
SELFTRAIN_FOLDS = 5


def pseudo(s: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Top-k% as positives, bottom-k% as negatives, middle discarded."""
    n = len(s)
    m = max(2, int(n * k / 100))
    o = np.argsort(s)
    neg, pos = o[:m], o[-m:]
    idx = np.concatenate([neg, pos])
    lab = np.concatenate([np.zeros(m, int), np.ones(m, int)])
    return idx, lab


def student(E: np.ndarray, idx: np.ndarray, lab: np.ndarray,
            seed: int = SELFTRAIN_SEED) -> np.ndarray:
    """Cross-fitted student: every tile scored by a model that did not see it."""
    out = np.empty(len(E), float)
    folds = list(StratifiedKFold(SELFTRAIN_FOLDS, shuffle=True,
                                 random_state=seed).split(E, np.zeros(len(E))))
    train_mask = np.zeros(len(E), bool)
    train_mask[idx] = True
    labmap = np.full(len(E), -1, int)
    labmap[idx] = lab
    for _f, (tr, te) in enumerate(folds):
        sel = tr[train_mask[tr]]
        if len(np.unique(labmap[sel])) < 2:
            out[te] = 0.0
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(E[sel], labmap[sel])
        out[te] = m.decision_function(E[te])
    return out
