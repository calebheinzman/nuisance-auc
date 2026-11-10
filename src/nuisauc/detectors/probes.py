"""Supervised probes and the one-class memory-bank detector.

`probe` / `cv_probe_auc`: standardise, PCA to at most 64 components, logistic
regression. `GPUBank` / `agg_top5`: PatchCore-style k-nearest-neighbour
distance of each spatial token to a bank of normal tokens, aggregated per
image as the mean of the top 5% of token scores.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize

PCA_DIM = 192
K_NEIGHBORS = 75
BANK_SIZE = 1500
MAX_FIT_PATCHES = 80_000


def probe(n_train: int):
    """Unfitted scaler + PCA(min(64, n_train - 1)) + logistic regression."""
    return make_pipeline(StandardScaler(),
                         PCA(n_components=min(64, max(1, n_train - 1)),
                             random_state=0),
                         LogisticRegression(max_iter=5000))


def cv_probe_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Cross-validated probe AUC on the labelled set (0.5 when not estimable)."""
    if len(np.unique(y)) < 2 or len(y) < 6:
        return 0.5
    k = min(3, int(min(np.bincount(y))))
    if k < 2:
        return 0.5
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        if len(np.unique(y[tr])) < 2:
            return 0.5
        m = probe(len(tr)).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, oof))


class GPUBank:
    """k-NN memory bank of normal spatial tokens, distances on the GPU.

    Fit: L2-normalise tokens, subsample at most MAX_FIT_PATCHES, PCA to
    PCA_DIM, keep BANK_SIZE rows. `fit_idx` selects the normal images without
    materialising a copy (exact: normalisation is per row and the same
    generator draws select the same rows).
    """

    def __init__(self, spatial_normals: np.ndarray, device: str, seed: int = 42,
                 fit_idx: np.ndarray | None = None):
        import torch
        rng = np.random.default_rng(seed)
        if fit_idx is not None:
            n_all, s, c = spatial_normals.shape
            total = len(fit_idx) * s
            flat = (rng.choice(total, MAX_FIT_PATCHES, replace=False)
                    if total > MAX_FIT_PATCHES else np.arange(total))
            rows = spatial_normals[np.asarray(fit_idx)[flat // s], flat % s, :]
            X = sk_normalize(rows.astype(np.float32), norm="l2")
        else:
            n, s, c = spatial_normals.shape
            X = sk_normalize(spatial_normals.reshape(-1, c).astype(np.float32),
                             norm="l2")
            if len(X) > MAX_FIT_PATCHES:
                X = X[rng.choice(len(X), MAX_FIT_PATCHES, replace=False)]
        self.pca = PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0]),
                       random_state=seed)
        Xp = self.pca.fit_transform(X)
        bank_idx = rng.choice(len(Xp), min(BANK_SIZE, len(Xp)), replace=False)
        self.B = torch.as_tensor(np.ascontiguousarray(Xp[bank_idx]),
                                 dtype=torch.float32, device=device)
        self.k = min(K_NEIGHBORS, len(bank_idx))
        self.device = device

    def score(self, spatial: np.ndarray, chunk: int = 256) -> np.ndarray:
        """(n, s) mean distance of each token to its k nearest bank rows."""
        import torch
        n, s, c = spatial.shape
        out = np.zeros((n, s), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, chunk):
                part = spatial[i:i + chunk]
                X = sk_normalize(part.reshape(-1, c).astype(np.float32),
                                 norm="l2")
                Q = torch.as_tensor(np.ascontiguousarray(self.pca.transform(X)),
                                    dtype=torch.float32, device=self.device)
                d = torch.cdist(Q, self.B).topk(self.k, largest=False).values
                out[i:i + chunk] = d.mean(1).reshape(len(part), -1).cpu().numpy()
                del Q, d
        return out


def agg_top5(ps: np.ndarray) -> np.ndarray:
    """Per-image mean of the top 5% (at least one) token scores."""
    k = max(1, int(ps.shape[1] * 0.05))
    return np.sort(ps, axis=1)[:, -k:].mean(axis=1)
