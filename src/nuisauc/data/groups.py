"""Dependence units: slide groups and near-duplicate clusters.

Kaggle tiles carry a slide/image identifier; Wellgen tiles a field identifier
(cached as groups__<corpus>__cap<cap>.npy); the other corpora have none, so
units are connected components of the cosine-similarity graph of DINOv2
ViT-S/14 vectors (deliberately not the model under test).
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from nuisauc.data.corpora import load_any
from nuisauc.paths import cache_dir, raw_dir

MAX_EDGES = 20_000_000
PRIMARY_TAU = 0.95


def kaggle_groups(cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicate load_corpus's keep-set (seed 42) without decoding images."""
    gdir = raw_dir() / "processed" / "Kaggle__local_128"
    meta = pd.read_csv(gdir / "meta.csv")
    leaf = meta["key"].astype(str).str.rsplit("/", n=1).str[-1]
    y = meta["label"].to_numpy().astype(int)
    groups = (leaf.str.replace(r"^-?\d+_", "", regex=True)
                  .str.replace(r"\.jpg_\d+\.jpg$", "", regex=True).to_numpy())
    rng = np.random.default_rng(42)
    keep = np.sort(np.concatenate([
        rng.choice(np.where(y == c)[0], cap, replace=False)
        if (y == c).sum() > cap else np.where(y == c)[0] for c in (0, 1)]))
    return y[keep], groups[keep]


def dinov2_vectors(corpus: str, cap: int, dev: str) -> np.ndarray:
    """One L2-normalised mean-pooled DINOv2 ViT-S/14 vector per tile (cached)."""
    f = cache_dir() / f"dino__{corpus}__cap{cap}.npy"
    if f.exists():
        return np.load(f)
    import torch

    from nuisauc.detectors.vlm import encode
    items, _y = load_any(corpus, cap)
    tok = encode("fm", "dinov2_vits14", items, dev)
    v = tok.mean(1)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    v = v.astype(np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, v)
    del items, tok
    gc.collect()
    torch.cuda.empty_cache()
    return v


def cluster(v: np.ndarray, tau: float, chunk: int = 1024) -> tuple[np.ndarray, bool]:
    """Connected components of the graph joining tiles with cosine >= tau.

    Returns (labels, collapsed). If the edge count exceeds MAX_EDGES the graph
    is reported as collapsed (all labels 0) instead of exhausting memory.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(v)
    rows, cols, total = [], [], 0
    for i in range(0, n, chunk):
        sim = v[i:i + chunk] @ v.T
        r, c = np.where(sim >= tau)
        total += len(r)
        if total > MAX_EDGES:
            return np.zeros(n, dtype=int), True
        rows.append(r + i)
        cols.append(c)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    _k, lab = connected_components(g, directed=False)
    return lab, False


def units_for(corpus: str, y: np.ndarray, cap: int) -> tuple[np.ndarray, str]:
    """Bootstrap units: Kaggle slide/image, Wellgen field, else duplicate cluster."""
    if corpus.startswith("Kaggle"):
        yk, g = kaggle_groups(cap)
        assert np.array_equal(yk, y)
        return g.astype(str), "slide/image"
    if corpus.startswith("Wellgen"):
        g = np.load(cache_dir() / f"groups__{corpus}__cap{cap}.npy",
                    allow_pickle=True)
        assert len(g) == len(y)
        return g.astype(str), "field"
    v = np.load(cache_dir() / f"dino__{corpus}__cap{cap}.npy")
    assert len(v) == len(y)
    lab, collapsed = cluster(v, PRIMARY_TAU)
    assert not collapsed
    return lab.astype(str), f"dup-cluster(tau={PRIMARY_TAU})"


def cluster_units_for(corpus: str, y: np.ndarray, cap: int) -> tuple[np.ndarray, str]:
    """Bootstrap units: Kaggle slide, else duplicate cluster (Wellgen included)."""
    if corpus.startswith("Kaggle"):
        yk, g = kaggle_groups(cap)
        assert np.array_equal(yk, y), "Kaggle keep-set mismatch"
        return g, "slide"
    v = np.load(cache_dir() / f"dino__{corpus}__cap{cap}.npy")
    assert len(v) == len(y), f"{corpus} dino cache length mismatch"
    lab, collapsed = cluster(v, PRIMARY_TAU)
    assert not collapsed
    return lab.astype(str), f"dup-cluster(tau={PRIMARY_TAU})"
