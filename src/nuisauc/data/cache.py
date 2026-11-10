"""Cached per-corpus inputs: labels, nuisance descriptors, nuisance-model sets.

Files live in `nuisauc.paths.cache_dir()` (env `NUISAUC_CACHE`), resolved at
call time:

    nuis__<corpus>__cap<cap>.npz     y and the corpus's confound descriptor C
                                     (62 columns where CONFOUND_FOR is C3,
                                     trivial-16 otherwise)
    nuisC3__<corpus>__cap<cap>.npy   62-feature descriptor (C1 corpora)
    nuisT16__<corpus>__cap<cap>.npy  trivial-16 (C3 corpora)
    freq__/morph__/dino__/img__/txt__/txt_r21__/groups__ ...  see the modules
                                     that read them
"""
from __future__ import annotations

import gc

import numpy as np

from nuisauc.data.corpora import CONFOUND_FOR, CORPORA, load_any
from nuisauc.data.descriptors import extended_nuisance, trivial16
from nuisauc.paths import cache_dir

__all__ = ["CORPORA", "corpus_cache", "descriptor62", "nuisance_models"]


def corpus_cache(corpus: str, cap: int) -> tuple[np.ndarray, np.ndarray, str]:
    """(y, C, pkind) with the corpus's confound descriptor, cached on first use."""
    f = cache_dir() / f"nuis__{corpus}__cap{cap}.npz"
    pkind = "gbm" if CONFOUND_FOR[corpus] == "C3" else "linear"
    if f.exists():
        z = np.load(f)
        return z["y"], z["C"], pkind
    items, y = load_any(corpus, cap)
    C = (extended_nuisance(items) if CONFOUND_FOR[corpus] == "C3"
         else trivial16(items))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, y=y, C=C)
    del items
    gc.collect()
    return y, C, pkind


def descriptor62(corpus: str, cap: int = 4000) -> np.ndarray:
    """The 62-feature colour/focus/texture/edge descriptor, from cache."""
    f = cache_dir() / f"nuisC3__{corpus}__cap{cap}.npy"
    if f.exists():
        return np.load(f)
    z = np.load(cache_dir() / f"nuis__{corpus}__cap{cap}.npz")
    C = z["C"]
    if C.shape[1] < 40:
        raise RuntimeError(f"{corpus}: cached descriptor has {C.shape[1]} "
                           "columns; the 62-feature one is required")
    return C


def nuisance_models(corpus: str, cap: int) -> list[tuple[str, np.ndarray, str]]:
    """C1 trivial-16 linear, C2 trivial-16 boosted, C3 62-feature boosted."""
    f16 = cache_dir() / f"nuisT16__{corpus}__cap{cap}.npy"
    T16 = (np.load(f16) if f16.exists()
           else np.load(cache_dir() / f"nuis__{corpus}__cap{cap}.npz")["C"])
    if T16.shape[1] != 16:
        T16 = T16[:, :16]
    D62 = descriptor62(corpus, cap)
    return [("C1", T16, "linear"), ("C2", T16, "gbm"), ("C3", D62, "gbm")]
