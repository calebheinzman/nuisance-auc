"""Corpus definitions and raw-image loaders.

Raw data live under `nuisauc.paths.raw_dir()` (env `NUISAUC_RAW`) with this
layout (the same relative layout as the original `data/` directory):

    <raw>/processed/<corpus>/meta.csv      one row per tile; columns `key`, `label`
    <raw>/processed/<corpus>/index.npy     (n, 2) byte offset and length per tile
    <raw>/processed/<corpus>/images.bin    concatenated encoded tile images
        for corpus in AFB__internal_81, Kaggle__local_128, Wellgen__local_128
    <raw>/external/pcam/pcam_train0.parquet          columns `image`, `label`
    <raw>/external/malaria/malaria_train.parquet     columns `image`, `label`
    <raw>/external/medmnist/pathmnist_224_train{0,1,2}.parquet
    <raw>/external/medmnist/bloodmnist_128_train.parquet

Labels: Kaggle and AFB store 0/1 in `label`; Wellgen labels come from the
`pos_`/`neg_` prefix of the key's leaf (slide-level, propagated by the
distributors). The malaria parquet stores label 1 for the uninfected class and
is flipped on load. Raw images are only needed to rebuild the cache.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from nuisauc.paths import raw_dir

CORPORA = ["Kaggle__local_128", "pcam", "malaria",
           "AFB__internal_81", "Wellgen__local_128"]
PANEL_A = ["AFB__internal_81", "Kaggle__local_128", "pcam"]
# C3 where the corpus is C3-conditionable; C1 otherwise.
CONFOUND_FOR = {"Kaggle__local_128": "C3", "pcam": "C3", "malaria": "C1",
                "AFB__internal_81": "C1", "Wellgen__local_128": "C1"}
ENCODERS = {
    "dinov2_vits14": ("fm", "dinov2_vits14"),
    "dinov2_vitb14": ("fm", "dinov2_vitb14"),
    "phikon": ("fm", "phikon"),
    "plip": ("fm", "plip"),
    "resnet18_imagenet": ("resnet", None),
}
# parquet files of the binary corpora, relative to raw_dir()
BINARY_CORPORA = {
    "pcam": "external/pcam/pcam_train0.parquet",
    "malaria": "external/malaria/malaria_train.parquet",
}
# Corpora whose stored label 1 is the NEGATIVE class (the malaria mirror was
# converted with ImageFolder: Parasitized -> 0, Uninfected -> 1).
LABEL_FLIP = {"malaria"}


def load_corpus(corpus: str, cap: int = 5000, seed: int = 42):
    """Stream tiles, deriving labels and slide groups per corpus.

    Returns (items, y, groups). `groups` is None when the corpus ships no
    slide identifier, in which case CV cannot be grouped.
    """
    from PIL import Image
    gdir = raw_dir() / "processed" / corpus
    meta = pd.read_csv(gdir / "meta.csv")
    keys = meta["key"].astype(str).to_numpy()
    leaf = pd.Series(keys).str.rsplit("/", n=1).str[-1]

    lab = meta["label"].to_numpy()
    if set(np.unique(lab)) <= {0, 1}:
        y = lab.astype(int)
    elif leaf.str.startswith(("pos_", "neg_")).all():
        # Wellgen: labels are propagated from the slide prefix, not per tile.
        y = leaf.str.startswith("pos_").astype(int).to_numpy()
    else:
        raise ValueError(f"cannot resolve labels for {corpus}")

    if corpus.startswith("Kaggle"):
        groups = (leaf.str.replace(r"^-?\d+_", "", regex=True)
                      .str.replace(r"\.jpg_\d+\.jpg$", "", regex=True).to_numpy())
    elif corpus.startswith("Wellgen"):
        groups = leaf.str.replace(r"\.jpg_\d+\.jpg$", "", regex=True).to_numpy()
    else:
        groups = None                      # AFB ships no slide identifier

    rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([
        rng.choice(np.where(y == c)[0], cap, replace=False)
        if (y == c).sum() > cap else np.where(y == c)[0] for c in (0, 1)]))

    idxf = np.load(gdir / "index.npy", allow_pickle=True)
    items = []
    with open(gdir / "images.bin", "rb") as fh:
        for i in keep:
            off, ln = int(idxf[i, 0]), int(idxf[i, 1])
            fh.seek(off)
            items.append({"image": Image.open(io.BytesIO(fh.read(ln)))
                          .convert("RGB")})
    return items, y[keep], (groups[keep] if groups is not None else None)


def load_binary(corpus: str = "pcam", cap: int = 8000, seed: int = 42):
    """Class-balanced subsample of a binary parquet corpus: (items, y)."""
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(raw_dir() / BINARY_CORPORA[corpus],
                      columns=["image", "label"])
    lab = np.asarray(t.column("label").to_pylist()).reshape(-1).astype(int)
    if corpus in LABEL_FLIP:
        lab = 1 - lab
    rng = np.random.default_rng(seed)
    per = cap // 2
    keep = np.sort(np.concatenate([
        rng.choice(np.where(lab == c)[0], min(per, int((lab == c).sum())),
                   replace=False) for c in (0, 1)]))
    col = t.column("image")
    items = []
    for i in keep:
        v = col[int(i)].as_py()
        b = v["bytes"] if isinstance(v, dict) else v
        items.append({"image": Image.open(io.BytesIO(b)).convert("RGB")})
    return items, lab[keep]


def load_any(name: str, cap: int):
    """(items, y) for any of the five corpora at the given per-class cap."""
    if name in ("pcam", "malaria"):
        return load_binary(name, cap=cap * 2)
    items, y, _g = load_corpus(name, cap=cap)
    return items, y


def draw_labels(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n label indices, extending the draw until both classes appear."""
    perm = rng.permutation(len(y))
    take = n
    while take < len(y) and len(np.unique(y[perm[:take]])) < 2:
        take += 1
    return perm[:take]


def pool_tokens_chunked(spatial: np.ndarray, mode: str = "meanmax",
                        chunk: int = 512) -> np.ndarray:
    """Per-token L2 normalisation then per-image mean/max pooling, in chunks.

    Normalisation is per token and pooling is per image, so chunking over
    images is exact.
    """
    n, s, c = spatial.shape
    out = []
    for i in range(0, n, chunk):
        blk = spatial[i:i + chunk].astype(np.float32, copy=False)
        b = blk.shape[0]
        x = blk.reshape(-1, c)
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x = (x / nrm).reshape(b, s, c)
        if mode == "mean":
            out.append(x.mean(1))
        elif mode == "max":
            out.append(x.max(1))
        else:
            out.append(np.concatenate([x.mean(1), x.max(1)], axis=1))
        del x, blk
    return np.concatenate(out)
