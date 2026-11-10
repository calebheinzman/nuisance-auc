"""MedMNIST pathology/microscopy corpora as one-vs-rest benchmark instances.

Each multi-class corpus yields one instance per class: the target class is the
anomaly, every other class is normal, at the natural one-vs-rest prevalence.
Neither mirror ships patient or slide identifiers, so CV cannot be grouped.
Parquet files are read from `<raw>/external/medmnist/`.
"""
from __future__ import annotations

import io

import numpy as np

from nuisauc.paths import raw_dir

MEDMNIST_SUBDIR = "external/medmnist"
MEDMNIST_CORPORA = {
    "PathMNIST": dict(
        files=[f"pathmnist_224_train{i}.parquet" for i in range(3)],
        n_classes=9,
        names=["adipose", "background", "debris", "lymphocytes", "mucus",
               "smooth muscle", "normal colon mucosa", "cancer stroma",
               "colorectal adenocarcinoma"]),
    "BloodMNIST": dict(
        files=["bloodmnist_128_train.parquet"],
        n_classes=8,
        names=["basophil", "eosinophil", "erythroblast",
               "immature granulocytes", "lymphocyte", "monocyte",
               "neutrophil", "platelet"]),
}


def load_pool(corpus: str, cap: int = 8000, seed: int = 42):
    """Return (items, labels) for a corpus, class-stratified to at most `cap`."""
    import pyarrow.parquet as pq
    from PIL import Image
    spec = MEDMNIST_CORPORA[corpus]
    root = raw_dir() / MEDMNIST_SUBDIR
    labs, blobs = [], []
    for f in spec["files"]:
        t = pq.read_table(root / f, columns=["label", "image"])
        # PathMNIST stores each label as a one-element list, BloodMNIST
        # as a scalar; flatten so both yield a 1-D label vector.
        labs.append(np.asarray(t.column("label").to_pylist(),
                               dtype=np.int64).reshape(-1))
        for v in t.column("image").to_pylist():
            blobs.append(v["bytes"] if isinstance(v, dict) else v)
    lab = np.concatenate(labs)
    assert lab.ndim == 1, lab.shape
    assert len(lab) == len(blobs), (len(lab), len(blobs))

    rng = np.random.default_rng(seed)
    per = max(1, cap // spec["n_classes"])
    keep = np.sort(np.concatenate([
        rng.choice(np.where(lab == c)[0], min(per, int((lab == c).sum())),
                   replace=False)
        for c in range(spec["n_classes"])]))
    items = [{"image": Image.open(io.BytesIO(blobs[i])).convert("RGB")}
             for i in keep]
    return items, lab[keep]


def instances(corpus: str, lab: np.ndarray, min_pos: int = 120) -> list[tuple[str, np.ndarray]]:
    """One-vs-rest instances: (name, binary y) with the target class as anomaly."""
    spec = MEDMNIST_CORPORA[corpus]
    out = []
    for c in range(spec["n_classes"]):
        y = (lab == c).astype(int)
        if int(y.sum()) < min_pos:
            continue
        out.append((f"{corpus}:{c}:{spec['names'][c][:18]}", y))
    return out
