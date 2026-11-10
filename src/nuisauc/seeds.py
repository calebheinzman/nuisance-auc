"""Deterministic seeding and run provenance.

Only the `sha256-v1` scheme is supported: a corpus's seed is the first four
bytes of the SHA-256 of its name. (An earlier scheme used Python's
process-dependent `hash()` and cannot be reproduced.)
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEME = "sha256-v1"


def corpus_seed(corpus: str) -> int:
    return int.from_bytes(hashlib.sha256(corpus.encode("utf-8")).digest()[:4], "big")


def array_hash(a) -> dict:
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def record_run(out_dir: Path, name: str, config: dict, inputs: dict | None = None) -> Path:
    """Write `<name>.provenance.json` next to the experiment's tables."""
    import importlib.metadata as md
    versions = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-image", "torch"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    rec = {"experiment": name, "seed_scheme": SCHEME, "config": config,
           "inputs": inputs or {}, "python": sys.version.split()[0],
           "platform": platform.platform(), "versions": versions,
           "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p = Path(out_dir) / f"{name}.provenance.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return p
