"""Filesystem locations, each overridable by an environment variable.

NUISAUC_CACHE    per-tile inputs derived from images (descriptors, embeddings, labels)
NUISAUC_RAW      raw image collections (only needed to rebuild the cache)
NUISAUC_MODELS   local Hugging Face model cache for the image encoders
NUISAUC_RESULTS  where `nuisauc run` writes result tables
NUISAUC_REFERENCE  reference result tables shipped with the package
"""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def cache_dir() -> Path:
    return _env("NUISAUC_CACHE", "data/cache")


def raw_dir() -> Path:
    return _env("NUISAUC_RAW", "data/raw")


def models_dir() -> Path:
    return _env("NUISAUC_MODELS", "data/models")


def results_dir() -> Path:
    return _env("NUISAUC_RESULTS", "runs")


def reference_dir() -> Path:
    return _env("NUISAUC_REFERENCE", "results")
