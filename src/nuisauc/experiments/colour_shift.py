"""A known-answer test on real images: an injected colour shortcut.

Shift      per-pixel channel gain on real tiles: R *= 1+delta, B *= 1-delta,
           delta in {0.10, 0.25}.
Assignment A_i ~ Bernoulli(p_{Y_i}); conditions ctrl (p1 = p0 = 0.5),
           with (p1 = 0.8, p0 = 0.2) and against (p1 = 0.2, p0 = 0.8). All
           three shift half the corpus in expectation; only the label
           dependence of A differs.
Detectors  zero-shot BiomedCLIP prompt ensemble on the observed images, and a
           supervised logistic probe on BiomedCLIP embeddings trained on the
           training fold of the observed corpus.
Models     for each base learner F in {C1 logistic, C2 trees}: F-blind on the
           16 features of the original images (omits A), F-obs on the 16
           features of the observed images (sees A through colour statistics).
Known answer
           oracle_F = pair-weighted within-stratum AUC with strata = the q
           quantile bins of F-blind's propensity crossed with A.
Selection  on the observed 16 features, cross-fitted log-loss picks a class.

The shifted images are re-embedded on the GPU (or CPU) at run time.
"""
from __future__ import annotations

import gc
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nuisauc.data.corpora import load_any
from nuisauc.data.descriptors import trivial16
from nuisauc.detectors.vlm import VLM, embed_cache, prompt_embeddings, score
from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.inference.selection import crossfit_scores, select_family
from nuisauc.io import write_table
from nuisauc.propensity import fit_any, propensity
from nuisauc.seeds import record_run

NAME = "colour_shift"
DESCRIPTION = "Injected colour shortcut on real images with a known answer"
PAPER = "§5, App F"
REQUIRES = "cache"
OUTPUTS = ["injected_shortcut"]

SEED = 61007
DELTAS = [0.10, 0.25]
CONDS = {"ctrl": (0.5, 0.5), "with": (0.8, 0.2), "against": (0.2, 0.8)}
N_SPLIT = 5
CORPORA = ["AFB__internal_81", "Kaggle__local_128", "pcam", "malaria",
           "Wellgen__local_128"]

log = logging.getLogger(f"nuisauc.{NAME}")


def shifted(items: list, delta: float) -> list:
    """Copies of the items with the R/B channel-gain shift applied."""
    out = []
    g = np.array([1 + delta, 1.0, 1 - delta])
    for it in items:
        a = np.asarray(it["image"].convert("RGB") if hasattr(it["image"], "convert")
                       else it["image"], dtype=np.float64)
        b = np.clip(a * g, 0, 255).astype(np.uint8)
        out.append({**it, "image": Image.fromarray(b)})
    return out


def oracle(y: np.ndarray, s: np.ndarray, e: np.ndarray, A: np.ndarray,
           q: int) -> float:
    """Pair-weighted AUC within strata = q quantile bins of e crossed with A."""
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.digitize(e, edges[1:-1]), 0, q - 1) * 2 + A
    num = den = 0.0
    for k in np.unique(b):
        m = b == k
        yk = y[m]
        n1, n0 = int(yk.sum()), int((1 - yk).sum())
        if n1 == 0 or n0 == 0:
            continue
        num += n1 * n0 * roc_auc_score(yk, s[m])
        den += n1 * n0
    return num / den


def per_corpus(corpus: str, cap: int, dev: str, rows: list, out_dir: Path) -> None:
    import torch
    items, y = load_any(corpus, cap)
    y = np.asarray(y).astype(int)
    T_orig = trivial16(items)
    E_orig, _ = embed_cache("biomedclip", corpus, cap, dev)
    assert len(E_orig) == len(y)
    T = prompt_embeddings("biomedclip", corpus, dev)
    Tp, Tn = T[f"{corpus}__pos"], T[f"{corpus}__neg"]

    def zs(E):
        return score(E, Tp, Tn, np.arange(len(Tp)), np.arange(len(Tn)))

    s_orig = zs(E_orig)
    splits = [next(iter(StratifiedKFold(4, shuffle=True, random_state=r)
                        .split(np.zeros(len(y)), y))) for r in range(N_SPLIT)]
    vlm = VLM("biomedclip", dev)
    chk = vlm.images(items[:64])
    assert np.abs(chk - E_orig[:64]).max() < 2e-2, "embedding cache misaligned"
    for delta in DELTAS:
        sh = shifted(items, delta)
        T_sh = trivial16(sh)
        E_sh = vlm.images(sh).astype(np.float32)
        del sh
        gc.collect()
        s_sh = zs(E_sh)
        d_load = float((s_sh - s_orig).mean() / s_orig.std())
        log.info("=== %s delta=%s: detector loading on shift d=%+.3f "
                 "(score SD units) ===", corpus, delta, d_load)
        for cname, (p1, p0) in CONDS.items():
            for rep, (tr, te) in enumerate(splits):
                # fresh shortcut assignment per fold (same draw for all models)
                rng = np.random.default_rng([SEED, zlib.crc32(corpus.encode()) % 2 ** 16,
                                             int(delta * 100),
                                             list(CONDS).index(cname), rep])
                A = (rng.random(len(y)) < np.where(y == 1, p1, p0)).astype(int)
                T_obs = np.where(A[:, None] == 1, T_sh, T_orig)
                E_obs = np.where(A[:, None] == 1, E_sh, E_orig)
                q = n_strata(len(te))
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=3000))
                m.fit(E_obs[tr], y[tr])
                sup = np.full(len(y), np.nan)
                sup[te] = m.decision_function(E_obs[te])
                so = m.decision_function(E_orig[te])
                d_sup = float((m.decision_function(E_sh[te]) - so).mean() / so.std())
                dets_f = {"zero_shot": np.where(A == 1, s_sh, s_orig),
                          "supervised": sup}
                dloads = {"zero_shot": d_load, "supervised": d_sup}
                e = {}
                for F, kind in (("C1", "linear"), ("C2", "gbm")):
                    e[f"{F}_blind"] = propensity(kind, T_orig, tr, y, rep).predict_proba(T_orig[te])[:, 1]
                    e[f"{F}_obs"] = propensity(kind, T_obs, tr, y, rep).predict_proba(T_obs[te])[:, 1]
                sc = crossfit_scores(T_obs, y, tr, rep)
                verdict, member, fam, z = select_family(sc, 0)
                for f_, (tag, _ll) in fam.items():
                    e[f"sel_{f_}"] = fit_any(tag, T_obs, y, tr, rep).predict_proba(T_obs[te])[:, 1]
                for dname, s_all in dets_f.items():
                    s = s_all[te]
                    row = dict(corpus=corpus, delta=delta, cond=cname,
                               detector=dname, rep=rep, d_load=dloads[dname],
                               raw_auc=float(roc_auc_score(y[te], s)),
                               A_auc=float(roc_auc_score(y[te], A[te])) if cname != "ctrl" else 0.5,
                               verdict=verdict, z_logloss=z,
                               member_lin=fam["linear"][0], member_tree=fam["tree"][0])
                    for F in ("C1", "C2"):
                        row[f"oracle_{F}"] = float(oracle(y[te], s, e[f"{F}_blind"], A[te], q))
                        for v in ("blind", "obs"):
                            c, dg = conditional_auc(y[te], s, e[f"{F}_{v}"], q)
                            ctl = conditional_auc(y[te], e[f"{F}_{v}"], e[f"{F}_{v}"], q)[0]
                            row[f"{F}_{v}"] = float(c)
                            row[f"{F}_{v}_degen"] = float(dg)
                            row[f"{F}_{v}_control"] = float(ctl)
                    for f_ in ("linear", "tree"):
                        row[f"sel_{f_}"] = float(conditional_auc(y[te], s, e[f"sel_{f_}"], q)[0])
                    rows.append(row)
            g = pd.DataFrame(rows)
            g = g[(g.corpus == corpus) & (g.delta == delta) & (g.cond == cname)]
            for dname, gg in g.groupby("detector"):
                log.info("  %-8s %-11s raw %.3f | C1 oracle %.3f blind %.3f obs %.3f | "
                         "C2 oracle %.3f blind %.3f obs %.3f | verdict %s",
                         cname, dname, gg.raw_auc.mean(), gg.oracle_C1.mean(),
                         gg.C1_blind.mean(), gg.C1_obs.mean(), gg.oracle_C2.mean(),
                         gg.C2_blind.mean(), gg.C2_obs.mean(),
                         gg.verdict.mode().iloc[0])
            write_table(pd.DataFrame(rows), out_dir, "injected_shortcut")
        del E_sh, T_sh
        gc.collect()
    vlm.close()
    del vlm, items
    gc.collect()
    torch.cuda.empty_cache()


def default_config() -> dict:
    return {"cap": 4000, "corpora": list(CORPORA)}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for corpus in cfg["corpora"]:
        per_corpus(corpus, cfg["cap"], dev, rows, out_dir)
    d = pd.DataFrame(rows)
    write_table(d, out_dir, "injected_shortcut")
    record_run(out_dir, NAME, cfg)
    return {"injected_shortcut": d}
