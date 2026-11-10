"""The same-feature panel on all five corpora with uncertainty and diagnostics.

Per corpus and per fold (the five repeated 4-fold splits of the screen): raw
AUC, the conditional AUC under C1 (logistic) and C2 (boosted) on the same 16
features and under the best member of each class picked by cross-fitted
log-loss, within-stratum standardized mean differences, the cross-model check
(each model's propensity scored inside the other's strata), calibration (ECE,
Hosmer-Lemeshow, slope), for eight detectors: zero-shot BiomedCLIP, PLIP and
QuiltNet, transductive and inductive self-trained students, and supervised
logistic probes on BiomedCLIP and DINOv2 embeddings.

The paired cluster bootstrap holds the fitted nuisance models fixed and
resamples test-side dependence units (one global unit draw shared by all
folds, both models and all detectors, so differences are paired). The
unit-isolated rerun refits on StratifiedGroupKFold outer folds.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nuisauc.checks import N_SPLIT
from nuisauc.data.cache import CORPORA, corpus_cache, nuisance_models
from nuisauc.data.groups import units_for
from nuisauc.detectors.selftrain import K_PSEUDO, N_ITER, pseudo, student
from nuisauc.detectors.vlm import embed_cache, prompt_embeddings, score
from nuisauc.diagnostics.balance import balance
from nuisauc.diagnostics.calibration import calibration
from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.inference.bootstrap import cauc, fast_auc
from nuisauc.inference.selection import crossfit_scores, select_family
from nuisauc.io import write_table
from nuisauc.paths import cache_dir
from nuisauc.propensity import fit_any, propensity
from nuisauc.seeds import record_run

NAME = "realdata_panel"
DESCRIPTION = "Same-feature C1/C2 panel on five corpora: folds, cluster bootstrap, grouped folds"
PAPER = "§5, Tab 4"
REQUIRES = "cache"
OUTPUTS = ["panel_folds", "panel_boot", "panel_grouped"]

N_BOOT = 1000
BOOT_SEED = 20261006
# (C1, C2) means of the zero-shot detector that must be reproduced (Table 4)
TABLE4 = {
    ("AFB__internal_81", "zero_shot"): (0.8937, 0.8384),
    ("Kaggle__local_128", "zero_shot"): (0.7533, 0.7053),
    ("pcam", "zero_shot"): (0.7210, 0.6768),
}

log = logging.getLogger(f"nuisauc.{NAME}")


def detectors(corpus: str, cap: int, dev: str, y: np.ndarray,
              splits: list) -> dict:
    """{name: per-fold test-score arrays (full length, nan off-test) or one
    full-corpus score}."""
    E, _ = embed_cache("biomedclip", corpus, cap, dev)
    T = prompt_embeddings("biomedclip", corpus, dev)
    Tp, Tn = T[f"{corpus}__pos"], T[f"{corpus}__neg"]
    s_zero = score(E, Tp, Tn, np.arange(len(Tp)), np.arange(len(Tn)))
    s_self = s_zero
    for _ in range(N_ITER):
        idx, plab = pseudo(s_self, K_PSEUDO)
        s_self = student(E, idx, plab)
    out = {"zero_shot": s_zero, "self_trained": s_self}
    for vlm in ("plip", "quiltnet"):
        Ev, _ = embed_cache(vlm, corpus, cap, dev)
        Tv = prompt_embeddings(vlm, corpus, dev)
        out[f"zs_{vlm}"] = score(Ev, Tv[f"{corpus}__pos"], Tv[f"{corpus}__neg"],
                                 np.arange(4), np.arange(4))
    D = np.load(cache_dir() / f"dino__{corpus}__cap{cap}.npy")
    ind, supb, supd = [], [], []
    for rep, (tr, te) in enumerate(splits):
        # inductive self-training: pseudo-labels and student on train only
        s_tr = s_zero[tr]
        for _ in range(N_ITER):
            idx, plab = pseudo(s_tr, K_PSEUDO)
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=3000))
            m.fit(E[tr][idx], plab)
            s_tr = m.decision_function(E[tr])
        f = np.full(len(y), np.nan)
        f[te] = m.decision_function(E[te])
        ind.append(f)
        for X, acc in ((E, supb), (D, supd)):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=3000))
            m.fit(X[tr], y[tr])
            f = np.full(len(y), np.nan)
            f[te] = m.decision_function(X[te])
            acc.append(f)
    out["self_trained_inductive"] = ind
    out["supervised_biomedclip"] = supb
    out["supervised_dinov2"] = supd
    return out


def fold_score(s, rep: int) -> np.ndarray:
    return s[rep] if isinstance(s, list) else s


def run_corpus(corpus: str, cap: int, dev: str, rows: list, boots: list,
               grows: list, n_boot: int = N_BOOT) -> None:
    import torch
    y, _C, _pk = corpus_cache(corpus, cap)
    T16 = [m for m in nuisance_models(corpus, cap) if m[0] == "C1"][0][1]
    splits = [next(iter(StratifiedKFold(4, shuffle=True, random_state=r)
                        .split(np.zeros(len(y)), y))) for r in range(N_SPLIT)]
    units, ukind = units_for(corpus, y, cap)
    dets = detectors(corpus, cap, dev, y, splits)
    log.info("=== %s n=%d unit=%s (%d units) ===", corpus, len(y), ukind,
             len(np.unique(units)))

    folds = []
    for rep, (tr, te) in enumerate(splits):
        q = n_strata(len(te))
        e = {}
        for tag, kind in (("C1", "linear"), ("C2", "gbm")):
            e[tag] = propensity(kind, T16, tr, y, rep).predict_proba(T16[te])[:, 1]
        sc = crossfit_scores(T16, y, tr, rep)
        verdict, member, fam, z = select_family(sc, 0)
        for f_, (tag, _ll) in fam.items():
            e[f"sel_{f_}"] = (e[tag] if tag in e else
                              fit_any(tag, T16, y, tr, rep).predict_proba(T16[te])[:, 1])
        diag = {}
        for tag in ("C1", "C2"):
            smd = balance(T16[te], y[te], e[tag], q)
            other = "C2" if tag == "C1" else "C1"
            ece, hl, hlp, slope = calibration(y[te], e[tag], q)
            diag[tag] = dict(
                smd_max=float(np.abs(smd).max()),
                smd_n_over_0p1=int((np.abs(smd) > 0.1).sum()),
                cross=float(cauc(y[te], e[other], e[tag], q)),
                ece=ece, hl=hl, hl_p=hlp, cal_slope=slope,
                prop_auc=float(roc_auc_score(y[te], e[tag])),
                control=float(cauc(y[te], e[tag], e[tag], q)),
                degen=float(conditional_auc(y[te], e[tag], e[tag], q)[1]))
        folds.append(dict(tr=tr, te=te, q=q, e=e))
        for dname, s_all in dets.items():
            s = fold_score(s_all, rep)[te]
            row = dict(corpus=corpus, detector=dname, rep=rep, unit=ukind,
                       raw_auc=float(roc_auc_score(y[te], s)),
                       verdict=verdict, member_ll=member, z_logloss=z,
                       member_lin=fam["linear"][0], member_tree=fam["tree"][0])
            for tag in ("C1", "C2", "sel_linear", "sel_tree"):
                row[tag] = float(conditional_auc(y[te], s, e[tag], q)[0])
            for tag in ("C1", "C2"):
                for k, v in diag[tag].items():
                    row[f"{tag}_{k}"] = v
            rows.append(row)
        log.info("  fold %d: verdict %s z=%+.1f C1 smd_max %.3f cross %.3f | "
                 "C2 smd_max %.3f cross %.3f", rep, verdict, z,
                 diag["C1"]["smd_max"], diag["C1"]["cross"],
                 diag["C2"]["smd_max"], diag["C2"]["cross"])

    # paired cluster bootstrap (fixed fitted models, global unit draw)
    uniq, inv = np.unique(units, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)),
                           minlength=len(uniq))
        w = mult[inv]
        res = {}
        for dname, s_all in dets.items():
            acc = {t: [] for t in ("C1", "C2", "sel_linear", "sel_tree", "raw")}
            for rep, fd in enumerate(folds):
                te = fd["te"]
                ii = np.repeat(np.arange(len(te)), w[te])
                yy = y[te][ii]
                ss = fold_score(s_all, rep)[te][ii]
                for t in ("C1", "C2", "sel_linear", "sel_tree"):
                    acc[t].append(cauc(yy, ss, fd["e"][t][ii], fd["q"]))
                acc["raw"].append(fast_auc(yy, ss))
            for t, v in acc.items():
                res[t] = float(np.mean(v))
            boots.append(dict(corpus=corpus, detector=dname, b=b, **res))
        if b % 200 == 0:
            log.info("  boot %d", b)

    # unit-isolated outer folds (refit), claim arms only
    gsplits = [next(iter(StratifiedGroupKFold(4, shuffle=True,
                                              random_state=100 + r)
                         .split(np.zeros(len(y)), y, groups=units)))
               for r in range(N_SPLIT)]
    for rep, (tr, te) in enumerate(gsplits):
        q = n_strata(len(te))
        for tag, kind in (("C1", "linear"), ("C2", "gbm")):
            ee = propensity(kind, T16, tr, y, rep).predict_proba(T16[te])[:, 1]
            for dname in ("zero_shot", "self_trained"):
                c, dg = conditional_auc(y[te], dets[dname][te], ee, q)
                ctl = conditional_auc(y[te], ee, ee, q)[0]
                grows.append(dict(corpus=corpus, detector=dname, rep=rep,
                                  model=tag, cauc=float(c), degen=float(dg),
                                  control=float(ctl)))
    del dets
    gc.collect()
    torch.cuda.empty_cache()


def default_config() -> dict:
    return {"corpora": list(CORPORA), "cap": 4000, "tag": "", "n_boot": N_BOOT}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Tables are named panel_folds<tag>, panel_boot<tag>, panel_grouped<tag>."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = cfg["tag"]
    rows, boots, grows = [], [], []
    for corpus in cfg["corpora"]:
        run_corpus(corpus, cfg["cap"], dev, rows, boots, grows, cfg["n_boot"])
        write_table(pd.DataFrame(rows), out_dir, f"panel_folds{tag}")
        write_table(pd.DataFrame(boots), out_dir, f"panel_boot{tag}")
        write_table(pd.DataFrame(grows), out_dir, f"panel_grouped{tag}")
    d = pd.DataFrame(rows)
    for (c, det), (c1, c2) in TABLE4.items():
        g = d[(d.corpus == c) & (d.detector == det)]
        if len(g):
            assert abs(g.C1.mean() - c1) < 5e-4 and abs(g.C2.mean() - c2) < 5e-4, \
                (c, det, g.C1.mean(), g.C2.mean())
    log.info("reproduces the Table 4 values: ok")
    record_run(out_dir, NAME, cfg)
    return {f"panel_folds{tag}": d, f"panel_boot{tag}": pd.DataFrame(boots),
            f"panel_grouped{tag}": pd.DataFrame(grows)}
