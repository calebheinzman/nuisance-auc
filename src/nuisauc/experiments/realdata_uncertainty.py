"""Paired per-fold uncertainty of the same-feature C1 - C2 disagreement.

C1 and C2 are evaluated on identical folds, so the disagreement is paired per
fold. For the Panel-A corpora and the zero-shot and self-trained BiomedCLIP
detectors, this records the per-fold conditional AUCs under C1 and C2 and
their difference.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import N_SPLIT
from nuisauc.data.cache import corpus_cache, nuisance_models
from nuisauc.data.corpora import PANEL_A
from nuisauc.detectors.selftrain import K_PSEUDO, N_ITER, pseudo, student
from nuisauc.detectors.vlm import GATED_VLM, embed_cache, prompt_embeddings, score
from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.io import write_table
from nuisauc.propensity import propensity
from nuisauc.seeds import record_run

NAME = "realdata_uncertainty"
DESCRIPTION = "Per-fold paired C1 - C2 differences on the corpora where both models pass"
PAPER = "§5, App F"
REQUIRES = "cache"
OUTPUTS = ["panel_a_uncertainty"]

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"cap": 4000}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cfg["cap"]

    rows = []
    for corpus in PANEL_A:
        y, _C, _pk = corpus_cache(corpus, cap)
        splits = [next(iter(StratifiedKFold(
            4, shuffle=True, random_state=r).split(np.zeros(len(y)), y)))
            for r in range(N_SPLIT)]
        models = {t: (C, k) for t, C, k in nuisance_models(corpus, cap)}
        E, _ = embed_cache(GATED_VLM, corpus, cap, dev)
        T = prompt_embeddings(GATED_VLM, corpus, dev)
        Tp, Tn = T[f"{corpus}__pos"], T[f"{corpus}__neg"]
        s_zero = score(E, Tp, Tn, np.arange(len(Tp)), np.arange(len(Tn)))
        s_self = s_zero
        for _ in range(N_ITER):
            idx, plab = pseudo(s_self, K_PSEUDO)
            s_self = student(E, idx, plab)

        for sname, s in (("zero_shot", s_zero), ("self_trained", s_self)):
            for rep, (tr, te) in enumerate(splits):
                q = n_strata(len(te))
                vals = {}
                for tag in ("C1", "C2"):
                    C, kind = models[tag]
                    m = propensity(kind, C, tr, y, rep)
                    e = m.predict_proba(C[te])[:, 1]
                    c, _ = conditional_auc(y[te], s[te], e, q)
                    vals[tag] = float(c)
                rows.append(dict(corpus=corpus, score=sname, rep=rep,
                                 C1=vals["C1"], C2=vals["C2"],
                                 diff=vals["C1"] - vals["C2"]))
            g = [r for r in rows
                 if r["corpus"] == corpus and r["score"] == sname]
            d = np.array([r["diff"] for r in g])
            log.info("  %-22s %-13s mean %+.4f SD %.4f  signs %d/%d positive",
                     corpus, sname, d.mean(), d.std(ddof=1),
                     int((d > 0).sum()), len(d))
        del E
        gc.collect()
        torch.cuda.empty_cache()

    d = pd.DataFrame(rows)
    write_table(d, out_dir, "panel_a_uncertainty")
    record_run(out_dir, NAME, cfg)
    return {"panel_a_uncertainty": d}


def summarize(tables: dict) -> str:
    """Same sign in >= 4/5 folds and |mean| > 2 SD, per instance."""
    d = tables["panel_a_uncertainty"]
    out = ["same sign in >= 4/5 folds and |mean| > 2*SD, per instance"]
    ok_all = True
    for (corpus, sname), g in d.groupby(["corpus", "score"]):
        dd = g["diff"].to_numpy()
        signs = max(int((dd > 0).sum()), int((dd < 0).sum()))
        ok = signs >= 4 and abs(dd.mean()) > 2 * dd.std(ddof=1)
        ok_all &= ok
        out.append(f"   {corpus:<22} {sname:<13} mean {dd.mean():+.4f} "
                   f"2SD {2*dd.std(ddof=1):.4f} signs {signs}/5  "
                   f"{'ok' if ok else 'FAIL'}")
    out.append(f"--> {'PASS' if ok_all else 'FAIL'}")
    return "\n".join(out)
