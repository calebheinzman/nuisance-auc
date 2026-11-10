"""Same-feature disagreement beyond two learners.

On the Panel-A corpora, the zero-shot and self-trained BiomedCLIP detectors
are evaluated under four nuisance learners on the SAME 16 features and the
SAME folds: C1 logistic, C2 boosted trees, C1e elastic-net logistic
(saga, l1_ratio 0.5, C = 1) and C2r random forest (400 trees,
min_samples_leaf 20). The question is whether screen-passing same-feature
estimators disagree beyond the C1/C2 pair.
"""
from __future__ import annotations

import gc
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import DEGEN_MAX, N_SPLIT, NEG_WINDOW
from nuisauc.data.cache import corpus_cache, nuisance_models
from nuisauc.data.corpora import PANEL_A
from nuisauc.detectors.selftrain import K_PSEUDO, N_ITER, pseudo, student
from nuisauc.detectors.vlm import GATED_VLM, embed_cache, prompt_embeddings, score
from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.io import write_table
from nuisauc.propensity import fit_learner, propensity
from nuisauc.seeds import record_run

NAME = "realdata_learners"
DESCRIPTION = "Adjusted AUC under four same-feature nuisance learners"
PAPER = "App F"
REQUIRES = "cache"
OUTPUTS = ["panel_a_learners"]

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
        T16 = [m for m in nuisance_models(corpus, cap)
               if m[0] == "C1"][0][1]
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
                for tag in ("C1", "C2", "C1e", "C2r"):
                    if tag in ("C1", "C2"):
                        m = propensity("linear" if tag == "C1" else "gbm",
                                       T16, tr, y, rep)
                    else:
                        m = fit_learner(tag, T16, y, tr, rep)
                    e = m.predict_proba(T16[te])[:, 1]
                    c, dg = conditional_auc(y[te], s[te], e, q)
                    nc, _ = conditional_auc(y[te], e, e, q)
                    rows.append(dict(corpus=corpus, score=sname, rep=rep,
                                     model=tag, cond=float(c),
                                     degen=float(dg), neg=float(nc)))
            log.info("  %-22s %s done", corpus, sname)
        del E
        gc.collect()
        torch.cuda.empty_cache()

    d = pd.DataFrame(rows)
    write_table(d, out_dir, "panel_a_learners")
    record_run(out_dir, NAME, cfg)
    return {"panel_a_learners": d}


def summarize(tables: dict) -> str:
    """Per instance: does some screen-passing same-feature pair disagree by
    >= 0.04 with the same sign in >= 4 of 5 folds?"""
    d = tables["panel_a_learners"]
    out = ["P1: some screen-passing same-feature pair disagrees by "
           ">= 0.04 with stable sign, per instance"]
    hits = 0
    for (corpus, sname), g in d.groupby(["corpus", "score"]):
        passing = []
        for tag, gg in g.groupby("model"):
            ok = (gg.degen.mean() <= DEGEN_MAX
                  and NEG_WINDOW[0] <= gg.neg.mean() <= NEG_WINDOW[1])
            if ok:
                passing.append(tag)
        best = None
        piv = g.pivot_table(index="rep", columns="model", values="cond")
        for t1, t2 in combinations(passing, 2):
            dd = (piv[t1] - piv[t2]).to_numpy()
            signs = max(int((dd > 0).sum()), int((dd < 0).sum()))
            cand = (abs(dd.mean()), t1, t2, signs)
            if abs(dd.mean()) >= 0.04 and signs >= 4:
                if best is None or cand > best:
                    best = cand
        hit = best is not None
        hits += hit
        detail = (f"{best[1]}-{best[2]} |diff|={best[0]:.4f} "
                  f"signs {best[3]}/5" if hit else "none >= 0.04")
        out.append(f"   {corpus:<22} {sname:<13} passing={passing}  {detail}"
                   f"  {'ok' if hit else '--'}")
    out.append(f"--> {hits}/6 instances  {'PASS' if hits >= 4 else 'FAIL'}")
    return "\n".join(out)
