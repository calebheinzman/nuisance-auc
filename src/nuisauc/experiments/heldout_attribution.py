"""After a split-family fire: which held-out features carry the leftover
within-stratum label signal, and what does adding them to the nuisance do?

Per corpus and held-out family B (frequency or morphology), over the five
repeated splits used on the real corpora:
  1. fit the boosted propensity on the 62-feature descriptor A, cut strata on
     the test fold, and score each held-out column B_j by its within-stratum
     label signal |adjusted AUC(B_j) - 1/2| (table `sff_attribution_rank`);
  2. add the top three columns to A and recompute the split-family statistic
     over 10 random halvings of the remaining columns (fire fraction at
     0.0167), and the zero-shot detector's adjusted AUC with propensities on
     A, A + top-3 and A + all of B, with the overlap screen for each (table
     `sff_attribution_enrich`).
This attributes the signal to columns; whether a column is a nuisance or a
disease proxy stays a clinical judgment.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import DEGEN_MAX, NEG_WINDOW
from nuisauc.data.cache import corpus_cache, descriptor62
from nuisauc.data.descriptors import frequency16, morph16
from nuisauc.diagnostics.heldout import exc_statistic
from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple
from nuisauc.seeds import record_run

NAME = "heldout_attribution"
DESCRIPTION = "Column attribution of split-family fires and the effect of adding the top columns"
PAPER = "App D"
REQUIRES = "cache"
OUTPUTS = ["sff_attribution_rank", "sff_attribution_enrich"]

THRESHOLD = 0.0167
CORPORA = ["AFB__internal_81", "Kaggle__local_128", "pcam", "malaria",
           "Wellgen__local_128"]
VLM_NAME = "biomedclip"
CAP = 4000
N_SPLIT = 5
N_TOP = 3
N_HALVINGS = 10
HALVING_SEED = 61012

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"corpora": list(CORPORA), "cap": CAP}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    import torch

    from nuisauc.detectors.vlm import embed_cache, prompt_embeddings, score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cfg["cap"]
    rank_rows, enr_rows = [], []
    for corpus in cfg["corpora"]:
        y, _C, _pk = corpus_cache(corpus, cap)
        A = descriptor62(corpus, cap)
        E, _ = embed_cache(VLM_NAME, corpus, cap, dev)
        T = prompt_embeddings(VLM_NAME, corpus, dev)
        s = score(E, T[f"{corpus}__pos"], T[f"{corpus}__neg"], np.arange(4), np.arange(4))
        splits = [next(iter(StratifiedKFold(4, shuffle=True, random_state=r)
                            .split(np.zeros(len(y)), y))) for r in range(N_SPLIT)]
        for fam, B in (("frequency", frequency16(corpus, cap)),
                       ("morphology", morph16(corpus, cap))):
            sig = np.zeros((N_SPLIT, B.shape[1]))
            for rep, (tr, te) in enumerate(splits):
                q = n_strata(len(te))
                e = fit_simple("gbm", A, y, tr, rep).predict_proba(A[te])[:, 1]
                for j in range(B.shape[1]):
                    c, _ = conditional_auc(y[te], B[te, j], e, q)
                    sig[rep, j] = abs(c - 0.5)
            order = np.argsort(-sig.mean(0))
            for r, j in enumerate(order):
                rank_rows.append(dict(corpus=corpus, family=fam, col=int(j), rank=r,
                                      signal=float(sig[:, j].mean()),
                                      signal_sd=float(sig[:, j].std(ddof=1))))
            top = order[:N_TOP]
            rest = order[N_TOP:]
            for setname, X in (("A", A), ("A+top3", np.c_[A, B[:, top]]),
                               ("A+allB", np.c_[A, B])):
                cs, dg, ctl, fires = [], [], [], []
                for rep, (tr, te) in enumerate(splits):
                    q = n_strata(len(te))
                    e = fit_simple("gbm", X, y, tr, rep).predict_proba(X[te])[:, 1]
                    c, d = conditional_auc(y[te], s[te], e, q)
                    cs.append(c); dg.append(d)
                    ctl.append(conditional_auc(y[te], e, e, q)[0])
                    if setname != "A+allB":
                        Bm = B[:, rest] if setname == "A+top3" else B
                        for hs in range(N_HALVINGS):
                            ex, _f = exc_statistic(X, Bm, y, tr, te, q, rep,
                                                   np.random.default_rng([HALVING_SEED, rep, hs]))
                            fires.append(ex > THRESHOLD)
                enr_rows.append(dict(corpus=corpus, family=fam, set=setname,
                                     top3=",".join(map(str, top)),
                                     cauc=float(np.mean(cs)), degen=float(np.mean(dg)),
                                     control=float(np.mean(ctl)),
                                     screen=bool(np.mean(dg) <= DEGEN_MAX
                                                 and NEG_WINDOW[0] <= np.mean(ctl) <= NEG_WINDOW[1]),
                                     fire_frac=float(np.mean(fires)) if fires else np.nan))
                r = enr_rows[-1]
                log.info("%s %s %s cAUC %.3f screen %s fire %.2f", corpus, fam,
                         setname, r["cauc"], r["screen"], r["fire_frac"])
    tables = {"sff_attribution_rank": pd.DataFrame(rank_rows),
              "sff_attribution_enrich": pd.DataFrame(enr_rows)}
    for name, df in tables.items():
        write_table(df, out_dir, name)
    record_run(out_dir, NAME, cfg)
    return tables
