"""Disagreement among screen-passing nuisance models on the real corpora.

If the overlap screen certified correctness, two nuisance models that both pass
it would have to agree. For the zero-shot BiomedCLIP detector, its self-trained
student, and a pure-nuisance reference score, this measures the conditional
AUC under C1 (trivial-16 logistic), C2 (trivial-16 boosted) and C3 (62-feature
boosted), whether each passes the screen, and the spread among the passing
models.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import N_SPLIT, screen_evaluate
from nuisauc.data.cache import CORPORA, corpus_cache, nuisance_models
from nuisauc.detectors.selftrain import K_PSEUDO, N_ITER, pseudo, student
from nuisauc.detectors.vlm import GATED_VLM, embed_cache, prompt_embeddings, score
from nuisauc.io import write_table
from nuisauc.seeds import record_run

NAME = "realdata_screen"
DESCRIPTION = "Spread of the adjusted AUC among screen-passing nuisance models on real corpora"
PAPER = "§5, Tab 4"
REQUIRES = "cache"
OUTPUTS = ["screen_realdata"]

Y1_SPREAD = 0.05
Y1_CORPORA = 3
CLAIM_ARMS = {"zero_shot", "self_trained"}

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"corpora": list(CORPORA), "cap": 4000}


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cfg["cap"]

    rows = []
    for corpus in cfg["corpora"]:
        y, _C, _pk = corpus_cache(corpus, cap)
        splits = [next(iter(StratifiedKFold(
            4, shuffle=True, random_state=r).split(np.zeros(len(y)), y)))
            for r in range(N_SPLIT)]
        models = nuisance_models(corpus, cap)
        E, _ = embed_cache(GATED_VLM, corpus, cap, dev)
        T = prompt_embeddings(GATED_VLM, corpus, dev)
        Tp, Tn = T[f"{corpus}__pos"], T[f"{corpus}__neg"]

        s_zero = score(E, Tp, Tn, np.arange(len(Tp)), np.arange(len(Tn)))
        s_self = s_zero
        for _ in range(N_ITER):
            idx, plab = pseudo(s_self, K_PSEUDO)
            s_self = student(E, idx, plab)
        T16 = [m for m in models if m[0] == "C1"][0][1]
        scores = {"zero_shot": s_zero, "self_trained": s_self,
                  "nuisance_ref": T16[:, 0]}

        log.info("=== %s  n=%d ===", corpus, len(y))
        for sname, s in scores.items():
            cells = {}
            for tag, C, kind in models:
                c, dgm, ncm, passes = screen_evaluate(s, y, C, kind, splits)
                cells[tag] = (c, passes, ncm)
            passing = [t for t in cells if cells[t][1]]
            vals = [cells[t][0] for t in passing]
            spread = (max(vals) - min(vals)) if len(vals) >= 2 else np.nan
            allv = [cells[t][0] for t in cells]
            rows.append(dict(
                corpus=corpus, score=sname,
                C1=cells["C1"][0], C2=cells["C2"][0], C3=cells["C3"][0],
                pass_C1=cells["C1"][1], pass_C2=cells["C2"][1],
                pass_C3=cells["C3"][1], n_pass=len(passing),
                spread_pass=spread, spread_all=max(allv) - min(allv)))
            tags = " ".join(f"{t}={cells[t][0]:.3f}"
                            f"{'*' if cells[t][1] else ''}" for t in cells)
            log.info("  %-14s %s   passing=%d  spread(pass)=%s", sname, tags,
                     len(passing), "n/a" if np.isnan(spread) else f"{spread:.3f}")
        del E
        gc.collect()

    d = pd.DataFrame(rows)
    write_table(d, out_dir, "screen_realdata")
    record_run(out_dir, NAME, cfg)
    return {"screen_realdata": d}


def summarize(tables: dict) -> str:
    """Spread among screen-passing models; * in the log marks a passing model."""
    d = tables["screen_realdata"]
    out = []
    ev = d[d.n_pass >= 2]
    out.append(f"evaluable cells (>= 2 nuisance models pass the screen): "
               f"{len(ev)} of {len(d)}")
    if len(ev) < 5:
        out.append("fewer than five evaluable cells; inconclusive")
        return "\n".join(out)

    hits = ev[ev.spread_pass >= Y1_SPREAD]
    corp = sorted(hits.corpus.unique())
    y1 = len(corp) >= Y1_CORPORA
    out.append(f"\nY1 some evaluable cell has spread >= {Y1_SPREAD} on >= "
               f"{Y1_CORPORA} of 5 corpora")
    for c in sorted(ev.corpus.unique()):
        g = ev[ev.corpus == c]
        best = g.spread_pass.max()
        arm = g.loc[g.spread_pass.idxmax(), "score"]
        out.append(f"   {c:<22} max spread {best:.3f}  ({arm})"
                   f"{'  ok' if best >= Y1_SPREAD else ''}")
    out.append(f"   --> {len(corp)}/5  {'PASS' if y1 else 'FAIL'}")

    claim = hits[hits.score.isin(CLAIM_ARMS)]
    y2 = len(claim) > 0
    out.append("\nY2 at least one such cell is a claim-bearing arm")
    for _i, r in claim.iterrows():
        out.append(f"   {r.corpus:<22} {r.score:<14} "
                   f"C1={r.C1:.3f} C2={r.C2:.3f} C3={r.C3:.3f}  "
                   f"spread(pass)={r.spread_pass:.3f}")
    out.append(f"   --> {'PASS' if y2 else 'FAIL'}")

    out.append("\nY3 magnitude over evaluable cells")
    out.append(f"   max {ev.spread_pass.max():.3f}   "
               f"mean {ev.spread_pass.mean():.3f}   n={len(ev)}")
    out.append("\nY4 screen-passing spread vs all-model spread")
    out.append(f"   mean spread among passing models: {ev.spread_pass.mean():.3f}")
    out.append(f"   mean spread among all models:     {ev.spread_all.mean():.3f}")
    out.append(f"\n--> overall {'PASS' if (y1 and y2) else 'FAIL'}")
    return "\n".join(out)
