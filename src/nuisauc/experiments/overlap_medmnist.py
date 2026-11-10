"""Overlap boundary on MedMNIST one-vs-rest instances, by prevalence and stratum count.

Whether an instance passes the overlap screen depends only on (y, propensity,
q), never on the detector's score (the control is the propensity itself). On
the same images this sweeps prevalence (by thinning the negatives, so the
positive count is fixed) and the number of strata, under C1 (trivial-16,
logistic) and C3 (62-feature, boosted), to separate a genuine lack of common
support from strata emptying of positives at low prevalence.

Reads raw images (NUISAUC_RAW: the MedMNIST parquet files and the processed
ZN corpora), not the cache.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import DEGEN_MAX, NEG_WINDOW
from nuisauc.data.corpora import load_corpus
from nuisauc.data.descriptors import extended_nuisance, trivial16
from nuisauc.data.medmnist import MEDMNIST_CORPORA, instances, load_pool
from nuisauc.estimand import conditional_auc
from nuisauc.io import write_table
from nuisauc.propensity import propensity
from nuisauc.seeds import record_run

NAME = "overlap_medmnist"
DESCRIPTION = "Screen pass/fail on MedMNIST one-vs-rest instances by prevalence and stratum count"
PAPER = "§4"
REQUIRES = "raw"
OUTPUTS = ["overlap_boundary"]

PREVALENCES = [0.125, 0.25, 0.50]
QS = [4, 8, 12, 20]
THIN_SEED = 7
# the reference run adds three ZN corpora (each a single binary instance) to the
# MedMNIST ones; the legacy CLI default was the MedMNIST corpora alone
ZN_CORPORA = ["Kaggle__local_128", "AFB__internal_81", "Wellgen__local_128"]

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"corpora": list(MEDMNIST_CORPORA) + ZN_CORPORA, "cap": 8000, "n_rep": 5}


def _load_any(name: str, cap: int):
    """MedMNIST corpora yield several one-vs-rest instances; a ZN corpus is a
    single binary instance. Returns (items, [(instance_name, y), ...])."""
    if name in MEDMNIST_CORPORA:
        items, lab = load_pool(name, cap=cap)
        return items, instances(name, lab)
    items, y, _groups = load_corpus(name, cap=cap)
    return items, [(name, y)]


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for corpus in cfg["corpora"]:
        items, inst = _load_any(corpus, cfg["cap"])
        T16, EXT = trivial16(items), extended_nuisance(items)
        n_img = len(items)
        del items
        log.info("=== %s  n=%d  instances=%d ===", corpus, n_img, len(inst))

        for name, y in inst:
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            for pi in PREVALENCES:
                rng = np.random.default_rng(THIN_SEED)
                # thin the NEGATIVES to reach the target prevalence, so the
                # positive count (and hence the pool's information) is fixed
                want_neg = int(round(len(pos) * (1 - pi) / pi))
                if want_neg > len(neg):
                    continue
                idx = np.sort(np.concatenate(
                    [pos, rng.choice(neg, want_neg, replace=False)]))
                yy = y[idx]
                for cname, (pkind, C) in (("C1", ("linear", T16)),
                                          ("C3", ("gbm", EXT))):
                    Cs = C[idx]
                    for rep in range(cfg["n_rep"]):
                        tr, te = next(iter(StratifiedKFold(
                            4, shuffle=True,
                            random_state=rep).split(np.zeros(len(yy)), yy)))
                        m = propensity(pkind, Cs, tr, yy, rep)
                        e = m.predict_proba(Cs[te])[:, 1]
                        for q in QS:
                            cond, degen = conditional_auc(yy[te], e, e, q)
                            rows.append(dict(
                                corpus=corpus, instance=name, prevalence=pi,
                                confound=cname, q=q, rep=rep,
                                n_test=len(te), n_pos_test=int(yy[te].sum()),
                                neg_ctrl_cond=float(cond),
                                degenerate=float(degen)))
            log.info("[%s] swept", name)

    d = pd.DataFrame(rows)
    write_table(d, out_dir, "overlap_boundary")
    record_run(out_dir, NAME, cfg)
    return {"overlap_boundary": d}


def summarize(tables: dict) -> str:
    d = tables["overlap_boundary"]
    g = d.groupby(["confound", "prevalence", "q"]).agg(
        degen=("degenerate", "mean"),
        neg=("neg_ctrl_cond", "mean"),
        pos_per_stratum=("n_pos_test", "mean")).reset_index()
    g["pos_per_stratum"] = g.pos_per_stratum / g.q
    g["conditionable"] = ((g.degen <= DEGEN_MAX)
                          & g.neg.between(*NEG_WINDOW))
    out = ["Would an instance pass the screen, by prevalence and stratum count? "
           "(means over instances)",
           f"  {'conf':<6}{'prev':>7}{'Q':>5}{'degen':>9}{'negctrl':>10}"
           f"{'pos/stratum':>13}{'  verdict'}"]
    for _, r in g.iterrows():
        out.append(f"  {r.confound:<6}{r.prevalence:>7.3f}{int(r.q):>5}"
                   f"{r.degen:>9.2f}{r.neg:>10.3f}{r.pos_per_stratum:>13.1f}"
                   f"{'  passes' if r.conditionable else '  no'}")

    out.append("\n  per-instance passing cells (any prevalence x Q), C3:")
    c3 = d[d.confound == "C3"].groupby(["instance", "prevalence", "q"]).agg(
        degen=("degenerate", "mean"), neg=("neg_ctrl_cond", "mean"))
    c3["ok"] = (c3.degen <= DEGEN_MAX) & c3.neg.between(*NEG_WINDOW)
    for inst, sub in c3.groupby(level=0):
        out.append(f"    {inst:<42}{int(sub.ok.sum())}/{len(sub)}")
    return "\n".join(out)
