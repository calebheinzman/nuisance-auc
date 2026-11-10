"""Bin-count sensitivity and non-cancelling balance of the same-feature panel.

(1) For each corpus and the zero-shot BiomedCLIP detector, recompute the C1
    (logistic, 16 features) and C2 (boosted trees, same 16) adjusted AUCs at
    q in {5, 10, 20, 40} equal-count propensity bins, with the overlap screen
    quantities, per-split check passes, and the share of all positive-negative
    test pairs that fall within a bin.
(2) Within each of 20 bins, the absolute standardized mean difference of each
    of the 16 features between the classes (bin's own pooled SD), summarised
    by its pair-weighted mean over bins, the max over features, and the
    pair-weighted share of (bin, feature) cells with |SMD| > 0.1, beside a
    within-bin label-permutation floor.

Folds, seeds, nuisance models, detector and estimator are those of
realdata_panel; q = 20 must reproduce its C1/C2 means (checked against the
reference panel_folds table when it is present).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import DEGEN_MAX, N_SPLIT, NEG_WINDOW
from nuisauc.data.cache import CORPORA, corpus_cache, nuisance_models
from nuisauc.detectors.vlm import embed_cache, prompt_embeddings, score
from nuisauc.estimand import conditional_auc, strata
from nuisauc.io import write_table
from nuisauc.paths import reference_dir
from nuisauc.propensity import propensity
from nuisauc.seeds import record_run

NAME = "bins_balance"
DESCRIPTION = "Bin-count sensitivity of C1/C2 and within-bin absolute feature balance"
PAPER = "§5, Tab 5, 9"
REQUIRES = "cache"
OUTPUTS = ["bins_sensitivity", "abs_balance"]

CAP = 4000
QS = (5, 10, 20, 40)
N_PERM = 20

log = logging.getLogger(f"nuisauc.{NAME}")


def within_pair_share(y: np.ndarray, e: np.ndarray, q: int) -> float:
    """Share of all positive-negative pairs that fall inside a bin."""
    b = strata(e, q)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    used = 0
    for k in range(q):
        yk = y[b == k]
        used += int(yk.sum()) * int(len(yk) - yk.sum())
    return used / (n1 * n0)


def abs_smd(X: np.ndarray, y: np.ndarray, e: np.ndarray,
            q: int) -> tuple[np.ndarray, float]:
    """Pair-weighted mean over bins of |SMD| per feature (bin pooled SD), and
    pair-weighted share of (bin, feature) cells with |SMD| > 0.1."""
    b = strata(e, q)
    acc = np.zeros(X.shape[1])
    over = 0.0
    den = 0.0
    for k in range(q):
        m = b == k
        yk = y[m]
        n1, n0 = int(yk.sum()), int(len(yk) - yk.sum())
        if n1 < 2 or n0 < 2:
            continue
        X1, X0 = X[m][yk == 1], X[m][yk == 0]
        sp = np.sqrt((X1.var(0, ddof=1) + X0.var(0, ddof=1)) / 2) + 1e-12
        d = np.abs(X1.mean(0) - X0.mean(0)) / sp
        w = n1 * n0
        acc += w * d
        over += w * (d > 0.1).mean()
        den += w
    per_feat = acc / den
    return per_feat, over / den


def default_config() -> dict:
    return {"corpora": list(CORPORA)}


def _names(corpora: list[str]) -> tuple[str, str]:
    """Full-panel table names, or *_rerun names for a subset of corpora."""
    if list(corpora) == CORPORA:
        return "bins_sensitivity", "abs_balance"
    return "bins_sensitivity_rerun", "abs_balance_rerun"


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpora = cfg["corpora"]
    n_bins, n_bal = _names(corpora)
    rows, brows = [], []
    for corpus in corpora:
        y, _C, _pk = corpus_cache(corpus, CAP)
        T16 = [m for m in nuisance_models(corpus, CAP) if m[0] == "C1"][0][1]
        E, _ = embed_cache("biomedclip", corpus, CAP, dev)
        T = prompt_embeddings("biomedclip", corpus, dev)
        Tp, Tn = T[f"{corpus}__pos"], T[f"{corpus}__neg"]
        s = score(E, Tp, Tn, np.arange(len(Tp)), np.arange(len(Tn)))
        splits = [next(iter(StratifiedKFold(4, shuffle=True, random_state=r)
                            .split(np.zeros(len(y)), y)))
                  for r in range(N_SPLIT)]
        for rep, (tr, te) in enumerate(splits):
            e = {tag: propensity(kind, T16, tr, y, rep).predict_proba(T16[te])[:, 1]
                 for tag, kind in (("C1", "linear"), ("C2", "gbm"))}
            for q in QS:
                row = dict(corpus=corpus, rep=rep, q=q)
                for tag in ("C1", "C2"):
                    c, dg = conditional_auc(y[te], s[te], e[tag], q)
                    ctl = conditional_auc(y[te], e[tag], e[tag], q)[0]
                    row[f"{tag}"] = float(c)
                    row[f"{tag}_degen"] = float(dg)
                    row[f"{tag}_control"] = float(ctl)
                    row[f"{tag}_pass"] = bool(dg <= DEGEN_MAX and
                                              NEG_WINDOW[0] <= ctl <= NEG_WINDOW[1])
                    row[f"{tag}_pairshare"] = within_pair_share(y[te], e[tag], q)
                rows.append(row)
            for tag in ("C1", "C2"):
                pf, share = abs_smd(T16[te], y[te], e[tag], 20)
                # sampling-noise floor: permute labels within each bin, which
                # keeps bin sizes and class counts but removes any imbalance
                b = strata(e[tag], 20)
                rng = np.random.default_rng([rep, 0 if tag == "C1" else 1])
                null_mean, null_share = [], []
                for _ in range(N_PERM):
                    yp = y[te].copy()
                    for k in range(20):
                        m = np.where(b == k)[0]
                        yp[m] = rng.permutation(yp[m])
                    pn, sn = abs_smd(T16[te], yp, e[tag], 20)
                    null_mean.append(pn.mean())
                    null_share.append(sn)
                brows.append(dict(corpus=corpus, rep=rep, model=tag,
                                  mean_abs_smd=float(pf.mean()),
                                  max_feat_abs_smd=float(pf.max()),
                                  share_cells_over_0p1=float(share),
                                  n_feat_over_0p1=int((pf > 0.1).sum()),
                                  null_mean_abs_smd=float(np.mean(null_mean)),
                                  null_share_over_0p1=float(np.mean(null_share))))
        log.info("%s done", corpus)
        write_table(pd.DataFrame(rows), out_dir, n_bins)
        write_table(pd.DataFrame(brows), out_dir, n_bal)

    d = pd.DataFrame(rows)
    ref_p = reference_dir() / "panel_folds.csv"
    if ref_p.exists():
        ref = pd.read_csv(ref_p)
        ref = ref[ref.detector == "zero_shot"].groupby("corpus")[["C1", "C2"]].mean()
        g = d[d.q == 20].groupby("corpus")[["C1", "C2"]].mean()
        assert (g - ref.loc[g.index]).abs().max().max() < 5e-4, (g, ref)
        log.info("q=20 reproduces panel_folds: ok")
    else:
        log.warning("no reference panel_folds table at %s; q=20 check skipped", ref_p)
    record_run(out_dir, NAME, cfg)
    return {n_bins: d, n_bal: pd.DataFrame(brows)}


def summarize(tables: dict) -> str:
    d = next(v for k, v in tables.items() if k.startswith("bins_sensitivity"))
    bal = next(v for k, v in tables.items() if k.startswith("abs_balance"))
    s = d.assign(gap=d.C1 - d.C2).groupby(["corpus", "q"]).agg(
        C1=("C1", "mean"), C2=("C2", "mean"), gap=("gap", "mean"),
        gap_pos=("gap", lambda v: int((v > 0).sum())),
        C1_degen=("C1_degen", "mean"), C2_degen=("C2_degen", "mean"),
        C1_ctl=("C1_control", "mean"), C2_ctl=("C2_control", "mean"),
        C1_pass=("C1_pass", "sum"), C2_pass=("C2_pass", "sum"),
        C1_pairs=("C1_pairshare", "mean"), C2_pairs=("C2_pairshare", "mean"))
    return (s.round(3).to_string() + "\n" +
            bal.groupby(["corpus", "model"]).mean(numeric_only=True)
            .drop(columns="rep").round(3).to_string())
