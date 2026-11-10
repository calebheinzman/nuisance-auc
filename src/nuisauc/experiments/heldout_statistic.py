"""The split-family statistic (EXC), its validation, and an endorsed-model audit.

Part B (simulation, table `<table>_sim`): an eight-column generative model in
which the analysis model sees columns 0-3 and the held-out family B is columns
4-7. EXC = max over both role assignments of HOB(excluded half) -
HOB(included half), with the propensity fitted on A plus the included half.
The threshold is the 95th percentile of EXC over the correct-condition
calibration cells.

Part B (real corpora, table `<table>_realfloor`): EXC of the 62-feature
descriptor against the frequency and morphology families, mean over five
repeated splits.

Part A (table `<table>`): the probe-versus-zero-shot comparison at n = 8 and
n = 64 labels, re-evaluated under every nuisance model (C1, C2, C3) that
passes the overlap screen on the same corpus and splits. Needs the raw images
and the image encoders (GPU recommended).
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from nuisauc.checks import DEGEN_MAX, N_SPLIT, NEG_WINDOW
from nuisauc.data.cache import corpus_cache, descriptor62
from nuisauc.data.corpora import (CORPORA, ENCODERS, draw_labels, load_any,
                                  pool_tokens_chunked)
from nuisauc.data.descriptors import (extended_nuisance, frequency16, morph16,
                                      trivial16)
from nuisauc.diagnostics.heldout import SIM_N, exc_statistic, null_threshold
from nuisauc.estimand import conditional_auc, n_strata
from nuisauc.io import read_table, write_table
from nuisauc.propensity import propensity
from nuisauc.seeds import record_run

NAME = "heldout_statistic"
DESCRIPTION = "Split-family statistic: simulation validation, real-corpus values, endorsed-model audit"
PAPER = "§5, App D"
REQUIRES = "cache"
# Part A (endorsed-model audit of a labelled-probe vs zero-shot comparison) is not
# used in the paper. It encodes raw images with five GPU encoders and picks an
# encoder by argmax, so GPU floating-point noise flips near-ties between runs
# (row-level differences up to ~0.4 at 8 labels; cell means agree to 0.02). It is
# off by default; enable with --set skip_a=false.
OUTPUTS = ["symmetry_floor_sim", "symmetry_floor_realfloor"]

BUDGETS_A = [8, 64]
N_DRAW = 50
A1_RATE = 0.90
PRIMARY_N = 64
PART_A_SEED = 3801

SIM_SEED = 3800
SIM_ALPHAS = [0.5, 1.0, 2.0]
SIM_LAMBDAS = [0.0, 1.0, 2.0]
SIM_REPS, SIM_CAL = 10, 20
SIM_CONDS = ["correct", "functional", "omitted_spread",
             "omitted_concentrated", "omitted_unavailable"]
F1_MIN, F2_MAX, F3_MAX = 0.80, 0.10, 0.20

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"corpora": list(CORPORA), "cap": 4000, "n_draw": N_DRAW,
            "skip_sim": False, "skip_real_floor": False, "skip_a": True,
            "table": "symmetry_floor"}


# ---------------------------------------------------------------- Part B sim

def sim8(cond: str, alpha: float, lam: float, rng: np.random.Generator):
    """Eight columns: the analysis model uses 0-3, B is 4-7."""
    Z = rng.standard_normal((SIM_N, 8))
    u, eps = rng.standard_normal((2, SIM_N))
    if cond == "correct":
        core = Z[:, 0] + Z[:, 1]
    elif cond == "functional":
        core = np.abs(Z[:, 0]) - 0.8
    elif cond == "omitted_spread":
        core = Z[:, 4:8].mean(axis=1) * 2.0
    elif cond == "omitted_concentrated":
        core = Z[:, 4]
    elif cond == "omitted_unavailable":
        core = rng.standard_normal(SIM_N)
    else:
        raise ValueError(cond)
    logit = alpha * core + u
    y = (rng.random(SIM_N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    s = u + lam * (Z[:, 0] - Z[:, 1]) + eps
    return Z, y, s


def sim_cell(cond: str, alpha: float, lam: float, rep: int) -> dict:
    rng = np.random.default_rng([SIM_SEED, SIM_CONDS.index(cond),
                                 int(alpha * 10), int(lam * 10), rep])
    Z, y, _s = sim8(cond, alpha, lam, rng)
    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                       .split(np.zeros(len(y)), y)))
    exc, floor = exc_statistic(Z[:, :4], Z[:, 4:8], y, tr, te,
                               n_strata(len(te)), rep, rng)
    return dict(cond=cond, alpha=alpha, lam=lam, rep=rep,
                exc=exc, floor=floor)


def run_sim() -> pd.DataFrame:
    rows = []
    for cond in SIM_CONDS:
        reps = SIM_CAL + SIM_REPS if cond == "correct" else SIM_REPS
        for alpha in SIM_ALPHAS:
            for lam in SIM_LAMBDAS:
                for rep in range(reps):
                    r = sim_cell(cond, alpha, lam, rep)
                    r["split"] = ("cal" if (cond == "correct"
                                            and rep < SIM_CAL) else "eval")
                    rows.append(r)
            log.info("sim %s alpha=%s done", cond, alpha)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Part B real

def run_real_floor(cap: int) -> pd.DataFrame:
    rows = []
    # quirk kept: always the five default corpora, regardless of cfg["corpora"]
    for corpus in CORPORA:
        y, _C, _pk = corpus_cache(corpus, cap)
        A = descriptor62(corpus, cap)
        for fam, B in (("frequency", frequency16(corpus, cap)),
                       ("morphology", morph16(corpus, cap))):
            es, fl = [], []
            for rep in range(N_SPLIT):
                rng = np.random.default_rng([SIM_SEED, 99, rep])
                tr, te = next(iter(StratifiedKFold(
                    4, shuffle=True, random_state=rep)
                    .split(np.zeros(len(y)), y)))
                e_, f_ = exc_statistic(A, B, y, tr, te,
                                       n_strata(len(te)), rep, rng)
                es.append(e_); fl.append(f_)
            rows.append(dict(corpus=corpus, family=fam,
                             exc=float(np.nanmean(es)),
                             floor=float(np.nanmean(fl))))
            log.info("SFF %s %s EXC=%+.4f floor=%.4f", corpus, fam,
                     np.nanmean(es), np.nanmean(fl))
        gc.collect()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Part A

def part_a(corpora: list[str], cap: int, n_draw: int, dev: str) -> pd.DataFrame:
    import torch

    from nuisauc.detectors.probes import (GPUBank, agg_top5, cv_probe_auc,
                                          probe)
    from nuisauc.detectors.vlm import encode
    rows = []
    for ci, corpus in enumerate(corpora):
        items, y = load_any(corpus, cap)
        T16 = trivial16(items)
        EXT = extended_nuisance(items)
        mods = [("C1", T16, "linear"), ("C2", T16, "gbm"), ("C3", EXT, "gbm")]

        scores, feats = {}, {}
        for k, (kind, spec) in ENCODERS.items():
            tok = encode(kind, spec, items, dev)
            feats[k] = pool_tokens_chunked(tok, "meanmax")
            scores[k] = agg_top5(GPUBank(
                tok, dev, seed=0, fit_idx=np.where(y == 0)[0]).score(tok))
            del tok
            gc.collect()
            torch.cuda.empty_cache()
        del items
        gc.collect()
        log.info("=== %s  n=%d ===", corpus, len(y))

        splits = [next(iter(StratifiedKFold(
            4, shuffle=True, random_state=r).split(np.zeros(len(y)), y)))
            for r in range(N_SPLIT)]

        # screen each model in this pipeline; audit only what passes here
        endorsed, e_cache = [], {}
        for tag, C, kind in mods:
            dg, nc = [], []
            for rep, (tr, te) in enumerate(splits):
                m = propensity(kind, C, tr, y, rep)
                e = m.predict_proba(C[te])[:, 1]
                e_cache[(tag, rep)] = e
                q = n_strata(len(te))
                n_, d_ = conditional_auc(y[te], e, e, q)
                dg.append(float(d_)); nc.append(float(n_))
            ok = (np.mean(dg) <= DEGEN_MAX
                  and NEG_WINDOW[0] <= np.mean(nc) <= NEG_WINDOW[1])
            if ok:
                endorsed.append(tag)
            log.info("screen %s: degen=%.3f self-control=%.3f  %s", tag,
                     np.mean(dg), np.mean(nc), "endorsed" if ok else "refused")

        for rep, (tr, te) in enumerate(splits):
            q = n_strata(len(te))
            zs_cond = {}
            for tag in endorsed:
                e_te = e_cache[(tag, rep)]
                zs_cond[tag] = {k: conditional_auc(
                    y[te], scores[k][te], e_te, q)[0] for k in ENCODERS}
            for n in BUDGETS_A:
                for draw in range(n_draw):
                    rng = np.random.default_rng([PART_A_SEED, ci, rep, n, draw])
                    idx = tr[draw_labels(y[tr], n, rng)]
                    yy = y[idx]
                    est = {k: float(roc_auc_score(yy, scores[k][idx]))
                           for k in ENCODERS}
                    kz = max(est, key=lambda a: est[a])
                    cvs = {k: cv_probe_auc(feats[k][idx], yy, rep)
                           for k in ENCODERS}
                    kp = max(cvs, key=lambda a: cvs[a])
                    mdl = probe(len(idx)).fit(feats[kp][idx], yy)
                    p = mdl.predict_proba(feats[kp][te])[:, 1]
                    for tag in endorsed:
                        e_te = e_cache[(tag, rep)]
                        pc, _ = conditional_auc(y[te], p, e_te, q)
                        rows.append(dict(
                            corpus=corpus, model=tag, rep=rep, n=n, draw=draw,
                            probe=float(pc), zeroshot=float(zs_cond[tag][kz])))
            log.info("%s rep %d done", corpus, rep)
        del scores, feats, T16, EXT, e_cache
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    base = cfg["table"]
    names = {"sim": f"{base}_sim", "real": f"{base}_realfloor", "a": base}
    tables = {}

    if not cfg["skip_sim"]:
        log.info("Part B: split-family statistic in simulation")
        tables[names["sim"]] = run_sim()
        write_table(tables[names["sim"]], out_dir, names["sim"])
    elif (Path(out_dir) / f"{names['sim']}.csv").exists():
        tables[names["sim"]] = read_table(out_dir, names["sim"])

    if not cfg["skip_real_floor"]:
        log.info("Part B: split-family statistic on the real corpora")
        tables[names["real"]] = run_real_floor(cfg["cap"])
        write_table(tables[names["real"]], out_dir, names["real"])
    elif (Path(out_dir) / f"{names['real']}.csv").exists():
        tables[names["real"]] = read_table(out_dir, names["real"])

    if not cfg["skip_a"]:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Part A: endorsed-model audit of the probe-vs-zero-shot claim")
        tables[names["a"]] = part_a(cfg["corpora"], cfg["cap"], cfg["n_draw"], dev)
        write_table(tables[names["a"]], out_dir, names["a"])
    elif (Path(out_dir) / f"{names['a']}.csv").exists():
        tables[names["a"]] = read_table(out_dir, names["a"])

    record_run(out_dir, NAME, cfg)
    return tables


def summarize(tables: dict) -> str:
    A = next((v for k, v in tables.items()
              if not k.endswith(("_sim", "_realfloor"))), pd.DataFrame())
    S = next((v for k, v in tables.items() if k.endswith("_sim")), pd.DataFrame())
    RF = next((v for k, v in tables.items() if k.endswith("_realfloor")),
              pd.DataFrame())
    out = []
    if len(A):
        g = (A.groupby(["corpus", "model", "n"])
             .apply(lambda x: float((x.probe - x.zeroshot).mean()),
                    include_groups=False).rename("diff").reset_index())
        g64 = g[g.n == PRIMARY_N]
        wins = int((g64["diff"] > 0).sum())
        a1 = wins / len(g64) >= A1_RATE if len(g64) else False
        out.append(f"probe beats zero-shot at n={PRIMARY_N} under every "
                   f"endorsed model (>= {A1_RATE:.0%} of cells)")
        for _i, r in g64.iterrows():
            out.append(f"  {r.corpus:<22} {r.model}  diff={r['diff']:+.4f}"
                       f"{'' if r['diff'] > 0 else '  zero-shot wins'}")
        out.append(f"  -> {wins}/{len(g64)}  {'pass' if a1 else 'fail'}")
        g8 = g[g.n == 8]
        out.append("the n = 8 direction per cell")
        for corpus, gg in g8.groupby("corpus"):
            dirs = {r.model: r["diff"] > 0 for _i, r in gg.iterrows()}
            un = len(set(dirs.values())) == 1
            out.append(f"  {corpus:<22} " +
                       "  ".join(f"{m}={'probe' if d else 'zs'}"
                                 for m, d in dirs.items()) +
                       ("" if un else "   split"))

    if len(S):
        cal = S[S.split == "cal"]
        ev = S[S.split == "eval"]
        t = null_threshold(cal.exc)
        out.append(f"split-family threshold from {len(cal)} calibration cells: {t:.4f}")
        rates = {}
        for cond in SIM_CONDS:
            gg = ev[ev.cond == cond]
            rates[cond] = float((gg.exc > t).mean()) if len(gg) else np.nan
            out.append(f"  {cond:<26} EXC mean={gg.exc.mean():+.4f}  "
                       f"fires={rates[cond]:.2f}")
        fpr = float((ev[ev.cond.isin(["correct", "functional"])].exc > t).mean())
        out.append(f"fires >= {F1_MIN} on omitted_spread: "
                   f"{rates['omitted_spread']:.2f} -> "
                   f"{'pass' if rates['omitted_spread'] >= F1_MIN else 'fail'}")
        out.append(f"false-positive rate <= {F2_MAX} on correct+functional: "
                   f"{fpr:.3f} -> {'pass' if fpr <= F2_MAX else 'fail'}")
        out.append(f"boundary <= {F3_MAX} on omitted_unavailable: "
                   f"{rates['omitted_unavailable']:.2f} -> "
                   f"{'as predicted' if rates['omitted_unavailable'] <= F3_MAX else 'not as predicted'}")
        out.append(f"omitted_concentrated fires: {rates['omitted_concentrated']:.2f}")
        if len(RF):
            out.append(f"real corpora, threshold {t:.4f}")
            for _i, r in RF.iterrows():
                out.append(f"  {r.corpus:<22} {r.family:<10} "
                           f"EXC={r.exc:+.4f}  floor={r.floor:.4f}"
                           f"{'   fires' if r.exc > t else ''}")
    return "\n".join(out)
