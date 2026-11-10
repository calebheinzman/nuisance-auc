"""Label-fit selection between the linear and tree nuisance classes: validation.

Part `semisynth`: semi-synthetic cells on each corpus's real trivial-16
geometry (synthetic labels, fold-matched truth theta_B) in three regime
classes (linear, nonlinear, outspan), alpha in {1, 2}, audit size in
{full corpus, 1000}. Per cell, inner 4-fold cross-fitted log-loss and Brier
on the outer training fold pick each class's member and the class; both
classes' conditional-AUC errors against theta_B are recorded.

Part `real`: the same selection on the six real Panel-A instances
(3 corpora x {zero_shot, self_trained} BiomedCLIP scores), five outer folds,
with each class's conditional AUC on the test fold.
"""
from __future__ import annotations

import gc
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.data.corpora import CORPORA, PANEL_A
from nuisauc.estimand import conditional_auc, n_strata, pair_weighted, qbins
from nuisauc.inference.selection import (
    GAP_MIN, N_SUBS, crossfit_scores, majority_verdict, select_family,
)
from nuisauc.io import write_table
from nuisauc.propensity import fit_any
from nuisauc.seeds import record_run
from nuisauc.simulate.gaussian import Q_FINE
from nuisauc.simulate.semisynthetic import (
    ADJ_ALPHAS, ADJ_SEED, adj_load_geometry, adj_regimes, adj_synth,
)

NAME = "selection_validation"
DESCRIPTION = ("Label-fit selection of the nuisance-model class: semi-synthetic "
               "validation and application to the real corpora")
PAPER = "§5, App E"
REQUIRES = "cache"
OUTPUTS = ["adjudication_semisynth", "adjudication_real"]
PART_TABLE = {"semisynth": "adjudication_semisynth", "real": "adjudication_real"}

N_REP = 6

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    # The legacy script ran one part per invocation (default semisynth);
    # both parts are run by default here so that every output is produced.
    return {"parts": ["semisynth", "real"], "n_rep": N_REP, "cap": 4000,
            "corpora": list(CORPORA)}


def run_semisynth(cfg: dict, draw_seed: int = ADJ_SEED,
                  seed_off: int = 0) -> pd.DataFrame:
    """Semi-synthetic validation cells (one row per corpus x regime x alpha x size x rep).

    The synthetic draws are keyed by `draw_seed`; rep r uses fold and
    estimator seed seed_off + r.
    """
    rows = []
    for corpus in cfg["corpora"]:
        T16, T16s, p1, p2, p3, a_lo, info = adj_load_geometry(corpus, cfg["cap"])
        log.info("%s: R2(a_lo|T16)=%.3f", corpus, info["r2_alo_T16"])
        for regime, (core_f, g_f) in adj_regimes(T16s, p1, p2, p3, a_lo).items():
            for alpha in ADJ_ALPHAS:
                for n_sub in N_SUBS:
                    for rep in range(cfg["n_rep"]):
                        seed = seed_off + rep
                        rng = np.random.default_rng(
                            [draw_seed, zlib.crc32(corpus.encode()) % 2 ** 16,
                             zlib.crc32(regime.encode()) % 2 ** 16,
                             int(alpha * 10), n_sub, rep])
                        if n_sub and n_sub < len(core_f):
                            keep = rng.choice(len(core_f), n_sub,
                                              replace=False)
                        else:
                            keep = np.arange(len(core_f))
                        core, g, X = core_f[keep], g_f[keep], T16[keep]
                        y, u = adj_synth(core, alpha, rng)
                        s = u + g + rng.standard_normal(len(core))
                        tr, te = next(iter(StratifiedKFold(
                            4, shuffle=True, random_state=seed)
                            .split(np.zeros(len(y)), y)))
                        q = n_strata(len(te))
                        tB = pair_weighted(y[te], s[te],
                                           qbins(core[te], Q_FINE))[0]
                        scores = crossfit_scores(X, y, tr, seed)
                        sel_ll, mem_ll, fam_ll, z = select_family(scores, 0)
                        sel_br, _, _, _zb = select_family(scores, 1)
                        errs = {}
                        for fam, (tag, _) in fam_ll.items():
                            m = fit_any(tag, X, y, tr, seed)
                            e_te = m.predict_proba(X[te])[:, 1]
                            est, _dg = conditional_auc(y[te], s[te], e_te, q)
                            errs[fam] = abs(float(est) - float(tB))
                        rows.append(dict(
                            corpus=corpus, regime=regime, alpha=alpha,
                            n_sub=n_sub or len(core_f), rep=rep,
                            theta_B=float(tB), member_ll=mem_ll,
                            sel_logloss=sel_ll, sel_brier=sel_br,
                            z_logloss=z,
                            err_linear=errs["linear"],
                            err_tree=errs["tree"],
                            ll_linear=fam_ll["linear"][1],
                            ll_tree=fam_ll["tree"][1]))
                    log.info("%s %s alpha=%s n=%s done", corpus, regime, alpha,
                             n_sub or "full")
        gc.collect()
    return pd.DataFrame(rows)


def panel_a_scores(corpus: str, cap: int, dev: str):
    """(y, T16, {score name: score}) for one Panel-A corpus."""
    from nuisauc.data.cache import corpus_cache, nuisance_models
    from nuisauc.detectors.selftrain import K_PSEUDO, N_ITER, pseudo, student
    from nuisauc.detectors.vlm import (
        GATED_VLM, embed_cache, prompt_embeddings, score,
    )
    y, _C, _pk = corpus_cache(corpus, cap)
    T16 = [m for m in nuisance_models(corpus, cap) if m[0] == "C1"][0][1]
    E, _ = embed_cache(GATED_VLM, corpus, cap, dev)
    T = prompt_embeddings(GATED_VLM, corpus, dev)
    Tp, Tn = T[f"{corpus}__pos"], T[f"{corpus}__neg"]
    s_zero = score(E, Tp, Tn, np.arange(len(Tp)), np.arange(len(Tn)))
    s_self = s_zero
    for _ in range(N_ITER):
        idx, plab = pseudo(s_self, K_PSEUDO)
        s_self = student(E, idx, plab)
    del E
    return y, T16, {"zero_shot": s_zero, "self_trained": s_self}


def run_real(cfg: dict, seed_off: int = 0) -> pd.DataFrame:
    """Real Panel-A selection; fold r uses random_state seed_off + r."""
    import torch

    from nuisauc.checks import N_SPLIT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for corpus in PANEL_A:
        y, T16, scores_by_name = panel_a_scores(corpus, cfg["cap"], dev)
        splits = [next(iter(StratifiedKFold(
            4, shuffle=True, random_state=seed_off + r)
            .split(np.zeros(len(y)), y))) for r in range(N_SPLIT)]
        for sname, s in scores_by_name.items():
            for rep, (tr, te) in enumerate(splits):
                seed = seed_off + rep
                q = n_strata(len(te))
                scores = crossfit_scores(T16, y, tr, seed)
                sel_ll, mem_ll, fam_ll, z = select_family(scores, 0)
                sel_br, _, _, _zb = select_family(scores, 1)
                row = dict(corpus=corpus, score=sname, rep=rep,
                           sel_logloss=sel_ll, sel_brier=sel_br,
                           member=mem_ll, z_logloss=z,
                           ll_linear=fam_ll["linear"][1],
                           ll_tree=fam_ll["tree"][1])
                for fam, (tag, _) in fam_ll.items():
                    m = fit_any(tag, T16, y, tr, seed)
                    e_te = m.predict_proba(T16[te])[:, 1]
                    c, _ = conditional_auc(y[te], s[te], e_te, q)
                    row[f"cond_{fam}"] = float(c)
                rows.append(row)
            log.info("%s %s done", corpus, sname)
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for part in cfg["parts"]:
        d = run_semisynth(cfg) if part == "semisynth" else run_real(cfg)
        write_table(d, out_dir, PART_TABLE[part])
        tables[PART_TABLE[part]] = d
    record_run(out_dir, NAME, cfg)
    return tables


def condition_table(d: pd.DataFrame) -> pd.DataFrame:
    """Per-condition mean class errors, oracle-best class and verdicts."""
    rows = []
    for key, g in d.groupby(["corpus", "regime", "alpha", "n_sub"]):
        el, et = float(g.err_linear.mean()), float(g.err_tree.mean())
        v_ll = majority_verdict(g.sel_logloss)
        v_br = majority_verdict(g.sel_brier)
        rows.append(dict(
            corpus=key[0], regime=key[1], alpha=key[2], n_sub=key[3],
            err_linear=el, err_tree=et, gap=abs(el - et),
            best="linear" if el < et else "tree",
            verdict=v_ll, verdict_brier=v_br,
            adj_frac=float((g.sel_logloss != "abstain").mean())))
    return pd.DataFrame(rows)


def report_semisynth(d: pd.DataFrame) -> list[str]:
    c = condition_table(d)
    inf = c[c.gap >= GAP_MIN]
    dec = inf[inf.verdict != "abstain"].copy()
    dec["correct"] = dec.verdict == dec.best
    dec["err_sel"] = np.where(dec.verdict == "linear",
                              dec.err_linear, dec.err_tree)
    dec["excess"] = dec.err_sel - dec[["err_linear", "err_tree"]].min(axis=1)

    out = ["=" * 92,
           "  Label-fit class selection: semi-synthetic validation (condition level)",
           "=" * 92,
           f"  conditions {len(c)}, informative (mean-gap >= {GAP_MIN}) "
           f"{len(inf)}, with verdict {len(dec)}"]

    acc = float(dec.correct.mean()) if len(dec) else float("nan")
    cov = len(dec) / len(inf) if len(inf) else float("nan")
    g1 = acc >= 0.80 and cov >= 0.50
    out.append(f"  G1 verdict accuracy {acc:.3f} "
               f"({int(dec.correct.sum())}/{len(dec)}), coverage {cov:.2f}  "
               f"--> {'PASS' if g1 else 'FAIL'}")

    exc = float(dec.excess.mean()) if len(dec) else float("nan")
    out.append(f"  G2 mean excess error of verdict class: {exc:.4f}  "
               f"--> {'PASS' if exc <= 0.005 else 'FAIL'}")

    out.append("  G3 per-regime accuracy on informative adjudicated "
               "conditions (need >= 0.75 each, both directions present)")
    g3 = True
    for regime, g in dec.groupby("regime"):
        a = float(g.correct.mean())
        ok = a >= 0.75
        g3 &= ok
        out.append(f"     {regime:<10} acc {a:.3f} ({int(g.correct.sum())}"
                   f"/{len(g)})  {'ok' if ok else 'FAIL'}")
    nl = int((inf.best == "linear").sum())
    nt = int((inf.best == "tree").sum())
    both = nl > 0 and nt > 0
    g3 &= both
    out.append(f"     informative conditions won by linear {nl} / tree {nt}; "
               f"both present: {both}  --> {'PASS' if g3 else 'FAIL'}")

    bb = inf[(inf.verdict != "abstain") & (inf.verdict_brier != "abstain")]
    agr = float((bb.verdict == bb.verdict_brier).mean()) if len(bb) \
        else float("nan")
    out.append(f"  G4 Brier/log-loss verdict agreement: {agr:.3f} "
               f"(n={len(bb)})  --> {'PASS' if agr >= 0.80 else 'FAIL'}")

    out.append("  condition table (informative marked *)")
    for _, r in c.sort_values(["regime", "alpha", "n_sub",
                               "corpus"]).iterrows():
        mark = "*" if r.gap >= GAP_MIN else " "
        hit = ("ok" if r.verdict == r.best else
               ("--" if r.verdict == "abstain" else "WRONG")) \
            if r.gap >= GAP_MIN else ""
        out.append(f"   {mark} {r.regime:<10} a={r.alpha} n={int(r.n_sub):<5} "
                   f"{r.corpus:<22} err L/T {r.err_linear:.4f}/{r.err_tree:.4f}"
                   f"  best {r.best:<6} verdict {r.verdict:<7} "
                   f"(adj {r.adj_frac:.2f})  {hit}")
    return out


def report_real(d: pd.DataFrame) -> list[str]:
    out = ["  G5 class-selection stability on real Panel-A instances "
           "(abstentions count against)"]
    stable = 0
    for (corpus, sname), g in d.groupby(["corpus", "score"]):
        counts = g[g.sel_logloss != "abstain"].sel_logloss.value_counts()
        if len(counts):
            fam, k = counts.index[0], int(counts.iloc[0])
        else:
            fam, k = "abstain", 0
        ok = fam != "abstain" and k >= 4
        stable += ok
        if fam != "abstain":
            other = "tree" if fam == "linear" else "linear"
            pref = g[f"cond_{fam}"].mean()
            extra = (f"preferred cAUC {pref:.4f} "
                     f"(vs {g[f'cond_{other}'].mean():.4f})")
        else:
            extra = ""
        nab = int((g.sel_logloss == "abstain").sum())
        out.append(f"     {corpus:<22} {sname:<13} {fam} {k}/{len(g)} folds "
                   f"(abstain {nab})  {extra}  "
                   f"{'stable' if ok else 'UNRESOLVED'}")
    out.append(f"  --> {stable}/6 instances stable  "
               f"{'PASS' if stable >= 5 else 'FAIL'}")
    return out


def summarize(tables: dict) -> str:
    out = []
    if "adjudication_semisynth" in tables:
        out += report_semisynth(tables["adjudication_semisynth"])
    if "adjudication_real" in tables:
        out += report_real(tables["adjudication_real"])
    return "\n".join(out)
