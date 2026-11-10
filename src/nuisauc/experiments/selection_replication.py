"""Frozen replication of the label-fit class selection, plus abstention calibration.

The `selection_validation` protocol is reused unchanged (same candidates,
cross-fit, member/class selection, |z| >= 2 abstention, condition-level
majority verdict, GAP_MIN, the same 60 conditions including the fixed random
directions keyed to ADJ_SEED). Fresh: every rep and fold seed (200 + r,
r = 0..23) and the synthetic-draw key (9600 in place of 4800); the real
Panel-A part uses the same fresh fold-seed block.

Reps 0..11 form the confirmatory replication. The calibration analysis uses
nested prefixes r in {6, 12, 24}; each condition's oracle gap is the
|mean err_linear - mean err_tree| over all 24 reps.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nuisauc.data.corpora import CORPORA
from nuisauc.experiments.selection_validation import run_real, run_semisynth
from nuisauc.inference.selection import GAP_MIN, majority_verdict
from nuisauc.io import read_table, write_table
from nuisauc.paths import reference_dir
from nuisauc.seeds import record_run

NAME = "selection_replication"
DESCRIPTION = ("Fresh-seed replication of the class selection and calibration "
               "of its abstentions")
PAPER = "App E"
REQUIRES = "cache"
OUTPUTS = ["adjudication_repl_semisynth", "adjudication_repl_real",
           "adjudication_calibration"]

REPL_SEED = 9600
SEED_OFF = 200          # per-rep seed offset: a block disjoint from the validation run
N_REP_FULL = 24         # total fresh reps (calibration)
N_REP_CONF = 12         # confirmatory prefix
R_GRID = (6, 12, 24)    # nested prefixes for calibration
VALIDATION_REAL = "adjudication_real"   # real-data verdicts to compare against

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    # The legacy script ran one part per invocation (default semisynth);
    # both parts are run by default here so that every output is produced.
    return {"parts": ["semisynth", "real"], "n_rep": N_REP_FULL, "cap": 4000,
            "corpora": list(CORPORA)}


def condition_table_prefix(d: pd.DataFrame, r: int) -> pd.DataFrame:
    """Condition table using only reps < r (nested prefix of the fresh block)."""
    g = d[d.rep < r]
    rows = []
    for key, gg in g.groupby(["corpus", "regime", "alpha", "n_sub"]):
        el, et = float(gg.err_linear.mean()), float(gg.err_tree.mean())
        rows.append(dict(
            corpus=key[0], regime=key[1], alpha=key[2], n_sub=key[3],
            err_linear=el, err_tree=et, gap=abs(el - et),
            best="linear" if el < et else "tree",
            verdict=majority_verdict(gg.sel_logloss),
            verdict_brier=majority_verdict(gg.sel_brier),
            mean_abs_z=float(gg.z_logloss.abs().mean()),
            adj_frac=float((gg.sel_logloss != "abstain").mean())))
    return pd.DataFrame(rows)


def calibration_tables(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Per-prefix condition tables against the 24-rep oracle, and the calibration table."""
    full = condition_table_prefix(d, N_REP_FULL)
    fullx = full.set_index(["corpus", "regime", "alpha", "n_sub"])
    oracle = fullx.gap
    oracle_best = fullx.best
    tabs = {}
    for r in R_GRID:
        t = condition_table_prefix(d, r).set_index(
            ["corpus", "regime", "alpha", "n_sub"])
        t["oracle_gap"] = oracle
        t["oracle_best"] = oracle_best
        t["decided"] = t.verdict != "abstain"
        t["harmful"] = (t.decided & (t.verdict != t.oracle_best)
                        & (t.oracle_gap >= GAP_MIN))
        tabs[r] = t
    t24 = tabs[24]
    cal = t24.reset_index()[["corpus", "regime", "alpha", "n_sub",
                             "oracle_gap", "oracle_best", "mean_abs_z",
                             "verdict", "adj_frac"]]
    for r in R_GRID:
        cal[f"decided_r{r}"] = tabs[r].decided.values
        cal[f"verdict_r{r}"] = tabs[r].verdict.values
    return tabs, cal


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    if "semisynth" in cfg["parts"]:
        d = run_semisynth(cfg, draw_seed=REPL_SEED, seed_off=SEED_OFF)
        write_table(d, out_dir, "adjudication_repl_semisynth")
        tables["adjudication_repl_semisynth"] = d
        if d.rep.max() + 1 >= N_REP_FULL:
            _tabs, cal = calibration_tables(d)
            write_table(cal, out_dir, "adjudication_calibration")
            tables["adjudication_calibration"] = cal
    if "real" in cfg["parts"]:
        d = run_real(cfg, seed_off=SEED_OFF)
        write_table(d, out_dir, "adjudication_repl_real")
        tables["adjudication_repl_real"] = d
    record_run(out_dir, NAME, cfg)
    return tables


def report_confirmatory(d: pd.DataFrame) -> list[str]:
    c = condition_table_prefix(d, N_REP_CONF)
    inf = c[c.gap >= GAP_MIN]
    dec = inf[inf.verdict != "abstain"].copy()
    dec["correct"] = dec.verdict == dec.best
    dec["err_sel"] = np.where(dec.verdict == "linear",
                              dec.err_linear, dec.err_tree)
    dec["excess"] = dec.err_sel - dec[["err_linear", "err_tree"]].min(axis=1)

    out = ["=" * 92,
           f"  Replication (frozen protocol, fresh seeds, reps 0..{N_REP_CONF - 1})",
           "=" * 92,
           f"  conditions {len(c)}, informative (mean-gap >= {GAP_MIN}) "
           f"{len(inf)}, with verdict {len(dec)}"]

    acc = float(dec.correct.mean()) if len(dec) else float("nan")
    out.append(f"  R1 verdict accuracy {acc:.3f} "
               f"({int(dec.correct.sum())}/{len(dec)})  "
               f"--> {'PASS' if acc >= 0.80 else 'FAIL'}")

    exc = float(dec.excess.mean()) if len(dec) else float("nan")
    out.append(f"  R2 mean excess error {exc:.4f}  "
               f"--> {'PASS' if exc <= 0.005 else 'FAIL'}")

    g3 = True
    out.append("  R3 per-regime accuracy (need >= 0.75 each, both directions)")
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
    out.append(f"     informative won by linear {nl} / tree {nt}; both: {both}"
               f"  --> {'PASS' if g3 else 'FAIL'}")

    bb = inf[(inf.verdict != "abstain") & (inf.verdict_brier != "abstain")]
    agr = float((bb.verdict == bb.verdict_brier).mean()) if len(bb) \
        else float("nan")
    out.append(f"  R4 Brier/log-loss agreement {agr:.3f} (n={len(bb)})  "
               f"--> {'PASS' if agr >= 0.80 else 'FAIL'}")

    cov = len(dec) / len(inf) if len(inf) else float("nan")
    out.append(f"  R7 coverage on informative conditions {cov:.3f} "
               f"({len(dec)}/{len(inf)})  "
               f"--> {'PASS' if cov >= 0.636 else 'FAIL'}")

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


def report_real_repl(d: pd.DataFrame, validation_real: pd.DataFrame | None) -> list[str]:
    out = ["=" * 92, "  Replication -- real Panel A (fresh fold seeds)", "=" * 92]
    stable = 0
    verdicts = {}
    for (corpus, sname), g in d.groupby(["corpus", "score"]):
        counts = g[g.sel_logloss != "abstain"].sel_logloss.value_counts()
        if len(counts):
            fam, k = counts.index[0], int(counts.iloc[0])
        else:
            fam, k = "abstain", 0
        ok = fam != "abstain" and k >= 4
        stable += ok
        verdicts[(corpus, sname)] = fam if ok else "abstain"
        if fam != "abstain":
            other = "tree" if fam == "linear" else "linear"
            extra = (f"preferred cAUC {g[f'cond_{fam}'].mean():.4f} "
                     f"(vs {g[f'cond_{other}'].mean():.4f})  "
                     f"|z| {g.z_logloss.abs().min():.1f}-"
                     f"{g.z_logloss.abs().max():.1f}")
        else:
            extra = ""
        nab = int((g.sel_logloss == "abstain").sum())
        out.append(f"     {corpus:<22} {sname:<13} {fam} {k}/{len(g)} folds "
                   f"(abstain {nab})  {extra}  "
                   f"{'stable' if ok else 'UNRESOLVED'}")
    out.append(f"  R5 {stable}/6 instances stable  "
               f"--> {'PASS' if stable >= 5 else 'FAIL'}")

    if validation_real is not None:
        flips = []
        for (corpus, sname), g in validation_real.groupby(["corpus", "score"]):
            counts = g[g.sel_logloss != "abstain"].sel_logloss.value_counts()
            v48 = counts.index[0] if len(counts) and counts.iloc[0] >= 4 \
                else "abstain"
            vr = verdicts.get((corpus, sname), "missing")
            if vr != v48:
                flips.append((corpus, sname, v48, vr))
        for f in flips:
            out.append(f"     FLIP: {f[0]} {f[1]}: validation={f[2]} "
                       f"replication={f[3]}")
        out.append(f"  R6 verdict flips vs the validation run: {len(flips)}  "
                   f"--> {'PASS' if not flips else 'FAIL'}")
    else:
        out.append(f"  R6 skipped: {VALIDATION_REAL} table not found")
    return out


def report_calibration(d: pd.DataFrame) -> list[str]:
    out = ["=" * 92, "  Selective-resolution calibration (C1-C4; nested prefixes)",
           "=" * 92]
    tabs, cal = calibration_tables(d)
    for r in R_GRID:
        t = tabs[r]
        n_inf = int((t.oracle_gap >= GAP_MIN).sum())
        n_dec_inf = int(((t.oracle_gap >= GAP_MIN) & t.decided).sum())
        out.append(f"  r={r:<3} decided {int(t.decided.sum())}/60 overall; "
                   f"informative (oracle) {n_dec_inf}/{n_inf}; "
                   f"harmful confident decisions {int(t.harmful.sum())}")

    h12, h24 = int(tabs[12].harmful.sum()), int(tabs[24].harmful.sum())
    c1 = h12 == 0 and h24 == 0
    out.append(f"  C1 harmful confident decisions r=12: {h12}, r=24: {h24}  "
               f"--> {'PASS' if c1 else 'FAIL'}")
    if not c1:
        for r in (12, 24):
            for _, row in tabs[r][tabs[r].harmful].reset_index().iterrows():
                out.append(f"     r={r} HARMFUL: {row.regime} a={row.alpha} "
                           f"n={int(row.n_sub)} {row.corpus} "
                           f"verdict={row.verdict} best={row.oracle_best} "
                           f"Delta={row.oracle_gap:.4f}")

    t24 = tabs[24]
    inf24 = t24[t24.oracle_gap >= GAP_MIN]
    m_abs = float(inf24[~inf24.decided].oracle_gap.median()) \
        if int((~inf24.decided).sum()) else float("nan")
    m_dec = float(inf24[inf24.decided].oracle_gap.median()) \
        if int(inf24.decided.sum()) else float("nan")
    c2 = m_abs <= 0.5 * m_dec if np.isfinite(m_abs) else True
    note = "" if np.isfinite(m_abs) else " (no informative abstentions)"
    out.append(f"  C2 informative-only median Delta: abstained {m_abs:.4f} vs "
               f"decided {m_dec:.4f} (need <= half){note}  "
               f"--> {'PASS' if c2 else 'FAIL'}")

    strong = t24[t24.oracle_gap >= 0.02]
    c3 = float(strong.decided.mean()) if len(strong) else float("nan")
    out.append(f"  C3 coverage on strongly informative (Delta >= 0.02): "
               f"{c3:.3f} ({int(strong.decided.sum())}/{len(strong)})  "
               f"--> {'PASS' if c3 >= 0.85 else 'FAIL'}")

    out.append("  C4 (descriptive) abstained informative conditions at r=24:")
    ab = inf24[~inf24.decided].reset_index()
    if not len(ab):
        out.append("     none")
    for _, row in ab.iterrows():
        g = d[(d.corpus == row.corpus) & (d.regime == row.regime)
              & (d.alpha == row.alpha) & (d.n_sub == row.n_sub)]
        gap = g.ll_linear - g.ll_tree
        se = gap.std(ddof=1) / np.sqrt(len(gap))
        out.append(f"     {row.regime:<10} a={row.alpha} n={int(row.n_sub):<5} "
                   f"{row.corpus:<22} Delta={row.oracle_gap:.4f} "
                   f"ll gap {gap.mean():+.4f} "
                   f"[{gap.mean() - 1.96 * se:+.4f},{gap.mean() + 1.96 * se:+.4f}]"
                   f" mean|z|={row.mean_abs_z:.2f}")

    out.append("  secondary diagnostics (global calibration of mean |z|):")
    a1 = float(roc_auc_score((t24.oracle_gap >= GAP_MIN).astype(int),
                             t24.mean_abs_z))
    out.append(f"     A1 AUROC mean|z| vs Delta >= {GAP_MIN} (all 60): {a1:.3f}")
    med_abs_all = float(t24[~t24.decided].oracle_gap.median())
    med_dec_all = float(t24[t24.decided].oracle_gap.median())
    out.append(f"     A2 all-conditions median Delta abstained {med_abs_all:.4f}"
               f" vs decided {med_dec_all:.4f}")
    out.append("     A3 == C3 above")

    conv = cal[~cal.decided_r6 & cal.decided_r24]
    if len(conv):
        out.append("  conditions converted abstain->decided by r=24 "
                   "(oracle gap shown):")
        for _, r in conv.iterrows():
            out.append(f"     {r.regime:<10} a={r.alpha} n={int(r.n_sub):<5} "
                       f"{r.corpus:<22} Delta={r.oracle_gap:.4f} "
                       f"-> {r.verdict_r24}")
    else:
        out.append("  no conditions converted abstain->decided by r=24")
    return out


def _validation_real(tables: dict) -> pd.DataFrame | None:
    """The validation run's real-data table: passed in, else the reference copy."""
    if VALIDATION_REAL in tables:
        return tables[VALIDATION_REAL]
    if (reference_dir() / f"{VALIDATION_REAL}.csv").exists():
        return read_table(reference_dir(), VALIDATION_REAL)
    return None


def summarize(tables: dict) -> str:
    out = []
    d = tables.get("adjudication_repl_semisynth")
    if d is not None:
        out += report_confirmatory(d)
        if d.rep.max() + 1 >= N_REP_FULL:
            out += report_calibration(d)
    if "adjudication_repl_real" in tables:
        out += report_real_repl(tables["adjudication_repl_real"],
                                _validation_real(tables))
    return "\n".join(out)
