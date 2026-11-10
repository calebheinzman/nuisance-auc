"""Manuscript audit: recompute the paper's headline numbers from the result
tables, format them as printed, and check that the sentence or table row that
should contain each one occurs in the manuscript source.

A check fails if a recomputed number differs from what the manuscript says, or
if the manuscript no longer contains the statement. Expected values are never
hard-coded: they come from the tables, and the text comes from the manuscript.
Run with `nuisauc audit`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nuisauc import paths
from nuisauc.io import read_table

def flatten(path: Path) -> str:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*\\input\{([^}]+)\}", line)
        if m:
            sub = path.parent / (m.group(1) if m.group(1).endswith(".tex")
                                 else m.group(1) + ".tex")
            out.append(flatten(sub))
        elif not line.lstrip().startswith("%"):
            out.append(line)
    return "\n".join(out)


def norm(s: str) -> str:
    s = s.replace("\\%", "%").replace("~", " ").replace("âˆ’", "-")
    s = re.sub(r"\\\\", " ", s)              # row ends
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)      # command names
    s = re.sub(r"[{}$]", "", s)
    return re.sub(r"\s+", " ", s).strip()






def f3(v):
    return f"{v:.3f}"


def s3(v):
    return f"{v:+.3f}"


def passes(degen, ctl):
    return (degen <= 0.20) & (ctl >= 0.45) & (ctl <= 0.55)


def checks(T: Path, MS: str) -> list[tuple[str, str, str]]:
    """Run every check; returns (name, phrase, status) rows."""
    rows: list[tuple[str, str, str]] = []

    def say(name, phrase):
        rows.append((name, phrase, "ok" if norm(phrase) in MS else "NOT IN MANUSCRIPT"))

    # ---- Section 4: checks miss material errors
    d = read_table(T, "estimand")
    sig = d[~d.pure].copy()
    sig["err"] = (sig.est - sig.theta_B).abs()
    bad = sig[sig.err > 0.05]
    f = read_table(T, "screen_families")
    n2 = f[(f.family == "nonlinear") & (f.fit == "mis") & (f.err > 0.05)]
    n3 = f[(f.family == "correlated") & (f.fit == "mis") & (f.err > 0.05)]
    say("Family I material cells passing", f"{int(bad.screen.sum())} of {len(bad)} material cells pass")
    say("Family I worst error", f"errs by up to {bad.err.max():.4f}")
    assert n2.screen.all() and n3.screen.all()
    say("Families II/III material cells (all pass)",
        f"pass in all {len(n2)} and {len(n3)} material cells")
    tot_pass = int(bad.screen.sum()) + len(n2) + len(n3)
    say("Families I-III", f"{tot_pass}/{len(bad) + len(n2) + len(n3)} pass overall")

    # ---- Section 3 / Table 2: deterministic three-regime grid
    s = read_table(T, "semisynthetic_regimes_stable")
    n = s[s.fit != "oracle"]
    pred = np.where(n.variant == "same", 1, -1)
    cm = n.groupby(["corpus", "variant", "lam", "fit"]).signed.mean()
    cp = np.where(cm.index.get_level_values("variant") == "same", 1, -1)
    assert (np.sign(cm.values) == cp).all()
    say("direction agreement",
        f"{int((np.sign(n.signed) == pred).sum())} of {len(n)} replicate cells and in all "
        f"{len(cm)} condition means, and {int(n.screen.sum())} cells pass both checks")
    rev = {}
    for lam in (3, 2):
        r = n[(n.fit == "lin16") & (n.variant == "opp") & (n.lam == lam)]
        rev[lam] = int(((r.est < 0.5) & (r.theta_B > 0.5)).sum())
    say("reversals, lambda=3", f"{rev[3]} of 50 replicates")
    say("reversals, lambda=2", f"at $\\lambda=2$ in {rev[2]} of 50")
    a = n[(n.corpus.str.startswith("AFB")) & (n.fit == "lin16") & (n.variant == "opp") & (n.lam == 3)]
    say("AFB lambda=3", f"mean {f3(a.est.mean())} against a truth of {f3(a.theta_B.mean())}")
    order = ["AFB__internal_81", "Kaggle__local_128", "Wellgen__local_128", "malaria", "pcam"]
    tr = n[(n.fit == "lin16") & (n.variant == "opp") & (n.lam == 3)].groupby("corpus").theta_B.mean()
    say("Table 2 truth row", "truth (oracle strata), AUC & against, $\\lambda=3$ & "
        + " & ".join(f3(tr[c]) for c in order))
    for fit, label in (("lin16", "logistic, 16 features"), ("gbm62", "boosted trees, 62 features")):
        for var, lam, lab in (("opp", 3, "against, $\\lambda=3$"), ("same", 2, "with, $\\lambda=2$")):
            g = n[(n.fit == fit) & (n.variant == var) & (n.lam == lam)]
            cells = []
            for c in order:
                gc_ = g[g.corpus == c]
                k = int(((gc_.est < 0.5) & (gc_.theta_B > 0.5)).sum())
                cells.append(s3(gc_.signed.mean()) + (f" [{k}/10]" if k else ""))
            say(f"Table 2 {fit} {var}", (label + " & " if lab.startswith("against") else " & ")
                + lab + " & " + " & ".join(cells))
    h = read_table(T, "semisynthetic_regimes_historical")
    hn = h[h.fit != "oracle"]
    hp = np.where(hn.variant == "same", 1, -1)
    hr = hn[(hn.fit == "lin16") & (hn.variant == "opp")]
    say("historical counts (App. G)",
        f"{int((np.sign(hn.signed) == hp).sum())} of 600 non-oracle cells")
    say("historical reversals (App. G)",
        f"in {int(((hr[hr.lam == 3].est < 0.5) & (hr[hr.lam == 3].theta_B > 0.5)).sum())} of 50 replicates")

    # ---- oracle-bin sensitivity
    ob = read_table(T, "oracle_bins")
    obp = np.where(ob.variant == "same", 1, -1)
    ag = {q: int((np.sign(ob.est - ob[f"truth{q}"]) == obp).sum()) for q in (50, 200)}
    say("oracle bins", f"replicate direction agreement is {ag[50]} and {ag[200]} of 600")

    # ---- Table 4: real-data panel
    p = read_table(T, "panel_folds")
    z = p[p.detector == "zero_shot"].copy()
    for t in ("C1", "C2"):
        z[f"{t}p"] = passes(z[f"{t}_degen"], z[f"{t}_control"])
    z["gap"] = z.C1 - z.C2
    mean = z.groupby("corpus")[["raw_auc", "C1", "C2", "gap"]].mean()
    b = read_table(T, "panel_boot")
    bz = b[b.detector == "zero_shot"].assign(g=lambda v: v.C1 - v.C2)
    q = bz.groupby("corpus").g.quantile([0.025, 0.975]).unstack()
    fm = z.groupby("corpus")[["C1_degen", "C1_control", "C2_degen", "C2_control"]].mean()
    sp = z.groupby("corpus")[["C1p", "C2p"]].sum()
    jg = z[z.C1p & z.C2p].groupby("corpus").gap.agg(["mean", "count"])
    gr = read_table(T, "panel_grouped")
    gr = gr[gr.detector == "zero_shot"].pivot_table(index=["corpus", "rep"], columns="model",
                                                    values="cauc")
    gg = (gr.C1 - gr.C2).groupby("corpus").mean()
    names = {"AFB__internal_81": "AFB", "Kaggle__local_128": "Kaggle", "pcam": "PCam",
             "Wellgen__local_128": "Wellgen", "malaria": "malaria"}
    for c, nm in names.items():
        chk = "/".join("pass" if passes(fm.loc[c, f"{t}_degen"], fm.loc[c, f"{t}_control"]) else "fail"
                       for t in ("C1", "C2"))
        grp = s3(gg[c]) if c in gg.index and np.isfinite(gg[c]) else "undefined"
        say(f"Table 4 row {nm}",
            f"{nm} & {f3(mean.loc[c, 'raw_auc'])} & {f3(mean.loc[c, 'C1'])} & {f3(mean.loc[c, 'C2'])} & "
            f"{s3(mean.loc[c, 'gap'])} [{f3(q.loc[c, 0.025])}, {f3(q.loc[c, 0.975])}] & {chk} & "
            f"{int(sp.loc[c, 'C1p'])}/{int(sp.loc[c, 'C2p'])} & "
            f"{s3(jg.loc[c, 'mean'])} ({int(jg.loc[c, 'count'])}) & {grp}")

    # ---- Table 5: fit diagnostics
    ab = read_table(T, "abs_balance").groupby(["corpus", "model"]).mean(numeric_only=True)
    dg = z.groupby("corpus")[["C1_ece", "C2_ece", "C1_cross", "C2_cross"]].mean()
    for c, nm in names.items():
        say(f"Table 5 row {nm}",
            f"{nm} & {ab.loc[(c, 'C1'), 'mean_abs_smd']:.2f}/{ab.loc[(c, 'C2'), 'mean_abs_smd']:.2f} "
            f"[{ab.loc[(c, 'C1'), 'null_mean_abs_smd']:.2f}/{ab.loc[(c, 'C2'), 'null_mean_abs_smd']:.2f}] & "
            f"{f3(dg.loc[c, 'C1_ece'])}/{f3(dg.loc[c, 'C2_ece'])} & {f3(dg.loc[c, 'C1_cross'])} & "
            f"{f3(dg.loc[c, 'C2_cross'])}")
    ex = (ab.mean_abs_smd - ab.null_mean_abs_smd).unstack()
    say("excess imbalance", f"(excess {ex.C2.min():.2f}--{ex.C2.max():.2f}), while C1's exceeds it by "
        f"{ex.C1.min():.2f}--{ex.C1.max():.2f}")

    # ---- bins table and text
    bs = read_table(T, "bins_sensitivity").assign(gap=lambda v: v.C1 - v.C2)
    gq = bs.groupby(["corpus", "q"]).agg(gap=("gap", "mean"), p1=("C1_pass", "sum"), p2=("C2_pass", "sum"))
    for c, nm in names.items():
        say(f"bins table row {nm}", f"{nm} & " + " & ".join(
            f"{s3(gq.loc[(c, k), 'gap'])} ({int(gq.loc[(c, k), 'p1'])}/{int(gq.loc[(c, k), 'p2'])})"
            for k in (5, 10, 20, 40)))
    afb = gq.loc["AFB__internal_81", "gap"]
    say("bins text AFB range", f"gap at {afb.min():.3f}--{afb.max():.3f} on AFB")
    for t in ("C1", "C2"):
        bs[f"{t}_ratio"] = bs[f"{t}_pairshare"] * bs.q
    pr_ = bs.groupby(["corpus", "q"])[["C1_ratio", "C2_ratio"]].mean()
    say("within-bin pair share", f"make up {pr_.min().min():.2f} to {pr_.max().max():.2f} times $1/q$")

    # ---- ties
    ti = read_table(T, "ties")
    mx = max((ti[f"{t}_{v}"] - ti[f"{t}_default"]).abs().max()
             for t in ("C1", "C2") for v in ("eqmass_w", "tiebreak"))
    say("ties/weights bound", f"by at most {math.ceil(mx * 1e4) / 1e4:.4f}")

    # ---- colour shift and held-out null
    inj = read_table(T, "injected_shortcut")
    folds = []
    for m in ("C1", "C2"):
        e = inj.assign(err=inj[f"{m}_blind"] - inj[f"oracle_{m}"],
                       ok=passes(inj[f"{m}_blind_degen"], inj[f"{m}_blind_control"]), model=m)
        folds.append(e[e.cond != "ctrl"])
    e = pd.concat(folds)
    c = e.groupby(["corpus", "delta", "cond", "detector", "model"]).agg(err=("err", "mean"), d=("d_load", "mean"))
    pr = np.sign(c.d) * np.where(c.index.get_level_values("cond") == "with", 1, -1)
    say("colour shift direction", f"in {int((np.sign(c.err) == pr).sum())} of {len(c)} label-dependent cells")
    big = e[e.err.abs() > 0.02]
    say("colour shift checks", f"{int(big.ok.sum())} of the {len(big)} fold evaluations")
    sn = read_table(T, "sff_null")
    say("held-out null", f"rejects on {100 * (sn.exc > 0.0167).mean():.1f}% of halvings")
    return rows


def main(manuscript: Path | None = None, tables: Path | None = None) -> int:
    ms = manuscript or Path("manuscript/main.tex")
    if not ms.exists():
        raise SystemExit(f"manuscript not found: {ms} (use --manuscript)")
    rows = checks(Path(tables or paths.reference_dir()), norm(flatten(ms)))
    w = max(len(r[0]) for r in rows)
    fails = [r for r in rows if r[2] != "ok"]
    print(f"manuscript: {ms}\n")
    for r in rows:
        print(f"{r[2]:<18} {r[0]:<{w}}  | {r[1][:90]}")
    print(f"\n{len(rows) - len(fails)}/{len(rows)} checks ok")
    return 1 if fails else 0
