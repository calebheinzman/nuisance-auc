"""Held-out balance (HOB): a specification test from features the model never saw.

Fit the propensity e on features A; ask whether held-out features B still
predict y within strata of e(A). If e captured the label-relevant nuisance, no
function of the nuisance -- including B -- predicts y within strata.

Simulation (table `<table>`): six Gaussian columns; a ladder of propensities
L1 (2 columns, logistic), L2 (2 columns, boosted), L3 (4 columns, boosted);
B = columns 4-5. Conditions place the label driver in a used column (linear or
non-linear), in B, or nowhere observable. Thresholds for HOB and for the
ladder spread are 95th percentiles over correctly specified calibration cells.

Real corpora (table `<table>_real`): HOB of the frequency family against the
62-feature boosted propensity, mean over five repeated splits.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nuisauc.data.cache import CORPORA, corpus_cache, descriptor62
from nuisauc.data.descriptors import frequency16
from nuisauc.diagnostics.heldout import hob, null_threshold
from nuisauc.estimand import conditional_auc, n_strata, pair_weighted, qbins
from nuisauc.io import write_table
from nuisauc.propensity import fit_simple, propensity
from nuisauc.seeds import record_run
from nuisauc.simulate.gaussian import Q_FINE

NAME = "heldout_hob"
DESCRIPTION = "Held-out balance test: simulation validation and real-corpus values"
PAPER = "App D"
REQUIRES = "cache"
OUTPUTS = ["heldout_balance_b", "heldout_balance_b_real"]

N = 20_000
N_REP = 10
ALPHAS = [0.5, 1.0, 2.0]
LAMBDAS = [0.0, 1.0, 2.0]
BETA = GAMMA = 1.0
SEED = 3500
BAD_TOL = 0.05
H1_SENS, H1_LADDER = 0.80, 0.20
H2_FPR = 0.10
H3_MAX = 0.20
N_CAL = 20                  # null-calibration replicates, disjoint from eval
CONDITIONS = ["correct", "functional", "omitted_available",
              "omitted_unavailable"]
LADDER = [("L1", 2, "linear"), ("L2", 2, "gbm"), ("L3", 4, "gbm")]
N_HELD = 2                  # columns 4, 5 are the held-out set B
REAL_CAP = 4000
REAL_SPLITS = 5

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"n_rep": N_REP, "skip_real": False, "table": "heldout_balance_b"}


def simulate(cond: str, alpha: float, lam: float, rng: np.random.Generator):
    """Z has six columns. L3 uses 0-3; B is 4-5."""
    Z = rng.standard_normal((N, 6))
    u, eps = rng.standard_normal((2, N))
    if cond == "correct":
        core = Z[:, 0] + Z[:, 1]
    elif cond == "functional":
        core = np.abs(Z[:, 0]) - 0.8
    elif cond == "omitted_available":
        core = Z[:, 4]
    elif cond == "omitted_unavailable":
        core = rng.standard_normal(N)
    else:
        raise ValueError(cond)
    logit = alpha * core + BETA * u
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    s = GAMMA * u + lam * (Z[:, 0] - Z[:, 1]) + eps
    return Z, y, s, core


def run_cell(cond: str, alpha: float, lam: float, rep: int) -> dict:
    rng = np.random.default_rng([SEED, CONDITIONS.index(cond),
                                 int(alpha * 10), int(lam * 10), rep])
    Z, y, s, core = simulate(cond, alpha, lam, rng)
    tB = pair_weighted(y, s, qbins(core, Q_FINE))[0]

    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                       .split(np.zeros(len(y)), y)))
    q = n_strata(len(te))
    ests = {}
    for name, ncol, kind in LADDER:
        X = Z[:, :ncol]
        m = fit_simple(kind, X, y, tr, rep)
        e = m.predict_proba(X[te])[:, 1]
        c, _ = conditional_auc(y[te], s[te], e, q)
        ests[name] = (float(c), e)

    vals = np.array([v[0] for v in ests.values()])
    spread = float(vals.max() - vals.min())
    err = abs(ests["L3"][0] - tB)

    # HOB uses the top rung's propensity and the columns it never saw
    e_top = ests["L3"][1]
    B_te = Z[te, 4:4 + N_HELD]
    h, uncond = hob(e_top, y[te], B_te, q)

    return dict(cond=cond, alpha=alpha, lam=lam, rep=rep, theta_B=float(tB),
                est_L3=ests["L3"][0], err=float(err),
                bad=bool(err > BAD_TOL), spread=spread, hob=h,
                b_probe_auc=uncond)


def real_corpora(cap: int = REAL_CAP) -> pd.DataFrame:
    """Is the 62-feature descriptor missing label-relevant nuisance?"""
    out = []
    for corpus in CORPORA:
        y, _C, _pk = corpus_cache(corpus, cap)
        D62 = descriptor62(corpus, cap)
        F = frequency16(corpus, cap)
        splits = [next(iter(StratifiedKFold(
            4, shuffle=True, random_state=r).split(np.zeros(len(y)), y)))
            for r in range(REAL_SPLITS)]
        hs, us = [], []
        for rep, (tr, te) in enumerate(splits):
            m = propensity("gbm", D62, tr, y, rep)
            e = m.predict_proba(D62[te])[:, 1]
            h, u = hob(e, y[te], F[te], n_strata(len(te)))
            hs.append(h); us.append(u)
        out.append(dict(corpus=corpus, hob=float(np.nanmean(hs)),
                        b_probe_auc=float(np.nanmean(us))))
        log.info("%s HOB=%.4f held-out probe raw AUC=%.4f", corpus,
                 np.nanmean(hs), np.nanmean(us))
        gc.collect()
    return pd.DataFrame(out)


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for cond in CONDITIONS:
        reps = N_CAL + cfg["n_rep"] if cond == "correct" else cfg["n_rep"]
        for alpha in ALPHAS:
            for lam in LAMBDAS:
                for rep in range(reps):
                    r = run_cell(cond, alpha, lam, rep)
                    r["split"] = ("cal" if (cond == "correct" and rep < N_CAL)
                                  else "eval")
                    rows.append(r)
            log.info("%s alpha=%s done", cond, alpha)
    d = pd.DataFrame(rows)
    tables = {cfg["table"]: d}
    write_table(d, out_dir, cfg["table"])

    if not cfg["skip_real"]:
        log.info("held-out balance on the real corpora")
        real = real_corpora()
        tables[f"{cfg['table']}_real"] = real
        write_table(real, out_dir, f"{cfg['table']}_real")
    record_run(out_dir, NAME, cfg)
    return tables


def summarize(tables: dict) -> str:
    d = next(v for k, v in tables.items() if not k.endswith("_real"))
    real = next((v for k, v in tables.items() if k.endswith("_real")),
                pd.DataFrame())
    cal = d[d.split == "cal"]
    ev = d[d.split == "eval"]
    t_hob = null_threshold(cal.hob)
    t_lad = null_threshold(cal.spread)
    out = [f"null-calibrated thresholds ({len(cal)} correctly-specified "
           f"calibration cells): HOB {t_hob:.4f}, ladder {t_lad:.4f}",
           f"  {'condition':<24}{'cells':>7}{'HOB mean':>11}"
           f"{'HOB fires':>12}{'ladder fires':>15}"]
    for c in CONDITIONS:
        g = ev[ev.cond == c]
        if not len(g):
            continue
        out.append(f"  {c:<24}{len(g):>7}{g.hob.mean():>11.4f}"
                   f"{float((g.hob > t_hob).mean()):>12.2f}"
                   f"{float((g.spread > t_lad).mean()):>15.2f}")

    oa = ev[ev.cond == "omitted_available"]
    fh = float((oa.hob > t_hob).mean())
    fl = float((oa.spread > t_lad).mean())
    h1 = fh >= H1_SENS and fl <= H1_LADDER
    out.append(f"omitted_available: HOB fires >= {H1_SENS}, ladder <= {H1_LADDER}: "
               f"{len(oa)} cells, HOB {fh:.2f}, ladder {fl:.2f} -> "
               f"{'pass' if h1 else 'fail'}")
    null_ev = ev[ev.cond.isin(["correct", "functional"])]
    fpr = float((null_ev.hob > t_hob).mean())
    h2 = fpr <= H2_FPR
    out.append(f"false positives on correct + functional <= {H2_FPR}: "
               f"{len(null_ev)} cells, HOB fires {fpr:.3f} -> "
               f"{'pass' if h2 else 'fail'}")
    ou = ev[ev.cond == "omitted_unavailable"]
    f3 = float((ou.hob > t_hob).mean())
    h3 = f3 <= H3_MAX
    out.append(f"boundary on omitted_unavailable <= {H3_MAX} (predicted failure "
               f"to detect): {len(ou)} cells, HOB fires {f3:.2f} -> "
               f"{'as predicted' if h3 else 'not as predicted'}")
    if not real.empty:
        out.append(f"real corpora against threshold {t_hob:.4f}")
        for _i, r in real.iterrows():
            mult = r.hob / t_hob if t_hob else float("nan")
            out.append(f"  {r.corpus:<22} HOB={r.hob:.4f} ({mult:.1f}x null)  "
                       f"probe raw AUC={r.b_probe_auc:.4f}"
                       f"{'   fires' if r.hob > t_hob else ''}")
    out.append(f"overall {'pass' if (h1 and h2 and h3) else 'fail'}")
    return "\n".join(out)
