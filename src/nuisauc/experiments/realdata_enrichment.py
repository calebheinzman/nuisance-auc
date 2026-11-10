"""Enriching the descriptor, and a sensitivity bound for the ordinal claims.

Part A: C4 = 62-feature descriptor + frequency16 + morphology16. Screen it,
check that it absorbs its own added families (held-out balance, HOB), measure
the conditional-AUC change C3 -> C4 for every detector arm.

Part B: Gamma* = sqrt(S+/S-) per ordinal claim -- the smallest pair-weight
distortion an unmeasured within-stratum confounder needs to flip the claim --
beside Gamma_obs, the distortion the measured omitted families induce.

Part C: in a Gaussian generative model, does small Gamma* mark the cells where
the estimated ordering is wrong?
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nuisauc.checks import cond_and_screen
from nuisauc.data.cache import CORPORA, corpus_cache, descriptor62, nuisance_models
from nuisauc.data.descriptors import frequency16, morph16
from nuisauc.detectors.selftrain import pseudo, student
from nuisauc.detectors.vlm import VLMS, embed_cache, prompt_embeddings, score
from nuisauc.diagnostics.heldout import hob
from nuisauc.estimand import n_strata, pair_weighted, qbins
from nuisauc.io import read_table, write_table
from nuisauc.propensity import fit_simple, propensity
from nuisauc.seeds import record_run

NAME = "realdata_enrichment"
DESCRIPTION = "Enriched descriptor C4, held-out balance, and a pair-weight sensitivity bound"
PAPER = "§5, App C"
REQUIRES = "cache"
OUTPUTS = ["remedy", "remedy_hob", "remedy_gamma", "remedy_gammaobs", "remedy_sim"]

N_SPLIT = 5
K_PSEUDO, N_ITER = 20, 2
HOB_NULL = 0.0238           # null threshold of HOB from the held-out-balance calibration
GAMMA_CAP = 1e6             # stands in for +inf when ranking
Q_FINE = 100

# Part C constants
SIM_SEED = 3700
SIM_N = 20_000
SIM_ALPHAS = [0.5, 1.0, 2.0]
SIM_LAMBDAS = [0.0, 1.0, 2.0]
SIM_REPS = 10
SIM_CONDS = ["correct", "omitted_available"]
G2_U, G2_CORE = 0.9, 0.6
R1_MIN = 4                  # corpora on which C4 must pass the screen
R2_MIN = 8                  # of 10 included-family cells absorbed
R4_AUC = 0.75
R4_MIN_WRONG = 10

log = logging.getLogger(f"nuisauc.{NAME}")


# ---------------------------------------------------------------- Gamma*

def pair_masses(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, e: np.ndarray,
                q: int) -> tuple[float, float]:
    """S+ and S- for the claim "s1 beats s2", within strata of e."""
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.digitize(e, edges[1:-1]), 0, q - 1)
    sp = sn = 0.0
    for k in range(q):
        m = b == k
        yk = y[m]
        if yk.sum() == 0 or (1 - yk).sum() == 0:
            continue
        a1, a2 = s1[m], s2[m]
        p, n = yk == 1, yk == 0
        d1 = ((a1[p][:, None] > a1[n][None, :]).astype(float)
              + 0.5 * (a1[p][:, None] == a1[n][None, :]))
        d2 = ((a2[p][:, None] > a2[n][None, :]).astype(float)
              + 0.5 * (a2[p][:, None] == a2[n][None, :]))
        delta = d1 - d2
        sp += float(delta[delta > 0].sum())
        sn += float(-delta[delta < 0].sum())
    return sp, sn


def gamma_star(sp: float, sn: float) -> tuple[int, float]:
    """(direction, Gamma*): direction is +1 when s1 wins, -1 when s2 wins."""
    if sp == sn:
        return 0, 1.0
    hi, lo = (sp, sn) if sp > sn else (sn, sp)
    g = float(np.sqrt(hi / lo)) if lo > 0 else float("inf")
    return (1 if sp > sn else -1), g


def med_gamma(vals) -> float:
    v = [min(g, GAMMA_CAP) for g in vals]
    return float(np.median(v))


# ---------------------------------------------------------------- Part C

def sim_cell(cond: str, alpha: float, lam: float, rep: int) -> dict:
    rng = np.random.default_rng([SIM_SEED, SIM_CONDS.index(cond),
                                 int(alpha * 10), int(lam * 10), rep])
    Z = rng.standard_normal((SIM_N, 6))
    u, e1, e2 = rng.standard_normal((3, SIM_N))
    if cond == "correct":
        core = Z[:, 0] + Z[:, 1]
    else:                                   # omitted_available
        core = Z[:, 4]
    logit = alpha * core + u
    y = (rng.random(SIM_N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    s1 = u + lam * (Z[:, 0] - Z[:, 1]) + e1
    s2 = G2_U * u + G2_CORE * core + e2

    # true direction: theta_B under fine stratification on core, full sample
    fb = qbins(core, Q_FINE)
    t1 = pair_weighted(y, s1, fb)[0]
    t2 = pair_weighted(y, s2, fb)[0]
    true_dir = 1 if t1 > t2 else -1

    # estimated direction and Gamma* under the boosted analysis model
    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=rep)
                       .split(np.zeros(len(y)), y)))
    m = fit_simple("gbm", Z[:, :4], y, tr, rep)
    e = m.predict_proba(Z[te, :4])[:, 1]
    sp, sn = pair_masses(y[te], s1[te], s2[te], e, n_strata(len(te)))
    est_dir, g = gamma_star(sp, sn)
    return dict(cond=cond, alpha=alpha, lam=lam, rep=rep,
                theta1=float(t1), theta2=float(t2),
                true_dir=true_dir, est_dir=est_dir,
                gamma=min(g, GAMMA_CAP),
                wrong=bool(est_dir != 0 and est_dir != true_dir))


def run_sim() -> pd.DataFrame:
    rows = []
    for cond in SIM_CONDS:
        for alpha in SIM_ALPHAS:
            for lam in SIM_LAMBDAS:
                for rep in range(SIM_REPS):
                    rows.append(sim_cell(cond, alpha, lam, rep))
            log.info("  sim %-20s alpha=%s done", cond, alpha)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Part A/B

def crossfit_logit(B: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """HOB's probe, but returning the logit predictions themselves."""
    p = np.empty(len(y), float)
    for tr, te in StratifiedKFold(4, shuffle=True,
                                  random_state=seed).split(B, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(B[tr], y[tr])
        p[te] = m.decision_function(B[te])
    return p


def gamma_obs(e: np.ndarray, y: np.ndarray, B: np.ndarray, q: int,
              seed: int) -> float:
    """Median over strata of exp(IQR of within-stratum probe logits)."""
    p = crossfit_logit(B, y, seed)
    edges = np.quantile(e, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.digitize(e, edges[1:-1]), 0, q - 1)
    iqrs = []
    for k in range(q):
        pk = p[b == k]
        if len(pk) >= 8:
            iqrs.append(float(np.subtract(*np.percentile(pk, [75, 25]))))
    return float(np.exp(np.median(iqrs))) if iqrs else float("nan")


def run_real(corpora: list[str], cap: int, dev: str):
    import torch
    hob_rows, theta_rows, gam_rows, obs_rows = [], [], [], []
    for corpus in corpora:
        y, _C, _pk = corpus_cache(corpus, cap)
        splits = [next(iter(StratifiedKFold(
            4, shuffle=True, random_state=r).split(np.zeros(len(y)), y)))
            for r in range(N_SPLIT)]
        D62 = descriptor62(corpus, cap)
        FF = frequency16(corpus, cap)
        MM = morph16(corpus, cap)
        C4 = np.concatenate([D62, FF, MM], axis=1)
        # frequency16 returns 18 columns despite its name (radial bands per
        # RGB channel + anisotropy); assert against actual widths
        assert C4.shape[1] == D62.shape[1] + FF.shape[1] + MM.shape[1]
        log.info("=== %s  n=%d ===", corpus, len(y))

        # ---- absorption and cross-family transfer
        for tag, X, fams in (
                ("C4", C4, [("frequency", FF, "included"),
                            ("morphology", MM, "included")]),
                ("62+freq", np.concatenate([D62, FF], axis=1),
                 [("morphology", MM, "cross")]),
                ("62+morph", np.concatenate([D62, MM], axis=1),
                 [("frequency", FF, "cross")])):
            for fam, B, role in fams:
                hs = []
                for rep, (tr, te) in enumerate(splits):
                    e = propensity("gbm", X, tr, y, rep).predict_proba(
                        X[te])[:, 1]
                    h, _u = hob(e, y[te], B[te], n_strata(len(te)))
                    hs.append(h)
                hv = float(np.nanmean(hs))
                hob_rows.append(dict(corpus=corpus, prop=tag, family=fam,
                                     role=role, hob=hv,
                                     fires=bool(hv > HOB_NULL)))
                log.info("  HOB[%-8s vs %-10s %-8s] = %.4f (%.1fx null)%s",
                         tag, fam, role, hv, hv / HOB_NULL,
                         "  FIRES" if hv > HOB_NULL else "")

        # ---- score arms
        sc = {}
        for v in VLMS:
            E, _ = embed_cache(v, corpus, cap, dev)
            T = prompt_embeddings(v, corpus, dev)
            Tp, Tn = T[f"{corpus}__pos"], T[f"{corpus}__neg"]
            z = score(E, Tp, Tn, np.arange(len(Tp)), np.arange(len(Tn)))
            sc[f"zs_{v}"] = z
            if v == "biomedclip":
                st = z
                for _ in range(N_ITER):
                    idx, pl = pseudo(st, K_PSEUDO)
                    st = student(E, idx, pl)
                sc["self_trained"] = st
                Eb = E
            else:
                del E
            gc.collect()

        rng = np.random.default_rng(37)
        probe = np.empty(len(y), float)
        for tr, te in StratifiedKFold(N_SPLIT, shuffle=True,
                                      random_state=36).split(Eb, y):
            pick = rng.permutation(tr)
            k = 64
            while k < len(pick) and len(np.unique(y[pick[:k]])) < 2:
                k += 1
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=3000))
            m.fit(Eb[pick[:k]], y[pick[:k]])
            probe[te] = m.decision_function(Eb[te])
        sc["probe_n64"] = probe

        # ---- theta + screen under C1..C4
        mods = nuisance_models(corpus, cap) + [("C4", C4, "gbm")]
        endorsed = {}
        for tag, C, kind in mods:
            vals, ok = {}, True
            for name, s in sc.items():
                c, passes = cond_and_screen(s, y, C, kind, splits)
                vals[name] = c
                theta_rows.append(dict(corpus=corpus, model=tag, arm=name,
                                       theta=c))
                if name == "zs_biomedclip":
                    ok = passes
            if ok:
                endorsed[tag] = vals
        log.info("  endorsed: %s", ", ".join(endorsed) or "none")

        # ---- Gamma* per endorsed model, median over folds
        for tag, C, kind in mods:
            if tag not in endorsed:
                continue
            best = max(VLMS, key=lambda v: endorsed[tag][f"zs_{v}"])
            claims = {
                "O1_probe_beats_zeroshot": ("probe_n64", [f"zs_{best}"]),
                "O2_selftrain_helps": ("self_trained", ["zs_biomedclip"]),
                "O3_biomedclip_best": ("zs_biomedclip",
                                       ["zs_plip", "zs_quiltnet"]),
            }
            for claim, (a1, rivals) in claims.items():
                per_fold = []
                for rep, (tr, te) in enumerate(splits):
                    e = propensity(kind, C, tr, y, rep).predict_proba(
                        C[te])[:, 1]
                    q = n_strata(len(te))
                    rr = [gamma_star(*pair_masses(y[te], sc[a1][te],
                                                  sc[a2][te], e, q))
                          for a2 in rivals]
                    # the claim holds on this fold only if EVERY rival is
                    # beaten; its Gamma* is the weakest rival margin. If any
                    # rival wins, the fold fails and Gamma* is the smallest
                    # distortion that would restore the claim.
                    losing = [g for d, g in rr if d <= 0]
                    if losing:
                        per_fold.append((-1, max(losing)))
                    else:
                        per_fold.append((1, min(g for _d, g in rr)))
                dirs = [d for d, _g in per_fold]
                gm = med_gamma([g for _d, g in per_fold])
                holds = sum(1 for d in dirs if d > 0)
                gam_rows.append(dict(corpus=corpus, model=tag, claim=claim,
                                     folds_holding=holds, gamma=gm))
                log.info("  Gamma*[%s] %-26s holds %d/%d folds  median=%.3f",
                         tag, claim, holds, N_SPLIT, gm)

        # ---- Gamma_obs anchors, C3 strata
        for fam, B in (("frequency", FF), ("morphology", MM)):
            gs = []
            for rep, (tr, te) in enumerate(splits):
                e = propensity("gbm", D62, tr, y, rep).predict_proba(
                    D62[te])[:, 1]
                gs.append(gamma_obs(e, y[te], B[te], n_strata(len(te)),
                                    3500 + rep))
            obs_rows.append(dict(corpus=corpus, family=fam,
                                 gamma_obs=float(np.nanmean(gs))))
            log.info("  Gamma_obs[%-10s] = %.3f", fam, np.nanmean(gs))

        del Eb, sc
        gc.collect()
        torch.cuda.empty_cache()
    return (pd.DataFrame(hob_rows), pd.DataFrame(theta_rows),
            pd.DataFrame(gam_rows), pd.DataFrame(obs_rows))


def default_config() -> dict:
    return {"corpora": list(CORPORA), "cap": 4000,
            "skip_real": False, "skip_sim": False}


def _read_if(out_dir: Path, name: str) -> pd.DataFrame:
    return (read_table(out_dir, name) if (Path(out_dir) / f"{name}.csv").exists()
            else pd.DataFrame())


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Parts C then A/B; a skipped part is read back from out_dir if present."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    SIM = pd.DataFrame()
    if not cfg["skip_sim"]:
        log.info("Part C: simulation validation of Gamma*")
        SIM = run_sim()
        write_table(SIM, out_dir, "remedy_sim")

    if not cfg["skip_real"]:
        H, TH, G, OBS = run_real(cfg["corpora"], cfg["cap"], dev)
        write_table(H, out_dir, "remedy_hob")
        write_table(TH, out_dir, "remedy")
        write_table(G, out_dir, "remedy_gamma")
        write_table(OBS, out_dir, "remedy_gammaobs")
    else:
        H = _read_if(out_dir, "remedy_hob")
        TH = _read_if(out_dir, "remedy")
        G = _read_if(out_dir, "remedy_gamma")
        OBS = _read_if(out_dir, "remedy_gammaobs")
    if cfg["skip_sim"]:
        SIM = _read_if(out_dir, "remedy_sim")

    record_run(out_dir, NAME, cfg)
    return {"remedy": TH, "remedy_hob": H, "remedy_gamma": G,
            "remedy_gammaobs": OBS, "remedy_sim": SIM}


def summarize(tables: dict) -> str:
    H, TH, G = tables["remedy_hob"], tables["remedy"], tables["remedy_gamma"]
    OBS, SIM = tables["remedy_gammaobs"], tables["remedy_sim"]
    out = []
    if len(H):
        # R1: a corpus passes when C4 produced Gamma* rows (endorsed)
        c4 = set(G[G.model == "C4"].corpus) if len(G) else set()
        r1 = len(c4) >= R1_MIN
        out.append(f"R1 C4 passes the screen on >= {R1_MIN} of 5 corpora")
        out.append(f"   endorsed on {len(c4)}: {', '.join(sorted(c4)) or '--'}"
                   f"   --> {'PASS' if r1 else 'FAIL'}")

        inc = H[(H.prop == "C4") & (H.role == "included")]
        n_abs = int((~inc.fires).sum())
        r2 = n_abs >= R2_MIN
        out.append(f"\nR2 C4 absorbs included families on >= {R2_MIN} of 10 cells")
        for _i, r in inc.iterrows():
            out.append(f"   {r.corpus:<22} {r.family:<10} HOB={r.hob:.4f}"
                       f"{'  STILL FIRES' if r.fires else '  absorbed'}")
        out.append(f"   --> {n_abs}/10  {'PASS' if r2 else 'FAIL'}")

        out.append("\nR3 cross-family transfer")
        for _i, r in H[H.role == "cross"].iterrows():
            out.append(f"   {r.corpus:<22} {r.prop:<9} vs {r.family:<10} "
                       f"HOB={r.hob:.4f} ({r.hob/HOB_NULL:.1f}x null)"
                       f"{'  FIRES' if r.fires else ''}")

        out.append("\nR3 theta span C3 -> C4 per arm")
        piv = TH.pivot_table(index=["corpus", "arm"], columns="model",
                             values="theta")
        if "C3" in piv.columns and "C4" in piv.columns:
            piv["span"] = piv["C4"] - piv["C3"]
            out.append(piv.round(3).to_string())

    if len(SIM):
        wrong = SIM[SIM.wrong]
        right = SIM[~SIM.wrong & (SIM.est_dir != 0)]
        out.append(f"\nR4 Gamma* separates wrong from right "
                   f"(need AUC >= {R4_AUC}, >= {R4_MIN_WRONG} wrong cells)")
        out.append(f"   wrong cells: {len(wrong)}   right cells: {len(right)}")
        for cond in SIM_CONDS:
            g = SIM[SIM.cond == cond]
            gw = g[g.wrong].gamma.median() if g.wrong.any() else float("nan")
            out.append(f"   {cond:<20} wrong {int(g.wrong.sum()):>3}/{len(g)}"
                       f"   median Gamma* wrong={gw:.3f}"
                       f" right={g[~g.wrong].gamma.median():.3f}")
        if len(wrong) >= R4_MIN_WRONG:
            lab = np.r_[np.ones(len(right)), np.zeros(len(wrong))]
            val = np.r_[right.gamma.to_numpy(), wrong.gamma.to_numpy()]
            auc = float(roc_auc_score(lab, val))
            out.append(f"   AUC(right > wrong) = {auc:.3f}   "
                       f"--> {'PASS' if auc >= R4_AUC else 'FAIL'}")
        else:
            out.append("   --> VOID (too few wrong cells); Gamma* is unvalidated")

    if len(G):
        out.append("\nR5 real-data Gamma* beside Gamma_obs")
        mx = OBS.groupby("corpus").gamma_obs.max()
        for corpus, g in G.groupby("corpus"):
            anchor = float(mx.get(corpus, float("nan")))
            out.append(f"   {corpus}  (max Gamma_obs = {anchor:.3f})")
            for _i, r in g.iterrows():
                verdict = ("SURVIVES" if r.gamma > anchor and
                           r.folds_holding == N_SPLIT else "")
                out.append(f"     {r.model:<4} {r.claim:<26} "
                           f"holds {int(r.folds_holding)}/{N_SPLIT} "
                           f"Gamma*={r.gamma:8.3f}  {verdict}")
    return "\n".join(out)
