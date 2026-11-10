"""Brute-force Monte Carlo verification of the theory's statements.

Simulates the Gaussian latent-index model directly (no quadrature code shared
with `theory_law`) and checks T1-T5, L1, P1 and P2 of
`nuisauc.theory.montecarlo`; each check asserts its claim. One generator
(seed `mc_seed`) is passed through the checks in the order T1, T2, T3/T4, T5,
L1, P1, P2.

Output table `theory_montecarlo`, one row per checked quantity:
    check     T1 | T2 | T3 | T4 | T5 | L1 | P1 | P2
    item      what the value is (cell coordinates included)
    value     the Monte Carlo (or matrix-formula) value
    target    what theory says it should be (closed form, quadrature value
              from `theory_bias_check*`, 1/2, 0, or the predicted covariance);
              empty where the claim is a monotonicity, sign or size statement
The quadrature tables are read from `input_dir`; when it is null, from the
run directory if `theory_law` wrote them there, otherwise from the reference.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nuisauc import paths
from nuisauc.io import read_table, write_table
from nuisauc.seeds import record_run
from nuisauc.theory.montecarlo import (MC_SEED, check_L1, check_P1, check_P2, check_T1,
                                       check_T2, check_T3_T4, check_T5)

NAME = "theory_montecarlo"
DESCRIPTION = "Brute-force Monte Carlo checks of the correlation law and its corollaries"
PAPER = "App A, B"
REQUIRES = "none"
OUTPUTS = ["theory_montecarlo"]

T2_RHOS = (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9)     # the rho grid of check_T2
NAN = float("nan")

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"mc_seed": MC_SEED, "input_dir": None}


def _input(cfg: dict, out_dir: Path, name: str) -> tuple[pd.DataFrame, Path]:
    if cfg.get("input_dir"):
        d = Path(cfg["input_dir"])
    elif (Path(out_dir) / f"{name}.csv").exists():
        d = Path(out_dir)
    else:
        d = paths.reference_dir()
    log.info("reading %s from %s", name, d)
    return read_table(d, name), d / f"{name}.csv"


def _cell(alpha: float, lam: float) -> str:
    return f"alpha={alpha:g},lam={lam:g}"


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    quad, p1 = _input(cfg, out_dir, "theory_bias_check")
    quad3, p3 = _input(cfg, out_dir, "theory_bias_check_familyIII")
    rng = np.random.default_rng(cfg["mc_seed"])
    rows: list[dict] = []

    def add(check: str, item: str, value: float, target: float = NAN) -> None:
        rows.append(dict(check=check, item=item, value=float(value), target=float(target)))

    t1 = check_T1(rng)
    add("T1", "rho_formula", t1["rho"])
    add("T1", "decile_corr_min", t1["decile_corr_min"], t1["rho"])
    add("T1", "decile_corr_max", t1["decile_corr_max"], t1["rho"])
    add("T1", "residual_corr", t1["residual_corr"], t1["rho"])

    t2 = check_T2(rng)
    for (m, v_l), vals in t2.items():
        for r, v in zip(T2_RHOS, vals):
            add("T2", f"auc,m={m:g},v_L={v_l:g},rho={r:g}", v, 0.5 if r == 0.0 else NAN)

    t4, t3 = check_T3_T4(rng, quad)
    for _, r in t4.iterrows():
        c = _cell(r.alpha, r.lam)
        add("T4", f"rho_true_matrix,{c}", r.rho_true_mat, r.rho_true_cf)
        add("T4", f"rho_mis_matrix,{c}", r.rho_mis_mat, r.rho_mis_cf)
        add("T4", f"theta_true,{c}", r.mc_true, r.quad_true)
        add("T4", f"theta_mis,{c}", r.mc_mis, r.quad_mis)
        add("T4", f"sign_ok,{c}", float(bool(r.sign_ok)), 1.0)
    add("T3", "theta_random_weights,alpha=2,lam=1", t3["theta_random_weights"])
    add("T3", "rho_mis,alpha=2,lam=1", t3["rho_mis"])

    t5 = check_T5(rng, quad3)
    for _, r in t5.iterrows():
        c = _cell(r.alpha, r.lam)
        add("T5", f"theta_true,{c}", r.mc_true, r.quad_true)
        add("T5", f"theta_mis,{c}", r.mc_mis, r.quad_mis)
        add("T5", f"rho_mis_matrix,{c}", r.rho_mis, r.rho_mis_script)

    l1 = check_L1(rng)
    add("L1", "exact_level_sets", l1["exact"], 0.5)
    for q in (10, 20, 100):
        add("L1", f"quantile_bins,q={q}", l1[q], 0.5)

    p1 = check_P1(rng)
    add("P1", "max_within_stratum_balance_diff", p1["max_balance_diff"])
    add("P1", "median_bin_width", p1["bin_width"])
    add("P1", "control", p1["control"], 0.5)
    add("P1", "theta_correct_strata", p1["theta_true"])
    add("P1", "theta_eA_strata", p1["theta_eA"])
    add("P1", "max_imbalance_wrong_form", p1["max_imbalance_wrong_form"])

    for r in check_P2(rng):
        c = f"k_inc={r['k_inc']},k_exc={r['k_exc']},rho={r['rho']:g}"
        add("P2", f"cov_inc,{c}", r["cov_inc"], 0.0)
        add("P2", f"cov_exc,{c}", r["cov_exc"], r["predicted_exc"])

    out = pd.DataFrame(rows, columns=["check", "item", "value", "target"])
    write_table(out, out_dir, "theory_montecarlo")
    record_run(out_dir, NAME, cfg, inputs={"theory_bias_check": str(p1),
                                           "theory_bias_check_familyIII": str(p3)})
    return {"theory_montecarlo": out}


def summarize(tables: dict) -> str:
    d = tables["theory_montecarlo"]
    lines = ["all assertions passed (T1-T5, L1; P1/P2 are reported, not asserted)"]
    for chk, g in d.groupby("check", sort=False):
        t = g.dropna(subset=["target"])
        if chk == "T2":
            lines.append(f"  T2 AUC(rho) increasing in all {g.shape[0] // len(T2_RHOS)} strata; "
                         f"max |AUC(0) - 1/2| {(t.value - t.target).abs().max():.4f}")
        elif len(t):
            lines.append(f"  {chk} max |value - target| {(t.value - t.target).abs().max():.4g} "
                         f"over {len(t)} quantities")
        else:
            lines.append(f"  {chk} " + "; ".join(f"{i} {v:.4f}" for i, v in zip(g.item, g.value)))
    return "\n".join(lines)
