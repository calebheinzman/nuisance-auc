"""The correlation law by quadrature, against the Family I and Family III simulations.

For every (alpha, lam) cell of the Family I simulation (table `estimand`) and
the Family III simulation (the "correlated" family of table `screen_families`)
this evaluates the law's truth and misspecified-stratifier values by
Gauss-Hermite quadrature (no simulation) and sets them beside the simulated
means. Family I reverses (adjusted AUC below 1/2) iff alpha * lam > 1.

The simulated tables are read from `input_dir`; when it is null, from the run
directory if `family_i` / `families_ii_iii` have written their tables there,
otherwise from the reference tables.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nuisauc import paths
from nuisauc.io import read_table, write_table
from nuisauc.seeds import record_run
from nuisauc.theory.law import family_I, family_III

NAME = "theory_law"
DESCRIPTION = "Correlation law by quadrature vs the Family I and Family III simulations"
PAPER = "§3, Tab 1, Fig 1"
REQUIRES = "none"
OUTPUTS = ["theory_bias_check", "theory_bias_check_familyIII"]

SIGN_TOL = 0.01       # sign agreement is only scored where |theory_err| > SIGN_TOL

log = logging.getLogger(f"nuisauc.{NAME}")


def default_config() -> dict:
    return {"input_dir": None}


def _input(cfg: dict, out_dir: Path, name: str) -> tuple[pd.DataFrame, Path]:
    """Read a simulation table from input_dir, else the run dir, else the reference."""
    if cfg.get("input_dir"):
        d = Path(cfg["input_dir"])
    elif (Path(out_dir) / f"{name}.csv").exists():
        d = Path(out_dir)
    else:
        d = paths.reference_dir()
    log.info("reading %s from %s", name, d)
    return read_table(d, name), d / f"{name}.csv"


def family_I_table(d: pd.DataFrame) -> pd.DataFrame:
    sig = d[~d.pure]
    rows = []
    for (alpha, lam), g in sig.groupby(["alpha", "lam"]):
        rho_t, th_t, rho_m, th_m = family_I(alpha, lam)
        sim_truth = g.theta_B.mean()
        sim_cor = g[g.fit == "correct"].est.mean()
        sim_mis = g[g.fit == "misspecified"].est.mean()
        sim_mis_sd = g[g.fit == "misspecified"].est.std()
        rows.append(dict(alpha=alpha, lam=lam, alpha_lam=alpha * lam,
                         rho_true=rho_t, theory_true=th_t, sim_theta_B=sim_truth,
                         sim_correct_est=sim_cor,
                         rho_mis=rho_m, theory_mis=th_m, sim_mis_est=sim_mis,
                         sim_mis_sd=sim_mis_sd,
                         reversal_predicted=bool(alpha * lam > 1),
                         reversal_observed=bool(sim_mis < 0.5),
                         theory_err=th_m - th_t, sim_err=sim_mis - sim_truth))
    return pd.DataFrame(rows)


def family_III_table(f: pd.DataFrame) -> pd.DataFrame:
    g3 = f[f.family == "correlated"]
    rows = []
    for (alpha, lam), g in g3.groupby(["alpha", "lam"]):
        rho_t, th_t, rho_m, th_m, extra = family_III(alpha, lam)
        rows.append(dict(alpha=alpha, lam=lam, rho_true=rho_t, theory_true=th_t,
                         sim_theta_B=g.theta_B.mean(),
                         sim_correct_est=g[g.fit == "correct"].est.mean(),
                         rho_mis=rho_m, theory_mis=th_m,
                         sim_mis_est=g[g.fit == "mis"].est.mean(),
                         theory_err=th_m - th_t,
                         sim_err=g[g.fit == "mis"].est.mean() - g.theta_B.mean(),
                         hidden_R2_absorbed=extra["absorbed_R2"]))
    return pd.DataFrame(rows)


def run(cfg: dict, out_dir: Path) -> dict[str, pd.DataFrame]:
    d, p1 = _input(cfg, out_dir, "estimand")
    out1 = family_I_table(d)
    write_table(out1, out_dir, "theory_bias_check")
    f, p3 = _input(cfg, out_dir, "screen_families")
    out3 = family_III_table(f)
    write_table(out3, out_dir, "theory_bias_check_familyIII")
    record_run(out_dir, NAME, cfg, inputs={"estimand": str(p1), "screen_families": str(p3)})
    return {"theory_bias_check": out1, "theory_bias_check_familyIII": out3}


def _sign_agree(out: pd.DataFrame) -> int:
    return int(((np.sign(out.sim_err) == np.sign(out.theory_err))
                | (out.theory_err.abs() <= SIGN_TOL)).sum())


def summarize(tables: dict) -> str:
    out = tables["theory_bias_check"]
    out3 = tables["theory_bias_check_familyIII"]
    with pd.option_context("display.width", 200):
        lines = [out.round(4).to_string(index=False), "",
                 "max |theory_true - sim_theta_B| : "
                 f"{round(float((out.theory_true - out.sim_theta_B).abs().max()), 4)}",
                 "max |theory_mis  - sim_mis_est| : "
                 f"{round(float((out.theory_mis - out.sim_mis_est).abs().max()), 4)}",
                 "reversal predicted == observed  : "
                 f"{int((out.reversal_predicted == out.reversal_observed).sum())} of {len(out)}",
                 f"sign(sim_err) == sign(theory_err) where |theory_err|>{SIGN_TOL}: "
                 f"{_sign_agree(out)} of {len(out)}",
                 "", "Family III (correlated, 6 of 12 columns seen)",
                 out3.round(4).to_string(index=False),
                 "max |theory_true - sim_theta_B| : "
                 f"{round(float((out3.theory_true - out3.sim_theta_B).abs().max()), 4)}",
                 "max |theory_mis  - sim_mis_est| : "
                 f"{round(float((out3.theory_mis - out3.sim_mis_est).abs().max()), 4)}",
                 f"sign agreement (|theory_err|>{SIGN_TOL}): {_sign_agree(out3)} of {len(out3)}"]
    return "\n".join(lines)
