"""Figures in the paper.

Figure 1 (`theory_law`): the correlation law against simulation for Families I
and III (left) and the two-property example as the detector loading grows
(right). Drawn from the `theory_law` experiment's tables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nuisauc.io import read_table


def theory_law(tables_dir: Path, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nuisauc.theory.law import family_I

    t1 = read_table(tables_dir, "theory_bias_check")
    t3 = read_table(tables_dir, "theory_bias_check_familyIII")
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))

    a = ax[0]
    a.plot([0.25, 0.8], [0.25, 0.8], color="0.6", lw=1, zorder=0)
    a.scatter(t1.theory_true, t1.sim_theta_B, marker="o", s=28, color="tab:blue",
              label="Family I, correct stratifier")
    a.scatter(t1.theory_mis, t1.sim_mis_est, marker="s", s=28, color="tab:red",
              label="Family I, omitted $Z_2$")
    a.scatter(t3.theory_true, t3.sim_theta_B, marker="o", s=28, facecolors="none",
              edgecolors="tab:blue", label="Family III, correct")
    a.scatter(t3.theory_mis, t3.sim_mis_est, marker="s", s=28, facecolors="none",
              edgecolors="tab:red", label="Family III, 6 of 12 used")
    a.axhline(0.5, color="0.8", lw=0.8, ls="--")
    a.axvline(0.5, color="0.8", lw=0.8, ls="--")
    a.set_xlabel("conditional AUC, predicted (quadrature)")
    a.set_ylabel("conditional AUC, simulation")
    a.set_xlim(0.25, 0.8)
    a.set_ylim(0.25, 0.8)
    a.legend(fontsize=6.5, loc="upper left", frameon=False)
    max_deviation = max((t1.theory_true - t1.sim_theta_B).abs().max(),
                        (t1.theory_mis - t1.sim_mis_est).abs().max(),
                        (t3.theory_true - t3.sim_theta_B).abs().max(),
                        (t3.theory_mis - t3.sim_mis_est).abs().max())
    a.set_title(f"36 cells, max deviation {max_deviation:.3f}", fontsize=9)

    b = ax[1]
    lams = np.linspace(0, 2.5, 41)
    for alpha, col in zip([0.5, 1.0, 2.0], ["tab:green", "tab:orange", "tab:purple"]):
        b.plot(lams, [family_I(alpha, lam)[3] for lam in lams], color=col, lw=1.6,
               label=rf"reported, $\alpha={alpha:g}$")
        b.plot(lams, [family_I(alpha, lam)[1] for lam in lams], color=col, lw=1.0, ls=":")
        sub = t1[t1.alpha == alpha]
        b.scatter(sub.lam, sub.sim_mis_est, color=col, s=26, zorder=3)
        b.axvline(1.0 / alpha, color=col, lw=0.6, ls="--", alpha=0.6)
    b.axhline(0.5, color="0.5", lw=0.8)
    b.set_xlabel(r"score loading on the omitted direction, $\lambda$")
    b.set_ylabel("conditional AUC")
    b.set_title(r"solid: omitted-$Z_2$ model; dotted: truth; dashed: $\alpha\lambda=1$",
                fontsize=8)
    b.legend(fontsize=6.5, frameon=False, loc="lower left")
    b.set_ylim(0.25, 0.72)

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_all(out_dir: Path, tables_dir: Path) -> list[Path]:
    return [theory_law(tables_dir, Path(out_dir) / "fig_theory_law.pdf")]
