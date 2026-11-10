"""Label-fit selection between the linear and tree nuisance-model classes.

On the training fold, inner 4-fold cross-fitted log-loss (or Brier score)
picks the best member of each class and the better class. A rep abstains
unless the paired per-tile log-loss difference between the two chosen members
has |z| >= 2; a condition's verdict is the strict majority among adjudicating
reps, provided at least half the reps adjudicate.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from nuisauc.propensity import CLASSES, fit_any

GAP_MIN = 0.01        # a condition is informative when class errors differ this much
N_SUBS = [0, 1000]    # audit sizes: 0 = full corpus; 1000 = small-audit regime


def crossfit_scores(X: np.ndarray, y: np.ndarray, tr: np.ndarray,
                    seed: int) -> dict:
    """Cross-fitted per-tile losses per candidate on the training fold.

    Returns {tag: (log_loss, brier, per_tile_logloss_vector)}.
    """
    out = {}
    probs = {t: np.empty(len(tr), float)
             for f in CLASSES.values() for t in f}
    for itr, ite in StratifiedKFold(4, shuffle=True,
                                    random_state=1000 + seed
                                    ).split(np.zeros(len(tr)), y[tr]):
        a_itr = tr[itr]
        for tag in probs:
            m = fit_any(tag, X, y, a_itr, seed)
            probs[tag][ite] = m.predict_proba(X[tr[ite]])[:, 1]
    yt = y[tr]
    for tag, p in probs.items():
        p = np.clip(p, 1e-7, 1 - 1e-7)
        pt = -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
        out[tag] = (float(pt.mean()), float(brier_score_loss(yt, p)), pt)
    return out


def select_family(scores: dict, idx: int) -> tuple[str, str, dict, float]:
    """Pick a class from `crossfit_scores` output; idx 0 = log-loss, 1 = Brier.

    Returns (verdict class or 'abstain', best member of the selected class,
    {class: (best member, loss)}, z) where z is the paired per-tile z-statistic
    (linear - tree) of the log-loss difference. Abstains when |z| < 2.
    """
    fam_best = {}
    for fam, tags in CLASSES.items():
        tag = min(tags, key=lambda t: scores[t][idx])
        fam_best[fam] = (tag, scores[tag][idx])
    fam = min(fam_best, key=lambda f: fam_best[f][1])
    dl = (scores[fam_best["linear"][0]][2]
          - scores[fam_best["tree"][0]][2])
    se = dl.std(ddof=1) / np.sqrt(len(dl))
    z = float(dl.mean() / se) if se > 0 else 0.0
    verdict = fam if abs(z) >= 2.0 else "abstain"
    return verdict, fam_best[fam][0], fam_best, z


def majority_verdict(sels) -> str:
    """Condition verdict from per-rep selections with abstentions."""
    sels = list(sels)
    n = len(sels)
    adj = [x for x in sels if x != "abstain"]
    if len(adj) < 0.5 * n:
        return "abstain"
    nl, nt = adj.count("linear"), adj.count("tree")
    if nl == nt:
        return "abstain"
    return "linear" if nl > nt else "tree"


def family_dl(scores: dict) -> tuple[np.ndarray, str, str]:
    """Per-tile log-loss difference: linear member minus tree member."""
    lin = min(("C1", "C1e"), key=lambda t: scores[t][0])
    tre = min(("C2", "C2r"), key=lambda t: scores[t][0])
    return scores[lin][2] - scores[tre][2], lin, tre
