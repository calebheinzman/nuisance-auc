"""Reading, writing and comparing result tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def write_table(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    p = Path(out_dir) / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def read_table(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / f"{name}.csv")


@dataclass
class Comparison:
    name: str
    equal: bool
    max_abs_diff: float = 0.0
    problems: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "identical" if self.equal else "DIFFERENT"
        extra = f" (max |diff| {self.max_abs_diff:.3g})" if self.max_abs_diff else ""
        lines = [f"{self.name}: {status}{extra}"]
        lines += [f"  - {p}" for p in self.problems]
        return "\n".join(lines)


def compare_tables(new: pd.DataFrame, ref: pd.DataFrame, name: str = "",
                   atol: float = 1e-9) -> Comparison:
    """Compare two tables column by column, row by row (in file order).

    Numeric columns must agree within `atol` (NaN equals NaN); other columns
    must be identical after converting to strings.
    """
    problems: list[str] = []
    if list(new.columns) != list(ref.columns):
        problems.append(f"columns differ: {list(new.columns)} vs {list(ref.columns)}")
        return Comparison(name, False, np.nan, problems)
    if len(new) != len(ref):
        problems.append(f"row count {len(new)} vs {len(ref)}")
        return Comparison(name, False, np.nan, problems)
    worst = 0.0
    for c in ref.columns:
        a, b = new[c], ref[c]
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            both_nan = np.isnan(av) & np.isnan(bv)
            d = np.where(both_nan, 0.0, np.abs(av - bv))
            d = np.where(np.isnan(d), np.inf, d)
            m = float(d.max()) if len(d) else 0.0
            worst = max(worst, m)
            if m > atol:
                problems.append(f"column {c}: max |diff| {m:.3g} in {int((d > atol).sum())} rows")
        elif not (a.astype(str).values == b.astype(str).values).all():
            n = int((a.astype(str).values != b.astype(str).values).sum())
            problems.append(f"column {c}: {n} rows differ")
    return Comparison(name, not problems, worst, problems)
