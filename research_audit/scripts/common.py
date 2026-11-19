"""Shared helpers for the claims audit (read-only: these scripts never write to results/).

Run every script from the repository root, e.g.
    python -B research_audit/scripts/claims_repair.py

Conventions (the same as the paper's tests):
* paired two-sided Wilcoxon signed-rank test over seeds (scipy defaults, zero
  differences dropped);
* Holm correction across the datasets (or units) of one comparison;
* "better / level / worse" = Holm-adjusted p < 0.05 and the sign of the mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def wilcox_p(d, alternative="two-sided") -> float:
    d = np.asarray(d, float)
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(d, alternative=alternative).pvalue)


def paired(df, a, b, col, units, unit_col="dataset", method_col="method", lower_better=False):
    """Per-unit paired test of a vs b on ``col`` over seeds; Holm across ``units``.

    Returns a frame with the mean difference (a - b), a 95% t-interval, raw and
    Holm p, the number of seeds where a is better, and a verdict for a.
    """
    rows = []
    for u in units:
        x = df[(df[unit_col] == u) & (df[method_col] == a)].set_index("seed")[col]
        y = df[(df[unit_col] == u) & (df[method_col] == b)].set_index("seed")[col]
        s = x.index.intersection(y.index)
        d = (x[s] - y[s]).to_numpy(float)
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        q = tdist.ppf(0.975, n - 1) if n > 1 else np.nan
        dz = d.mean() / d.std(ddof=1) if n > 1 and d.std(ddof=1) > 0 else np.nan
        better = (d < 0) if lower_better else (d > 0)
        rows.append({"unit": u, "n": n, "a": x[s].mean(), "b": y[s].mean(), "diff": d.mean(),
                     "lo": d.mean() - q * se, "hi": d.mean() + q * se, "dz": dz,
                     "p": wilcox_p(d), "a_better_seeds": int(better.sum())})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    sign = -1 if lower_better else 1
    out["verdict"] = np.where(out.p_holm < 0.05,
                              np.where(sign * out["diff"] > 0, "better", "worse"), "level")
    return out


def tally(out) -> str:
    v = out.verdict.value_counts()
    b, l, w = v.get("better", 0), v.get("level", 0), v.get("worse", 0)
    return (f"{b} better / {l} level / {w} worse; better: {list(out[out.verdict == 'better'].unit)}"
            f"; worse: {list(out[out.verdict == 'worse'].unit)}")


def rng_str(values, fmt="{:.2f}") -> str:
    v = np.asarray(list(values), float)
    v = v[np.isfinite(v)]
    return f"{fmt.format(v.min())}–{fmt.format(v.max())}" if len(v) else "n/a"


def section(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
