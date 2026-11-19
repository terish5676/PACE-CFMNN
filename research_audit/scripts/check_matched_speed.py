"""Check the paper's claim (Section 7, abstract, contributions) that, given the same rows,
KernelSHAP took 3.5-10x as long as PACE-SHAP.

The claim comes from the conditional-replacement run (seeds 40-49), where KernelSHAP used the
same 200 background rows as PACE-SHAP, every feature and 200 coalitions. The times are
milliseconds per explanation from that run's ``runs.csv``. In that run each paired method's
time is the model calls it shares with its pair plus its own part
(``scripts/run_faithfulness.py``), and other runs shared the laptop, so the ratios are rough.

Read-only. Run from the repository root:
    python -B research_audit/scripts/check_matched_speed.py > research_audit/scripts/check_matched_speed_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, rng_str, section

sys.stdout.reconfigure(encoding="utf-8")

runs = pd.read_csv(RES / "extensions_v1/conditional/runs.csv")
times = runs[runs.method.isin(["KernelSHAP", "PACE-SHAP"])]
print("seeds", sorted(int(s) for s in times.seed.unique()), "datasets", times.dataset.nunique())

# One ratio per dataset and seed.
per_seed = times.pivot_table(index=["dataset", "seed"], columns="method", values="ms_per_explanation")
ratio = per_seed["KernelSHAP"] / per_seed["PACE-SHAP"]

# Ratio of the per-dataset mean times (the paper's 3.5-10x).
means = times.groupby(["dataset", "method"]).ms_per_explanation.mean().unstack()
mean_ratio = means["KernelSHAP"] / means["PACE-SHAP"]

section("KernelSHAP time / PACE-SHAP time, same 200 rows, every feature, 200 coalitions")
table = pd.DataFrame({
    "PACE-SHAP ms (mean)": means["PACE-SHAP"],
    "KernelSHAP ms (mean)": means["KernelSHAP"],
    "ratio of means": mean_ratio,
    "median ratio": ratio.groupby(level="dataset").median(),
    "min ratio": ratio.groupby(level="dataset").min(),
    "max ratio": ratio.groupby(level="dataset").max(),
    "seeds": ratio.groupby(level="dataset").size(),
})
print(table.round(2).to_string())

section("Summary")
print("ratio of means, range over datasets:", rng_str(mean_ratio))
print("median ratio, range over datasets:", rng_str(ratio.groupby(level="dataset").median()))
print("per dataset-seed ratio, range:", rng_str(ratio))
print(f"KernelSHAP slower in {int((ratio > 1).sum())} of {len(ratio)} dataset-seed pairs")
