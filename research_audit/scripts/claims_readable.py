"""Recompute every number the paper takes from the readable-baselines run (seeds 70-89).

The run, its plan and its own summary are in ``results/readable_baselines/``
(PREREGISTRATION.md, runs.csv, summary.md). This script does not use the run's summary
code: it redoes each test with ``common.paired`` (paired two-sided Wilcoxon over seeds,
Holm across the 13 datasets, verdict for the first model), so the two can be compared.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_readable.py > research_audit/scripts/claims_readable_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, paired, rng_str, section, tally

sys.stdout.reconfigure(encoding="utf-8")

DIR = RES / "readable_baselines"
runs = pd.read_csv(DIR / "runs.csv")

NAMES = {"S1": "S1", "S2": "S2", "S3": "S3", "wine": "Wine", "breast_cancer": "Breast Cancer",
         "digits": "Digits", "thyroid": "Thyroid", "shuttle": "Shuttle", "skin": "Skin",
         "magic": "MAGIC", "letter": "Letter", "landsat": "Landsat", "page_blocks": "Page Blocks"}
DATASETS = list(NAMES)
MODELS = ["CFMNN-0.3", "CFMNN-val", "Tree", "LogReg", "EBM", "GFMM-0.3", "kNN", "Random Forest"]
runs["dataset"] = pd.Categorical(runs.dataset, DATASETS, ordered=True)

section("The run is complete")
print("rows", len(runs), "| expected", 13 * 20 * 8)
print("datasets", runs.dataset.nunique(), "| seeds", sorted(int(s) for s in runs.seed.unique()))
print("models", sorted(runs.model.unique()))
print("duplicate (dataset, seed, model):", int(runs.duplicated(["dataset", "seed", "model"]).sum()))
print("missing accuracy or macro-F1:", int(runs[["accuracy", "macro_f1"]].isna().sum().sum()))
log = (DIR / "run_log.txt").read_text(encoding="utf-8").splitlines()
print(log[0])
print([line for line in log if line.startswith("Finished")])
print("lines with FAIL:", sum("FAIL" in line.upper() for line in log))

section("Mean test accuracy (%) over the 20 seeds")
acc = runs.pivot_table(index="dataset", columns="model", values="accuracy", observed=True)[MODELS]
print(acc.rename(index=NAMES).round(1).to_string())

COMPARISONS = [
    ("P1", "CFMNN-val", "Tree", "accuracy"),
    ("P2", "CFMNN-val", "LogReg", "accuracy"),
    ("P3", "CFMNN-val", "EBM", "accuracy"),
    ("P4", "CFMNN-0.3", "GFMM-0.3", "accuracy"),
    ("S1a", "CFMNN-0.3", "Tree", "accuracy"),
    ("S1b", "CFMNN-0.3", "LogReg", "accuracy"),
    ("S1c", "CFMNN-0.3", "EBM", "accuracy"),
    ("S2", "CFMNN-val", "CFMNN-0.3", "accuracy"),
    ("S3-P1", "CFMNN-val", "Tree", "macro_f1"),
    ("S3-P2", "CFMNN-val", "LogReg", "macro_f1"),
    ("S3-P3", "CFMNN-val", "EBM", "macro_f1"),
    ("S3-P4", "CFMNN-0.3", "GFMM-0.3", "macro_f1"),
]
results = {}
for code, a, b, col in COMPARISONS:
    out = paired(runs, a, b, col, DATASETS, method_col="model")
    out["unit"] = out.unit.map(NAMES)
    results[code] = out
    section(f"{code}: {a} vs {b} ({col})")
    print(tally(out))
    print(out[["unit", "a", "b", "diff", "lo", "hi", "p", "p_holm", "a_better_seeds", "verdict"]]
          .round(4).to_string(index=False))

section("Verdict counts (better / level / worse for the first model)")
for code, a, b, col in COMPARISONS:
    v = results[code].verdict.value_counts()
    print(f"{code:6s} {a} vs {b} ({col}): {v.get('better', 0)} / {v.get('level', 0)} / {v.get('worse', 0)}")

section("S2: where theta by validation beat theta = 0.3, the theta it chose and its box count")
s2 = results["S2"].set_index("unit")
val = runs[runs.model == "CFMNN-val"]
fixed = runs[runs.model == "CFMNN-0.3"]
rows = []
for ds in DATASETS:
    chosen = val[val.dataset == ds].setting.value_counts().sort_index()
    boxes_val = val[val.dataset == ds].rules.mean()
    boxes_fix = fixed[fixed.dataset == ds].rules.mean()
    rows.append({"dataset": NAMES[ds], "verdict": s2.loc[NAMES[ds], "verdict"],
                 "diff": s2.loc[NAMES[ds], "diff"],
                 "theta chosen": ", ".join(f"{t:g}×{n}" for t, n in chosen.items()),
                 "boxes val": boxes_val, "boxes 0.3": boxes_fix, "ratio": boxes_val / boxes_fix})
s2t = pd.DataFrame(rows)
print(s2t.round(2).to_string(index=False))
better = s2t[s2t.verdict == "better"]
print("better on", len(better), ":", list(better.dataset))
print("gain where better:", rng_str(better["diff"]))
small = better[better.dataset != "S1"]
print("box ratio where better, S1 excluded (S1 chose larger theta):", rng_str(small.ratio, "{:.1f}"))
print("S1 box ratio:", round(float(better[better.dataset == "S1"].ratio.iloc[0]), 2))
n005 = {NAMES[ds]: int((val[val.dataset == ds].setting == 0.05).sum()) for ds in DATASETS}
print("seeds choosing theta = 0.05 (the smallest value tried):", n005)

section("Grid edges: datasets where a tuned model chose the end of its grid in most seeds")
GRID_END = {"CFMNN-val": (0.05, "smallest theta"), "Tree": (1024, "most leaves"),
            "LogReg": (100.0, "largest C"), "kNN": (1, "k = 1")}
for model, (edge, label) in GRID_END.items():
    m = runs[runs.model == model]
    hit = [NAMES[ds] for ds in DATASETS if (m[m.dataset == ds].setting == edge).sum() > 10]
    print(f"{model} at {label} ({edge:g}) in more than 10 of 20 seeds: {len(hit)} {hit}")

section("Gap to the better of the tuned kNN and the Random Forest (points, per-dataset means)")
best_ref = acc[["kNN", "Random Forest"]].max(axis=1)
gap = pd.DataFrame({"CFMNN-0.3": acc["CFMNN-0.3"] - best_ref, "CFMNN-val": acc["CFMNN-val"] - best_ref,
                    "best of kNN, RF": best_ref}).rename(index=NAMES)
print(gap.round(3).to_string())
print("CFMNN-0.3 gap:", rng_str(gap["CFMNN-0.3"], "{:.2f}"), "| CFMNN-val gap:", rng_str(gap["CFMNN-val"], "{:.2f}"))
print("datasets where CFMNN-0.3 mean >= best reference:", list(gap.index[gap["CFMNN-0.3"] >= 0]))
print("datasets where CFMNN-val mean >= best reference:", list(gap.index[gap["CFMNN-val"] >= 0]))

section("Sizes: mean rules (boxes, leaves, terms, non-zero weights) and conditions per rule")
size = runs.pivot_table(index="dataset", columns="model", values="rules", observed=True)
cond = runs.pivot_table(index="dataset", columns="model", values="conditions_per_rule", observed=True)
print("rules"); print(size[["CFMNN-0.3", "CFMNN-val", "GFMM-0.3", "Tree", "EBM", "LogReg"]].rename(index=NAMES).round(1).to_string())
print("conditions per rule"); print(cond[["CFMNN-0.3", "CFMNN-val", "GFMM-0.3", "Tree", "EBM"]].rename(index=NAMES).round(2).to_string())
print("Tree leaves, range of dataset means:", rng_str(size["Tree"], "{:.1f}"))
print("CFMNN-0.3 boxes, range of dataset means:", rng_str(size["CFMNN-0.3"], "{:.1f}"))
print("CFMNN-val boxes, range of dataset means:", rng_str(size["CFMNN-val"], "{:.1f}"))
print("Tree conditions per leaf, range of dataset means:", rng_str(cond["Tree"], "{:.2f}"))
# With main effects only, EBM has one term per feature, so its term count is the number of features.
share = cond["CFMNN-0.3"] / size["EBM"]
print("CFMNN-0.3 conditions per box as a share of the features, range:", rng_str(share, "{:.2f}"))
fewer = [NAMES[ds] for ds in DATASETS if size.loc[ds, "Tree"] < size.loc[ds, "CFMNN-0.3"]]
print(f"datasets where the tuned tree has fewer leaves than CFMNN-0.3 has boxes: {len(fewer)} {fewer}")
fewer_val = [NAMES[ds] for ds in DATASETS if size.loc[ds, "Tree"] < size.loc[ds, "CFMNN-val"]]
print(f"datasets where the tuned tree has fewer leaves than CFMNN-val has boxes: {len(fewer_val)} {fewer_val}")
