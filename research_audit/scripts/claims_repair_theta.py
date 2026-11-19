"""Recompute every number the paper takes from the repair-at-theta-val run (seeds 70-89).

The run, its plan and its own summary are in ``results/repair_theta/``
(PREREGISTRATION.md, runs.csv, summary.md). This script does not use the run's summary
code: it redoes each test with ``common.paired`` (paired two-sided Wilcoxon over seeds,
Holm across the 13 datasets, verdict for the first model), and checks its verdict counts
against the ones in summary.md.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_repair_theta.py > research_audit/scripts/claims_repair_theta_out.txt
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from common import RES, paired, rng_str, section, tally

sys.stdout.reconfigure(encoding="utf-8")

DIR = RES / "repair_theta"
runs = pd.read_csv(DIR / "runs.csv")
readable = pd.read_csv(RES / "readable_baselines" / "runs.csv")

NAMES = {"S1": "S1", "S2": "S2", "S3": "S3", "wine": "Wine", "breast_cancer": "Breast Cancer",
         "digits": "Digits", "thyroid": "Thyroid", "shuttle": "Shuttle", "skin": "Skin",
         "magic": "MAGIC", "letter": "Letter", "landsat": "Landsat", "page_blocks": "Page Blocks"}
DATASETS = list(NAMES)
MODELS = ["B0-0.3", "SR-0.3", "B0-val", "SR-val"]
runs["dataset"] = pd.Categorical(runs.dataset, DATASETS, ordered=True)

section("The run is complete")
print("rows", len(runs), "| expected", 13 * 20 * 4)
print("datasets", runs.dataset.nunique(), "| seeds", sorted(int(s) for s in runs.seed.unique()))
print("models", sorted(runs.model.unique()))
print("duplicate (dataset, seed, model):", int(runs.duplicated(["dataset", "seed", "model"]).sum()))
print("missing accuracy or macro-F1:", int(runs[["accuracy", "macro_f1"]].isna().sum().sum()))
log = (DIR / "run_log.txt").read_text(encoding="utf-8").splitlines()
print(log[0])
print([line for line in log if line.startswith("Finished")])
print("lines with FAIL or Traceback:", sum(("FAIL" in line.upper()) or ("Traceback" in line) for line in log))

section("Check: the unrepaired models reproduce the readable-baselines run")
ref = readable[readable.model.isin(["CFMNN-0.3", "CFMNN-val"])].copy()
ref["model"] = ref.model.map({"CFMNN-0.3": "B0-0.3", "CFMNN-val": "B0-val"})
b0 = runs[runs.model.str.startswith("B0")].merge(
    ref[["dataset", "seed", "model", "accuracy", "macro_f1", "setting"]],
    on=["dataset", "seed", "model"], suffixes=("", "_ref"))
print("B0 rows matched to a readable-baselines row:", len(b0), "of", int(runs.model.str.startswith("B0").sum()))
print("max |accuracy difference|:", float((b0.accuracy - b0.accuracy_ref).abs().max()))
print("max |macro-F1 difference|:", float((b0.macro_f1 - b0.macro_f1_ref).abs().max()))
print("theta differs from the readable-baselines setting:", int((b0.theta - b0.setting).abs().gt(1e-12).sum()))

section("Mean test accuracy and macro-F1 (%) over the 20 seeds")
acc = runs.pivot_table(index="dataset", columns="model", values="accuracy", observed=True)[MODELS]
f1 = runs.pivot_table(index="dataset", columns="model", values="macro_f1", observed=True)[MODELS]
print("accuracy"); print(acc.rename(index=NAMES).round(2).to_string())
print("macro-F1"); print(f1.rename(index=NAMES).round(2).to_string())

COMPARISONS = [
    ("P1", "SR-0.3", "B0-val", "accuracy"),
    ("P2", "SR-val", "B0-val", "accuracy"),
    ("S1", "SR-0.3", "B0-0.3", "accuracy"),
    ("S2", "SR-val", "SR-0.3", "accuracy"),
    ("S3-P1", "SR-0.3", "B0-val", "macro_f1"),
    ("S3-P2", "SR-val", "B0-val", "macro_f1"),
    ("S3-S1", "SR-0.3", "B0-0.3", "macro_f1"),
    ("S3-S2", "SR-val", "SR-0.3", "macro_f1"),
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

section("Verdict counts (better / level / worse for the first model), and the run's summary.md")
summary = (DIR / "summary.md").read_text(encoding="utf-8")
theirs = [tuple(int(x) for x in m) for m in re.findall(r"better (\d+), level (\d+), worse (\d+)", summary)]
mine = []
for code, a, b, col in COMPARISONS:
    v = results[code].verdict.value_counts()
    mine.append((int(v.get("better", 0)), int(v.get("level", 0)), int(v.get("worse", 0))))
    print(f"{code:6s} {a} vs {b} ({col}): {mine[-1][0]} / {mine[-1][1]} / {mine[-1][2]}")
print("summary.md gives the same counts in the same order:", mine == theirs)

section("Where each comparison is better or worse: the mean differences (points)")
for code in results:
    out = results[code]
    for verdict in ("better", "worse"):
        sel = out[out.verdict == verdict]
        if len(sel):
            print(f"{code:6s} {verdict:6s} {len(sel):2d}: {rng_str(sel['diff'], '{:+.2f}')} "
                  f"{dict(zip(sel.unit, sel['diff'].round(2)))}")

section("Seeds where theta-val is 0.3 (the two theta give the same model there)")
val = runs[runs.model == "B0-val"]
same = val[val.theta == 0.3].groupby("dataset", observed=True).size()
print(same.rename(index=NAMES).to_string(), "| total", int(same.sum()))
d = runs.pivot_table(index=["dataset", "seed"], columns="model", values="accuracy", observed=True)
th = val.set_index(["dataset", "seed"]).theta
print("on those seeds, max |B0-val - B0-0.3| and |SR-val - SR-0.3|:",
      float((d["B0-val"] - d["B0-0.3"])[th == 0.3].abs().max()),
      float((d["SR-val"] - d["SR-0.3"])[th == 0.3].abs().max()))

section("Boxes: means, the ratio between the two theta, and change under repair")
boxes = runs.pivot_table(index="dataset", columns="model", values="boxes", observed=True)[MODELS]
boxes["val / 0.3"] = boxes["B0-val"] / boxes["B0-0.3"]
print(boxes.rename(index=NAMES).round(2).to_string())
grew = (runs.pivot_table(index=["dataset", "seed"], columns="model", values="boxes", observed=True))
print("pairs where repair added boxes, at 0.3 and at val:",
      int((grew["SR-0.3"] > grew["B0-0.3"]).sum()), int((grew["SR-val"] > grew["B0-val"]).sum()))
small_theta = [ds for ds in DATASETS if (val[val.dataset == ds].theta <= 0.1).sum() > 10]
print("datasets where theta-val is 0.1 or less in most seeds:", [NAMES[ds] for ds in small_theta])
print("box ratio on those:", rng_str(boxes.loc[small_theta, "val / 0.3"], "{:.1f}"))

section("Training errors, and the share inside the winning wrong-class box (the ones repair can cut)")
te = runs.pivot_table(index="dataset", columns="model", values="train_errors", observed=True)[MODELS]
print(te.rename(index=NAMES).round(2).to_string())
b0rows = runs[runs.model.str.startswith("B0")]
print("B0 rows where every training error is inside a winning wrong-class box:",
      int((b0rows.train_errors_in_winning_box == b0rows.train_errors).sum()), "of", len(b0rows))
print("B0 rows where every training error also has a full box of its own class (a cross-class tie):",
      int((b0rows.train_errors_own_box_full == b0rows.train_errors).sum()), "of", len(b0rows),
      "| B0 rows with at least one training error:", int((b0rows.train_errors > 0).sum()))
print("total training errors over the B0 rows:", int(b0rows.train_errors.sum()))
none = [NAMES[ds] for ds in DATASETS if te.loc[ds, "B0-0.3"] < 1]
print("datasets with a mean of less than one training error at theta = 0.3:", none)
pct = runs.pivot_table(index="dataset", columns="model", values="train_error_pct", observed=True)
print("training error % at B0-0.3 and B0-val:")
print(pct[["B0-0.3", "B0-val"]].rename(index=NAMES).round(3).to_string())

section("Repair: edits accepted, test predictions fixed and broken (means), seconds")
sr = runs[runs.model.str.startswith("SR")]
rep = sr.pivot_table(index="dataset", columns="model",
                     values=["edits_logged", "edits_already_right", "edits_accepted",
                             "test_fixed", "test_broken", "seconds"], observed=True)
print(rep.rename(index=NAMES).round(1).to_string())
print("budget reached (300 edits logged) in pairs, at 0.3 and at val:",
      int(((sr.model == "SR-0.3") & (sr.edits_logged == 300)).sum()),
      int(((sr.model == "SR-val") & (sr.edits_logged == 300)).sum()))
print("budget reached, by dataset and model:",
      {f"{NAMES[k[0]]} {k[1]}": v for k, v in sr[sr.edits_logged == 300].groupby(["dataset", "model"], observed=True).size().items()})
rq3 = ["thyroid", "skin", "magic", "letter", "landsat", "page_blocks"]
s1f = results["S3-S1"].set_index("unit").loc[[NAMES[d] for d in rq3]]
print("S3-S1 (macro-F1, repair at 0.3) on RQ3's six datasets:", list(s1f.verdict), rng_str(s1f["diff"], "{:+.2f}"))
mins = sr.groupby(["dataset", "model"], observed=True).seconds.mean().unstack() / 60
print("repair minutes per seed, mean (six runs shared the CPU):")
print(mins.rename(index=NAMES).round(1).to_string())

section("Gap to the better of the tuned kNN and the Random Forest (points, per-dataset means)")
ref_acc = readable.pivot_table(index="dataset", columns="model", values="accuracy")[["kNN", "Random Forest"]]
best = ref_acc.max(axis=1).reindex(DATASETS)
gap = acc.sub(best, axis=0)
print(gap.rename(index=NAMES).round(2).to_string())
for m in MODELS:
    print(f"{m} gap: {rng_str(gap[m], '{:.2f}')}; within 0.5 points on "
          f"{int((gap[m] >= -0.5).sum())} datasets {list(gap.index[gap[m] >= -0.5].map(NAMES))}")
