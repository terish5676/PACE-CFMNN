"""Recompute every number the write-up takes from Run L (Covertype, θ by validation, then self-repair).

Run L: results/covertype_theta/runs.csv (descriptive, seeds 210-214, a stratified 10% of
Covertype per seed, one task at a time alone, home PC). No test: means and standard deviations
over the 5 seeds, ranges over seeds, tie and centroid counts pooled over them. Gains are per-seed
differences, written +x. The CSV is read with float_precision="round_trip", so every value is the
one written.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_covertype_theta.py > research_audit/scripts/claims_covertype_theta_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, section

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

L = pd.read_csv(RES / "covertype_theta" / "runs.csv", float_precision="round_trip")
BAD = []
MODELS = ["B0-0.3", "SR-0.3", "B0-val", "SR-val"]


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def r(x, nd=1):
    return round(float(x), nd)


section("Completeness and setup")
check("rows", len(L), 20)
check("seeds", sorted(L.seed.unique().tolist()), list(range(210, 215)))
check("models", sorted(L.model.unique().tolist()), sorted(MODELS))
check("share of the rows", L.subsample.unique().tolist(), [0.1])
check("training and test rows", (int(L.n_train.iloc[0]), int(L.n_test.iloc[0])), (46480, 11621))
check("tuning on every training row", sorted(L.tune_rows.unique().tolist()), [46480])
check("θ chosen by validation, every seed", sorted(L[L.model == "B0-val"].theta.unique().tolist()), [0.05])
cv = L[L.model == "B0-val"].cv_scores.str.split(";").map(lambda s: [float(v) for v in s])
check("CV accuracy is highest at θ = 0.05, the end of the grid, on every seed",
      bool(all(max(range(len(s)), key=s.__getitem__) == len(s) - 1 for s in cv)), True)
check("CV accuracy at θ = 0.05 (%), range over seeds", (r(100 * min(s[-1] for s in cv)), r(100 * max(s[-1] for s in cv))),
      (72.4, 73.4))
print("CV accuracy per θ (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05), per seed:")
for seed, s in zip(L[L.model == "B0-val"].seed, cv):
    print(f"  {seed}: " + ", ".join(f"{100 * v:.1f}" for v in s))

section("The four models, mean and standard deviation over seeds")
cols = ["test_accuracy", "test_macro_f1", "boxes", "train_error_pct", "pace_flip_ms", "pace_shap_ms"]
m = L.groupby("model")[cols].agg(["mean", "std"]).loc[MODELS]
print(m.round(2).to_string())
check("accuracy means", [r(m.loc[x, ("test_accuracy", "mean")]) for x in MODELS], [50.7, 53.9, 72.9, 73.4])
check("macro-F1 means", [r(m.loc[x, ("test_macro_f1", "mean")]) for x in MODELS], [39.5, 41.4, 64.3, 64.8])
check("boxes, means", [r(m.loc[x, ("boxes", "mean")]) for x in MODELS], [82.8, 78.0, 3009.6, 3009.6])
check("training error (%), means", [r(m.loc[x, ("train_error_pct", "mean")]) for x in MODELS], [48.9, 45.6, 19.6, 18.9])
check("PACE-flip ms, means", [r(m.loc[x, ("pace_flip_ms", "mean")], 2) for x in MODELS], [0.84, 0.69, 8.66, 8.94])
check("PACE-SHAP s, means (B0-0.3, B0-val)",
      [r(m.loc[x, ("pace_shap_ms", "mean")] / 1000) for x in ("B0-0.3", "B0-val")], [0.3, 9.3])
check("PACE-SHAP mode", L.pace_shap_mode.unique().tolist(), ["sampled"])

section("Per seed: θ by validation against θ = 0.3, and self-repair against its start (+x)")
P = {c: L.pivot(index="seed", columns="model", values=c) for c in ("test_accuracy", "test_macro_f1", "boxes")}
gains = pd.DataFrame({
    "acc val - 0.3": P["test_accuracy"]["B0-val"] - P["test_accuracy"]["B0-0.3"],
    "acc SR-0.3 - B0-0.3": P["test_accuracy"]["SR-0.3"] - P["test_accuracy"]["B0-0.3"],
    "acc SR-val - B0-val": P["test_accuracy"]["SR-val"] - P["test_accuracy"]["B0-val"],
    "F1 SR-0.3 - B0-0.3": P["test_macro_f1"]["SR-0.3"] - P["test_macro_f1"]["B0-0.3"],
    "F1 SR-val - B0-val": P["test_macro_f1"]["SR-val"] - P["test_macro_f1"]["B0-val"],
    "boxes val / 0.3": P["boxes"]["B0-val"] / P["boxes"]["B0-0.3"],
})
print(gains.round(3).to_string())
rng = lambda s, nd=1: (r(s.min(), nd), r(s.max(), nd))  # noqa: E731
check("accuracy, θ by validation over θ = 0.3, range", rng(gains["acc val - 0.3"]), (20.4, 24.8))
check("accuracy gain of SR-0.3: mean, range", (r(gains["acc SR-0.3 - B0-0.3"].mean()), rng(gains["acc SR-0.3 - B0-0.3"])),
      (3.1, (0.5, 5.3)))
check("accuracy gain of SR-val: mean, range", (r(gains["acc SR-val - B0-val"].mean()), rng(gains["acc SR-val - B0-val"])),
      (0.4, (0.1, 0.6)))
check("macro-F1 gain of SR-0.3: mean, range", (r(gains["F1 SR-0.3 - B0-0.3"].mean()), rng(gains["F1 SR-0.3 - B0-0.3"])),
      (1.9, (0.6, 4.2)))
check("macro-F1 gain of SR-val: mean, range", (r(gains["F1 SR-val - B0-val"].mean()), rng(gains["F1 SR-val - B0-val"])),
      (0.5, (0.1, 1.4)))
check("every repair gain is positive (accuracy and macro-F1, both θ, every seed)",
      bool((gains[[c for c in gains if c.startswith(("acc SR", "F1 SR"))]] > 0).all().all()), True)
check("boxes at θ by validation over θ = 0.3, range (times)", rng(gains["boxes val / 0.3"], 0), (34.0, 42.0))
check("repair never added a box", bool((P["boxes"]["SR-0.3"] <= P["boxes"]["B0-0.3"]).all()
                                      and (P["boxes"]["SR-val"] <= P["boxes"]["B0-val"]).all()), True)

section("Ties and centroid-decided predictions, pooled over seeds")
pool = L.groupby("model")[["train_errors", "train_errors_tied_inside_both", "test_centroid_decided", "n_test"]].sum().loc[MODELS]
pool["tie %"] = 100 * pool.train_errors_tied_inside_both / pool.train_errors
pool["centroid %"] = 100 * pool.test_centroid_decided / pool.n_test
print(pool.round(2).to_string())
check("training errors before repair (θ = 0.3, θ by validation), pooled",
      [int(pool.loc[x, "train_errors"]) for x in ("B0-0.3", "B0-val")], [113545, 45542])
check("... all ties inside a box of both classes",
      [int(pool.loc[x, "train_errors_tied_inside_both"]) for x in ("B0-0.3", "B0-val")], [113545, 45542])
check("tie share after repair (%)", [r(pool.loc[x, "tie %"]) for x in ("SR-0.3", "SR-val")], [96.8, 95.6])
check("test predictions decided by the nearest centroid (%)", [r(pool.loc[x, "centroid %"]) for x in MODELS],
      [97.3, 93.8, 41.0, 39.3])

section("Repair: edits and seconds")
sr = L[L.model.str.startswith("SR")]
e = sr.groupby("model")[["edits_logged", "edits_already_right", "edits_accepted", "repair_seconds"]].agg(["mean", "min", "max"])
print(e.round(1).to_string())
check("tries per seed", sorted(sr.edits_logged.unique().tolist()), [300.0])
check("edits accepted at θ = 0.3: mean, range", (r(e.loc["SR-0.3", ("edits_accepted", "mean")]),
      (int(e.loc["SR-0.3", ("edits_accepted", "min")]), int(e.loc["SR-0.3", ("edits_accepted", "max")]))), (25.8, (18, 39)))
check("edits accepted at θ by validation: mean, range", (r(e.loc["SR-val", ("edits_accepted", "mean")]),
      (int(e.loc["SR-val", ("edits_accepted", "min")]), int(e.loc["SR-val", ("edits_accepted", "max")]))), (281.4, (272, 286)))
check("repair seconds, means", [r(e.loc[x, ("repair_seconds", "mean")], 0) for x in ("SR-0.3", "SR-val")], [25.0, 448.0])
b0 = L[L.model.str.startswith("B0")].groupby("model").fit_seconds.mean()
check("training seconds, means (θ = 0.3, θ by validation)", [r(b0["B0-0.3"]), r(b0["B0-val"])], [0.9, 8.8])
t = L.drop_duplicates("seed")
check("tuning seconds, range over seeds", (r(t.tune_seconds.min(), 0), r(t.tune_seconds.max(), 0)), (158.0, 278.0))
check("minutes per seed, range", (r(t.task_seconds.min() / 60), r(t.task_seconds.max() / 60)), (35.7, 40.8))
check("peak memory (MB), largest", r(L.peak_memory_mb.max(), 0), 3998.0)

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
