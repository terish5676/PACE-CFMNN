"""Recompute every number the write-up takes from Run O (re-tune theta or repair?).

Run O: results/theta_churn/runs.csv (B0 at each theta of the grid 0.5 ... 0.05, seeds 70-89, the
13 accuracy datasets, gamma = 4; flips against B0 at theta = 0.3 on the same test rows), home PC,
12 jobs. Repair's flips (SR-0.3 against B0-0.3, SR-val against B0-val) come from
results/repair_theta/runs.csv, as the plan says. NFR = broken / test rows, PFR = fixed / test rows,
in %; fix share = fixed / changed. P1 is recomputed here with common.paired (paired two-sided
Wilcoxon over seeds, Holm across the 13 datasets, lower is better), not read from the summary.
The CSVs are read with float_precision="round_trip".

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_theta_churn.py > research_audit/scripts/claims_theta_churn_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, paired, section

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

rd = lambda p: pd.read_csv(p, float_precision="round_trip")  # noqa: E731
T = rd(RES / "theta_churn" / "runs.csv")
R = rd(RES / "repair_theta" / "runs.csv")
BAD = []
ORDER = ["S1", "S2", "S3", "wine", "breast_cancer", "digits", "thyroid", "shuttle", "skin", "magic",
         "letter", "landsat", "page_blocks"]
GRID = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def r(x, nd=3):
    return round(float(x), nd)


section("The run's rows and its reproduction check")
check("rows", len(T), 13 * 20 * 7)
check("datasets", sorted(T.dataset.unique()), sorted(ORDER))
check("seeds", sorted(T.seed.unique().tolist()), list(range(70, 90)))
check("theta grid", sorted(T.theta.unique().tolist()), sorted(GRID))
checked = T[(T.is_fixed | T.is_val) & T.readable_accuracy.notna()]
check("B0 rows checked against the readable-baselines run", len(checked), 503)
check("... largest |accuracy - readable| and |macro-F1 - readable| are at most 1e-9",
      bool(max((checked.accuracy - checked.readable_accuracy).abs().max(),
               (checked.macro_f1 - checked.readable_macro_f1).abs().max()) <= 1e-9), True)
check("one theta-val row per dataset and seed", int(T.is_val.sum()), 260)
check("B0 at theta = 0.3 changes nothing against itself", int(T[T.theta == 0.3].test_changed.sum()), 0)
check("fixed + broken <= changed on every row", bool((T.test_fixed + T.test_broken <= T.test_changed).all()), True)
log = (RES / "theta_churn" / "run_log.txt").read_text(encoding="utf-8", errors="replace")
check("the log's only error-like lines are the pandas warning block",
      [ln.strip() for ln in log.splitlines() if "Error" in ln], ["+ FullyQualifiedErrorId : NativeCommandError"])

section("P1 (primary): NFR, repair (SR-0.3) vs re-tune (B0 at theta-val), lower is better")
sr = R[(R.model.isin(["SR-0.3", "SR-val"])) & R.seed.between(70, 89)].copy()
for col, src in (("changed_pct", "test_changed"), ("pfr", "test_fixed"), ("nfr", "test_broken")):
    sr[col] = 100 * sr[src] / sr.n_test
sr["fix_share"] = 100 * sr.test_fixed / sr.test_changed.where(sr.test_changed > 0)
check("SR rows read from the repair-at-theta run", len(sr), 13 * 20 * 2)
val = T[T.is_val].assign(model="B0-val")
cols = ["dataset", "seed", "model", "accuracy", "boxes", "changed_pct", "pfr", "nfr", "fix_share"]
P = pd.concat([sr[sr.model == "SR-0.3"][cols], val[cols]], ignore_index=True)
res = paired(P, "SR-0.3", "B0-val", "nfr", ORDER, method_col="model", lower_better=True).set_index("unit")
print(res[["n", "a", "b", "diff", "p", "p_holm", "verdict"]].round(4).to_string())
v = res.verdict.value_counts()
check("P1 tally (better, level, worse)", (int(v.get("better", 0)), int(v.get("level", 0)), int(v.get("worse", 0))), (10, 0, 3))
check("P1 worse on", sorted(res[res.verdict == "worse"].index), sorted(["shuttle", "skin", "letter"]))
check("... by (mean differences)", [r(res.loc[d, "diff"]) for d in ("shuttle", "skin", "letter")], [0.065, 0.185, 0.929])
check("largest Holm p", r(res.p_holm.max(), 4), 0.0143)
check("NFR of repair on S1, S2, S3, Wine, Digits (repair changes nothing there)",
      [r(res.loc[d, "a"]) for d in ("S1", "S2", "S3", "wine", "digits")], [0.0] * 5)
check("... and of re-tuning there", [r(res.loc[d, "b"]) for d in ("S1", "S2", "S3", "wine", "digits")],
      [6.730, 10.455, 8.380, 0.833, 1.319])
check("NFR of re-tuning, range over the 13", (r(res.b.min()), r(res.b.max())), (0.020, 10.455))
check("NFR of repair, range over the 13", (r(res.a.min()), r(res.a.max())), (0.0, 7.760))
check("MAGIC: repair 7.760, re-tune 8.905", (r(res.loc["magic", "a"]), r(res.loc["magic", "b"])), (7.760, 8.905))
changed_none = sorted(d for d in ORDER if sr[(sr.model == "SR-0.3") & (sr.dataset == d)].test_changed.sum() == 0)
check("datasets where repair changes no test prediction", changed_none, sorted(["S1", "S2", "S3", "wine", "digits"]))

section("Descriptive: accuracy, PFR and fix share after each update")
m = P.groupby(["dataset", "model"])[["accuracy", "boxes", "changed_pct", "pfr", "nfr", "fix_share"]].mean().unstack()
print(m.round(2).to_string())
acc = paired(P, "B0-val", "SR-0.3", "accuracy", ORDER, method_col="model").set_index("unit")
va = acc.verdict.value_counts()
check("accuracy, re-tune vs repair (the repair-at-theta run's P1, 1 / 5 / 7 for repair)",
      (int(va.get("better", 0)), int(va.get("level", 0)), int(va.get("worse", 0))), (7, 5, 1))
check("re-tune more accurate on", sorted(acc[acc.verdict == "better"].index),
      sorted(["S1", "shuttle", "skin", "magic", "letter", "landsat", "page_blocks"]))
check("repair more accurate on", list(acc[acc.verdict == "worse"].index), ["thyroid"])
check("accuracy after the update, repair / re-tune: Letter, MAGIC, Page Blocks, Landsat",
      [(r(m.loc[d, ("accuracy", "SR-0.3")], 2), r(m.loc[d, ("accuracy", "B0-val")], 2)) for d in ("letter", "magic", "page_blocks", "landsat")],
      [(77.59, 95.12), (76.71, 81.76), (87.56, 95.37), (86.34, 90.03)])
check("fix share of re-tuning: Letter, Page Blocks, Skin",
      [r(m.loc[d, ("fix_share", "B0-val")], 1) for d in ("letter", "page_blocks", "skin")], [85.3, 81.1, 95.6])
check("fix share of repair: Thyroid, Skin, Letter",
      [r(m.loc[d, ("fix_share", "SR-0.3")], 1) for d in ("thyroid", "skin", "letter")], [90.2, 90.8, 44.3])
check("share of test predictions changed, re-tune, range", (r(m[("changed_pct", "B0-val")].min(), 2), r(m[("changed_pct", "B0-val")].max(), 2)), (0.21, 27.94))
check("... repair, range", (r(m[("changed_pct", "SR-0.3")].min(), 2), r(m[("changed_pct", "SR-0.3")].max(), 2)), (0.0, 20.59))
ratio = m[("boxes", "B0-val")] / m[("boxes", "SR-0.3")]
print("boxes, re-tune / repair:", ratio.round(2).to_dict())
more = [d for d in ORDER if acc.loc[d, "verdict"] == "better"]
check("boxes, re-tune / repair, where re-tune is more accurate, except S1", (r(ratio[[d for d in more if d != "S1"]].min(), 1), r(ratio[[d for d in more if d != "S1"]].max(), 1)), (6.3, 23.2))
check("... on S1 (theta-val is larger than 0.3 there)", r(ratio["S1"], 2), 0.52)

section("theta chosen by validation")
small = sorted(d for d in ORDER if (val[val.dataset == d].theta == 0.05).sum() == 20)
check("theta-val is 0.05 on all 20 seeds", small, sorted(["shuttle", "skin", "magic", "letter"]))

section("Descriptive: boxes needed to match repair (the fewest mean boxes reaching SR-0.3's mean accuracy)")
g = T.groupby(["dataset", "theta"])[["accuracy", "boxes"]].mean()
sra = sr[sr.model == "SR-0.3"].groupby("dataset").accuracy.mean()
need = {}
for d in ORDER:
    ok = g.loc[d][g.loc[d].accuracy >= sra[d]]
    need[d] = None if ok.empty else (float(ok.boxes.idxmin()), r(ok.boxes.min() / g.loc[(d, 0.3), "boxes"], 2))
print(need)
check("not reached on", [d for d in ORDER if need[d] is None], ["thyroid"])
check("Skin, Shuttle, MAGIC: (theta, boxes / boxes at 0.3)", [need[d] for d in ("skin", "shuttle", "magic")],
      [(0.05, 23.21), (0.05, 6.24), (0.15, 4.94)])
check("Letter", need["letter"], (0.2, 2.38))
check("range of the ratio where reached", (min(v[1] for v in need.values() if v), max(v[1] for v in need.values() if v)), (0.48, 23.21))
check("datasets needing more boxes than at 0.3, and their range", ([d for d in ORDER if need[d] and need[d][1] > 1], min(v[1] for v in need.values() if v and v[1] > 1)),
      (["breast_cancer", "shuttle", "skin", "magic", "letter", "landsat", "page_blocks"], 1.84))
check("datasets needing fewer boxes than at 0.3 (a larger theta)", [d for d in ORDER if need[d] and need[d][1] < 1], ["S1", "S2", "S3"])

section("Repair at theta-val: its flips against B0-val")
sv = sr[sr.model == "SR-val"].groupby("dataset")[["nfr", "pfr"]].mean()
print(sv.round(3).to_string())
check("largest mean NFR of SR-val against B0-val", (sv.nfr.idxmax(), r(sv.nfr.max())), ("page_blocks", 0.406))
check("... the next largest", r(sv.nfr.drop("page_blocks").max()), 0.285)

section("Verdict")
print("ALL OK" if not BAD else f"{len(BAD)} BAD: {BAD}")
