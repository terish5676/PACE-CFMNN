"""Recompute every number the paper takes from Run H (repair order) and Run T (repair timing).

Run H: results/repair_order/ (pre-registered, seeds 130-139, amendment 1 before the run).
Run T: results/repair_timing/ (descriptive, seed 50, one task at a time on the home PC).
Each Run H test is redone with ``common.paired`` (paired two-sided Wilcoxon over the 10 seeds,
Holm across the 6 datasets), not with the run's own summary code.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_order_timing.py > research_audit/scripts/claims_order_timing_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, paired, section

sys.stdout.reconfigure(encoding="utf-8")

H = pd.read_csv(RES / "repair_order" / "runs.csv")
T = pd.read_csv(RES / "repair_timing" / "runs.csv")
DS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]


def check(label, got, want):
    ok = got == want
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (paper: {want})"))


def test(a, b, col="test_macro_f1", lower=False):
    out = paired(H, a, b, col, DS, method_col="arm", lower_better=lower).set_index("unit")
    print(f"{a} vs {b} ({col}):")
    print(out[["a", "b", "diff", "p", "p_holm", "verdict"]].round(4).to_string())
    return out


def verdicts(out):
    return {v: sorted(out.index[out.verdict == v]) for v in ("better", "worse") if (out.verdict == v).any()}


section("Run H: completeness")
check("rows", len(H), 540)
check("seeds", sorted(H.seed.unique().tolist()), list(range(130, 140)))
check("arms per dataset and seed", int(H.groupby(["dataset", "seed"]).size().unique()[0]), 9)

section("Run H primary: greedy-300 vs random-300, macro-F1")
p = test("greedy-300", "random-300")
check("better / worse", verdicts(p), {"better": ["letter", "magic"], "worse": ["skin"]})
check("MAGIC, Letter, Skin differences", [round(float(p.loc[d, "diff"]), 1) for d in ("magic", "letter", "skin")],
      [4.9, 3.0, -1.1])

section("Run H amendment A1: greedy-300 vs random-300-real, macro-F1")
a1 = test("greedy-300", "random-300-real")
check("better / worse", verdicts(a1), {"better": ["letter", "magic"]})
check("MAGIC, Letter, Skin differences", [round(float(a1.loc[d, "diff"]), 1) for d in ("magic", "letter", "skin")],
      [4.7, 2.7, -0.2])

section("Run H secondary: greedy-300 vs random-all, macro-F1")
s = test("greedy-300", "random-all")
check("better / worse", verdicts(s), {"better": ["magic"]})
check("MAGIC difference", round(float(s.loc["magic", "diff"]), 1), 8.0)

section("Run H secondary: box order")
b = test("box-300", "random-300")
check("box-300 vs random-300", verdicts(b), {"worse": ["letter", "skin"]})
b2 = test("box-1000", "random-1000")
check("box-1000 vs random-1000", verdicts(b2), {})
g2 = test("greedy-1000", "random-1000")
check("greedy-1000 vs random-1000", verdicts(g2), {"better": ["letter", "magic"]})

section("Run H amendment E1 and E2: greedy-300 vs random-all, seconds and edits that reached the gate")
e1 = test("greedy-300", "random-all", "seconds", lower=True)
check("E1: slower on", sorted(e1.index[e1.verdict == "worse"]), sorted(DS))
e2 = test("greedy-300", "random-all", "edits_gated", lower=True)
check("E2: fewer on", sorted(e2.index[e2.verdict == "better"]), ["letter", "magic"])
m = H.groupby(["arm", "dataset"]).seconds.mean()
check("mean seconds, Skin: greedy-300, random-all, random-300",
      (round(m["greedy-300", "skin"]), round(m["random-all", "skin"], 1), round(m["random-300", "skin"], 1)),
      (1160, 26.6, 8.8))
check("mean seconds, MAGIC and Letter: greedy-300 against random-300",
      (round(m["greedy-300", "magic"]), round(m["random-300", "magic"], 1),
       round(m["greedy-300", "letter"]), round(m["random-300", "letter"], 1)), (241, 5.2, 420, 11.8))

section("Run H descriptive: tries wasted, and Skin")
g = H.groupby(["arm", "dataset"])[["edits_gated", "skip_already_right", "test_macro_f1"]].mean()
check("random-300 tries that reached the gate on Skin", round(g.loc[("random-300", "skin"), "edits_gated"]), 39)
check("Skin macro-F1: random-300, random-300-real, random-all",
      tuple(round(g.loc[(a, "skin"), "test_macro_f1"], 1) for a in ("random-300", "random-300-real", "random-all")),
      (98.9, 98.0, 98.0))
start = H[H.arm == "B0"].groupby("dataset").train_errors_start.mean()
check("training errors at the start, MAGIC, Letter, Skin",
      [round(start[d]) for d in ("magic", "letter", "skin")], [2396, 2108, 4992])

section("Run T: time of each setup, seed 50, one task at a time")
check("rows", len(T), 30)
check("every setup's macro-F1 equals RQ3's", bool(T.matches_committed.all()), True)
s_ = T.pivot(index="setup", columns="dataset", values="seconds")[DS]
print(s_.round(2).to_string())
sr = s_.loc["SR, cached gate"]
check("SR, cached gate (s)", f"{sr.min():.1f}-{sr.max():.1f}", "0.2-4.7")
check("train CFMNN (s)", f"{s_.loc['Train CFMNN'].min():.2f}-{s_.loc['Train CFMNN'].max():.1f}", "0.04-1.9")
al = s_.loc["AL"]
check("AL (s)", f"{al.min():.1f}-{al.max():.0f}", "3.9-142")
r = al / sr
check("AL as a multiple of SR, cached gate", f"{r.min():.0f}-{r.max():.0f}", "5-56")
both = s_.loc["SR+AL"]
check("SR+AL (s) and multiple of SR", (f"{both.min():.1f}-{both.max():.0f}", f"{(both / sr).min():.1f}-{(both / sr).max():.0f}"),
      ("4.6-145", "5.5-57"))
old = s_.loc["SR, old gate"] / sr
check("old gate as a multiple of the cached gate", f"{old.min():.0f}-{old.max():.0f}", "2-45")

section("Numbers used in the paper's wording")
save = 100 * (1 - sr / both)
check("time saved by leaving out AL (% of SR+AL)", f"{save.min():.0f}-{save.max():.0f}%", "82-98%")
k = H.groupby(["arm", "dataset"]).skip_already_right.mean()
check("tries on rows already right: random-300 on MAGIC and Skin; greedy-300 on MAGIC and Letter",
      (round(k["random-300", "magic"]), round(k["random-300", "skin"]),
       round(k["greedy-300", "magic"]), round(k["greedy-300", "letter"])), (108, 260, 0, 0))
check("greedy-300 vs random-all: not significantly different on", int((s.verdict == "level").sum()), 5)
