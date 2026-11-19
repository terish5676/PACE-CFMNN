"""Recompute every number the write-up takes from Run F (self-repair against contraction).

Run F: results/contraction/runs.csv (pre-registered, seeds 120-129, amendment 1 before the run,
home PC). Each test is redone with ``common.paired`` (paired two-sided Wilcoxon over the 10 seeds,
Holm across the 6 datasets; p = 1 when every difference is zero), not with the run's own
summary code. Values are rounded here from the raw CSV.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_contraction.py > research_audit/scripts/claims_contraction_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, paired, section

sys.stdout.reconfigure(encoding="utf-8")

F = pd.read_csv(RES / "contraction" / "runs.csv")
DS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
BAD = []


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def test(a, b, col="test_macro_f1", lower=False):
    out = paired(F, a, b, col, DS, method_col="arm", lower_better=lower).set_index("unit")
    print(f"\n{a} vs {b} ({col}):")
    print(out[["a", "b", "diff", "p", "p_holm", "a_better_seeds", "verdict"]].round(4).to_string())
    return out


def verdicts(out):
    return {v: sorted(out.index[out.verdict == v]) for v in ("better", "worse") if (out.verdict == v).any()}


def diffs(out, nd=1):
    return [round(float(out.loc[d, "diff"]), nd) for d in DS]


section("Completeness")
check("rows", len(F), 420)
check("seeds", sorted(F.seed.unique().tolist()), list(range(120, 130)))
check("arms per dataset and seed", F.groupby(["dataset", "seed"]).size().unique().tolist(), [7])

section("P1 (primary): SR vs SC-all, test macro-F1")
p1 = test("SR", "SC-all")
check("P1 higher on", verdicts(p1), {"better": sorted(DS)})
check("P1 differences (Thyroid, Landsat, Page Blocks, MAGIC, Letter, Skin)", diffs(p1),
      [4.5, 2.4, 10.0, 14.9, 2.6, 22.5])
check("P1 largest Holm p", round(float(p1.p_holm.max()), 4), 0.0195)

section("S1: SR vs SC-err (ungated contraction at the same training errors), macro-F1")
s1 = test("SR", "SC-err")
check("S1 higher on", verdicts(s1), {"better": sorted(DS)})
check("S1 differences", diffs(s1), [4.5, 1.7, 5.3, 5.1, 2.4, 9.0])

section("S2a and S2b: SR vs the GFMM-rule arms, macro-F1")
s2a = test("SR", "GC-all")
check("S2a higher on", verdicts(s2a), {"better": sorted(DS)})
s2b = test("SR", "GC-err")
check("S2b higher on", verdicts(s2b), {"better": sorted(DS)})

section("S3: SR vs GFMM trained from scratch, macro-F1")
s3 = test("SR", "GFMM")
check("S3 higher / lower", verdicts(s3), {"better": ["magic", "skin", "thyroid"], "worse": ["letter"]})
check("S3 Letter difference", round(float(s3.loc["letter", "diff"]), 1), -5.4)
check("S3 Letter: GFMM, SR macro-F1", (round(float(s3.loc["letter", "b"]), 1), round(float(s3.loc["letter", "a"]), 1)),
      (82.3, 76.9))

section("S4a-f: each arm vs the starting model B0, macro-F1")
s4 = {arm: test(arm, "B0") for arm in ("SR", "SC-all", "SC-err", "GC-all", "GC-err", "GFMM")}
check("S4a SR higher on", verdicts(s4["SR"]), {"better": sorted(DS)})
check("S4a differences", diffs(s4["SR"]), [4.5, 2.0, 6.3, 1.9, 2.4, 7.6])
check("S4b SC-all lower on", verdicts(s4["SC-all"]), {"worse": ["magic", "skin"]})
check("S4b MAGIC, Skin", [round(float(s4["SC-all"].loc[d, "diff"]), 1) for d in ("magic", "skin")], [-13.0, -14.9])
check("S4c SC-err lower on", verdicts(s4["SC-err"]), {"worse": ["magic"]})
check("S4d GC-all", verdicts(s4["GC-all"]), {"worse": ["landsat", "letter", "magic", "skin", "thyroid"]})
check("S4f GC-err", verdicts(s4["GC-err"]), {"worse": ["letter", "magic", "thyroid"]})
check("S4e GFMM", verdicts(s4["GFMM"]), {"better": ["letter", "skin"], "worse": ["thyroid"]})

section("S5, S9: the primary pair on lowest class recall and accuracy")
s5 = test("SR", "SC-all", "test_min_class_recall")
check("S5 higher on", verdicts(s5), {"better": sorted(DS)})
s9 = test("SR", "SC-all", "test_accuracy")
check("S9 higher on", verdicts(s9), {"better": ["landsat", "letter", "magic", "skin", "thyroid"]})

section("S6, S7, S8: boxes, broken predictions, evicted rows (SR - SC-all)")
s6 = test("SR", "SC-all", "boxes_end", lower=True)
check("S6 (fewer boxes = better): SR more boxes on", verdicts(s6), {"worse": ["magic"]})
s7 = test("SR", "SC-all", "test_broken_pct", lower=True)
check("S7 (fewer broken = better)", verdicts(s7), {"better": ["magic", "page_blocks", "skin"], "worse": ["letter"]})
s8 = test("SR", "SC-all", "rows_evicted_pct", lower=True)
check("S8 (fewer evicted = better): SR evicts more on / fewer on", verdicts(s8),
      {"better": ["magic", "skin"], "worse": ["landsat", "letter", "thyroid"]})

section("Evicted training rows, % of training rows, mean over seeds")
ev = F.pivot_table(index="arm", columns="dataset", values="rows_evicted_pct", aggfunc="mean")[DS]
print(ev.round(2).to_string())
check("SR evicted %", [round(float(ev.loc["SR", d]), 1) for d in DS], [59.5, 23.5, 5.8, 32.6, 9.5, 2.8])
check("SC-all evicted %", [round(float(ev.loc["SC-all", d]), 1) for d in DS], [0.0, 12.5, 27.3, 89.3, 0.8, 17.8])
check("GC-all evicted %", [round(float(ev.loc["GC-all", d]), 1) for d in DS], [59.6, 62.2, 69.7, 89.6, 89.4, 21.1])

section("Thyroid: Simpson's test")
th = F[F.dataset == "thyroid"]
check("SC-all contractions kept, every Thyroid seed", sorted(th[th.arm == "SC-all"].edits_kept.unique().tolist()), [0])
check("SC-err contractions kept, every Thyroid seed", sorted(th[th.arm == "SC-err"].edits_kept.unique().tolist()), [0])
b0 = th[th.arm == "B0"].set_index("seed").test_macro_f1
sc = th[th.arm == "SC-all"].set_index("seed").test_macro_f1
check("Thyroid SC-all macro-F1 equals B0 on every seed", bool(np.allclose(b0, sc[b0.index], atol=0)), True)

section("Mean test macro-F1 (%), by arm")
mf = F.pivot_table(index="arm", columns="dataset", values="test_macro_f1", aggfunc="mean")[DS]
print(mf.round(2).to_string())
check("B0 macro-F1", [round(float(mf.loc["B0", d]), 1) for d in DS], [85.7, 82.6, 61.8, 72.5, 74.5, 91.3])
check("SR macro-F1", [round(float(mf.loc["SR", d]), 1) for d in DS], [90.1, 84.5, 68.1, 74.4, 76.9, 98.9])
check("SC-all macro-F1", [round(float(mf.loc["SC-all", d]), 1) for d in DS], [85.7, 82.1, 58.1, 59.5, 74.3, 76.5])
check("GFMM macro-F1", [round(float(mf.loc["GFMM", d]), 1) for d in DS], [75.7, 83.1, 66.1, 72.4, 82.3, 97.9])

section("Paper wording (revs 455-478): ranges and whole-number percentages")
check("P1 range", (round(float(p1["diff"].min()), 1), round(float(p1["diff"].max()), 1)), (2.4, 22.5))
check("S1 range", (round(float(s1["diff"].min()), 1), round(float(s1["diff"].max()), 1)), (1.7, 9.0))
check("SR evicted %, whole numbers (Thyroid, Landsat, Letter, MAGIC, Skin)",
      [round(float(ev.loc["SR", d])) for d in ("thyroid", "landsat", "letter", "magic", "skin")], [59, 24, 9, 33, 3])
check("SC-all evicted % (Thyroid, Landsat, Letter to 1 dp, MAGIC, Skin)",
      [round(float(ev.loc["SC-all", "thyroid"])), round(float(ev.loc["SC-all", "landsat"])),
       round(float(ev.loc["SC-all", "letter"]), 1), round(float(ev.loc["SC-all", "magic"])),
       round(float(ev.loc["SC-all", "skin"]))], [0, 12, 0.8, 89, 18])
check("GC-all evicted %, range", (round(float(ev.loc["GC-all"].min())), round(float(ev.loc["GC-all"].max()))), (21, 90))

section("Self-repair budget")
sr = F[F.arm == "SR"]
check("SR tries at most 300 on every task", int(sr.edits_tried.max()) <= 300, True)
print(sr.groupby("dataset").edits_tried.agg(["min", "max"]).to_string())

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
