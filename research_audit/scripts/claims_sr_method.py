"""Recompute every number the paper's "self-repair as the method" text takes from saved results.

Paper (doc rev 453): Section 9 "Self-repair as the method" paragraph and table, the last two
sentences of "Order of the training errors", Section 10 "Self-repair alone (post hoc)", and the
conclusion's "at 5.5-57 times the time".

Sources, all committed:
* results/refinement_v2_confirmatory/runs.csv (RQ3, pre-registered, seeds 50-59);
* results/repair_timing/runs.csv (Run T, descriptive, seed 50, home PC);
* results/repair_order/runs.csv (Run H, pre-registered, seeds 130-139, home PC).

Each test is redone with ``common.paired`` (paired two-sided Wilcoxon over the 10 seeds, Holm
across the datasets), not with the runs' own summary code. The bold cells of the table are the
pre-registered S3a (SR over B0) and P3 (SR+AL over SR); RQ4 for self-repair alone is post hoc.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_sr_method.py > research_audit/scripts/claims_sr_method_out.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from common import RES, paired, section

sys.stdout.reconfigure(encoding="utf-8")

R = pd.read_csv(RES / "refinement_v2_confirmatory" / "runs.csv")
T = pd.read_csv(RES / "repair_timing" / "runs.csv")
H = pd.read_csv(RES / "repair_order" / "runs.csv")
DS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]


def check(label, got, want):
    ok = got == want
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (paper: {want})"))


def r1(values):
    return [round(float(v), 1) for v in values]


def gain(col, a, b="B0"):
    """Mean over seeds of the paired difference a - b, per dataset."""
    p = R.pivot_table(index=["dataset", "seed"], columns="arm", values=col)
    return [(p[a] - p[b]).loc[d].mean() for d in DS]


def mean(col, arm):
    return [R[(R.arm == arm) & (R.dataset == d)][col].mean() for d in DS]


def span(values, lo="{:.1f}", hi="{:.1f}"):
    v = [float(x) for x in values]
    return f"{lo.format(min(v))}-{hi.format(max(v))}"


def test(df, a, b, col, units=DS, lower=False):
    out = paired(df, a, b, col, units, method_col="arm", lower_better=lower).set_index("unit")
    print(f"{a} vs {b} ({col}):")
    print(out[["a", "b", "diff", "p", "p_holm", "verdict"]].round(4).to_string())
    return out


section("RQ3 table: macro-F1 (points; starting model in %)")
check("rows per arm and dataset", sorted(R.groupby(["arm", "dataset"]).size().unique().tolist()), [10])
check("starting model", r1(mean("test_macro_f1", "B0")), [84.0, 82.0, 67.6, 72.6, 74.3, 93.1])
check("self-repair", r1(gain("test_macro_f1", "SR")), [6.0, 1.5, 2.6, 3.3, 2.2, 5.7])
check("active learning", r1(gain("test_macro_f1", "AL")), [6.4, 1.9, 4.9, 5.1, 1.9, 6.1])
check("SR then AL", r1(gain("test_macro_f1", "SR+AL")), [6.6, 2.6, 4.9, 5.3, 3.0, 6.2])
check("SR then AL, over self-repair", r1(gain("test_macro_f1", "SR+AL", "SR")), [0.6, 1.1, 2.3, 2.0, 0.7, 0.6])

section("RQ3 table: bold cells (pre-registered S3a and P3, recomputed)")
s3a = test(R, "SR", "B0", "test_macro_f1")
check("self-repair over the starting model: significant (better) on",
      sorted(s3a.index[s3a.verdict == "better"]), sorted(DS))
p3 = test(R, "SR+AL", "SR", "test_macro_f1")
check("SR then AL over self-repair: significant (better) on",
      sorted(p3.index[p3.verdict == "better"]), ["landsat", "letter", "magic"])
check("SR then AL over self-repair: significantly worse on", sorted(p3.index[p3.verdict == "worse"]), [])

section("RQ3 table: lowest class recall (points; starting model in %)")
check("starting model", r1(mean("test_min_class_recall", "B0")), [59.2, 47.3, 52.4, 71.9, 49.7, 80.9])
check("self-repair", r1(gain("test_min_class_recall", "SR")), [19.4, 6.4, 0.1, -2.7, 2.9, 16.7])
check("active learning", r1(gain("test_min_class_recall", "AL")), [19.2, 8.0, -1.9, 0.4, 1.9, 17.4])
check("SR then AL", r1(gain("test_min_class_recall", "SR+AL")), [20.5, 9.1, -1.3, -1.2, 3.8, 17.9])

section("RQ3 table: labels and seconds")
lab = {a: [int(x) for x in mean("total_labels", a)] for a in ("SR", "AL", "SR+AL")}
check("labels, self-repair", lab["SR"], [0] * 6)
check("labels, AL", lab["AL"], [1020, 944, 821, 2202, 2300, 24806])
check("labels, SR then AL", lab["SR+AL"], [1020, 944, 821, 2202, 2300, 24806])
check("labels constant over seeds (AL, SR then AL)",
      bool((R[R.arm.isin(["AL", "SR+AL"])].groupby(["arm", "dataset"]).total_labels.nunique() == 1).all()), True)
sec = T.pivot_table(index="setup", columns="dataset", values="seconds")[DS]
check("Run T verdict: every setup matches RQ3", bool(T.matches_committed.astype(bool).all()), True)
check("Run T seed", sorted(T.seed.unique().tolist()), [50])
check("seconds, self-repair (cached gate)", r1(sec.loc["SR, cached gate"]), [0.2, 1.4, 0.8, 2.1, 4.7, 2.5])
check("seconds, SR then AL", r1(sec.loc["SR+AL"]), [6.8, 17.1, 4.6, 33.3, 117.5, 145.0])

section("RQ3 paragraph: numbers in the wording")
over = gain("test_macro_f1", "SR+AL", "SR")
check("labels on top of self-repair add (points)", span(over), "0.6-2.3")
check("significantly on", f"{(p3.verdict == 'better').sum()} of 6", "3 of 6")
check("labels read", f"{min(lab['AL']):,}-{max(lab['AL']):,}", "821-24,806")
ratio = sec.loc["SR+AL"] / sec.loc["SR, cached gate"]
check("SR then AL as a multiple of self-repair", span(ratio, "{:.1f}", "{:.0f}"), "5.5-57")
al_sr = dict(zip(DS, gain("test_macro_f1", "AL", "SR")))
check("active learning ahead of self-repair: Page Blocks, MAGIC",
      (round(al_sr["page_blocks"], 1), round(al_sr["magic"], 1)), (2.3, 1.8))
check("self-repair lowered MAGIC's lowest class recall by",
      round(-dict(zip(DS, gain("test_min_class_recall", "SR")))["magic"], 1), 2.7)

section("RQ4 for self-repair alone (post hoc): fix share, broken predictions, explanations")
check("fix share, self-repair (%)", span(mean("churn_fix_share", "SR")), "59.0-87.5")
check("fix share, shuffled retraining (%)", span(mean("churn_fix_share", "Retrain, shuffled")), "40.5-52.4")
fx = test(R, "SR", "Retrain, shuffled", "churn_fix_share")
check("fix share higher on every dataset", bool((fx["diff"] > 0).all()), True)
check("fix share difference (points)", span(fx["diff"]), "6.6-47.0")
check("fix share significant on", sorted(fx.index[fx.verdict == "better"]), ["letter", "magic", "skin", "thyroid"])
br = test(R, "SR", "Retrain, shuffled", "churn_broken_share", lower=True)
check("fewer broken predictions, significant on", sorted(br.index[br.verdict == "better"]), sorted(DS))
check("broken: retrain minus self-repair (points of the test rows)", span(-br["diff"]), "0.8-8.4")
five = [d for d in DS if d != "skin"]
ex = test(R, "SR", "Retrain, shuffled", "expl_topk_change_share", units=five, lower=True)
check("fewer explanation changes, significant on", sorted(ex.index[ex.verdict == "better"]), sorted(five))
check("explanations: retrain minus self-repair (points of the explained rows)",
      span(-100 * ex["diff"]), "22.9-64.5")
check("Skin's top 3 never changes (3 features)",
      float(R[(R.dataset == "skin") & R.arm.isin(["SR", "Retrain, shuffled"])].expl_topk_change_share.abs().max()), 0.0)
sr = R[R.arm == "SR"]
check("self-repair's accepted edits", int(sr.cert_edits.sum()), 6233)
check("misses (tight, plain)", (int(sr.cert_tight_misses.sum()), int(sr.cert_loose_misses.sum())), (0, 0))

section("Run H: last two sentences of 'Order of the training errors'")
s = H.pivot_table(index="arm", columns="dataset", values="seconds")[DS]
g = s.loc["greedy-300"] / s.loc["random-300"]
print(g.round(2).to_string())
check("greedy-300 as a multiple of random-300, MAGIC and Letter", span(g[["magic", "letter"]], "{:.0f}", "{:.0f}"), "36-46")
check("greedy-300 as a multiple of random-300, 6 datasets", span(g, "{:.1f}", "{:.0f}"), "4.5-132")
pr = test(H, "greedy-300", "random-300", "test_macro_f1")
check("greedy-300 better than random-300 on", sorted(pr.index[pr.verdict == "better"]), ["letter", "magic"])
check("no gain on the other 4", sorted(pr.index[pr.verdict != "better"]),
      ["landsat", "page_blocks", "skin", "thyroid"])

section("FINDINGS §5.5 only: extra numbers")
check("accuracy gain, self-repair (points)", span(gain("test_accuracy", "SR")), "1.1-3.9")
check("accuracy gain, SR then AL (points)", span(gain("test_accuracy", "SR+AL")), "1.2-5.9")
check("AL raised MAGIC's lowest class recall by", round(dict(zip(DS, gain("test_min_class_recall", "AL")))["magic"], 1), 0.4)
m = R[(R.dataset == "magic")]
pv = m.pivot_table(index="seed", columns="arm", values="test_min_class_recall")
check("seeds where SR lowered MAGIC's lowest class recall", int((pv["SR"] < pv["B0"]).sum()), 9)
check("fix share Holm p, Landsat and Page Blocks", (round(float(fx.loc["landsat", "p_holm"]), 3),
      round(float(fx.loc["page_blocks", "p_holm"]), 3)), (0.074, 0.074))
pb = sr[sr.dataset == "page_blocks"]
check("Page Blocks: SR accepted edits per seed", round(float(pb.cert_edits.mean()), 1), 12.5)
check("Page Blocks: seeds with SR fix share 0%", int((pb.churn_fix_share == 0).sum()), 1)
check("time saved by leaving out AL (% of SR then AL)", span(100 * (1 - 1 / ratio), "{:.0f}", "{:.0f}"), "82-98")
