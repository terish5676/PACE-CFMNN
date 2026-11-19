"""Recompute every number the write-up takes from Run G (CFMNN + PACE-SHAP against EBM and a tree).

Run G: results/systems/runs.csv and timing.csv (pre-registered, seeds 120-129; amendment 1 before
the run; amendment 2, the TreeSHAP threshold fix, after a failed first attempt and before the
reported run; home PC). Each test is redone with ``common.paired`` (paired two-sided Wilcoxon over
the 10 seeds, Holm across the datasets tested), not with the run's own summary code. The
near-zero rule and amendment 1's untied reading are re-implemented here from the plan.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_systems.py > research_audit/scripts/claims_systems_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, paired, section

sys.stdout.reconfigure(encoding="utf-8")

G = pd.read_csv(RES / "systems" / "runs.csv")
T = pd.read_csv(RES / "systems" / "timing.csv")
DS = ["S1", "S2", "S3", "wine", "breast_cancer", "thyroid", "shuttle", "magic", "letter", "landsat", "page_blocks"]
CF, EBM, TREE = "CFMNN + PACE-SHAP", "EBM + terms", "Tree + TreeSHAP"
NEAR_ZERO, MIN_UNTIED, TIE_HEAVY = 0.05, 10, 0.10
BAD = []


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def test(df, b, col, units=DS, lower=False):
    out = paired(df, CF, b, col, units, method_col="model", lower_better=lower).set_index("unit")
    print(f"\n{CF} vs {b} ({col}):")
    print(out[["a", "b", "diff", "p", "p_holm", "verdict"]].round(4).to_string())
    return out


def verdicts(out):
    return {v: sorted(out.index[out.verdict == v]) for v in ("better", "worse") if (out.verdict == v).any()}


def counts(out):
    v = out.verdict.value_counts()
    return (int(v.get("better", 0)), int(v.get("level", 0)), int(v.get("worse", 0)))


def unit_frame(col):
    syn = G[G.dataset.isin(["S1", "S2", "S3"])].groupby(["seed", "model"], as_index=False)[col].mean()
    return pd.concat([syn.assign(dataset="Synthetic"), G[~G.dataset.isin(["S1", "S2", "S3"])]])


UNITS = ["Synthetic"] + DS[3:]

section("Completeness")
check("rows", len(G), 330)
check("timing rows", len(T), 33)
check("seeds", sorted(G.seed.unique().tolist()), list(range(120, 130)))
check("timing seed", T.seed.unique().tolist(), [120])

for other, name in ((EBM, "P1"), (TREE, "P2")):
    section(f"{name} (primary): normalised deletion lead, CFMNN + PACE-SHAP vs {other}")
    lead = test(G, other, "deletion_lead")
    gain = test(G, other, "deletion_gain")
    flag = set(G[G.model.isin([CF, other]) & (G.random_deletion_auc.abs() < NEAR_ZERO)].dataset)
    rule = lead.verdict.copy()
    for d in flag:
        rule[d] = gain.verdict[d]
    print(f"flagged by the near-zero rule: {sorted(flag)}; rule verdicts: {rule.to_dict()}")
    want = {EBM: ({"better": ["S1", "page_blocks"], "worse": ["breast_cancer", "letter", "shuttle"]}, (2, 6, 3), (2, 5, 4)),
            TREE: ({"better": ["S1", "S2", "page_blocks"], "worse": ["breast_cancer", "letter", "shuttle", "wine"]},
                   (3, 4, 4), (3, 3, 5))}[other]
    check(f"{name} verdicts (lead)", verdicts(lead), want[0])
    check(f"{name} better / level / worse (lead)", counts(lead), want[1])
    check(f"{name} flagged datasets", sorted(flag), ["landsat"])
    check(f"{name} Landsat from the gain", rule["landsat"], "worse")
    check(f"{name} better / level / worse (near-zero rule)",
          (int((rule == "better").sum()), int((rule == "level").sum()), int((rule == "worse").sum())), want[2])
    unit = test(unit_frame("deletion_lead"), other, "deletion_lead", UNITS)
    check(f"{name} synthetic family as one unit", unit.verdict["Synthetic"], "better")

    section(f"{name}, amendment 1: untied points")
    low = G.groupby("dataset").n_untied.min()
    elig = [d for d in DS if low[d] >= MIN_UNTIED]
    check("eligible datasets (every seed of every system >= 10 untied points)", elig,
          ["S1", "S2", "S3", "wine", "breast_cancer", "thyroid", "shuttle", "letter", "landsat"])
    un = test(G, other, "deletion_lead_untied", elig)
    share = G[G.model == CF].groupby("dataset").share_zero_margin.mean()
    heavy = [d for d in DS if share[d] >= TIE_HEAVY]
    check("tie-heavy datasets (CFMNN's mean tied share >= 10%)", heavy, ["magic", "letter", "landsat", "page_blocks"])
    check("tied shares (%) of MAGIC, Letter, Landsat, Page Blocks",
          [round(100 * float(share[d]), 1) for d in heavy], [77.0, 53.6, 25.8, 74.6])
    read = {d: ("not checked" if d not in elig else
                ("holds" if lead.verdict[d] == un.verdict[d] else f"depends ({lead.verdict[d]} -> {un.verdict[d]})"))
            for d in heavy}
    check(f"{name} reading on tie-heavy datasets", read,
          {"magic": "not checked", "letter": "depends (worse -> better)", "landsat": "holds", "page_blocks": "not checked"})
    check(f"{name} untied, better / level / worse", counts(un), {EBM: (2, 5, 2), TREE: (3, 3, 3)}[other])
    ung = test(G, other, "deletion_gain_untied", elig)
    check(f"{name} untied gain, better / level / worse", counts(ung), (1, 1, 7))

    section(f"Secondaries against {other}")
    ins = test(G, other, "insertion_lead")
    check("S1 insertion lead", counts(ins), {EBM: (5, 5, 1), TREE: (6, 1, 4)}[other])
    acc = test(G, other, "accuracy")
    check("S2 accuracy", counts(acc), {EBM: (0, 1, 10), TREE: (2, 0, 9)}[other])
    if other == TREE:
        check("S2 accuracy, better on", verdicts(acc).get("better"), ["breast_cancer", "wine"])
    f1 = test(G, other, "macro_f1")
    check("S2 macro-F1", counts(f1), {EBM: (0, 2, 9), TREE: (2, 1, 8)}[other])
    ms = test(G, other, "ms_per_explanation", lower=True)
    check("S3 milliseconds (lower is better)", counts(ms), (0, 0, 11))
    dg = test(G, other, "deletion_gain")
    check("S4 deletion gain", counts(dg), {EBM: (0, 1, 10), TREE: (0, 2, 9)}[other])
    ig = test(G, other, "insertion_gain")
    check("S4 insertion gain", counts(ig), {EBM: (1, 2, 8), TREE: (0, 2, 9)}[other])

section("Accuracy, size and leads, means over seeds")
acc = G.pivot_table(index="dataset", columns="model", values="accuracy").loc[DS]
size = G.pivot_table(index="dataset", columns="model", values="rules").loc[DS]
lead = G.pivot_table(index="dataset", columns="model", values="deletion_lead").loc[DS]
gain = G.pivot_table(index="dataset", columns="model", values="deletion_gain").loc[DS]
print(acc.round(1).to_string(), size.round(1).to_string(), lead.round(3).to_string(), gain.round(3).to_string(), sep="\n\n")
check("CFMNN accuracy", [round(float(acc.loc[d, CF]), 1) for d in DS],
      [89.4, 80.1, 88.1, 97.8, 95.9, 96.9, 99.8, 71.9, 75.6, 85.4, 83.4])
check("EBM accuracy", [round(float(acc.loc[d, EBM]), 1) for d in DS],
      [100.0, 90.6, 99.9, 97.5, 97.5, 99.6, 100.0, 86.0, 87.8, 90.6, 97.5])
check("Tree accuracy", [round(float(acc.loc[d, TREE]), 1) for d in DS],
      [97.1, 88.4, 100.0, 92.5, 92.4, 99.6, 100.0, 85.1, 86.7, 86.6, 96.9])
check("CFMNN boxes", [round(float(size.loc[d, CF])) for d in DS],
      [1516, 1538, 1529, 31, 42, 53, 14, 284, 563, 200, 26])
check("EBM terms", [round(float(size.loc[d, EBM])) for d in DS], [10, 10, 10, 13, 30, 21, 9, 10, 16, 36, 10])
check("tree leaves", [round(float(size.loc[d, TREE])) for d in DS], [58, 32, 4, 6, 7, 16, 32, 96, 1024, 83, 22])

M = [CF, EBM, TREE]
check("paper table: deletion gain (2 dp), CFMNN / EBM / tree",
      {d: [round(float(gain.loc[d, m]), 2) for m in M] for d in DS},
      {"S1": [0.43, 0.52, 0.54], "S2": [0.39, 0.54, 0.56], "S3": [0.38, 0.53, 0.56], "wine": [0.3, 0.36, 0.53],
       "breast_cancer": [0.29, 0.41, 0.46], "thyroid": [0.2, 0.21, 0.18], "shuttle": [0.17, 0.31, 0.29],
       "magic": [0.17, 0.59, 0.55], "letter": [0.3, 0.37, 0.38], "landsat": [0.3, 0.41, 0.53],
       "page_blocks": [0.12, 0.17, 0.15]})
check("paper table: deletion lead (2 dp), CFMNN / EBM / tree",
      {d: [round(float(lead.loc[d, m]), 2) for m in M] for d in DS},
      {"S1": [1.52, 1.32, 1.24], "S2": [1.61, 1.39, 1.29], "S3": [1.27, 1.16, 1.17], "wine": [1.21, 1.24, 1.86],
       "breast_cancer": [0.7, 0.92, 0.86], "thyroid": [0.28, 0.27, 0.21], "shuttle": [0.3, 0.52, 0.44],
       "magic": [1.33, 1.45, 1.34], "letter": [0.79, 1.03, 1.04], "landsat": [58.9, 12.74, 623.11],
       "page_blocks": [1.03, 0.24, 0.2]})
check("paper table: tree leaves (1 dp)", [round(float(size.loc[d, TREE]), 1) for d in DS],
      [57.7, 32.0, 4.0, 6.5, 6.7, 16.2, 31.8, 96.0, 1024.0, 83.2, 22.4])

section("Timing pass (seed 120, one task at a time): milliseconds per explanation")
tm = T.pivot_table(index="dataset", columns="model", values="ms_per_explanation").loc[DS]
print(tm.round(3).to_string())
check("PACE-SHAP ms", [round(float(tm.loc[d, CF])) for d in DS], [1775, 1812, 1802, 7, 85, 59, 7, 248, 508, 636, 22])
check("EBM ms, largest", round(float(tm[EBM].max()), 2), 0.03)
check("Tree ms, largest except Letter", round(float(tm[TREE].drop("letter").max()), 2), 0.43)
check("Tree ms, Letter", round(float(tm.loc["letter", TREE]), 1), 8.5)
rt, re_ = tm[CF] / tm[TREE], tm[CF] / tm[EBM]
check("PACE-SHAP / tree, range", (round(float(rt.min())), round(float(rt.max()), -3)), (35, 69000.0))
check("paper table: EBM and TreeSHAP ms (2 dp)", {d: (round(float(tm.loc[d, EBM]), 2), round(float(tm.loc[d, TREE]), 2)) for d in DS},
      {"S1": (0.01, 0.07), "S2": (0.01, 0.07), "S3": (0.01, 0.03), "wine": (0.01, 0.03), "breast_cancer": (0.02, 0.03),
       "thyroid": (0.02, 0.03), "shuttle": (0.01, 0.19), "magic": (0.01, 0.15), "letter": (0.02, 8.46),
       "landsat": (0.03, 0.43), "page_blocks": (0.01, 0.08)})
check("PACE-SHAP / EBM, range", (round(float(re_.min()), -2), round(float(re_.max()), -3)), (500.0, 169000.0))

print(f"\n{'ALL OK' if not BAD else 'BAD: ' + ', '.join(BAD)}")
