"""Recompute every repair number in the paper (RQ3, RQ4) from the raw runs.csv files.

Sources:
* results/refinement_confirmatory/runs.csv      (v1, seeds 10-19)
* results/refinement_v2_explore/runs.csv        (E1-E4, seeds 0-2)
* results/refinement_v2_confirmatory/runs.csv   (v2, seeds 50-59)

    python -B research_audit/scripts/claims_repair.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RES, holm, rng_str, section, wilcox_p

DS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]


def piv(runs, value, agg="mean"):
    return runs.pivot_table(index="arm", columns="dataset", values=value, aggfunc=agg)[DS]


def diffs(runs, value, a, b):
    p = runs.pivot_table(index=["dataset", "seed"], columns="arm", values=value)
    return {d: (p.loc[d, a] - p.loc[d, b]).dropna().to_numpy() for d in DS}


def endpoint(runs, value, a, b, kind="two", margin=0.0, exclude=()):
    rows = []
    for d, x in diffs(runs, value, a, b).items():
        if d in exclude:
            continue
        p = wilcox_p(x + margin, "greater") if kind == "ni" else wilcox_p(x)
        rows.append({"dataset": d, "mean": x.mean(), "wins": int((x > 0).sum()),
                     "losses": int((x < 0).sum()), "n": len(x), "p": p})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t["sig"] = t.p_holm < 0.05
    return t


# ----------------------------------------------------------------------------- v2
v2 = pd.read_csv(RES / "refinement_v2_confirmatory" / "runs.csv")
section("v2 confirmatory: design facts")
print("seeds:", sorted(v2.seed.unique()), " datasets:", sorted(v2.dataset.unique()))
print(v2.groupby("dataset")[["n_train", "n_pool", "n_accept", "n_test", "budget"]].first().loc[DS])
print("audit rows used for certificates = min(2000, n_test):",
      {d: int(min(2000, v2[v2.dataset == d].n_test.iloc[0])) for d in DS})

section("Paper table 'Repair, second run (RQ3)': test macro-F1 (%), mean over 10 seeds, 1 dp")
arms = ["B0", "SR", "AL", "SR+AL", "Retrain, shuffled", "kNN, every label", "RF, every label"]
mf = piv(v2, "test_macro_f1")
print(mf.loc[arms].round(1))
print("\nunrounded (to catch .x5 rounding):")
print(mf.loc[arms].round(3))
lab = piv(v2, "total_labels")
print("\nlabels beyond training rows, AL:", lab.loc["AL"].to_dict(), " range",
      rng_str(lab.loc["AL"], "{:,.0f}"))

section("Endpoints (pre-registered tests)")
E = {
    "P3  SR+AL - SR (macro-F1)": ("test_macro_f1", "SR+AL", "SR", "two", 0, ()),
    "S3a SR - B0 (macro-F1)": ("test_macro_f1", "SR", "B0", "two", 0, ()),
    "S3b SR - AL > -1 (NI)": ("test_macro_f1", "SR", "AL", "ni", 1.0, ()),
    "S3c SR+AL - B0 min recall > -2 (NI)": ("test_min_class_recall", "SR+AL", "B0", "ni", 2.0, ()),
    "S3d SR - B0 min recall > -2 (NI)": ("test_min_class_recall", "SR", "B0", "ni", 2.0, ()),
    "P4  fix share SR+AL - Retrain shuffled": ("churn_fix_share", "SR+AL", "Retrain, shuffled", "two", 0, ()),
    "S4a broken Retrain shuffled - SR+AL": ("churn_broken_share", "Retrain, shuffled", "SR+AL", "two", 0, ()),
    "S4b top-3 change Retrain shuffled - SR+AL": ("expl_topk_change_share", "Retrain, shuffled", "SR+AL", "two", 0,
                                                  ("skin",)),
    "S4c macro-F1 SR+AL - Retrain shuffled": ("test_macro_f1", "SR+AL", "Retrain, shuffled", "two", 0, ()),
}
res = {}
for name, (val, a, b, kind, m, ex) in E.items():
    t = endpoint(v2, val, a, b, kind, m, ex)
    res[name] = t
    print(f"\n{name}")
    print(t.round(4).to_string(index=False))
    print("  significant on:", list(t[t.sig].dataset), " range of means:", rng_str(t["mean"]))

section("Ranges and descriptive claims")
b0 = mf.loc["B0"]
rf = mf.loc["RF, every label"]
for arm in ["SR", "AL", "SR+AL", "Retrain, shuffled", "Retrain, appended", "B9"]:
    gap = 100 * (mf.loc[arm] - b0) / (rf - b0)
    print(f"gap to RF closed, {arm:18s}", gap.round(1).to_dict(), " range", rng_str(gap, "{:.1f}"))
print("\nRF every label - SR+AL:", (rf - mf.loc["SR+AL"]).round(2).to_dict())
print("kNN every label - SR+AL:", (mf.loc["kNN, every label"] - mf.loc["SR+AL"]).round(2).to_dict())
print("Retrain appended - B0:", (mf.loc["Retrain, appended"] - b0).round(3).to_dict(),
      " range", rng_str(mf.loc["Retrain, appended"] - b0, "{:+.2f}"))
print("Retrain shuffled - B0:", (mf.loc["Retrain, shuffled"] - b0).round(3).to_dict(),
      " range", rng_str(mf.loc["Retrain, shuffled"] - b0, "{:+.2f}"))
t = endpoint(v2, "test_macro_f1", "Retrain, shuffled", "B0")
print("  post hoc Wilcoxon+Holm, Retrain shuffled - B0:\n", t.round(4).to_string(index=False))
t = endpoint(v2, "test_macro_f1", "Retrain, appended", "B0")
print("  post hoc Wilcoxon+Holm, Retrain appended - B0:\n", t.round(4).to_string(index=False))
print("SR+AL - Retrain appended (macro-F1):", (mf.loc["SR+AL"] - mf.loc["Retrain, appended"]).round(3).to_dict(),
      " range", rng_str(mf.loc["SR+AL"] - mf.loc["Retrain, appended"]))
mr = piv(v2, "test_min_class_recall")
print("min class recall, SR+AL - B0:", (mr.loc["SR+AL"] - mr.loc["B0"]).round(2).to_dict())

section("Paper table 'Certificates and targeted change (RQ4)' (means over 10 seeds)")
rows = {
    "Fix share, SR then AL (%)": ("churn_fix_share", "SR+AL", 1),
    "Fix share, retrained, shuffled (%)": ("churn_fix_share", "Retrain, shuffled", 1),
    "Broken, SR then AL (% of test rows)": ("churn_broken_share", "SR+AL", 2),
    "Broken, retrained, shuffled": ("churn_broken_share", "Retrain, shuffled", 2),
    "Broken, retrained, appended": ("churn_broken_share", "Retrain, appended", 2),
    "Tight certificate per edit, SR then AL (%)": ("cert_tight_mean_size_share", "SR+AL", 1),
    "Certified rows that changed, SR then AL (%)": ("cert_tight_precision", "SR+AL", 1),
}
for label, (val, arm, dp) in rows.items():
    v = piv(v2, val).loc[arm]
    print(f"{label:46s}", [round(x, dp) for x in v], " unrounded", [round(x, 4) for x in v])
fs = piv(v2, "churn_fix_share")
print("\nfix share SR+AL range", rng_str(fs.loc["SR+AL"], "{:.1f}"),
      "; retrain shuffled range", rng_str(fs.loc["Retrain, shuffled"], "{:.1f}"),
      "; retrain appended", fs.loc["Retrain, appended"].round(1).to_dict())
print("fix share SR+AL > appended on means:", (fs.loc["SR+AL"] > fs.loc["Retrain, appended"]).to_dict())
d = diffs(v2, "churn_fix_share", "SR+AL", "Retrain, appended")
print("  per-seed wins SR+AL over appended:", {k: f"{(x > 0).sum()}/{len(x)}" for k, x in d.items()})
br = piv(v2, "churn_broken_share")
print("broken SR+AL > appended on means:", (br.loc["SR+AL"] > br.loc["Retrain, appended"]).to_dict())
d = diffs(v2, "churn_broken_share", "SR+AL", "Retrain, appended")
print("  per-seed SR+AL breaks more than appended:", {k: f"{(x > 0).sum()}/{len(x)}" for k, x in d.items()})
ch = piv(v2, "churn_changed_share")
print("changed share, shuffled:", rng_str(ch.loc["Retrain, shuffled"]), " appended:",
      rng_str(ch.loc["Retrain, appended"]))
tk = piv(v2, "expl_topk_change_share")
print("top-3 changed SR+AL (no Skin):", tk.loc["SR+AL"].round(3).to_dict(),
      " range", rng_str(tk.loc["SR+AL"].drop("skin")))
print("top-3 changed, shuffled vs SR+AL:", (tk.loc["Retrain, shuffled"] - tk.loc["SR+AL"]).round(3).to_dict())

edit = v2[v2.arm.isin(["AL", "SR", "SR+AL"])]
print("\nC4 misses: tight", int(edit.cert_tight_misses.sum()), " loose", int(edit.cert_loose_misses.sum()),
      " edits", int(edit.cert_edits.sum()), " runs", len(edit))
for kind in ("tight", "loose"):
    for col in ("mean_size_share", "precision", "union_share"):
        v = piv(v2, f"cert_{kind}_{col}").loc["SR+AL"]
        print(f"SR+AL cert {kind} {col}:", v.round(3).to_dict(), " range", rng_str(v, "{:.2f}"))
    v = piv(v2, f"expl_{kind}_out_top3").loc["SR+AL"]
    print(f"SR+AL top-3 changed outside {kind} cert:", v.round(3).to_dict())

# ------------------------------------------------------------------------ explore
section("Exploratory checks E1-E4 (seeds 0-2)")
ex = pd.read_csv(RES / "refinement_v2_explore" / "runs.csv")
m = ex.pivot_table(index="arm", columns="dataset", values="test_macro_f1")[DS]
b0 = m.loc["B0"]
for arm in ["SR-300-train", "OE | A+B+C'-volume", "SR-all + OE | A+B+C'-volume", "CFMNN refit, +bought +acceptance",
            "RF, train only", "OE | A+B+C-oracle", "OE | A+B+C-paceshap", "OE | A+B+C-fewest"]:
    print(f"{arm:36s} - B0:", (m.loc[arm] - b0).round(2).to_dict())
print("SR-300-train - AL (OE|A+B+C'-volume):", (m.loc["SR-300-train"] - m.loc["OE | A+B+C'-volume"]).round(2).to_dict())
print("oracle cut - volume cut:", (m.loc["OE | A+B+C-oracle"] - m.loc["OE | A+B+C'-volume"]).round(3).to_dict(),
      " range", rng_str(m.loc["OE | A+B+C-oracle"] - m.loc["OE | A+B+C'-volume"]))
print("acceptance rows per dataset:", ex.groupby("dataset").n_accept.first().loc[DS].to_dict())
ee = ex[ex.cert_edits.notna() & (ex.cert_edits > 0)]
print("explore C4: edits", int(ee.cert_edits.sum()), " runs", len(ee), " tight misses",
      int(ee.cert_tight_misses.sum()), " loose misses", int(ee.cert_loose_misses.sum()))

# ------------------------------------------------------------------------------ v1
section("v1 confirmatory (seeds 10-19)")
v1 = pd.read_csv(RES / "refinement_confirmatory" / "runs.csv")
P = {"C": "order_ensemble | A+B+C", "vol": "order_ensemble | A+B+C'-volume",
     "rand": "order_ensemble | A+B+C'-random", "AB": "order_ensemble | A+B", "rsel": "random | A+B+C"}
for name, (a, b) in {"R1 C - C'vol": ("C", "vol"), "R2 C - A+B": ("C", "AB"),
                     "R3 C - C'rand": ("C", "rand"), "S1 OE - random sel": ("C", "rsel")}.items():
    t = endpoint(v1, "test_macro_f1", P[a], P[b])
    print(f"{name}: sig on {list(t[t.sig].dataset)}; means {t['mean'].round(2).tolist()}; range {rng_str(t['mean'])}")
