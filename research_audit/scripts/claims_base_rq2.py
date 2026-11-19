"""Recompute the paper's base-model (Section 3, Section 7 'Base model') and RQ2 numbers (rev 262).

Read-only. Run from research_audit/scripts:
    python -B claims_base_rq2.py > claims_base_rq2_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, holm, rng_str, section, wilcox_p

sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------ base model
section("G0: the two fixes and the paper-parity table (g0/runs.csv, seeds 0-4)")
g0 = pd.read_csv(RES / "g0/runs.csv")
acc = g0.groupby(["dataset", "model"]).accuracy.mean().unstack()
print(acc.round(2).to_string())
print("Shuttle original -> fixed: %.2f -> %.2f" % (acc.loc["shuttle", "CFMNN-original"], acc.loc["shuttle", "CFMNN-fixed"]))
print("Digits fixed -> fixed+search: %.2f -> %.2f" % (acc.loc["digits", "CFMNN-fixed"], acc.loc["digits", "CFMNN-fixed+search"]))
print("Skin fixed -> fixed+search: %.2f -> %.2f" % (acc.loc["skin", "CFMNN-fixed"], acc.loc["skin", "CFMNN-fixed+search"]))
print("Sepsis original %.2f (sd %.2f); majority %.2f" % (
    acc.loc["sepsis", "CFMNN-original"], g0[(g0.dataset == "sepsis") & (g0.model == "CFMNN-original")].accuracy.std(),
    acc.loc["sepsis", "Majority"]))
for ds in ["thyroid", "shuttle", "skin"]:
    print(f"  parity {ds}: paper's rules {acc.loc[ds, 'CFMNN-original']:.1f}, both fixes {acc.loc[ds, 'CFMNN-fixed+search']:.1f}")
print("  paper's ensemble values are quoted from the base paper (g0/summary.md line: thyroid 96.90, shuttle 94.04, "
      "skin 97.63, sepsis 68.76)")

section("Accuracy table (review_v1/accuracy/runs.csv, seeds 0-4)")
ac = pd.read_csv(RES / "review_v1/accuracy/runs.csv")
m = ac.groupby(["dataset", "model"]).accuracy.agg(["mean", "std"]).unstack()
tab = pd.DataFrame({k: m["mean"][k].round(1).astype(str) + " ± " + m["std"][k].round(1).astype(str)
                    for k in ["CFMNN", "10-NN", "Random Forest"]})
tab.insert(0, "n_train", ac.groupby("dataset").n_train.first())
gap = m["mean"]["CFMNN"] - m["mean"][["10-NN", "Random Forest"]].max(axis=1)
tab["gap"] = gap.round(1)
print(tab.to_string())
print("gap range:", rng_str(-gap, "{:.1f}"), "; CFMNN below the better reference on", int((gap < 0).sum()), "of", len(gap))
print("gap < 3:", sorted(gap[gap > -3].index.tolist()), " 3-9:", sorted(gap[(gap <= -3) & (gap > -9)].index.tolist()),
      " >= 9:", sorted(gap[gap <= -9].index.tolist()))

# ------------------------------------------------------------------ RQ2
SYN = ["S1", "S2", "S3", "S2-imbalanced", "S2-label-noise", "S2-shifted"]
REAL = ["digits", "thyroid", "skin", "magic", "letter", "landsat", "page_blocks"]
UNITS = ["Synthetic"] + REAL


def units(d, col="auprc"):
    syn = d[d.dataset.isin(SYN)].groupby(["seed", "method"])[col].mean().reset_index()
    syn["dataset"] = "Synthetic"
    return pd.concat([syn, d[d.dataset.isin(REAL)][["dataset", "seed", "method", col]]], ignore_index=True)


def compare(u, a, b, un=UNITS, col="auprc"):
    rows = []
    for x_ in un:
        x = u[(u.dataset == x_) & (u.method == a)].set_index("seed")[col]
        y = u[(u.dataset == x_) & (u.method == b)].set_index("seed")[col]
        s = x.index.intersection(y.index)
        d = (x[s] - y[s]).to_numpy(float)
        rows.append(dict(unit=x_, a=x[s].mean(), b=y[s].mean(), diff=d.mean(), a_higher_seeds=int((d > 0).sum()),
                         p=wilcox_p(d)))
    o = pd.DataFrame(rows)
    o["p_holm"] = holm(o.p)
    o["verdict"] = np.where(o.p_holm < 0.05, np.where(o["diff"] > 0, "higher", "lower"), "level")
    return o


def show(o, a, b):
    v = o.verdict.value_counts().to_dict()
    print(f"\n{a} vs {b}: {v}")
    print(o.round(4).to_string(index=False))


section("RQ2 confirmatory run (error_detection_confirmatory, seeds 10-19)")
ed = pd.read_csv(RES / "error_detection_confirmatory/runs.csv")
print("seeds", sorted(ed.seed.unique()), "datasets", sorted(ed.dataset.unique()))
mn = ed[ed.method == "Margin"].groupby("dataset").n_errors.min()
print("min errors per seed:", mn.to_dict())
u = units(ed)
cols = ["Margin", "PD + U*", "PRS*", "Trust Score", "kNN disagreement", "Order ensemble"]
tab = u[u.method.isin(cols)].groupby(["dataset", "method"]).auprc.mean().unstack()[cols].reindex(UNITS)
br = units(ed, "base_rate")
tab.insert(0, "share of errors", br[br.method == "Margin"].groupby("dataset").base_rate.mean().reindex(UNITS))
print(tab.round(2).to_string())
print("unrounded:\n", tab.round(4).to_string())
for a, b, un in [("PD + U*", "Margin", UNITS), ("PRS*", "PD + U*", UNITS), ("Trust Score", "PD + U*", UNITS),
                 ("kNN disagreement", "PD + U*", UNITS), ("Order ensemble", "PD + U*", UNITS)]:
    show(compare(u, a, b, un), a, b)
tie = ed[ed.method == "Margin"].groupby("dataset").tie_share.mean()
print("\ntie share per dataset:", tie.round(3).to_dict())
real_ties = [d for d in REAL if tie.get(d, 0) >= 0.01]
print("real datasets with >= 1% ties:", real_ties)
show(compare(u, "U*", "U", real_ties), "U*", "U")
show(compare(u, "U", "Random", UNITS), "U", "Random")
ei = compare(u, "EI", "Random", UNITS)
show(ei, "EI", "Random")
kr = pd.read_csv(RES / "error_detection_confirmatory/knn_reference.csv")
k = kr.groupby("dataset")[["cfmnn", "knn"]].mean()
print("\n10-NN vs CFMNN accuracy (knn_reference.csv):\n", k.round(1).to_string())
print("10-NN more accurate on", int((k.knn > k.cfmnn).sum()), "of", len(k), "; CFMNN ahead on:",
      list(k[k.knn <= k.cfmnn].index))

section("RQ2 combination run (error_detection_v2, seeds 20-29)")
v2 = pd.read_csv(RES / "error_detection_v2/runs.csv")
u2 = units(v2)
cols = ["kNN", "Trust Score", "LR: kNN + TS", "LR: kNN + TS + order ens."]
tab = u2[u2.method.isin(cols)].groupby(["dataset", "method"]).auprc.mean().unstack()[cols].reindex(UNITS)
br = units(v2, "base_rate")
tab.insert(0, "share of errors", br[br.method == "kNN"].groupby("dataset").base_rate.mean().reindex(UNITS))
print(tab.round(2).to_string())
print("unrounded:\n", tab.round(4).to_string())
for a, b in [("LR: kNN + TS + order ens.", "LR: kNN + TS"), ("LR: kNN + TS + order ens.", "kNN"),
             ("LR: kNN + TS + order ens.", "Trust Score"), ("LR: all", "LR: kNN + TS + order ens.")]:
    show(compare(u2, a, b), a, b)
pb = ed[(ed.dataset == "page_blocks") & (ed.method == "Margin")].sort_values("seed")
print("\nPage Blocks base rate per seed, confirmatory run:", pb.base_rate.round(3).tolist())
pb2 = v2[(v2.dataset == "page_blocks") & (v2.method == "kNN")].sort_values("seed")
print("Page Blocks base rate per seed, combination run:", pb2.base_rate.round(3).tolist())

section("RQ2 ensembles test (review_v1/ensembles, seeds 60-69)")
en = pd.read_csv(RES / "review_v1/ensembles/runs.csv")
u3 = units(en)
for a, b in [("LR: kNN + TS + order ens.", "LR: kNN + TS"), ("LR: kNN + TS + bagged ens.", "LR: kNN + TS"),
             ("LR: kNN + TS + bagged ens.", "LR: kNN + TS + order ens."),
             ("LR: kNN + TS + order ens. + bagged ens.", "LR: kNN + TS + bagged ens."),
             ("LR: kNN + TS + SGD order ens.", "LR: kNN + TS")]:
    show(compare(u3, a, b), a, b)
