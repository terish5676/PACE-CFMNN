"""Recompute every number the write-up takes from tough-review batch 1 (Runs P1, P3, P7, P8).

Sources (home PC, research_audit/HOME_PC_RUNBOOK.md section 12):
  P1  results/time_to_accuracy/runs.csv   (part a: seeds 300-302; part b: seed 303)
  P3  results/skin_dedup/*.csv            (seeds 260-269)
  P7  results/flip_intervals/runs.csv     (seeds 340-349)
  P8  results/recurrence/runs.csv         (seeds 350-352; the rerun after Amendment 1)
Every CSV is read with float_precision="round_trip". The expected values are those written in
results/FINDINGS.md section 23 and the paper; a BAD line means the text must change.

Read-only. Run from the repository root (home PC):
    python -B research_audit/scripts/claims_tough_batch1.py > research_audit/scripts/claims_tough_batch1_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, section, wilcox_p

sys.stdout.reconfigure(encoding="utf-8")
BAD = []


def check(label, got, want):
    ok = got == want
    if not ok:
        BAD.append(label)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (expected: {want})"))


def r(x, nd=1):
    return round(float(x), nd)


def read(*parts):
    return pd.read_csv(RES.joinpath(*parts), float_precision="round_trip")


def boot_low(kms, pms, n_boot=2000, seed=0):
    """The lower end of run_time_to_accuracy.ratio_ci's 95% bootstrap interval (same draws)."""
    kms, pms = np.asarray(kms, float), np.asarray(pms, float)
    idx = np.random.default_rng(seed).integers(0, len(kms), size=(n_boot, len(kms)))
    return float(np.quantile(kms[idx].mean(axis=1) / pms[idx].mean(axis=1), 0.025))


# ---------------------------------------------------------------------------------------------- P1
section("P1 part (a): KernelSHAP's time to 1% of the exact values")
T = read("time_to_accuracy", "runs.csv")
T["budget"] = T.budget.astype(str)
A = T[T.part == "a"]
SIX = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
check("part (a) datasets", sorted(A.dataset.unique()), sorted(SIX))
check("part (a) seeds", sorted(A.seed.unique().tolist()), [300, 301, 302])
check("part (a) points per dataset and arm", int(A[A.arm == "PACE-SHAP"].groupby("dataset").size().unique()[0]), 30)
K = A[A.arm == "KernelSHAP"]
med = K.groupby(["dataset", "budget"]).rel_l1.median().unstack()
print((100 * med).round(2).loc[SIX].to_string())
check("no sampled budget reaches 1% median error", bool((med.drop(columns="all").stack() > 0.01).all()), True)
check("every coalition gives 0 error (median, 1e-9)", bool((med["all"].abs() <= 1e-9).all()), True)
check("Shuttle has no 700 budget", bool(np.isnan(med.loc["shuttle", "700"])), True)
check("median error at 700, 5 datasets, range (%)",
      (r(100 * med["700"].drop("shuttle").min(), 2), r(100 * med["700"].drop("shuttle").max(), 2)), (4.03, 7.25))
check("Shuttle median error at 400 (%)", r(100 * med.loc["shuttle", "400"], 2), 1.33)
ratios, lows = {}, {}
for ds in SIX:
    g = A[A.dataset == ds]
    pm = g[g.arm == "PACE-SHAP"].sort_values(["seed", "point"]).ms.to_numpy()
    km = g[(g.arm == "KernelSHAP") & (g.budget == "all")].sort_values(["seed", "point"]).ms.to_numpy()
    ratios[ds], lows[ds] = km.mean() / pm.mean(), boot_low(km, pm)
check("ratio at b* per dataset", [r(ratios[d]) for d in SIX], [16.9, 17.3, 18.0, 20.1, 26.4, 20.7])
check("ratio at b*, range", (r(min(ratios.values())), r(max(ratios.values()))), (16.9, 26.4))
check("every bootstrap interval above 1", bool(min(lows.values()) > 1), True)

section("P1 part (b): the clean re-time")
B = T[T.part == "b"]
check("part (b) seed", B.seed.unique().tolist(), [303])
check("part (b) datasets", B.dataset.nunique(), 13)
rb, lb = {}, {}
for ds, g in B.groupby("dataset"):
    pm = g[g.arm == "PACE-SHAP"].sort_values("point").ms.to_numpy()
    km = g[g.arm == "KernelSHAP"].sort_values("point").ms.to_numpy()
    rb[ds], lb[ds] = km.mean() / pm.mean(), boot_low(km, pm)
print({k: r(v, 2) for k, v in rb.items()})
check("KernelSHAP / PACE-SHAP, range", (r(min(rb.values())), r(max(rb.values()))), (3.3, 8.0))
check("median", r(np.median(list(rb.values()))), 4.9)
check("every interval above 1", bool(min(lb.values()) > 1), True)

# ---------------------------------------------------------------------------------------------- P3
section("P3: Skin without its duplicate rows")
R = read("skin_dedup", "refinement.csv")
check("seeds", sorted(R.seed.unique().tolist()), list(range(260, 270)))
check("dataset", R.dataset.unique().tolist(), ["skin_dedup"])
pv = R.pivot_table(index="seed", columns="arm", values="test_macro_f1")
d = (pv["SR"] - pv["B0"]).to_numpy()
check("SR - B0 macro-F1 (2 dp)", r(d.mean(), 2), 6.20)
check("seeds SR better", int((d > 0).sum()), 10)
check("p (4 dp)", r(wilcox_p(d), 4), 0.0020)
dd = (pv["SR+AL"] - pv["SR"]).to_numpy()
check("SR+AL - SR (2 dp), seeds better", (r(dd.mean(), 2), int((dd > 0).sum())), (0.62, 10))
F = read("skin_dedup", "reference.csv")
pr = F.pivot_table(index="seed", columns="arm", values="test_macro_f1")
dr = (pr["SR"] - pr["B0"]).to_numpy()
check("reference (original Skin) SR - B0 (2 dp), seeds better", (r(dr.mean(), 2), int((dr > 0).sum())), (7.56, 10))
Acc = read("skin_dedup", "accuracy.csv").groupby("model").accuracy.mean()
check("accuracy CFMNN / 10-NN / RF (2 dp)", [r(Acc[m], 2) for m in ("CFMNN", "10-NN", "Random Forest")],
      [92.92, 99.73, 99.80])
check("gap to the Random Forest (1 dp)", r(Acc["Random Forest"] - Acc["CFMNN"]), 6.9)
Th = read("skin_dedup", "theta.csv")
val = Th[Th.is_val]
check("theta by validation on every seed", sorted(val.theta.unique().tolist()), [0.05])
check("theta 0.05: accuracy (2 dp), boxes (mean, 0 dp)",
      (r(val.accuracy.mean(), 2), int(round(val.boxes.mean()))), (99.43, 2011))
Rp = read("skin_dedup", "repair.csv")
sr = Rp[Rp.model == "SR-0.3"].set_index("seed")
nfr_sr = 100 * sr.test_broken / sr.n_test
nfr_val = val.set_index("seed").nfr
dn = (nfr_sr - nfr_val.loc[nfr_sr.index]).to_numpy()
check("NFR, SR-0.3 and B0 at theta-val (2 dp)", (r(nfr_sr.mean(), 2), r(nfr_val.mean(), 2)), (1.27, 0.34))
check("repair breaks more, by (2 dp), on seeds", (r(dn.mean(), 2), int((dn > 0).sum())), (0.93, 10))

# ---------------------------------------------------------------------------------------------- P7
section("P7: exact flip sets")
P7 = read("flip_intervals", "runs.csv")
check("pairs", len(P7), 85680)
check("datasets, seeds", (P7.dataset.nunique(), sorted(P7.seed.unique().tolist())), (11, list(range(340, 350))))
dense = P7[P7.dense]
check("densely checked pairs", len(dense), 17500)
check("dense: no flip outside, none missed", bool((dense.dense_outside == 0).all() and (dense.dense_missed == 0).all()), True)
check("the 32 values agree with the exact set on every pair", bool(P7.grid32_agrees.all()), True)
check("values at a breakpoint left out", int(P7.grid32_at_break.sum()), 2414)
fl = P7[P7.exact_flips]
miss = fl[~fl.grid32_flips]
check("flipping pairs, missed by the 32 values", (len(fl), len(miss)), (34832, 6356))
check("missed share overall (1 dp, %)", r(100 * len(miss) / len(fl)), 18.2)
share = (100 * miss.groupby("dataset").size() / fl.groupby("dataset").size()).fillna(0)
print(share.round(1).to_string())
check("datasets above 10%", sorted(share[share > 10].index), sorted(["magic", "thyroid", "page_blocks", "shuttle"]))
check("their range (1 dp)", (r(share[share > 10].min()), r(share[share > 10].max())), (13.9, 56.2))
check("largest share under 10% (1 dp)", r(share[share <= 10].max()), 8.3)

# ---------------------------------------------------------------------------------------------- P8
section("P8: the recurrence exact mode")
P8 = read("recurrence", "runs.csv")
MAT, REC, BRU, EXA = "PACE-SHAP, matmul", "PACE-SHAP, recurrence", "Brute force, vectorised", "shap ExactExplainer"
check("seeds", sorted(P8.seed.unique().tolist()), [350, 351, 352])
check("recurrence and brute force within 1e-12 of matmul",
      bool(P8[P8.arm.isin([REC, BRU])].max_abs_diff_matmul.max() <= 1e-12), True)
check("PACE-SHAP modes within 1e-9 of ExactExplainer (Wine, Letter)",
      bool(P8[P8.arm.isin([MAT, REC])].max_abs_diff_exact.dropna().max() <= 1e-9), True)
ms = P8.pivot_table(index="dataset", columns="arm", values="ms_per_explanation", aggfunc="mean")
mr = ms[MAT] / ms[REC]
br = ms[BRU] / ms[REC]
print(pd.DataFrame({"matmul/rec": mr.round(2), "brute/rec": br.round(1)}).to_string())
ten = [d for d in ms.index if d not in ("wine", "letter")]
check("matmul / recurrence at 9-10 features, range", (r(mr[ten].min(), 2), r(mr[ten].max(), 2)), (1.03, 1.23))
check("matmul / recurrence, Wine and Letter", (r(mr["wine"], 2), r(mr["letter"], 2)), (2.13, 1.40))
check("recurrence faster everywhere", bool((mr > 1).all()), True)
check("brute force / recurrence, range", (int(round(br.min())), int(round(br.max()))), (47, 112))
check("Letter recurrence, s per explanation", r(ms.loc["letter", REC] / 1000), 26.1)
check("decision rule: Letter above 5 s", bool(ms.loc["letter", REC] > 5000), True)

section("Verdict")
print("ALL OK" if not BAD else f"{len(BAD)} BAD: {BAD}")
