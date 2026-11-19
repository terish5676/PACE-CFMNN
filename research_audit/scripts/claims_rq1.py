"""Recompute every RQ1 number in the paper (rev 262) from the committed raw CSVs.

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_rq1.py > research_audit/scripts/claims_rq1_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, holm, paired, rng_str, section, tally, wilcox_p

sys.stdout.reconfigure(encoding="utf-8")

D11 = ["S1", "S2", "S3", "wine", "breast_cancer", "thyroid", "shuttle", "magic", "letter", "landsat", "page_blocks"]
D13 = ["S1", "S2", "S3", "S1c", "S2c", "wine", "breast_cancer", "thyroid", "shuttle", "magic", "letter", "landsat",
       "page_blocks"]
REAL8 = D11[3:]
DEL, INS = ("margin_auc", True), ("insertion_margin_auc", False)


def both_tests(df, a, b, units, cols=(DEL, INS)):
    for col, lower in cols:
        o = paired(df, a, b, col, units, lower_better=lower)
        print(f"  {a} vs {b} [{col}]: {tally(o)}")
    return o


# ------------------------------------------------------------------ v3 (second explanation run)
section("v3 faithfulness (seeds 30-39): counts")
v3 = pd.read_csv(RES / "faithfulness_v3/runs.csv")
print("seeds", sorted(v3.seed.unique()), "n per dataset/seed:", v3.groupby("dataset").n.first().to_dict())
for b in ["KernelSHAP", "KernelSHAP-strong", "LIME", "PACE-flip"]:
    both_tests(v3, "PACE-SHAP", b, D11)
both_tests(v3, "PACE-SHAP-20", "KernelSHAP", D11)
print("  any method significantly better than PACE-SHAP in v3?")
for b in v3.method.unique():
    if b in ("PACE-SHAP",):
        continue
    for col, lower in (DEL, INS):
        o = paired(v3, "PACE-SHAP", b, col, D11, lower_better=lower)
        if (o.verdict == "worse").any():
            print(f"    PACE-SHAP worse than {b} on {list(o[o.verdict == 'worse'].unit)} [{col}]")

section("v3: effect sizes, PACE-SHAP vs KernelSHAP by deletion (Figure 2)")
for b in ["KernelSHAP", "KernelSHAP-strong"]:
    o = paired(v3, "PACE-SHAP", b, "margin_auc", D11, lower_better=True)
    m = v3.groupby(["dataset", "method"]).margin_auc.mean().unstack()
    o["rel_gap_%"] = [100 * (m.loc[u, b] - m.loc[u, "PACE-SHAP"]) / (m.loc[u, "Random"] - m.loc[u, b]) for u in o.unit]
    o["ci_excludes_0"] = (o.hi < 0) | (o.lo > 0)
    print(f"-- vs {b}")
    print(o[["unit", "diff", "lo", "hi", "dz", "p", "p_holm", "verdict", "rel_gap_%", "ci_excludes_0"]]
          .round(4).to_string(index=False))
    print("   CI excludes 0 but not significant:", list(o[o.ci_excludes_0 & (o.verdict == "level")].unit))
    sig = o[o.verdict == "better"]
    print("   d_z among significant wins:", rng_str(sig.dz.abs()))

section("v3: mean deletion-margin AUC table (paper table after Figure 2)")
cols = ["PACE-SHAP", "KernelSHAP", "KernelSHAP-strong", "LIME", "PACE-flip", "Random"]
tab = v3.groupby(["dataset", "method"]).margin_auc.mean().unstack().reindex(D11)[cols]
print(tab.round(4).to_string())
print("lowest per row:", tab.idxmin(axis=1).to_dict())

section("v3: stability, ground truth and flip mass")
st = v3.groupby(["dataset", "method"]).stability_top3.mean().unstack()
sampled = ["wine", "breast_cancer", "thyroid", "letter", "landsat"]
print("PACE-SHAP sampled-mode datasets (d > 12):", sampled)
print("  PACE-SHAP stability on sampled:", rng_str(st.loc[sampled, "PACE-SHAP"]), " exact:",
      rng_str(st.loc[[d for d in D11 if d not in sampled], "PACE-SHAP"]))
for meth in ["LIME", "KernelSHAP", "PACE-flip", "PACE-SHAP-20"]:
    print(f"  {meth}: {rng_str(st[meth])}")
o = paired(v3, "PACE-SHAP-20", "KernelSHAP", "stability_top3", D11)
print("  post hoc: PACE-SHAP-20 vs KernelSHAP stability:", tally(o))
gt = v3[v3.dataset.isin(["S1", "S2", "S3"])].groupby(["dataset", "method"]).gt_top3_precision.mean().unstack()
for meth in ["LIME", "PACE-SHAP", "KernelSHAP", "PACE-flip"]:
    print(f"  ground truth top-3 {meth}: {rng_str(gt[meth])}")
fm = pd.read_csv(RES / "faithfulness_v3/flip_mass.csv")
print("  flip mass on relevant features:", rng_str(100 * fm.groupby("dataset").flip_mass_relevant.mean(), "{:.1f}"))
o = paired(v3, "LIME", "PACE-SHAP", "margin_auc", D11, lower_better=True)
print("  LIME vs PACE-SHAP deletion:", tally(o))
o = paired(v3, "LIME", "PACE-SHAP", "insertion_margin_auc", D11)
print("  LIME vs PACE-SHAP insertion:", tally(o))

section("v3 timing (3 seeds)")
t3 = pd.read_csv(RES / "faithfulness_v3/timing.csv")
T3 = t3.groupby(["dataset", "method"]).ms_per_explanation.mean().unstack()
r = T3["PACE-SHAP"] / T3["KernelSHAP"]
print("PACE-SHAP / KernelSHAP time:", r.round(2).to_dict())
print("  slower on:", list(r[r > 1].index), "range", rng_str(r[r > 1]))
r20 = T3["KernelSHAP"] / T3["PACE-SHAP-20"]
print("KernelSHAP / PACE-SHAP-20 time:", rng_str(r20))

# ------------------------------------------------------------------ first run
section("v2 first explanation run (seeds 0-9)")
v2 = pd.read_csv(RES / "faithfulness/runs.csv")
print("seeds", sorted(v2.seed.unique()), "methods", v2.method.unique().tolist())
for col, lower in (DEL, INS):
    o = paired(v2, "KernelSHAP", "PACE-flip", col, D11, lower_better=lower)
    print(f"  KernelSHAP vs PACE-flip [{col}]: {tally(o)}")
occ = [m for m in v2.method.unique() if m.startswith("Occlusion")]
if occ:
    a = v2[v2.method == occ[0]].set_index(["dataset", "seed"]).margin_auc
    b = v2[v2.method == "PACE-flip"].set_index(["dataset", "seed"]).margin_auc
    print(f"  max |{occ[0]} - PACE-flip| deletion AUC over dataset-seeds: {np.abs(a - b.loc[a.index]).max():.2e}")

section("Speed: PACE-flip vs occlusion, LIME, KernelSHAP (faithfulness/timing.csv)")
tm = pd.read_csv(RES / "faithfulness/timing.csv")
print("seeds", sorted(tm.seed.unique()))
T = tm.groupby(["dataset", "method"]).ms_per_explanation.mean().unstack()
print("PACE-flip ms per explanation:", rng_str(T["PACE-flip"], "{:.3f}"))
for meth in [c for c in T.columns if c.startswith("Occlusion")] + ["LIME", "KernelSHAP"]:
    print(f"  {meth} / PACE-flip: {rng_str(T[meth] / T['PACE-flip'], '{:.0f}')}")
ob = pd.read_csv(RES / "review_v1/occlusion_batched/runs.csv")
g = ob.groupby("dataset")[["flip_ms", "occlusion_loop_ms", "occlusion_batched_ms"]].mean()
print("occlusion_batched seeds", sorted(ob.seed.unique()))
print("  loop / batched:", rng_str(g.occlusion_loop_ms / g.occlusion_batched_ms))
print("  batched / PACE-flip:", rng_str(g.occlusion_batched_ms / g.flip_ms, "{:.0f}"))
print("  max diff batched vs flip:", ob.max_diff_batched_vs_flip.max())
su = pd.read_csv(RES / "speed_uci/runs.csv")
print("speed_uci datasets", sorted(su.dataset.unique()))
print("  share of boxes skipped:", rng_str(100 * su.groupby("dataset").share_of_boxes_skipped.mean(), "{:.0f}"))
sk_cols = [c for c in v3.columns if "skip" in c]
print("  (v3 has skip columns?)", sk_cols)

# ------------------------------------------------------------------ exact values
section("Exact PACE-SHAP vs ExactExplainer (review_v1/exact)")
ex = pd.read_csv(RES / "review_v1/exact/runs.csv")
e = ex[ex.part == "exact"]
print("datasets", sorted(e.dataset.unique()), "seeds", sorted(e.seed.unique()), "n", e.n.unique())
print("  max abs diff (all exact rows):", e.max_abs_diff.max())
E = e.groupby(["dataset", "method"]).ms_per_explanation.mean().unstack()
print(E.round(1).to_string())
print("  ExactExplainer / PACE-SHAP:", (E["shap ExactExplainer"] / E["PACE-SHAP"]).round(1).to_dict())
s = ex[ex.part == "sampled"]
S = s.groupby(["dataset", "method", "orders_per_row"], dropna=False)[["rel_l1_error", "top3_overlap",
                                                                        "ms_per_explanation"]].mean()
print(S.round(4).to_string())

# ------------------------------------------------------------------ ties, raw curves, untied, LIME-200, stability
section("Planned test, seeds 50-59 (review_v1/faithfulness)")
rv = pd.read_csv(RES / "review_v1/faithfulness/runs.csv")
for col, lower in [("margin_auc_raw", True), ("insertion_margin_auc_raw", False)]:
    print(f"  raw PACE-SHAP vs KernelSHAP [{col}]: {tally(paired(rv, 'PACE-SHAP', 'KernelSHAP', col, D11, lower_better=lower))}")
ps = rv[rv.method == "PACE-SHAP"]
untied_ok = [d for d in D11 if ((ps[ps.dataset == d].n - ps[ps.dataset == d].n_tied) >= 10).all()]
print("  datasets with >= 10 untied points in every seed:", len(untied_ok), untied_ok)
for col, lower in [("margin_auc_untied", True), ("insertion_margin_auc_untied", False)]:
    print(f"  untied PACE-SHAP vs KernelSHAP [{col}]: "
          f"{tally(paired(rv, 'PACE-SHAP', 'KernelSHAP', col, untied_ok, lower_better=lower))}")
for col, lower in [("margin_auc", True), ("insertion_margin_auc", False)]:
    print(f"  LIME (200 rows) vs PACE-SHAP [{col}]: {tally(paired(rv, 'LIME', 'PACE-SHAP', col, D11, lower_better=lower))}")
for col in ["stability_fixed", "stability_new"]:
    print(f"  stability PACE-SHAP vs KernelSHAP [{col}]: {tally(paired(rv, 'PACE-SHAP', 'KernelSHAP', col, D11))}")
print("  any method better than PACE-SHAP here?")
for b in rv.method.unique():
    if b == "PACE-SHAP":
        continue
    for col, lower in [("margin_auc_raw", True), ("insertion_margin_auc_raw", False), ("margin_auc_untied", True),
                       ("insertion_margin_auc_untied", False)]:
        un = untied_ok if "untied" in col else D11
        o = paired(rv, "PACE-SHAP", b, col, un, lower_better=lower)
        if (o.verdict == "worse").any():
            print(f"    worse than {b} on {list(o[o.verdict == 'worse'].unit)} [{col}]")
ties_file = RES / "review_v1/ties_v3.csv"
tv = pd.read_csv(ties_file)
print("  ties_v3.csv columns:", tv.columns.tolist())
num = [c for c in tv.columns if "tie" in c and tv[c].dtype != object]
print(tv.groupby("dataset")[num].mean().round(3).to_string())

# ------------------------------------------------------------------ conditional (B3)
section("Conditional replacement (extensions_v1/conditional, seeds 40-49)")
c = pd.read_csv(RES / "extensions_v1/conditional/runs.csv")
for a, b in [("PACE-SHAP-cond", "KernelSHAP-cond"), ("PACE-SHAP", "PACE-SHAP-cond")]:
    both_tests(c, a, b, D13, cols=[("cond_margin_auc", True), ("cond_insertion_margin_auc", False)])
print("  matched comparison, usual tests:")
both_tests(c, "PACE-SHAP", "KernelSHAP", D13)
o = paired(c, "PACE-SHAP-cond", "PACE-SHAP", "cond_margin_auc", ["shuttle"], lower_better=True)
print("  Shuttle cond deletion: cond %.3f vs usual %.3f, cond better in %d seeds" % (o.a[0], o.b[0], o.a_better_seeds[0]))
rl = pd.read_csv(RES / "extensions_v1/conditional/realism.csv")
rt = rl.groupby(["dataset", "method"]).off_share.mean().unstack() * 100
print(rt.round(1).to_string())
ns = [d for d in D13 if d != "shuttle"]
print("  outside Shuttle: cond max %.1f, usual max %.1f" % (rt.loc[ns, "shap_cond"].max(), rt.loc[ns, "shap_int"].max()))
print("  test inputs off (usual game) on Shuttle/Letter/Landsat:", rt.loc[["shuttle", "letter", "landsat"], "test_int"]
      .round(1).to_dict())
for path in ["posthoc_sigma.csv", "posthoc_rows.csv"]:
    p = pd.read_csv(RES / "extensions_v1/conditional" / path)
    key = "sigma" if "sigma" in p else "rows"
    print(f"-- {path}: datasets {sorted(p.dataset.unique())}, seeds {sorted(p.seed.unique())}")
    print(p.groupby(["dataset", key])[["eff_rows_small", "off_share", "margin_auc", "cond_margin_auc"]].mean()
          .round(3).to_string())
sg = pd.read_csv(RES / "extensions_v1/conditional/posthoc_sigma.csv")
if "eff_rows_small" in sg:
    pass
print("  datasets where the 0.1 game rests on about one row (eff rows, large coalitions) -- from runs? see extra:")
ex2 = pd.read_csv(RES / "extensions_v1/conditional/extra.csv")
print("  extra.csv columns:", ex2.columns.tolist())

# ------------------------------------------------------------------ counterfactuals
section("Counterfactuals (extensions_v1/counterfactuals, seeds 40-49)")
cf = pd.read_csv(RES / "extensions_v1/counterfactuals/runs.csv")
rows = []
for ds in D11:
    per = []
    for s in sorted(cf.seed.unique()):
        g = cf[(cf.dataset == ds) & (cf.seed == s)]
        P, S, D, N = (g[g.method == m].set_index("point") for m in ["PACE", "PACE-sparse", "DiCE",
                                                                    "Nearest unlike neighbour"])
        bPD = P.index[P.valid.astype(bool) & D.valid.reindex(P.index).fillna(False).astype(bool)]
        bSD = S.index[S.valid.astype(bool) & D.valid.reindex(S.index).fillna(False).astype(bool)]
        bSP = S.index[S.valid.astype(bool) & P.valid.reindex(S.index).fillna(False).astype(bool)]
        per.append(dict(seed=s, nboth=len(bPD), vP=P.valid.mean(), vD=D.valid.mean(), vS=S.valid.mean(),
                        lP=P.loc[bPD, "l1"].mean(), lD=D.loc[bPD, "l1"].mean(),
                        fP=P.loc[bPD, "n_changed"].mean(), fD=D.loc[bPD, "n_changed"].mean(),
                        pP=P.loc[bPD, "plausibility"].mean(), pD=D.loc[bPD, "plausibility"].mean(),
                        mP=P.loc[bPD, "ms"].median(), mD=D.loc[bPD, "ms"].median(), mS=S.ms.median(),
                        fS=S.loc[bSD, "n_changed"].mean(), fD2=D.loc[bSD, "n_changed"].mean(),
                        lS=S.loc[bSD, "l1"].mean(), lD2=D.loc[bSD, "l1"].mean(),
                        pS=S.loc[bSD, "plausibility"].mean(), pD2=D.loc[bSD, "plausibility"].mean(),
                        oneS=(S.loc[bSD, "n_changed"] == 1).mean(), oneD=(D.loc[bSD, "n_changed"] == 1).mean(),
                        pS3=S.loc[bSP, "plausibility"].mean(), pP3=P.loc[bSP, "plausibility"].mean(),
                        fN=N.n_changed.mean(), lockN=N.locked_kept.mean() if N.locked_kept.notna().any() else np.nan,
                        lockP=P.locked_kept.mean(), lockD=D.locked_kept.mean()))
    q = pd.DataFrame(per)
    r = dict(ds=ds, nboth=f"{q.nboth.min()}-{q.nboth.max()}", vP=100 * q.vP.mean(), vD=100 * q.vD.mean(),
             vS=100 * q.vS.mean())
    for k, (x, y) in {"l1": ("lP", "lD"), "feat": ("fP", "fD"), "plaus": ("pP", "pD"), "ms": ("mP", "mD"),
                      "sF": ("fS", "fD2"), "sL": ("lS", "lD2"), "sP": ("pS", "pD2"), "sT": ("mS", "mD")}.items():
        r[k + "_P"], r[k + "_D"], r[k + "_p"] = q[x].mean(), q[y].mean(), wilcox_p(q[x] - q[y])
    r.update(oneS=100 * q.oneS.mean(), oneD=100 * q.oneD.mean(), farther=100 * (q.pS3.mean() / q.pP3.mean() - 1),
             fN=q.fN.mean(), lockN=100 * q.lockN.mean(), lockP=q.lockP.mean(), lockD=q.lockD.mean())
    rows.append(r)
o = pd.DataFrame(rows)
for k in ["l1", "feat", "plaus", "ms", "sF", "sL", "sP", "sT"]:
    o[k + "_h"] = holm(o[k + "_p"])
print(o[["ds", "nboth", "vP", "vD", "vS", "l1_P", "l1_D", "l1_h", "feat_P", "feat_D", "feat_h", "plaus_P", "plaus_D",
         "plaus_h", "ms_P", "ms_D", "ms_h"]].round(3).to_string(index=False))
print("speed-up DiCE/PACE:", rng_str(o.ms_D / o.ms_P, "{:.0f}"))
print("PACE-sparse vs DiCE:")
o["sF_diff"] = o.sF_P - o.sF_D
print(o[["ds", "sF_P", "sF_D", "sF_diff", "sF_h", "sL_P", "sL_D", "sL_h", "sP_P", "sP_D", "sP_h", "sT_P", "sT_D", "sT_h",
         "oneS", "oneD", "farther"]].round(3).to_string(index=False))
print("nearest unlike neighbour features changed:", rng_str(o.fN, "{:.1f}"), " Thyroid locks kept:",
      o.set_index("ds").loc["thyroid", "lockN"])
print("locks kept PACE/DiCE on Thyroid:", o.set_index("ds").loc["thyroid", ["lockP", "lockD"]].to_dict())

# ------------------------------------------------------------------ global importance
section("Global importance")
for path in ["global_importance/runs.csv", "extensions_v1/global_importance/runs.csv"]:
    gi = pd.read_csv(RES / path)
    col = "rho_Permutation importance"
    print(f"-- {path} seeds {sorted(gi.seed.unique())}")
    pairs = [("PACE-flip, averaged", "Mean abs KernelSHAP")]
    if "Mean abs PACE-SHAP" in gi.method.unique():
        pairs.append(("Mean abs PACE-SHAP", "Mean abs KernelSHAP"))
    for a, b in pairs:
        o = paired(gi, a, b, col, D11)
        print(f"  {a} vs {b}: higher mean on {(o['diff'] > 0).sum()}/11; {tally(o)}")
        print("   lower mean on:", list(o[o['diff'] < 0].unit))
    print("  mean seconds per dataset, max over datasets:", gi.groupby(["method", "dataset"]).seconds.mean()
          .groupby("method").max().round(1).to_dict())
    print("  max seconds in one run:", gi.groupby("method").seconds.max().round(1).to_dict())
    gt3 = gi[gi.dataset.isin(["S1", "S2", "S3"])].groupby("method").gt_top3.min() if "gt_top3" in gi else None
    print("  min ground-truth top3 per method on S1-S3:", None if gt3 is None else gt3.round(3).to_dict())

# ------------------------------------------------------------------ interactions
section("Interactions (extensions_v1/interactions)")
it = pd.read_csv(RES / "extensions_v1/interactions/runs.csv")
for ds in sorted(it.dataset.unique()):
    top01, leads = 0, []
    for s, g in it[it.dataset == ds].groupby("seed"):
        g = g.sort_values("strength", ascending=False)
        top01 += int((g.iloc[0].i, g.iloc[0].j) == (0, 1))
        s01 = g[(g.i == 0) & (g.j == 1)].strength.iloc[0]
        leads.append(s01 / g[~((g.i == 0) & (g.j == 1))].strength.iloc[0])
    print(f"  {ds}: (x0,x1) strongest in {top01}/{it[it.dataset == ds].seed.nunique()} seeds; "
          f"lead ratio {np.mean(leads):.2f} ({min(leads):.2f}-{max(leads):.2f})")
