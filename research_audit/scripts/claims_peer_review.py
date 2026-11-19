"""Recompute every number the peer-review revision (paper revs 327-439) added to the paper.

The revision answered reviews/fable_peer_review.md with no new runs: equivalence tests, effect
sizes, the synthetic sets as one unit, the other PACE methods, ties, where the base model errs,
the rows a shrink cuts off, Figure 1, the power floor and the S1-S3 generators. Its own analysis
script is scripts/peer_review_reanalysis.py (output results/peer_review/reanalysis.md); this
script redoes each number from the raw runs.csv files with ``common.paired`` and its own
equivalence test, as the other ledger scripts do. The one exception is the tie count on the base
model (seeds 0-4), which needs a refit; it is read from results/peer_review/ties_all.csv, which
that script saved. Figure 1's case is refitted here (a 2-D problem, about 15 seconds).

Read-only. Run from the repository root:
    python -B research_audit/scripts/claims_peer_review.py > research_audit/scripts/claims_peer_review_out.txt
"""

from __future__ import annotations

import inspect
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import RES, ROOT, holm, paired, section, tally

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

V3 = pd.read_csv(RES / "faithfulness_v3" / "runs.csv")                 # seeds 30-39
COND = pd.read_csv(RES / "extensions_v1" / "conditional" / "runs.csv")   # seeds 40-49
TIE = pd.read_csv(RES / "review_v1" / "faithfulness" / "runs.csv")      # seeds 50-59
V2 = pd.read_csv(RES / "refinement_v2_confirmatory" / "runs.csv")       # seeds 50-59
EDITS = pd.read_csv(RES / "refinement_v2_confirmatory" / "edits.csv.gz")
RT = pd.read_csv(RES / "repair_theta" / "runs.csv")                     # seeds 70-89
RB = pd.read_csv(RES / "readable_baselines" / "runs.csv")               # seeds 70-89
ED = pd.read_csv(RES / "error_detection_confirmatory" / "runs.csv")     # seeds 10-19
CF = pd.read_csv(RES / "cf_robustness" / "runs.csv")                    # seeds 110-119
TIES0 = pd.read_csv(RES / "peer_review" / "ties_all.csv")               # seeds 0-4

RQ1 = sorted(V3.dataset.unique())                       # 11 explanation datasets
D13 = sorted(COND.dataset.unique())                     # the 11 plus S1c and S2c
RQ3 = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SYN = {"v3": ["S1", "S2", "S3"], "cond": ["S1", "S2", "S3", "S1c", "S2c"]}


def check(label, got, want):
    ok = got == want
    print(f"{'OK ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (paper: {want})"))


def counts(out):
    v = out.verdict.value_counts()
    return f"{v.get('better', 0)}/{v.get('level', 0)}/{v.get('worse', 0)}"


def tost(df, a, b, col, units, margins, method_col="method"):
    """Holm-adjusted equivalence p per unit: the larger of two one-sided Wilcoxon tests."""
    ps = []
    for u in units:
        g = df[df.dataset == u]
        x = g[g[method_col] == a].set_index("seed")[col]
        y = g[g[method_col] == b].set_index("seed")[col]
        d = (x - y.reindex(x.index)).dropna().to_numpy(float)
        m = float(margins[u])
        p1 = wilcoxon(d + m, alternative="greater").pvalue if np.any(d + m != 0) else 1.0
        p2 = wilcoxon(d - m, alternative="less").pvalue if np.any(d - m != 0) else 1.0
        ps.append(max(p1, p2))
    return pd.Series(holm(ps), index=units)


def lead_margins(df, col, lower, units):
    """A tenth of KernelSHAP's own lead over the random ranking, per unit."""
    m = df.groupby(["dataset", "method"])[col].mean().unstack()
    lead = (m["Random"] - m["KernelSHAP"]) if lower else (m["KernelSHAP"] - m["Random"])
    return (0.10 * lead.abs()).reindex(units)


def family(df, members):
    """S1-S3 (and S1c, S2c) averaged into one unit per seed and method."""
    num = df.select_dtypes("number").columns.difference(["seed"])
    syn = (df[df.dataset.isin(members)].groupby(["seed", "method"], as_index=False)[list(num)]
           .mean().assign(dataset="synthetic"))
    return pd.concat([syn, df[~df.dataset.isin(members)]], ignore_index=True)


METRICS = (("margin_auc", True, "deletion"), ("insertion_margin_auc", False, "insertion"))

section("PR1. Equivalence, like for like and usual setting (§7, abstract, contributions)")
for col, lower, name in METRICS:
    t = tost(COND, "PACE-SHAP", "KernelSHAP", col, D13, lead_margins(COND, col, lower, D13))
    check(f"same rows, {name}: equivalent on", f"{int((t < 0.05).sum())} of {len(t)}", "12 of 13")
    check(f"same rows, {name}: the one not equivalent", list(t[t >= 0.05].index), ["landsat"])
out = paired(COND, "PACE-SHAP", "KernelSHAP", "margin_auc", ["landsat"], lower_better=True)
r = out.iloc[0]
check("Landsat deletion difference (95% interval)",
      f"{r['diff']:.3f} ({r.lo:.3f} to {r.hi:.3f})", "-0.045 (-0.053 to -0.036)")   # rev 439 said -0.054
t = tost(V3, "PACE-SHAP", "KernelSHAP", "margin_auc", RQ1, lead_margins(V3, "margin_auc", True, RQ1))
check("usual setting, deletion: equivalent on", f"{int((t < 0.05).sum())} of 11", "6 of 11")

section("PR2. The synthetic sets as one unit (§7)")
units = ["synthetic"] + [d for d in RQ1 if d not in SYN["v3"]]
fam = {}
for key, df in (("usual", family(V3, SYN["v3"])), ("same rows", family(COND, SYN["cond"]))):
    for col, lower, name in METRICS:
        out = paired(df, "PACE-SHAP", "KernelSHAP", col, units, lower_better=lower)
        fam[key, name] = out
        print(f"{key}, {name}: {tally(out)}")
check("usual setting, better on (deletion, insertion)",
      (int((fam["usual", "deletion"].verdict == "better").sum()),
       int((fam["usual", "insertion"].verdict == "better").sum())), (6, 6))
check("same rows, deletion, better on", int((fam["same rows", "deletion"].verdict == "better").sum()), 3)
ps = {}
for k, out in fam.items():
    d = out["diff"].to_numpy() * (1 if k[1] == "deletion" else -1)   # negative favours PACE-SHAP
    ps[k] = wilcoxon(d).pvalue
    print(f"across 9 units, {k}: {int((d < 0).sum())} of 9 favour PACE-SHAP, "
          f"median {np.median(-d) if k[1] == 'insertion' else np.median(d):+.4f}, p = {ps[k]:.4f}")
check("across units p: usual deletion, usual insertion, same-rows deletion, same-rows insertion",
      tuple(round(ps[k], 3) if round(ps[k], 2) == 0 else round(ps[k], 2) for k in
            (("usual", "deletion"), ("usual", "insertion"), ("same rows", "deletion"),
             ("same rows", "insertion"))), (0.02, 0.02, 0.2, 0.004))
d = fam["same rows", "insertion"]["diff"]
check("same rows, insertion: units favouring PACE-SHAP, median", (int((d > 0).sum()), round(float(d.median()), 4)),
      (9, 0.0015))

section("PR3. The other PACE explanations against KernelSHAP and random (§7)")
want = {"PACE-contrastive": 9, "PACE-decomposition": 11, "PACE-multi": 6}
for m, w in want.items():
    k = paired(V3, m, "KernelSHAP", "margin_auc", RQ1, lower_better=True)
    rnd = paired(V3, m, "Random", "margin_auc", RQ1, lower_better=True)
    check(f"{m} worse than KernelSHAP on", int((k.verdict == "worse").sum()), w)
    print(f"    vs random (better/level/worse): {counts(rnd)}; level on {list(rnd[rnd.verdict == 'level'].unit)}")
rnd = paired(V3, "PACE-decomposition", "Random", "margin_auc", RQ1, lower_better=True)
check("score breakdown not significantly better than random on", int((rnd.verdict != "better").sum()), 2)

section("PR4. Tied and untied points side by side (§7 table; tie test, seeds 50-59)")
s = TIE[TIE.method == "PACE-SHAP"]
tied = (s.n_tied / s.n).groupby(s.dataset).mean() * 100
untied_min = (s.n - s.n_tied).groupby(s.dataset).min()
elig = sorted(untied_min[untied_min >= 10].index)
raw = paired(TIE, "PACE-SHAP", "KernelSHAP", "margin_auc_raw", RQ1, lower_better=True).set_index("unit")
unt = paired(TIE, "PACE-SHAP", "KernelSHAP", "margin_auc_untied", elig, lower_better=True).set_index("unit")
paper = {"S1": ("0%", "-0.0007, better", "-0.0071, better"),
         "S2": ("0%", "-0.0013, better", "-0.0104, better"),
         "S3": ("0%", "-0.0019, better", "-0.0132, better"),
         "wine": ("0%", "-0.0025, better", "-0.0065, level"),
         "breast_cancer": ("1%", "-0.0104, better", "-0.0603, better"),
         "thyroid": ("4%", "-0.0010, better", "-0.0423, better"),
         "shuttle": ("15%", "-0.0008, better", "+0.0007, level"),
         "landsat": ("26%", "-0.0155, better", "-0.1236, better"),
         "letter": ("50%", "-0.0042, better", "-0.0202, better"),
         "page_blocks": ("70%", "-0.0003, level", "too few untied points"),
         "magic": ("77%", "-0.0005, better", "too few untied points")}
for ds, w in paper.items():
    got = (f"{tied[ds]:.0f}%", f"{raw.loc[ds, 'diff']:+.4f}, {raw.loc[ds, 'verdict']}".replace("+0.0000", "0.0000"),
           f"{unt.loc[ds, 'diff']:+.4f}, {unt.loc[ds, 'verdict']}" if ds in elig else "too few untied points")
    got = tuple(g.replace("-", "-") for g in got)
    check(ds, got, w)
check("datasets with at least 10 untied points in every seed", len(elig), 9)

section("PR5. Test rows decided by the nearest-centroid rule (§7 Ties; base model, seeds 0-4)")
t = TIES0.groupby("dataset").sum(numeric_only=True)
share = 100 * t.n_tied / t.n                 # unrounded: Python's round() goes to even at .5
err = 100 * t.n_wrong_tied / t.n_wrong
print("tied % of test rows:", share.round(2).to_dict())
print("tied % of test errors:", err.round(2).to_dict())
check("0% on S1-S3 and Wine", all(share[d] == 0 for d in ["S1", "S2", "S3", "wine"]), True)
check("under 1% on Breast Cancer and Digits", all(share[d] < 1 for d in ["breast_cancer", "digits"]), True)
check("Thyroid, Shuttle, Skin, Landsat, Letter, Page Blocks, MAGIC (rounded)",
      [int(round(share[d])) for d in ["thyroid", "shuttle", "skin", "landsat", "letter", "page_blocks", "magic"]],
      [5, 13, 20, 25, 57, 73, 76])
check("errors on tied rows, Skin/MAGIC/Letter/Page Blocks within 90-100%",
      all(90 <= err[d] <= 100 for d in ["skin", "magic", "letter", "page_blocks"]), True)
check("errors on tied rows, Shuttle", int(round(err["shuttle"])), 83)
check("errors on tied rows, Landsat and Thyroid within 53-60%",
      (int(round(min(err["landsat"], err["thyroid"]))), int(round(max(err["landsat"], err["thyroid"])))), (53, 60))

section("PR6. Repair with intervals and equivalence (§9)")
sr = paired(V2, "SR", "B0", "test_macro_f1", RQ3, method_col="arm").set_index("unit")
want = {"thyroid": "6.0 (2.6 to 9.5)", "landsat": "1.5 (0.2 to 2.8)", "page_blocks": "2.6 (0.8 to 4.4)",
        "magic": "3.3 (2.2 to 4.4)", "letter": "2.2 (1.8 to 2.7)", "skin": "5.7 (3.7 to 7.6)"}  # rev 439: 2.3
for ds, w in want.items():
    r = sr.loc[ds]
    check(f"SR - B0 macro-F1, {ds}", f"{r['diff']:.1f} ({r.lo:.1f} to {r.hi:.1f})", w)
eq = tost(V2, "SR", "AL", "test_macro_f1", RQ3, pd.Series(1.0, index=RQ3), method_col="arm")
check("SR and AL equivalent within 1 point on", int((eq < 0.05).sum()), 0)
for arm, w in (("Retrain, appended", ("-2.6 to +0.8", ["letter"])),   # rev 439: +0.9
               ("Retrain, shuffled", ("-3.4 to +1.8", []))):
    out = paired(V2, arm, "B0", "test_macro_f1", RQ3, method_col="arm")
    check(f"{arm} - B0: range, significant on", (f"{out['diff'].min():.1f} to {out['diff'].max():+.1f}",
                                                list(out[out.verdict != "level"].unit)), w)
out = paired(V2, "Retrain, appended", "B0", "test_macro_f1", ["letter"], method_col="arm")
check("appended retraining on Letter", f"{out['diff'].iloc[0]:+.2f}", "+0.85")

section("PR7. Fix share and broken predictions (§10 table)")
g = V2.groupby(["arm", "dataset"]).churn_broken_share.mean()
for arm, w in (("SR+AL", [0.19, 2.36, 0.82, 6.85, 3.06, 0.22]),
               ("Retrain, shuffled", [1.03, 6.09, 6.81, 15.60, 10.48, 3.83]),
               ("Retrain, appended", [0.39, 1.77, 3.23, 2.97, 2.33, 0.49])):
    check(f"broken, % of test rows, {arm}", [round(float(g[arm, d]), 2) for d in RQ3], w)
fs = V2.groupby(["arm", "dataset"]).churn_fix_share.mean()
check("fix share, SR+AL", [round(float(fs["SR+AL", d]), 1) for d in RQ3], [85.3, 64.2, 81.9, 64.8, 66.2, 94.1])
check("fix share, shuffled", [round(float(fs["Retrain, shuffled", d]), 1) for d in RQ3],
      [51.8, 52.4, 42.6, 47.2, 50.9, 40.5])
net = V2.assign(net=V2.churn_fixed_share - V2.churn_broken_share).groupby(["arm", "dataset"]).net.mean()
print("fixed - broken (points), shuffled:", {d: round(float(net["Retrain, shuffled", d]), 2) for d in RQ3})

section("PR8. Cost of the trust signals (§8)")
t = ED.drop_duplicates(["dataset", "seed"]).groupby("dataset")[["signals_s", "baselines_s"]].mean()
check("signals slower than the three standard methods on", f"{int((t.signals_s > t.baselines_s).sum())} of {len(t)}",
      "14 of 16")
check("Letter seconds per seed (signals, standard)",
      (round(float(t.signals_s["letter"])), round(float(t.baselines_s["letter"]))), (87, 9))

section("PR9. Where the base model errs; boxes and conditions (§1, §6; seeds 70-89)")
b = RT[RT.model == "B0-0.3"].assign(rpb=lambda d: d.n_train / d.boxes)
m = b.groupby("dataset")[["train_error_pct", "boxes", "rpb"]].mean()
print(m.round(1).to_string())
check("training error 0% on S1-S3, Wine, Digits",
      all(round(m.train_error_pct[d], 1) == 0 for d in ["S1", "S2", "S3", "wine", "digits"]), True)
check("under 2% on Breast Cancer, Thyroid, Shuttle",
      all(m.train_error_pct[d] < 2 for d in ["breast_cancer", "thyroid", "shuttle"]), True)
check("Skin", int(round(m.train_error_pct["skin"])), 4)
check("Landsat and Page Blocks (range)",
      (int(round(m.train_error_pct[["landsat", "page_blocks"]].min())),
       int(round(m.train_error_pct[["landsat", "page_blocks"]].max()))), (10, 14))
check("Letter and MAGIC (range)",
      (int(round(m.train_error_pct[["letter", "magic"]].min())),
       int(round(m.train_error_pct[["letter", "magic"]].max()))), (21, 27))
check("training rows per box on S1-S3", round(float(m.rpb[["S1", "S2", "S3"]].mean()), 1), 2.6)
real = [d for d in m.index if d not in ("S1", "S2", "S3", "wine", "digits")]
check("rows per box, other real datasets", f"{m.rpb[real].min():.1f}-{m.rpb[real].max():,.0f}", "10.7-3,336")
check("boxes at theta = 0.3 (mean per dataset)", f"{m.boxes.min():.0f}-{m.boxes.max():,.0f}", "14-1,533")
r = RB[RB.model == "CFMNN-0.3"].groupby("dataset")[["rules", "conditions_per_rule"]].mean()
print("readable-baselines run, same seeds: boxes", f"{r.rules.min():.0f}-{r.rules.max():,.0f}")
nfeat = ED.drop_duplicates("dataset").set_index("dataset").n_features.to_dict()
nfeat["digits"] = nfeat.get("digits", 64)
sh = (100 * r.conditions_per_rule / pd.Series({d: nfeat[d] for d in r.index})).round(0)
check("share of features a box limits", f"{sh.min():.0f}-{sh.max():.0f}%", "79-100%")

section("PR10. Training rows a shrink cuts off (§5; second repair run, seeds 50-59)")
e = EDITS[(EDITS.arm == "SR") & (EDITS.accepted.astype(bool)) & EDITS.shrink_n_cut.notna()]
cut = e.groupby(["dataset", "seed"]).shrink_n_cut.sum()
ntr = V2.drop_duplicates(["dataset", "seed"]).set_index(["dataset", "seed"]).n_train
pct = (100 * cut / ntr.reindex(cut.index)).groupby("dataset").mean()
check("Skin, Page Blocks, Letter, Landsat, MAGIC, Thyroid (%)",
      [int(round(pct[d])) for d in ["skin", "page_blocks", "letter", "landsat", "magic", "thyroid"]],
      [2, 6, 10, 22, 32, 48])

section("PR11. How often a counterfactual survives a refit (§7; seeds 110-119)")
r3 = CF.groupby(["dataset", "method"]).r3.mean().unstack() * 100
for m, w in (("PACE", "17-55% (98% on Shuttle)"), ("PACE-sparse", "24-42% (98% on Shuttle)")):
    x = r3[m].drop("shuttle")
    check(f"{m}, share of refits still giving the target class",
          f"{x.min():.0f}-{x.max():.0f}% ({r3[m]['shuttle']:.0f}% on Shuttle)", w)

section("PR12. Figure 1 (refit of the 2-D case)")
from figure_boxes import GRID, find_case  # noqa: E402
from pace_cfmnn.audit import impact_set  # noqa: E402

seed, X, y, before, after, r, k, i, info = find_case()
gg = np.linspace(0, 1, GRID)
G = np.array(np.meshgrid(gg, gg)).reshape(2, -1).T
cert = impact_set(before, after, G)
changed = before.predict(G) != after.predict(G)
check("training rows cut off with the shrink", int(info["n_cut"]), 2)
check("certificate, share of the square", f"{cert.mean():.0%}", "17%")
check("prediction changed, share of the square", f"{changed.mean():.1%}", "0.2%")
check("changes outside the certificate", int((changed & ~cert).sum()), 0)
check("the circled row is class 0 (blue) inside a class-1 (orange) box",
      (int(y[r]), int(before.box_class_[k])), (0, 1))

section("PR13. What the tests can detect (§6)")
p10 = wilcoxon(np.arange(1, 11)).pvalue
p9 = wilcoxon(np.r_[-1, np.arange(2, 11)]).pvalue          # 9 of 10 agree, smallest rank against
p9b = wilcoxon(np.r_[1, -2, np.arange(3, 11)]).pvalue      # 9 of 10 agree, second rank against
p8 = wilcoxon(np.r_[-1, -2, np.arange(3, 11)]).pvalue      # 8 of 10 agree, the two smallest against
print(f"exact two-sided p: all 10 agree {p10:.5f}; 9 agree (rank 1 against) {p9:.5f}; "
      f"9 agree (rank 2 against) {p9b:.5f}; 8 agree (best case) {p8:.5f}; Holm thresholds "
      f"0.05/11 = {0.05 / 11:.5f}, 0.05/13 = {0.05 / 13:.5f}")
check("smallest p with 10 seeds", round(p10, 3), 0.002)
check("Holm across 11 reachable with 9 of 10, not with 8", (p9 < 0.05 / 11, p8 < 0.05 / 11), (True, False))
check("Holm across 13 needs all 10", (p10 < 0.05 / 13, p9 < 0.05 / 13), (True, False))

section("PR14. The S1-S3 generators (§6 Setup)")
from pace_cfmnn import data  # noqa: E402

check("weights on x0, x1, x2", data.SYNTH_WEIGHTS.tolist(), [2.0, 1.5, -1.0])
check("threshold (the median of the score)", data.synth_threshold(), 1.25)
sig = inspect.signature(data.make_synthetic).parameters
check("rows, features, S1 margin, S2 noise sd",
      (sig["n"].default, sig["d"].default, sig["margin"].default, sig["noise"].default), (5000, 10, 0.1, 0.25))
check("S3 margin", inspect.signature(data.make_box_synthetic).parameters["margin"].default, 0.02)
Xs = np.random.default_rng(0).uniform(size=(20000, 3))
rule = ((Xs[:, 0] > 0.5) & (Xs[:, 1] < 0.4)) | (Xs[:, 2] > 0.8)
check("S3 rule (x0 > 0.5 and x1 < 0.4) or x2 > 0.8", bool((data.s3_rule(Xs) == rule).all()), True)
for name in ("S1", "S2", "S3"):
    ds = data.load(name, seed=0)
    check(f"{name} shape", ds.X.shape, (5000, 10))

section("PR15. Software versions in Appendix C against requirements-lock.txt")
lock = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)$", (ROOT / "requirements-lock.txt").read_text(), re.M))
lock = {k.lower().replace("_", "-"): v for k, v in lock.items()}
paper = {"numpy": "2.5.3", "scipy": "1.18.1", "scikit-learn": "1.9.1", "pandas": "3.0.5", "shap": "0.54.1",
         "lime": "0.2.0.1", "dice-ml": "0.12", "interpret-core": "0.7.8", "hyperbox-brain": "0.1.6"}
check("versions", {k: lock.get(k) for k in paper}, paper)
