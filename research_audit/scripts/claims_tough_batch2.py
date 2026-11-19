"""Recompute every number the write-up takes from tough-review batch 2 (Runs P2, P4, P5, P6, P11).

Sources (home PC, research_audit/HOME_PC_RUNBOOK.md section 14):
  P2   results/fmm_baselines/runs.csv     (seeds 250-259; tests over seeds, Holm across all 24)
  P4   results/gfmm_exact/check.csv, timing.csv, first_run/check.csv   (seeds 310-319)
  P5   results/decision/points.csv        (seeds 320-329; per-seed means, Holm across 3 datasets)
  P6   results/shap_certificate/edits.csv (seeds 330-339)
  P11  results/cannot_hurt/edits.csv      (seeds 390-399)
Every CSV is read with float_precision="round_trip". The tests follow each run's own script:
paired two-sided Wilcoxon over seeds (scipy defaults), Holm as the plan says. The expected values
are those in results/FINDINGS.md section 23 and the paper; a BAD line means the text must change.

Read-only. Run from the repository root (home PC):
    python -B research_audit/scripts/claims_tough_batch2.py > research_audit/scripts/claims_tough_batch2_out.txt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES, holm, section, wilcox_p

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


def tests(rows):
    """rows: list of dicts with the per-seed differences under 'd'; Holm across all of them."""
    t = pd.DataFrame(rows)
    t["p"] = [wilcox_p(d) for d in t.d]
    t["diff"] = [float(np.mean(d)) for d in t.d]
    t["p_holm"] = holm(t.p.to_numpy())
    return t


def verdicts(t, lower=False):
    good = (t["diff"] < 0) if lower else (t["diff"] > 0)
    return np.where(t.p_holm >= 0.05, "nsd", np.where(good, "better", "worse"))


def count(v):
    s = pd.Series(v).value_counts()
    return (int(s.get("better", 0)), int(s.get("nsd", 0)), int(s.get("worse", 0)))


# ---------------------------------------------------------------------------------------------- P2
section("P2: CFMNN with self-repair against tuned fuzzy min-max networks")
P2 = read("fmm_baselines", "runs.csv")
DS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
FAM = ["EFMNN", "KNEFMNN", "RFMNN", "IOL-GFMM"]
check("seeds", sorted(P2.seed.unique().tolist()), list(range(250, 260)))
check("Skin in full (skin_share 1.0 everywhere)", sorted(P2.skin_share.unique().tolist()), [1.0])
check("rows", len(P2), 720)


def compare(a, suffix, col):
    rows = []
    for f in FAM:
        for ds in DS:
            g = P2[P2.dataset == ds].pivot_table(index="seed", columns="model", values=col)
            d = (g[a] - g[f + suffix]).dropna().to_numpy()
            rows.append({"baseline": f, "dataset": ds, "d": d, "wins": int((d > 0).sum())})
    t = tests(rows)
    t["verdict"] = verdicts(t)
    return t


prim = compare("CFMNN-SR-val", "-val", "macro_f1")
print(prim.drop(columns="d").round(4).to_string())
check("primary: better / nsd / worse", count(prim.verdict), (6, 17, 1))
better = sorted(zip(prim[prim.verdict == "better"].baseline, prim[prim.verdict == "better"].dataset))
check("primary: better", better, sorted([(f, "thyroid") for f in FAM] + [("IOL-GFMM", "landsat"), ("IOL-GFMM", "letter")]))
worse = prim[prim.verdict == "worse"]
check("primary: worse", list(zip(worse.baseline, worse.dataset)), [("IOL-GFMM", "magic")])
th = prim[prim.dataset == "thyroid"]["diff"]
check("Thyroid gains, range (2 dp)", (r(th.min(), 2), r(th.max(), 2)), (14.59, 29.40))
iol = prim[prim.baseline == "IOL-GFMM"].set_index("dataset")
check("IOL-GFMM: Landsat, Letter, MAGIC (2 dp)", [r(iol.loc[d, "diff"], 2) for d in ("landsat", "letter", "magic")],
      [2.34, 3.17, -1.22])
check("MAGIC: seeds better, Holm p (4 dp)", (int(iol.loc["magic", "wins"]), r(iol.loc["magic", "p_holm"], 4)), (0, 0.0469))
acc = compare("CFMNN-SR-val", "-val", "accuracy")
check("accuracy: better / nsd / worse", count(acc.verdict), (6, 17, 1))
check("accuracy: worse", list(zip(acc[acc.verdict == "worse"].baseline, acc[acc.verdict == "worse"].dataset)),
      [("IOL-GFMM", "magic")])
fixed = compare("CFMNN-SR-0.3", "-0.3", "macro_f1")
check("theta = 0.3: better / nsd / worse", count(fixed.verdict), (7, 11, 6))
check("theta = 0.3: worse", sorted(zip(fixed[fixed.verdict == "worse"].baseline, fixed[fixed.verdict == "worse"].dataset)),
      sorted([("KNEFMNN", "letter"), ("RFMNN", "letter"), ("RFMNN", "skin"),
              ("IOL-GFMM", "landsat"), ("IOL-GFMM", "magic"), ("IOL-GFMM", "letter")]))
norep = compare("CFMNN-val", "-val", "macro_f1")
check("tuned CFMNN without repair: better / nsd / worse", count(norep.verdict), (5, 15, 4))
box = P2.pivot_table(index="dataset", columns="model", values="boxes", aggfunc="mean").loc[DS]
for f in ("EFMNN", "KNEFMNN", "RFMNN"):
    check(f"fewer boxes than {f}-val on all 6", bool((box["CFMNN-SR-val"] < box[f + "-val"]).all()), True)
fewer = box["CFMNN-SR-val"] < box["IOL-GFMM-val"]
check("fewer boxes than IOL-GFMM-val on", sorted(box.index[fewer]), sorted(["thyroid", "letter", "skin"]))
check("Letter boxes, CFMNN-SR-val and EFMNN-val", (int(round(box.loc["letter", "CFMNN-SR-val"])),
                                                   int(round(box.loc["letter", "EFMNN-val"]))), (5343, 15075))
val = P2[P2.setting == "val"].drop_duplicates(["dataset", "seed", "family"])
for ds in ("letter", "skin"):
    check(f"theta-val is 0.05 for every model on {ds}", sorted(val[val.dataset == ds].theta.unique().tolist()), [0.05])

# ---------------------------------------------------------------------------------------------- P4
section("P4: exact PACE-SHAP for GFMM-family networks")
C = read("gfmm_exact", "check.csv")
check("seeds", sorted(C.seed.unique().tolist()), list(range(310, 320)))
check("models", sorted(C.model.unique().tolist()), ["IOL-GFMM", "Online GFMM"])
check("datasets", C.dataset.nunique(), 6)
check("adapter predicts like hyperbox-brain everywhere", bool(C.adapter_predicts_alike.all()), True)
mx = C[C.arm == "PACE-SHAP-min"].max_abs_diff_brute.max()
check("largest difference from the brute force", f"{mx:.1e}", "6.8e-16")
F1 = read("gfmm_exact", "first_run", "check.csv")
bad = F1[~F1.adapter_predicts_alike].drop_duplicates(["dataset", "seed", "model"])
check("first check run: tasks where the adapter differed", len(bad), 10)
check("first check run: all IOL-GFMM", bad.model.unique().tolist(), ["IOL-GFMM"])
check("first check run: PACE-SHAP-min within 7e-16 of brute force",
      bool(F1[F1.arm == "PACE-SHAP-min"].max_abs_diff_brute.max() <= 7e-16), True)
Tm = read("gfmm_exact", "timing.csv")
check("timing tasks finished (3 seeds x 6 datasets x 2 models x 3 arms)", len(Tm), 108)
check("ExactExplainer within 1e-9 of PACE-SHAP-min",
      bool(Tm[Tm.arm == "shap ExactExplainer"].max_abs_diff_pace.max() <= 1e-9), True)
ms = Tm.pivot_table(index=["dataset", "model"], columns="arm", values="ms_per_explanation", aggfunc="mean")
bf = ms["Brute force"] / ms["PACE-SHAP-min"]
ex = ms["shap ExactExplainer"] / ms["PACE-SHAP-min"]
print(pd.DataFrame({"brute/pace": bf.round(1), "exact/pace": ex.round(1)}).to_string())
syn = [i for i in bf.index if i[0] in ("S1", "S2", "S3")]
real = [i for i in bf.index if i[0] not in ("S1", "S2", "S3")]
check("S1-S3: both ratios, range", (r(min(bf[syn].min(), ex[syn].min())), r(max(bf[syn].max(), ex[syn].max()))), (10.7, 10.8))
check("Shuttle, MAGIC, Page Blocks: brute force, range", (r(bf[real].min()), r(bf[real].max())), (33.5, 94.2))
check("Shuttle, MAGIC, Page Blocks: ExactExplainer, range", (r(ex[real].min()), r(ex[real].max())), (33.4, 90.3))

# ---------------------------------------------------------------------------------------------- P5
section("P5: explaining the decision at ties")
P5 = read("decision", "points.csv")
D5 = ["magic", "letter", "page_blocks"]
check("seeds", sorted(P5.seed.unique().tolist()), list(range(320, 330)))
check("points per (dataset, seed, method)", P5.groupby(["dataset", "seed", "method"]).size().unique().tolist(), [50])
check("exact check within 1e-9", bool(P5.check_max_diff.dropna().max() <= 1e-9), True)
tied = P5[P5.method == "PACE-decision"].groupby(["dataset", "seed"]).tied.sum()
check("tied points per seed, range", (int(tied.min()), int(tied.max())), (22, 46))


def p5(col, lower, subset):
    sel = P5[P5.tied] if subset == "tied" else P5[~P5.tied]
    per = sel.groupby(["dataset", "seed", "method"])[col].mean().unstack("method")
    rows = []
    for ds in D5:
        x = per.loc[ds].dropna()
        d = (x["PACE-decision"] - x["PACE-SHAP"]).to_numpy()
        rows.append({"dataset": ds, "d": d, "wins": int(((d < 0) if lower else (d > 0)).sum())})
    t = tests(rows)
    t["verdict"] = verdicts(t, lower)
    return t.set_index("dataset")


pri = p5("decision_deletion_auc", True, "tied")
print(pri.drop(columns="d").round(4).to_string())
check("primary: better on all 3", pri.verdict.tolist(), ["better"] * 3)
check("primary differences (4 dp)", [r(pri.loc[d, "diff"], 4) for d in D5], [-0.0506, -0.0161, -0.1085])
check("primary: 10 of 10 seeds, Holm p (4 dp)", (pri.wins.tolist(), sorted({r(p, 4) for p in pri.p_holm})),
      ([10, 10, 10], [0.0059]))
check("untied decision deletion: better on all 3", p5("decision_deletion_auc", True, "untied").verdict.tolist(), ["better"] * 3)
check("tied decision insertion: better on all 3", p5("decision_insertion_auc", False, "tied").verdict.tolist(), ["better"] * 3)
md = p5("margin_deletion_auc", True, "tied")
check("margin deletion verdicts (MAGIC, Letter, Page Blocks)", md.verdict.tolist(), ["nsd", "worse", "nsd"])
check("margin deletion, Letter (4 dp), Holm p", (r(md.loc["letter", "diff"], 4), r(md.loc["letter", "p_holm"], 4)), (0.0268, 0.0059))
mi = p5("margin_insertion_auc", False, "tied")
check("margin insertion verdicts", mi.verdict.tolist(), ["nsd", "worse", "nsd"])
check("margin insertion, Letter (4 dp), Holm p", (r(mi.loc["letter", "diff"], 4), r(mi.loc["letter", "p_holm"], 4)), (-0.0246, 0.0410))

# ---------------------------------------------------------------------------------------------- P6
section("P6: the PACE-SHAP certificate")
P6 = read("shap_certificate", "edits.csv")
check("seeds", sorted(P6.seed.unique().tolist()), list(range(330, 340)))
check("accepted edits, (edit, row) pairs", (len(P6), int(P6.n.sum())), (6085, 608500))
check("no change outside the certificate above 1e-12", bool(P6.max_change_outside.max() <= 1e-12), True)
cs = (100 * P6.certified / P6.n).groupby(P6.dataset).mean()
ch = (100 * P6.changed / P6.n).groupby(P6.dataset).mean()
print(pd.DataFrame({"certified %": cs.round(1), "changed %": ch.round(1)}).to_string())
check("certified share, range (1 dp)", (r(cs.min()), r(cs.max())), (99.1, 100.0))
check("certified share at least 90% everywhere", bool((cs >= 90).all()), True)
check("changed share: Letter, Skin (1 dp)", (r(ch["letter"]), r(ch["skin"])), (31.7, 46.2))
oth = ch.drop(["letter", "skin"])
check("changed share elsewhere, range (1 dp)", (r(oth.min()), r(oth.max())), (80.3, 82.7))
rm = P6.groupby("dataset").rel_change_median.median()
rx = P6.groupby("dataset").rel_change_max.max()
check("median relative change, range", (f"{rm.min():.1e}", f"{rm.max():.1e}"), ("2.0e-05", "2.0e-03"))
check("largest relative change, range", (f"{rx.min():.1e}", f"{rx.max():.1e}"), ("1.5e+00", "2.3e+01"))

# ---------------------------------------------------------------------------------------------- P11
section("P11: edits that provably cannot hurt")
P11 = read("cannot_hurt", "edits.csv")
check("seeds", sorted(P11.seed.unique().tolist()), list(range(390, 400)))
acc11 = P11[P11.accepted]
check("accepted edits", len(acc11), 6110)
check("edits meeting S: all, accepted", (int(P11.safe.sum()), int(acc11.safe.sum())), (116, 116))
check("edits meeting S that broke a correct training prediction", int((P11[P11.safe].train_broken > 0).sum()), 0)
check("share meeting S overall (1 dp, %)", r(100 * acc11.safe.mean()), 1.9)
sh = 100 * acc11.groupby("dataset").safe.mean()
check("share meeting S, range (1 dp, %)", (r(sh.min()), r(sh.max())), (0.0, 8.0))
check("share above 10% anywhere", bool((sh >= 10).any()), False)
cut = 100 * (acc11.n_slab > 0).groupby(acc11.dataset).mean()
check("accepted edits cutting an own-class row, range (0 dp, %)", (int(round(cut.min())), int(round(cut.max()))), (90, 100))

section("Verdict")
print("ALL OK" if not BAD else f"{len(BAD)} BAD: {BAD}")
