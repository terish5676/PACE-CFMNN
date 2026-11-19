"""Re-analysis of existing results for the peer review (reviews/fable_peer_review.md).

Writes results/peer_review/reanalysis.md. Everything comes from saved results, except the tie
counts on all 13 datasets (section 10), which refit the base model on seeds 0-4 and are saved to
results/peer_review/ties_all.csv.

Post hoc: none of these analyses was pre-registered. Each answers a weakness of the review (codes
in the headings).

Equivalence margins, fixed here before any number below was computed:

* faithfulness (margin AUC): 10% of KernelSHAP's own lead over the random ranking on that dataset
  (mean over seeds);
* accuracy and macro-F1: 1 point, the margin RQ3's plan used for "no worse than".

Two methods are called equivalent when both one-sided Wilcoxon tests against the margin reject
(TOST), after Holm correction across the datasets.

    python -B scripts/peer_review_reanalysis.py --jobs 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import review_reanalysis as rr  # noqa: E402
from accuracy_table import DATASETS as D_ACC  # noqa: E402
from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_error_detection import holm  # noqa: E402

RES = ROOT / "results"
OUT = RES / "peer_review"
RQ1 = rr.RQ1
D13 = rr.D13
RQ3 = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SYN_V3 = ["S1", "S2", "S3"]
SYN_COND = ["S1", "S2", "S3", "S1c", "S2c"]


# ------------------------------------------------------------------ helpers
def tost(df, a, b, metric, datasets, margins):
    """Holm-adjusted TOST p-value per dataset: max of the two one-sided Wilcoxon tests."""
    ps = []
    for ds in datasets:
        g = df[df.dataset == ds]
        x = g[g.method == a].set_index("seed")[metric]
        y = g[g.method == b].set_index("seed")[metric]
        d = (x - y.reindex(x.index)).dropna().to_numpy()
        m = float(margins[ds])
        lo, hi = d + m, d - m
        p1 = stats.wilcoxon(lo, alternative="greater").pvalue if (lo != 0).any() else 1.0
        p2 = stats.wilcoxon(hi, alternative="less").pvalue if (hi != 0).any() else 1.0
        ps.append(max(p1, p2))
    return holm(np.array(ps))


def block(L, df, a, b, metric, lower, datasets, margins, title, digits=4):
    """One comparison: difference, 95% interval, effect size, Holm verdict and equivalence."""
    t = rr.compare(df, a, b, metric, lower, datasets)
    t["margin"] = [float(margins[d]) for d in datasets]
    t["p_tost"] = tost(df, a, b, metric, datasets, margins)
    f = f"{{:+.{digits}f}}"
    L += ["", f"**{title}** ({a} − {b}; {'lower' if lower else 'higher'} is better)", "",
          rr.counts(t, a).replace("level", "not significantly different"),
          f"Equivalent within the margin on {int((t.p_tost < 0.05).sum())} of {len(t)}.", "",
          "| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | "
          "Equivalence Holm p |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        v = {"level": "not significant"}.get(r.verdict, r.verdict)
        eq = f"{r.p_tost:.3f}" + (" (equivalent)" if r.p_tost < 0.05 else "")
        L.append(f"| {r.dataset} | {f.format(r['diff'])} | {f.format(r['diff'] - r.ci)} to "
                 f"{f.format(r['diff'] + r.ci)} | {'–' if pd.isna(r.dz) else format(r.dz, '+.2f')} | {r.wins} of {r.n} | {r.p_holm:.3f} | "
                 f"{v} | {r.margin:.{digits}f} | {eq} |")
    return t


def lead_margins(df, metric, lower, datasets, share=0.10):
    """10% of KernelSHAP's lead over the random ranking, per dataset."""
    m = df.groupby(["dataset", "method"])[metric].mean().unstack()
    lead = (m["Random"] - m["KernelSHAP"]) if lower else (m["KernelSHAP"] - m["Random"])
    return (share * lead.abs()).reindex(datasets)


def family(df, members, name="synthetic"):
    """Average the synthetic sets into one unit per seed and method (as RQ2 does)."""
    num = df.select_dtypes("number").columns.difference(["seed"])
    syn = (df[df.dataset.isin(members)].groupby(["seed", "method"], as_index=False)[list(num)]
           .mean().assign(dataset=name))
    return pd.concat([syn, df[~df.dataset.isin(members)]], ignore_index=True)


def across(L, t, label):
    """Signed-rank test of the per-unit mean differences, across units."""
    d = t["diff"].to_numpy()
    p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
    L.append(f"- {label}: {int((d < 0).sum())} of {len(d)} units negative, "
             f"median difference {np.median(d):+.4f}, Wilcoxon p = {p:.4f}")


def mean_table(L, m, digits=4, first="Dataset"):
    L += ["", f"| {first} | " + " | ".join(map(str, m.columns)) + " |", "|---|" + "---|" * m.shape[1]]
    for ds, r in m.iterrows():
        L.append(f"| {ds} | " + " | ".join("–" if pd.isna(v) else f"{v:.{digits}f}" for v in r) + " |")


# ------------------------------------------------------------------ sections
def s_rq1(L):
    v3 = pd.read_csv(RES / "faithfulness_v3" / "runs.csv")
    cond = pd.read_csv(RES / "extensions_v1" / "conditional" / "runs.csv")
    L += ["", "## 1. RQ1 with effect sizes and equivalence (C1, C2)",
          "", "The second explanation run (seeds 30–39, 11 datasets) and the like-for-like comparison "
          "from the conditional run (seeds 40–49; 13 datasets: the 11, plus S1c and S2c, which are "
          "S1 and S2 with feature 3 made a noisy copy of feature 0)."]
    out = {}
    for metric, lower, name in (("margin_auc", True, "deletion"),
                                ("insertion_margin_auc", False, "insertion")):
        mg = lead_margins(v3, metric, lower, RQ1)
        out[f"v3 {name}"] = block(L, v3, "PACE-SHAP", "KernelSHAP", metric, lower, RQ1, mg,
                                  f"Usual setting, {name}")
        block(L, v3, "PACE-SHAP", "KernelSHAP-strong", metric, lower, RQ1, mg,
              f"Stronger KernelSHAP, {name}")
        mg = lead_margins(cond, metric, lower, D13)
        out[f"same rows {name}"] = block(L, cond, "PACE-SHAP", "KernelSHAP", metric, lower, D13, mg,
                                         f"Same 200 rows, every feature, {name}")
    L += ["", "## 2. The synthetic family as one unit (C5)",
          "", "RQ2 averages the synthetic sets because they come from one generator. The same rule in "
          "RQ1 gives 9 units."]
    units = ["synthetic"] + RQ1[3:]
    fam = {}
    for metric, lower, name in (("margin_auc", True, "deletion"),
                                ("insertion_margin_auc", False, "insertion")):
        a = family(v3, SYN_V3)
        fam[f"v3 {name}"] = block(L, a, "PACE-SHAP", "KernelSHAP", metric, lower, units,
                                  lead_margins(a, metric, lower, units), f"Usual setting, {name}, 9 units")
        b = family(cond, SYN_COND)
        fam[f"same rows {name}"] = block(L, b, "PACE-SHAP", "KernelSHAP", metric, lower, units,
                                         lead_margins(b, metric, lower, units),
                                         f"Same rows, {name}, 9 units")
    L += ["", "## 3. Tests across datasets (C3)",
          "", "One value per unit (the mean difference over seeds), signed-rank test across the 9 "
          "units. For deletion a negative difference favours PACE-SHAP; insertion is given with its "
          "sign flipped so that negative again favours PACE-SHAP.", ""]
    for k, t in fam.items():
        t = t.copy()
        if "insertion" in k:
            t["diff"] = -t["diff"]
        across(L, t, f"PACE-SHAP vs KernelSHAP, {k}")


def s_repair(L):
    v2 = pd.read_csv(RES / "refinement_v2_confirmatory" / "runs.csv").rename(columns={"arm": "method"})
    one = pd.Series(1.0, index=RQ3)
    L += ["", "## 4. Repair with effect sizes and equivalence (C1, C2)",
          "", "The second repair run (seeds 50–59), test macro-F1 in points."]
    t1 = block(L, v2, "SR", "B0", "test_macro_f1", False, RQ3, one, "Self-repair vs the starting model", 2)
    block(L, v2, "SR+AL", "SR", "test_macro_f1", False, RQ3, one, "SR then AL vs self-repair", 2)
    block(L, v2, "SR", "AL", "test_macro_f1", False, RQ3, one, "Self-repair vs active learning", 2)
    rt = pd.read_csv(RES / "repair_theta" / "runs.csv").rename(columns={"model": "method"})
    ds = list(dict.fromkeys(rt.dataset))
    ds = [d for d in D_ACC if d in ds]
    one = pd.Series(1.0, index=ds)
    t2 = block(L, rt, "SR-0.3", "B0-val", "accuracy", False, ds, one,
               "Repair at θ = 0.3 vs θ chosen by validation, no repair (seeds 70–89)", 2)
    block(L, rt, "SR-val", "B0-val", "accuracy", False, ds, one,
          "Repair at the chosen θ vs no repair there", 2)
    L += ["", "Across datasets:", ""]
    for label, t in (("Self-repair vs the starting model (6 datasets; positive favours repair)", t1),
                     ("Repair at θ = 0.3 vs the chosen θ (13 datasets; negative favours the chosen θ)", t2)):
        d = t["diff"].to_numpy()
        L.append(f"- {label}: median {np.median(d):+.2f} points, Wilcoxon p = "
                 f"{stats.wilcoxon(d).pvalue:.4f}")

    L += ["", "## 5. Fix share against the change in accuracy (A11)",
          "", "Fixed − broken is the net change in correct test predictions, so an update that does "
          "not change accuracy has a fix share near 50% by arithmetic. Means over the 10 seeds; "
          "broken is the negative-flip rate.", "",
          "| Dataset | Arm | Fixed (% of test rows) | Broken (% of test rows) | Net (points) | Fix share (%) |",
          "|---|---|---|---|---|---|"]
    arms = ["SR", "SR+AL", "Retrain, appended", "Retrain, shuffled"]
    g = v2[v2.method.isin(arms)].groupby(["dataset", "method"])[
        ["churn_fixed_share", "churn_broken_share", "churn_fix_share"]].mean()
    for d in RQ3:
        for a in arms:
            r = g.loc[(d, a)]
            L.append(f"| {d} | {a} | {r.churn_fixed_share:.2f} | {r.churn_broken_share:.2f} | "
                     f"{r.churn_fixed_share - r.churn_broken_share:+.2f} | {r.churn_fix_share:.1f} |")


def s_omitted(L):
    v3 = pd.read_csv(RES / "faithfulness_v3" / "runs.csv")
    L += ["", "## 6. Faithfulness of every PACE method (C7)",
          "", "Mean deletion-margin AUC in the second run (seeds 30–39); lower is more faithful. The "
          "paper's table leaves out the last three columns."]
    order = ["PACE-SHAP", "KernelSHAP", "LIME", "PACE-flip", "PACE-multi", "PACE-contrastive",
             "PACE-decomposition", "Random"]
    mean_table(L, v3.groupby(["dataset", "method"]).margin_auc.mean().unstack().reindex(RQ1)[order])
    L += ["", "Against the random ranking and against KernelSHAP (Holm across the 11 datasets; "
          "better / not significant / worse for the PACE method):", "",
          "| Method | vs random | vs KernelSHAP |", "|---|---|---|"]
    for m in ("PACE-flip", "PACE-multi", "PACE-contrastive", "PACE-decomposition"):
        c = []
        for other in ("Random", "KernelSHAP"):
            v = rr.compare(v3, m, other, "margin_auc", True, RQ1).verdict.value_counts()
            c.append(f"{v.get('better', 0)} / {v.get('level', 0)} / {v.get('worse', 0)}")
        L.append(f"| {m} | {c[0]} | {c[1]} |")

    cf = pd.read_csv(RES / "cf_robustness" / "runs.csv")
    L += ["", "## 7. How often a counterfactual survives a refit (A8)",
          "", "Share of 4 CFMNNs, fitted on the same rows in other orders, that still give the target "
          "class at the answer; mean over points and seeds 110–119, as in the run's summary. The paper gives only the "
          "counts against DiCE's random search."]
    cols = ["PACE", "PACE-sparse", "DiCE random", "DiCE genetic", "DiCE kdtree",
            "Nearest unlike neighbour", "Brute force"]
    ds = [d for d in RQ1 if d in set(cf.dataset)]
    mean_table(L, cf.groupby(["dataset", "method"]).r3.mean().unstack().reindex(ds)[cols], 2)

    ed = pd.read_csv(RES / "error_detection_confirmatory" / "runs.csv")
    t = ed.drop_duplicates(["dataset", "seed"]).groupby("dataset")[["signals_s", "baselines_s"]].mean()
    L += ["", "## 8. Cost of the trust signals (C7)",
          "", "Seconds per seed in the confirmatory run. The signal time covers PD, EI and U on the "
          "calibration and evaluation rows; the baseline time covers Trust Score, nearest-neighbour "
          "disagreement and the order ensemble on the evaluation rows.",
          "", f"The signals took longer on {int((t.signals_s > t.baselines_s).sum())} of {len(t)} datasets."]
    mean_table(L, t.rename(columns={"signals_s": "PACE signals (s)", "baselines_s": "Baselines (s)"}), 1)


def s_base(L):
    rt = pd.read_csv(RES / "repair_theta" / "runs.csv")
    b = rt[rt.model == "B0-0.3"].assign(rows_per_box=lambda d: d.n_train / d.boxes)
    t = b.groupby("dataset")[["train_error_pct", "boxes", "rows_per_box", "accuracy"]].mean()
    sr = rt[rt.model == "SR-0.3"].groupby("dataset")[["edits_accepted", "accuracy"]].mean()
    t["edits accepted"] = sr.edits_accepted
    t["accuracy after repair"] = sr.accuracy
    t = t.reindex([d for d in D_ACC if d in t.index])
    t.columns = ["Training error (%)", "Boxes", "Training rows per box", "Test accuracy (%)",
                 "Repair edits accepted", "Test accuracy after repair (%)"]
    L += ["", "## 9. Where the base model errs (A5)",
          "", "The model at θ = 0.3, means over seeds 70–89. Proposition 1 covers training errors "
          "only. Where the training error is 0, self-repair has nothing to fix, yet test errors "
          "remain."]
    mean_table(L, t, 1)


def tie_rows(name, seed):
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])        # as the accuracy table
    sc = data.fit_scaler(ds.X[order])
    m = CFMNN().fit(sc.transform(ds.X[order]), ds.y[order])
    yte = ds.y[sp["test"]]
    det = m.predict_details(sc.transform(ds.X[sp["test"]]))
    s = np.where(det.best_box >= 0, det.support_rescaled, -np.inf)
    rows = np.arange(len(yte))
    own = s[rows, det.cls].copy()
    s[rows, det.cls] = -np.inf
    tied = own - s.max(axis=1) <= 2 * float(m.tie_tol)        # the runner-up class ties the winner
    wrong = det.y_pred != yte
    print(f"{name} seed {seed}: tied {tied.sum()} of {len(yte)}", flush=True)
    return [{"dataset": name, "seed": seed, "n": len(yte), "n_tied": int(tied.sum()),
             "n_wrong": int(wrong.sum()), "n_wrong_tied": int((wrong & tied).sum())}]


def s_ties(L, jobs):
    csv = OUT / "ties_all.csv"
    tasks = [(d, s) for d in D_ACC for s in range(5)]
    run_all(tie_rows, pending(csv, tasks, resume=True), csv, jobs, {"skin": 3, "letter": 2})
    t = pd.read_csv(csv).groupby("dataset").sum(numeric_only=True).reindex(D_ACC)
    L += ["", "## 10. Test predictions decided by the nearest-centroid rule (A4)",
          "", "The base model on the paper's 80/20 split, seeds 0–4, every test row. A row is tied "
          "when the best box of another class has the winner's membership, so the class comes from "
          "the nearest centroid, which PACE-SHAP's margin ignores.", "",
          "| Dataset | Test rows | Tied (%) | Error rate, tied rows (%) | Error rate, other rows (%) | "
          "Share of all errors that are tied (%) |", "|---|---|---|---|---|---|"]
    pct = lambda a, b: "–" if b == 0 else f"{100 * a / b:.1f}"   # noqa: E731
    for ds, r in t.iterrows():
        L.append(f"| {ds} | {int(r.n)} | {pct(r.n_tied, r.n)} | {pct(r.n_wrong_tied, r.n_tied)} | "
                 f"{pct(r.n_wrong - r.n_wrong_tied, r.n - r.n_tied)} | {pct(r.n_wrong_tied, r.n_wrong)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1, help="parallel jobs for the tie counts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Peer review: re-analysis of existing results", "",
         "Written by `scripts/peer_review_reanalysis.py`. Post hoc: nothing here was pre-registered.",
         "", "Each comparison is paired over seeds. Intervals are 95% t-intervals of the mean "
         "difference, not adjusted for the number of datasets. Verdicts use a two-sided Wilcoxon "
         "test with Holm correction across the datasets, as in the paper. With 10 seeds the smallest "
         "possible two-sided p is 0.002, so after Holm across 11 datasets a verdict needs at least 9 "
         "of 10 seeds to agree, and across 13 all 10.",
         "", "Equivalence margins were fixed in the script before the numbers were computed: for "
         "faithfulness, 10% of KernelSHAP's lead over the random ranking on that dataset; for "
         "accuracy and macro-F1, 1 point. \"Equivalent\" means both one-sided tests against the "
         "margin reject after Holm correction."]
    s_rq1(L)
    s_repair(L)
    s_omitted(L)
    s_base(L)
    s_ties(L, args.jobs)
    (OUT / "reanalysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'reanalysis.md'}")


if __name__ == "__main__":
    main()
