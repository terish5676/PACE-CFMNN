"""Re-analysis of existing results for the pre-submission review.

Writes results/review_v1/reanalysis.md. Everything comes from saved results,
except the tie counts on v3's seeds (review M3): these refit v3's models once
and are saved to results/review_v1/ties_v3.csv, which later runs reuse.

Post hoc: none of these analyses was pre-registered. Each one answers an issue
of the review (codes in the headings).

    python -B scripts/review_reanalysis.py --jobs 6
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

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_error_detection import holm  # noqa: E402

RES = ROOT / "results"
OUT = RES / "review_v1"
RQ1 = rf.DATASETS
D13 = ["S1", "S2", "S3", "S1c", "S2c"] + RQ1[3:]
REAL = ["wine", "breast_cancer", "thyroid", "shuttle", "magic", "letter", "landsat", "page_blocks"]
EXACT = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
RQ2_SYN = ["S1", "S2", "S3", "S2-imbalanced", "S2-label-noise", "S2-shifted"]
RQ2_REAL = ["digits", "thyroid", "skin", "magic", "letter", "landsat", "page_blocks"]


def compare(df, a, b, metric, lower, datasets):
    """a vs b per dataset, paired over seeds: Wilcoxon, Holm across the datasets,
    and 95% t-intervals, unadjusted and Bonferroni-adjusted."""
    rows = []
    for ds in datasets:
        g = df[df.dataset == ds]
        x = g[g.method == a].set_index("seed")[metric]
        y = g[g.method == b].set_index("seed")[metric]
        d = (x - y.reindex(x.index)).dropna()
        n, sd = len(d), d.std(ddof=1)
        rows.append({"dataset": ds, "a": x.mean(), "b": y.mean(), "diff": d.mean(), "sd": sd, "n": n,
                     "dz": d.mean() / sd if sd > 0 else np.nan,
                     "wins": int((d < 0).sum() if lower else (d > 0).sum()),
                     "p": stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.to_numpy())
    half = lambda alpha: stats.t.ppf(1 - alpha / 2, t.n - 1) * t.sd / np.sqrt(t.n)   # noqa: E731
    t["ci"], t["ci_bonf"] = half(0.05), half(0.05 / len(t))
    good = t["diff"] < 0 if lower else t["diff"] > 0
    t["verdict"] = np.where(t.p_holm < 0.05, np.where(good, "better", "worse"), "level")
    return t


def counts(t, a):
    v = t.verdict.value_counts()
    return (f"**{a} is better on {v.get('better', 0)}, level on {v.get('level', 0)} and worse on "
            f"{v.get('worse', 0)}** (Holm across {len(t)})")


def write_table(L, t, a, b, metric, lower, gap=None):
    L += ["", f"{a} minus {b} in `{metric}` ({'lower' if lower else 'higher'} is better). "
          + counts(t, a) + ".", "",
          f"| Dataset | {a} | {b} | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by {a} | p (Holm) |"
          + (" Share of the gap to Random |" if gap is not None else ""),
          "|---|---|---|---|---|---|---|---|---|" + ("---|" if gap is not None else "")]
    for i, r in t.iterrows():
        mark = lambda h: "" if abs(r["diff"]) <= h else " (excludes 0)"   # noqa: E731
        L.append(f"| {r.dataset} | {r.a:.4f} | {r.b:.4f} | {r['diff']:+.4f} | ±{r.ci:.4f}{mark(r.ci)} | "
                 f"±{r.ci_bonf:.4f}{mark(r.ci_bonf)} | {r.dz:+.2f} | {r.wins} of {r.n} | {r.p_holm:.3g} |"
                 + (f" {gap[i]:.0%} |" if gap is not None else ""))


def means(L, df, methods, metric, datasets, digits=4):
    m = df[df.method.isin(methods)].groupby(["dataset", "method"])[metric].mean().unstack()
    L += ["", "| Dataset | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for ds in datasets:
        L.append(f"| {ds} | " + " | ".join(f"{m.loc[ds, x]:.{digits}f}" for x in methods) + " |")


def b1(L):
    c = pd.read_csv(RES / "extensions_v1" / "conditional" / "runs.csv")
    L += ["## B1. PACE-SHAP vs KernelSHAP on the same 200 rows",
          "", "From B3 (`extensions_v1/conditional/runs.csv`, seeds 40–49, 13 datasets). There, KernelSHAP "
          "is this repository's solver on the same 200 explainer rows as PACE-SHAP, with 200 coalitions and "
          "no feature selection. v3's KernelSHAP used 20 of the rows and shap's default feature selection. "
          "The tests are the usual interventional deletion and insertion tests."]
    for metric, lower in [("margin_auc", True), ("insertion_margin_auc", False)]:
        write_table(L, compare(c, "PACE-SHAP", "KernelSHAP", metric, lower, D13), "PACE-SHAP", "KernelSHAP",
                    metric, lower)
    ms = c.groupby(["dataset", "method"]).ms_per_explanation.mean().unstack()
    r = (ms["KernelSHAP"] / ms["PACE-SHAP"]).reindex(D13)
    L += ["", "Milliseconds per explanation, KernelSHAP divided by PACE-SHAP (runs shared the CPU; each pair "
          "of games shares its model calls): " + ", ".join(f"{d} {v:.1f}" for d, v in r.items()) + "."]


def m4(L, v3):
    L += ["", "## M4 and m4. Absolute AUCs, effect sizes and adjusted intervals (v3)",
          "", "From `faithfulness_v3/runs.csv` (seeds 30–39). Share of the gap to Random: how much of "
          "KernelSHAP's distance from Random PACE-SHAP adds (positive: PACE-SHAP is further from Random). "
          "Figure 2 draws the unadjusted 95% intervals while its filled markers use Holm; the Bonferroni "
          "interval is the adjusted version (11 datasets)."]
    methods = ["PACE-SHAP", "PACE-SHAP-20", "KernelSHAP", "KernelSHAP-strong", "LIME", "PACE-flip", "Random"]
    for metric, title in [("margin_auc", "Mean deletion margin AUC (lower is more faithful)"),
                          ("insertion_margin_auc", "Mean insertion margin AUC (higher is more faithful)")]:
        L += ["", f"### {title}"]
        means(L, v3, methods, metric, RQ1)
    mm = {k: v3.groupby(["dataset", "method"])[k].mean().unstack() for k in ("margin_auc", "insertion_margin_auc")}
    for b in ["KernelSHAP", "KernelSHAP-strong"]:
        for metric, lower in [("margin_auc", True), ("insertion_margin_auc", False)]:
            t = compare(v3, "PACE-SHAP", b, metric, lower, RQ1)
            m = mm[metric].reindex(RQ1)
            sign = 1 if lower else -1
            gap = (sign * (m[b] - m["PACE-SHAP"]) / (sign * (m["Random"] - m[b]))).to_numpy()
            L += ["", f"### PACE-SHAP vs {b}, {'deletion' if lower else 'insertion'}"]
            write_table(L, t, "PACE-SHAP", b, metric, lower, gap)


def m2(L, v3):
    L += ["", "## M2. Stability when both methods re-draw 20 rows (v3)",
          "", "v3's re-run re-drew the 20 rows of KernelSHAP and of PACE-SHAP-20, but kept PACE-SHAP's 200. "
          "Stability: Jaccard overlap of the top 3 features between the two runs."]
    means(L, v3, ["PACE-SHAP", "PACE-SHAP-20", "KernelSHAP", "LIME", "PACE-flip"], "stability_top3", RQ1, 3)
    write_table(L, compare(v3, "PACE-SHAP-20", "KernelSHAP", "stability_top3", False, RQ1),
                "PACE-SHAP-20", "KernelSHAP", "stability_top3", False)


def tie_rows(name, seed):
    _, _, Xte, model, _ = rf.setup(name, seed, 50)
    det = model.predict_details(Xte)
    m = rf._margin(det, det.cls)
    print(f"{name} seed {seed}: tied {(m <= 1e-12).sum()} of {len(m)}", flush=True)
    return [{"dataset": name, "seed": seed, "n": len(m), "n_tied": int((m <= 1e-12).sum()),
             "n_near_tie": int((m < 0.01).sum()), "median_margin": float(np.median(m))}]


def m3(L, jobs):
    csv = OUT / "ties_v3.csv"
    tasks = [(d, s) for d in RQ1 for s in range(30, 40)]
    run_all(tie_rows, pending(csv, tasks, resume=True), csv, jobs, {"letter": 3, "magic": 2})
    t = pd.read_csv(csv).groupby("dataset").sum(numeric_only=True).reindex(RQ1)
    L += ["", "## M3. Tied explained points (v3's models, seeds 30–39)",
          "", "A point is tied when its margin is 0 (at most 1e-12): its class is then decided by the nearest "
          "centroid, which the explained margin ignores.", "",
          "| Dataset | Points | Tied | Margin below 0.01 |", "|---|---|---|---|"]
    for ds, r in t.iterrows():
        L.append(f"| {ds} | {int(r.n)} | {r.n_tied / r.n:.0%} | {r.n_near_tie / r.n:.0%} |")


def m1(L):
    s = pd.read_csv(RES / "faithfulness_v3" / "settings.csv")
    s = s[(s.R == 200) & (s["mode"] == "sampled")]
    L += ["", "## M1. PACE-SHAP's sampled mode (from `faithfulness_v3/settings.md`, seed 99)",
          "", "Error against a reference with 128 orders per row (not exact values; Test 1 adds that).", "",
          "| Dataset | d | Orders per row | Relative L1 error | Top-3 overlap |", "|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.dataset} | {r.d} | {int(r.P)} | {r.rel_l1_error:.3f} | {r.top3_overlap:.2f} |")


def m6(L):
    b = pd.read_csv(RES / "extensions_v1" / "counterfactuals" / "runs.csv")
    rows = []
    for (ds, s), g in b[b.method.isin(["PACE", "DiCE"])].groupby(["dataset", "seed"]):
        P, D = (g[g.method == m].set_index("point") for m in ("PACE", "DiCE"))
        both = P.index[P.valid.eq(True) & D.reindex(P.index).valid.eq(True)]
        for m, X in (("PACE", P), ("DiCE", D)):
            rows.append({"dataset": ds, "seed": s, "method": m, "n_both": len(both),
                         **{k: X.loc[both, k].mean() for k in ("l1", "n_changed", "plausibility")}})
    q = pd.DataFrame(rows)
    L += ["", "## M6. Counterfactuals: PACE vs DiCE, tested over seeds",
          "", "From B1 (`extensions_v1/counterfactuals/runs.csv`, seeds 40–49), on the points where both "
          "found a valid counterfactual. The first counterfactual run tested over points pooled from 3 "
          "models; here each seed is one observation."]
    for metric, title in [("l1", "Size of the change (L1)"), ("n_changed", "Features changed"),
                          ("plausibility", "Distance to real data (plausibility)")]:
        t = compare(q, "PACE", "DiCE", metric, True, RQ1)
        real = t[t.dataset.isin(REAL)].verdict.value_counts()
        L += ["", f"### {title}", "", f"On the 8 real datasets: PACE better on {real.get('better', 0)}, "
              f"level on {real.get('level', 0)}, worse on {real.get('worse', 0)}."]
        write_table(L, t, "PACE", "DiCE", metric, True)


def rq2_units(df, metric="auprc"):
    syn = df[df.dataset.isin(RQ2_SYN)].groupby(["seed", "method"], as_index=False)[metric].mean()
    return pd.concat([syn.assign(dataset="Synthetic family"),
                      df[df.dataset.isin(RQ2_REAL)][["dataset", "seed", "method", metric]]])


def m6_rq2(L):
    u = rq2_units(pd.read_csv(RES / "error_detection_v2" / "runs.csv"))
    L += ["", "## m6. Do the geometry signals add anything? (RQ2 combination run, seeds 20–29)"]
    write_table(L, compare(u, "LR: all", "LR: kNN + TS + order ens.", "auprc", False,
                           ["Synthetic family"] + RQ2_REAL), "LR: all", "LR: kNN + TS + order ens.",
                "auprc", False)


def m7(L):
    c = pd.read_csv(RES / "extensions_v1" / "conditional" / "runs.csv")
    L += ["", "## m7. Conditional vs interventional PACE-SHAP on the conditional test (B3)"]
    for metric, lower in [("cond_margin_auc", True), ("cond_insertion_margin_auc", False)]:
        write_table(L, compare(c, "PACE-SHAP-cond", "PACE-SHAP", metric, lower, D13), "PACE-SHAP-cond",
                    "PACE-SHAP", metric, lower)


def m8(L):
    it = pd.read_csv(RES / "extensions_v1" / "interactions" / "runs.csv")
    L += ["", "## m8. Figure 4: the lead of the (x0, x1) interaction over the next pair (B4)", "",
          "| Dataset | Lead as a ratio | Lead as a difference |", "|---|---|---|"]
    for ds in ["S1", "S2", "S3"]:
        rat, dif = [], []
        for _, g in it[it.dataset == ds].groupby("seed"):
            pair = (g.i == 0) & (g.j == 1)
            top, nxt = g[pair].strength.iloc[0], g[~pair].strength.max()
            rat.append(top / nxt)
            dif.append(top - nxt)
        L.append(f"| {ds} | {min(rat):.2f}–{max(rat):.2f} | {min(dif):.4f}–{max(dif):.4f} |")


def m13(L):
    L += ["", "## m13. Page Blocks' share of errors in the two RQ2 runs", "",
          "| Run | Seeds | Share of errors per seed | Mean |", "|---|---|---|---|"]
    for name, path in [("Confirmatory run", "error_detection_confirmatory"),
                       ("Combination run", "error_detection_v2")]:
        d = pd.read_csv(RES / path / "runs.csv").drop_duplicates(["dataset", "seed"])
        pb = d[d.dataset == "page_blocks"].sort_values("seed")
        share = pb.n_errors / pb.n_eval
        L.append(f"| {name} | {pb.seed.min()}–{pb.seed.max()} | "
                 + ", ".join(f"{s}: {v:.3f}" for s, v in zip(pb.seed, share)) + f" | {share.mean():.3f} |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Re-analysis of existing results for the review")
    ap.add_argument("--jobs", type=int, default=1, help="parallel jobs for the tie counts")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v3 = pd.read_csv(RES / "faithfulness_v3" / "runs.csv")
    L = ["# Re-analysis of existing results for the pre-submission review", "",
         "Post hoc: written after the review, from results that already existed. Each section answers one "
         "of the review's issues (its code is in the heading). Tests are paired over seeds: two-sided "
         "Wilcoxon signed-rank, Holm across the datasets, as in the paper. Made by "
         "`scripts/review_reanalysis.py`.", ""]
    b1(L)
    m4(L, v3)
    m2(L, v3)
    m3(L, a.jobs)
    m1(L)
    m6(L)
    m6_rq2(L)
    m7(L)
    m8(L)
    m13(L)
    text = "\n".join(L) + "\n"
    (OUT / "reanalysis.md").write_text(text, encoding="utf-8")
    print(f"Wrote {OUT / 'reanalysis.md'}")


if __name__ == "__main__":
    main()
