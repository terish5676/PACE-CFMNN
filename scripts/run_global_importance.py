"""Global feature importance benchmark (blueprint 5.4, 10).

Same models and splits as the RQ1 benchmark (paper's 80/20 split, main model).
Five global rankings per dataset and seed:

* PACE data-based: mean |contrastive term| over the test rows (``global_importance.data_based``);
* PACE model-only: class separation of the boxes, no data (``global_importance.model_only``);
* PACE-flip, averaged: mean PACE-flip score over the test rows;
* permutation importance: accuracy drop when a feature is shuffled on the test rows (sklearn);
* mean abs KernelSHAP over 50 test rows.

Reported: Spearman rank agreement of each PACE ranking with permutation importance
(the model-based reference) and with mean |KernelSHAP|; on S1-S3 the share of each
ranking's top 3 that are the truly relevant features x0-x2; time per ranking.

KernelSHAP settings (``--shap``):
* ``v2`` (default, fair): KernelSHAP explains the predicted class's margin with
  20 real background rows (as in the fair faithfulness protocol); 10 seeds.
* ``v1``: the original setting (pseudo-probability target, 10 k-means centres;
  3 seeds), kept so ``results/global_importance_v1`` reproduces.

The v2 summary adds a paired test: does PACE-flip, averaged, agree with
permutation importance better than mean abs KernelSHAP does? (Wilcoxon over
seeds, Holm across datasets.) ``--jobs`` runs dataset x seed pairs in parallel;
the rankings do not depend on it, but the seconds then include CPU sharing.

``--extended`` (RQ1 extension B2, pre-registered in
results/extensions_v1/PREREGISTRATION.md; seeds 40-49) adds three rankings:
* mean abs PACE-SHAP: exact Shapley values of the margin (PaceShap) on the
  same 50 points and the same 200 rows as PACE-flip;
* mean abs PACE-SHAP-20: the same, on KernelSHAP's own 20 rows;
* PACE-flip, normalised: each point's PACE-flip scores scaled to sum 1
  before averaging (``global_importance.normalised_mean``).

Usage (from the project folder):
    python scripts/run_global_importance.py --jobs 4
    python scripts/run_global_importance.py --shap v1 --out results/global_importance_v1
    python scripts/run_global_importance.py --datasets S1 wine --seeds 0
    python scripts/run_global_importance.py --extended --jobs 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn import global_importance as gi  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceFlip, PaceShap  # noqa: E402

DATASETS = ["S1", "S2", "S3", "wine", "breast_cancer", "thyroid", "shuttle",
            "magic", "letter", "landsat", "page_blocks"]
DATASETS_V1 = DATASETS[:7]
PACE = ["PACE data-based", "PACE model-only", "PACE-flip, averaged"]
PACE_EXT = ["PACE-flip, normalised", "Mean abs PACE-SHAP", "Mean abs PACE-SHAP-20"]
REFS = ["Permutation importance", "Mean abs KernelSHAP"]
SHAP_BG_ROWS = 20
EXT_COMPARISONS = [("Mean abs PACE-SHAP", "Mean abs KernelSHAP"),        # primary
                   ("Mean abs PACE-SHAP-20", "Mean abs KernelSHAP"),
                   ("PACE-flip, normalised", "PACE-flip, averaged"),
                   ("PACE-flip, averaged", "Mean abs KernelSHAP")]


def run_one(name: str, seed: int, n_shap: int = 50, shap: str = "v2",
            extended: bool = False) -> list[dict]:
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    Xte, yte = np.clip(sc.transform(ds.X[sp["test"]]), 0.0, 1.0), ds.y[sp["test"]]
    model = CFMNN().fit(Xtr, ytr)
    bg = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(200, len(Xtr)), replace=False)]
    pts = Xte[np.random.default_rng(seed).permutation(len(Xte))[:n_shap]]

    rankings, secs = {}, {}

    def timed(key, fn):
        t0 = time.perf_counter()
        rankings[key] = np.asarray(fn(), float)
        secs[key] = time.perf_counter() - t0

    timed("PACE data-based", lambda: gi.data_based(model, Xte))
    timed("PACE model-only", lambda: gi.model_only(model))
    timed("PACE-flip, averaged", lambda: PaceFlip(model, bg).explain(Xte).mean(axis=0))
    timed("Permutation importance", lambda: permutation_importance(
        model, Xte, yte, n_repeats=10, random_state=seed, scoring="accuracy").importances_mean)
    shap_kw = dict(target="margin", bg_rows=SHAP_BG_ROWS) if shap == "v2" else {}
    timed("Mean abs KernelSHAP",
          lambda: np.abs(xb.kernel_shap(model, pts, bg, seed=seed, **shap_kw)).mean(axis=0))
    if extended:
        # KernelSHAP's own 20 rows (as xai_baselines._kernel_shap_margin picks them).
        rows20 = (bg if len(bg) <= SHAP_BG_ROWS else
                  bg[np.random.default_rng(seed).choice(len(bg), SHAP_BG_ROWS, replace=False)])
        timed("PACE-flip, normalised", lambda: gi.normalised_mean(PaceFlip(model, bg).explain(Xte)))
        timed("Mean abs PACE-SHAP",
              lambda: np.abs(PaceShap(model, bg, seed=seed).explain(pts)).mean(axis=0))
        timed("Mean abs PACE-SHAP-20",
              lambda: np.abs(PaceShap(model, rows20, seed=seed).explain(pts)).mean(axis=0))

    relevant = set(ds.info.get("relevant", [])) if name in ("S1", "S2", "S3") else None
    rows = []
    for key, g in rankings.items():
        top3 = np.argsort(-g, kind="stable")[:3]
        row = {"dataset": name, "seed": seed, "method": key, "seconds": secs[key],
               "gt_top3": len(set(top3) & relevant) / 3 if relevant else np.nan}
        for ref in REFS:
            row[f"rho_{ref}"] = float(stats.spearmanr(g, rankings[ref])[0])
        rows.append(row)
    return rows


def ci95(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def summarise(df: pd.DataFrame, out: Path, shap: str = "v2") -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    f = lambda c, k=2: (f"{c.mean():.{k}f} ± {ci95(c):.{k}f}" if c.notna().any() else "–")
    extended = PACE_EXT[1] in set(df.method)
    L = ["# Global feature importance" + (" (RQ1 extension B2)" if extended else ""), "",
         f"Paper's 80/20 split, main model (fixed centroid + search); seeds {seeds}; mean ± 95% CI "
         "over seeds. Spearman rank agreement with permutation importance (accuracy drop when a "
         "feature is shuffled on the test rows) and with mean abs KernelSHAP over 50 test rows. "
         "Ground truth = share of the top 3 that are the truly relevant features (S1-S3 only). "
         + ("KernelSHAP explains the predicted class's margin with 20 real background rows (fair "
            "setting); seconds were measured with several runs sharing the CPU, so compare them "
            "within a dataset only." if shap == "v2" else
            "KernelSHAP: original setting (pseudo-probability, 10 k-means centres)."), "",
         "| Dataset | Ranking | Agreement with permutation importance | Agreement with mean abs SHAP | "
         "Ground truth top 3 | Seconds |",
         "|---|---|---|---|---|---|"]
    if extended:
        L[4:4] = ["Extension rankings: mean abs PACE-SHAP (exact Shapley values of the margin on "
                  "the same 50 points, 200 rows; sampled with 32 orders per row when d > 12), "
                  "mean abs PACE-SHAP-20 (KernelSHAP's own 20 rows) and PACE-flip, normalised "
                  "(each point's scores scaled to sum 1 before averaging).", ""]
    for ds in df.dataset.unique():
        for m in PACE + (PACE_EXT if extended else []) + REFS:
            g = df[(df.dataset == ds) & (df.method == m)]
            perm = "1 (reference)" if m == REFS[0] else f(g["rho_Permutation importance"])
            agree = "1 (reference)" if m == REFS[1] else f(g["rho_Mean abs KernelSHAP"])
            L.append(f"| {ds} | {m} | {perm} | {agree} | {f(g.gt_top3)} | {f(g.seconds, 3)} |")
    if shap == "v2" and df.seed.nunique() > 1:
        from run_error_detection import paired
        pairs = EXT_COMPARISONS if extended else [("PACE-flip, averaged", "Mean abs KernelSHAP")]
        for k, (a, b) in enumerate(pairs):
            res = paired(df, a, b, metric="rho_Permutation importance")
            tag = (" (primary)" if k == 0 else " (secondary)") if extended else ""
            L += ["", f"## {a} minus {b}: agreement with permutation importance{tag}",
                  "", "Spearman rho with permutation importance; paired over seeds; two-sided Wilcoxon "
                  f"signed-rank test, Holm-corrected across datasets. Positive = {a} agrees better.", "",
                  f"| Dataset | Difference | Seeds where {a} is higher | p | p (Holm) |",
                  "|---|---|---|---|---|"]
            for _, r in res.iterrows():
                L.append(f"| {r.dataset} | {r['diff']:+.2f} ± {r.ci:.2f} | {r.wins} of {r.n} | "
                         f"{r.p:.3g} | {r.p_holm:.3g} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", choices=["v1", "v2"], default="v2")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--extended", action="store_true",
                    help="RQ1 extension B2: add mean abs PACE-SHAP and normalised PACE-flip")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    v2 = a.shap == "v2"
    if a.extended and not v2:
        raise SystemExit("--extended needs --shap v2")
    datasets = a.datasets or (DATASETS if v2 else DATASETS_V1)
    seeds = a.seeds or (list(range(40, 50)) if a.extended else list(range(10)) if v2 else [0, 1, 2])
    out = Path(a.out or ROOT / "results" / ("extensions_v1/global_importance" if a.extended else
                                            "global_importance" if v2 else "global_importance_v1"))
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        tasks = [(n, s) for n in datasets for s in seeds]
        done = {}
        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                fut = {ex.submit(run_one, n, s, 50, a.shap, a.extended): (n, s) for n, s in tasks}
                for f in as_completed(fut):
                    done[fut[f]] = f.result()
                    print(f"{fut[f][0]} seed {fut[f][1]} done", flush=True)
        else:
            for t in tasks:
                t0 = time.perf_counter()
                done[t] = run_one(*t, 50, a.shap, a.extended)
                print(f"{t[0]} seed {t[1]}: {time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame([r for t in tasks for r in done[t]]).to_csv(out / "runs.csv", index=False)
    print(summarise(pd.read_csv(out / "runs.csv"), out, a.shap))


if __name__ == "__main__":
    main()
