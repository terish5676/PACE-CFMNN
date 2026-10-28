"""RQ1 primary endpoint: deletion faithfulness and time per explanation.

For each explained test point, features are "deleted" one at a time in the
order a method ranks them (largest |attribution| first) by copying that
feature's value from a real background row (averaged over 10 rows). We track
whether the model still predicts the original class, and the normalised margin
of that class. A more faithful ranking drops both sooner, so LOWER AUCs are
better. A random ranking is the reference.

Usage: python scripts/run_faithfulness.py [--datasets S1 S2 wine] [--n 50] [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.explain import contrastive, contrastive_multi, decomposition, pace_flip  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402


def _margin(det, cls):
    """Rescaled support of the original class minus the best other class."""
    s = det.support_rescaled.copy()
    rows = np.arange(len(cls))
    a = s[rows, cls].copy()
    s[rows, cls] = -np.inf
    return a - s.max(axis=1)


def deletion_auc(model, X, attr, background, reps=10, seed=0):
    """Per-point deletion AUCs (class kept, and margin) over k = 0..d deletions.

    A deleted feature takes its value from a real background row (averaged over
    ``reps`` rows), so binary features stay binary.
    """
    n, d = X.shape
    det0 = model.predict_details(X)
    orig = det0.cls
    m0 = np.maximum(_margin(det0, orig), 1e-12)
    order = np.argsort(-np.abs(attr), axis=1, kind="stable")
    rng = np.random.default_rng(seed)
    kept_all, marg_all = [], []
    for _ in range(reps):
        B = background[rng.integers(len(background), size=n)]
        Z = X.copy()
        kept, marg = [np.ones(n)], [np.ones(n)]
        for k in range(d):
            idx = order[:, k]
            Z[np.arange(n), idx] = B[np.arange(n), idx]
            det = model.predict_details(Z)
            kept.append((det.cls == orig).astype(float))
            marg.append(np.clip(_margin(det, orig) / m0, -1, 1))
        kept_all.append(np.trapezoid(np.array(kept), dx=1.0 / d, axis=0))
        marg_all.append(np.trapezoid(np.array(marg), dx=1.0 / d, axis=0))
    return np.mean(kept_all, axis=0), np.mean(marg_all, axis=0)


def run(name, seed, n_explain):
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    te = np.random.default_rng(seed).permutation(sp["test"])[:n_explain]
    Xte = sc.transform(ds.X[te])
    model = CFMNN().fit(Xtr, ytr)
    fill = Xtr.mean(axis=0)
    bg = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(200, len(Xtr)), replace=False)]

    methods = {
        "PACE-contrastive": lambda r: contrastive(model, Xte).e,
        "PACE-multi": lambda r: contrastive_multi(model, Xte),
        "PACE-flip": lambda r: pace_flip(model, Xte, bg),
        "Occlusion (brute force)": lambda r: xb.occlusion(model, Xte, bg),
        "PACE-decomposition": lambda r: decomposition(model, Xte, bg).phi,
        "KernelSHAP": lambda r: xb.kernel_shap(model, Xte, bg, seed=r),
        "LIME": lambda r: xb.lime_tabular(model, Xte, Xtr, seed=r),
        "Random": lambda r: np.random.default_rng(r).uniform(size=Xte.shape),
    }
    relevant = set(ds.info.get("relevant", [])) if name in ("S1", "S2", "S3") else None
    rows = []
    for m, fn in methods.items():
        t0 = time.perf_counter()
        attr = fn(seed)
        dt = (time.perf_counter() - t0) / len(Xte)
        auc, mauc = deletion_auc(model, Xte, attr, bg, seed=seed)
        top3 = np.argsort(-np.abs(attr), axis=1, kind="stable")[:, :3]
        # Stability: same points, explainer re-run with another random seed.
        top3b = np.argsort(-np.abs(fn(seed + 1000)), axis=1, kind="stable")[:, :3]
        stab = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(top3, top3b)])
        gt = (np.mean([len(set(t) & relevant) / 3 for t in top3]) if relevant else np.nan)
        rows.append({"dataset": name, "seed": seed, "method": m, "deletion_auc": auc.mean(),
                     "margin_auc": mauc.mean(), "stability_top3": stab, "gt_top3_precision": gt,
                     "ms_per_explanation": 1000 * dt, "n": len(Xte)})
        print(f"{name} seed {seed} {m}: AUC {auc.mean():.3f} margin {mauc.mean():.3f} "
              f"stab {stab:.2f} gt {gt:.2f}, {1000 * dt:.2f} ms", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["S1", "S2", "S3", "wine", "breast_cancer",
                                                      "thyroid", "shuttle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results" / "faithfulness"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r for ds in a.datasets for s in a.seeds for r in run(ds, s, a.n)])
    df.to_csv(out / "runs.csv", index=False)
    g = df.groupby(["dataset", "method"], sort=False)[["deletion_auc", "margin_auc", "stability_top3", "gt_top3_precision", "ms_per_explanation"]]
    mean, sd = g.mean(), g.std()
    lines = ["# Deletion faithfulness (lower AUC = more faithful)", "",
             f"{a.n} test points per dataset and seed; seeds {a.seeds}; features deleted by "
             "copying values from real background rows (10 draws); main model (fixed centroid + search).", "",
             "Stability = top-3 overlap (Jaccard) when the explainer is re-run with another seed (1 = identical). Ground truth = share of top-3 features that are truly relevant (synthetic sets only).", "",
             "| Dataset | Method | Class-kept AUC | Margin AUC | Stability | Ground truth top-3 | ms / explanation |", "|---|---|---|---|---|---|---|"]
    for (dsn, m), r in mean.iterrows():
        s = sd.loc[(dsn, m)]
        lines.append(f"| {dsn} | {m} | {r.deletion_auc:.3f} ± {s.deletion_auc:.3f} | "
                     f"{r.margin_auc:.3f} ± {s.margin_auc:.3f} | {r.stability_top3:.2f} | "
                     f"{'–' if pd.isna(r.gt_top3_precision) else f'{r.gt_top3_precision:.2f}'} | "
                     f"{r.ms_per_explanation:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
