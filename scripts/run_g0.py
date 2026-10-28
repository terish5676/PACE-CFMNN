"""Gate G0: baseline validity of the reference CFMNN.

Runs CFMNN (original and fixed centroid rule, plus the fixed rule with
Simpson-style expansion search) with the paper's protocol:
stratified 80/20 split, min-max scaling fitted on training rows, theta = 0.3,
gamma = 4. Training rows are fed in a seeded random order, because CFMNN is a
single-pass, order-dependent learner. Random Forest and a majority-class
predictor are included for context.

Usage (from the project folder):
    python scripts/run_g0.py
    python scripts/run_g0.py --datasets S1 wine thyroid --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics, majority_baseline, timer  # noqa: E402

# Accuracy of MRCFMNN (Max_Occurrences ensemble) in the paper's Table 2, for context only:
PAPER_ACC = {"thyroid": 96.90, "shuttle": 94.04, "skin": 97.63}

# (name, centroid rule, expansion rule). The first is the paper as written.
VARIANTS = [("CFMNN-original", "original", "best"),
            ("CFMNN-fixed", "fixed", "best"),
            ("CFMNN-original+search", "original", "search"),
            ("CFMNN-fixed+search", "fixed", "search")]


def run_one(ds: data.Dataset, seed: int) -> list[dict]:
    split = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(split["train"])
    scaler = data.fit_scaler(ds.X[order])
    Xtr, ytr = scaler.transform(ds.X[order]), ds.y[order]
    Xte, yte = scaler.transform(ds.X[split["test"]]), ds.y[split["test"]]
    base = {"dataset": ds.name, "seed": seed, "n_train": len(ytr), "n_test": len(yte),
            "n_features": ds.X.shape[1], "n_classes": len(np.unique(ds.y))}
    rows, preds = [], {}

    for name, rule, expansion in VARIANTS:
        t = {}
        model = CFMNN(theta=0.3, gamma=4.0, centroid_rule=rule, expansion=expansion)
        with timer(t, "fit_s"):
            model.fit(Xtr, ytr)
        with timer(t, "predict_s"):
            det = model.predict_details(Xte)
        preds[(rule, expansion)] = det.y_pred
        rows.append({**base, "model": name, "expansion": expansion,
                     **classification_metrics(yte, det.y_pred),
                     "n_boxes": model.n_boxes_, **t,
                     "test_tied_share": 100 * float((det.n_tied > 1).mean()),
                     **{f"fit_{k}": v for k, v in model.fit_stats_.items() if k != "n_train"}})

    # Share of test predictions the centroid fix changes, for each expansion rule.
    for r in rows:
        e = r["expansion"]
        r["pred_diff_orig_vs_fixed"] = 100 * float(
            (preds[("original", e)] != preds[("fixed", e)]).mean())

    t = {}
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    with timer(t, "fit_s"):
        rf.fit(Xtr, ytr)
    with timer(t, "predict_s"):
        p = rf.predict(Xte)
    rows.append({**base, "model": "RandomForest", **classification_metrics(yte, p), **t})
    rows.append({**base, "model": "Majority",
                 **classification_metrics(yte, majority_baseline(ytr, yte))})
    return rows


def summarise(df: pd.DataFrame) -> str:
    cols = ["accuracy", "macro_f1", "balanced_accuracy", "min_class_recall", "n_boxes", "fit_s"]
    g = df.groupby(["dataset", "model"], sort=False)
    mean, sd = g[cols].mean(), g[cols].std()
    lines = ["| Dataset | Model | Accuracy | Macro-F1 | Balanced acc. | Min class recall | Boxes | Fit (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for (dsn, model), m in mean.iterrows():
        s = sd.loc[(dsn, model)]
        cell = lambda c, f="{:.2f}": "–" if pd.isna(m[c]) else (f.format(m[c]) + " ± " + f.format(s[c]))
        lines.append(f"| {dsn} | {model} | {cell('accuracy')} | {cell('macro_f1')} | "
                     f"{cell('balanced_accuracy')} | {cell('min_class_recall')} | "
                     f"{cell('n_boxes', '{:.0f}')} | {cell('fit_s', '{:.2f}')} |")

    cf = df[df.model.str.startswith("CFMNN")]
    extra = cf.groupby(["dataset", "model"], sort=False)[
        ["fit_n_inside", "fit_n_best_tied", "fit_n_floor_tied", "test_tied_share",
         "pred_diff_orig_vs_fixed", "n_train"]].mean()
    lines += ["", "Training and tie statistics (means over seeds):", "",
              "| Dataset | Model | Inside points (% of train) | Best-box ties in training (% of train) "
              "| of which at the 0.5 floor | Tied test predictions (%) | Predictions changed by fix (%) |",
              "|---|---|---|---|---|---|---|"]
    for (dsn, model), m in extra.iterrows():
        n = m["n_train"]
        lines.append(f"| {dsn} | {model} | {100 * m.fit_n_inside / n:.1f} | "
                     f"{100 * m.fit_n_best_tied / n:.1f} | {100 * m.fit_n_floor_tied / n:.1f} | "
                     f"{m.test_tied_share:.1f} | {m.pred_diff_orig_vs_fixed:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+",
                    default=list(data.BUILTIN) + list(data.UCI))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(ROOT / "results" / "g0"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for name in args.datasets:
        try:
            ds = data.load(name, seed=0)
        except FileNotFoundError as e:
            skipped.append(f"{name}: {e}")
            continue
        for seed in args.seeds:
            rows.extend(run_one(ds, seed))
            print(f"{name} seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    text = ["# G0 baseline results", "",
            "Protocol: stratified 80/20 split, scaling fitted on training rows, theta = 0.3, "
            f"gamma = 4, training order shuffled by seed; seeds {args.seeds}. "
            "Mean ± standard deviation over seeds.", "",
            "Variants: *original* = the paper's centroid rule (points already inside a box "
            "are skipped); *fixed* = every accepted point updates the centroid; *+search* = "
            "expand the best same-class box that can still expand (Simpson's FMNN) instead of "
            "testing only the best box (the paper). Only CFMNN-original is the paper as written.",
            "", summarise(df)]
    paper = {k: v for k, v in PAPER_ACC.items() if k in set(df.dataset)}
    if paper:
        text += ["", "Paper (MRCFMNN Spark ensemble, Max_Occurrences) accuracy for context: "
                 + ", ".join(f"{k} {v:.2f}%" for k, v in paper.items())]
    if skipped:
        text += ["", "Skipped (data not downloaded yet):", ""] + [f"- {s}" for s in skipped]
    (out / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
