"""Test 4 of results/review_v1/PREREGISTRATION.md (descriptive): how accurate is the explained model?

CFMNN (the main model: fixed centroid + search expansion), 10-nearest
neighbours and a Random Forest (200 trees, as in G0) on all 13 datasets, with
the paper's 80/20 split and seeds 0-4 (as G0). Every model gets the same
min-max-scaled training rows, in the same order.

    python -B scripts/accuracy_table.py
    python -B scripts/accuracy_table.py --resume        # continue after a stop
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

DATASETS = ["S1", "S2", "S3", "wine", "breast_cancer", "digits", "thyroid", "shuttle", "skin",
            "magic", "letter", "landsat", "page_blocks"]
SEEDS = [0, 1, 2, 3, 4]
MODELS = ["CFMNN", "10-NN", "Random Forest"]
OUT = ROOT / "results" / "review_v1" / "accuracy"


def run_one(name, seed):
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])        # as G0 and RQ1
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    Xte, yte = sc.transform(ds.X[sp["test"]]), ds.y[sp["test"]]
    make = {"CFMNN": lambda: CFMNN(),
            "10-NN": lambda: KNeighborsClassifier(n_neighbors=10),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=seed,
                                                            n_jobs=-1)}
    rows = []
    for m in MODELS:
        t0 = time.perf_counter()
        pred = make[m]().fit(Xtr, ytr).predict(Xte)
        rows.append({"dataset": name, "seed": seed, "model": m,
                     "accuracy": 100 * float(np.mean(pred == yte)), "n_train": len(ytr),
                     "n_test": len(yte), "seconds": time.perf_counter() - t0})
    print(f"{name} seed {seed}: " + ", ".join(f"{r['model']} {r['accuracy']:.1f}%" for r in rows),
          flush=True)
    return rows


def summarise(df, out: Path) -> str:
    L = ["# Review Test 4 (descriptive): accuracy of the explained model and two references", "",
         "Rules: [`../PREREGISTRATION.md`](../PREREGISTRATION.md) (Test 4). The paper's 80/20 split, "
         f"seeds {sorted(int(s) for s in df.seed.unique())}; accuracy (%) on the test rows, mean ± "
         "standard deviation over seeds.", "",
         "| Dataset | Training rows | " + " | ".join(MODELS) + " | CFMNN minus the better reference |",
         "|---|---|" + "---|" * len(MODELS) + "---|"]
    for ds, g in df.groupby("dataset", sort=False):
        acc = {m: g[g.model == m].accuracy for m in MODELS}
        gap = acc["CFMNN"].mean() - max(acc["10-NN"].mean(), acc["Random Forest"].mean())
        L.append(f"| {ds} | {int(g.n_train.iloc[0])} | "
                 + " | ".join(f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"
                              for v in acc.values()) + f" | {gap:+.1f} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Review Test 4: accuracy table")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(d, s) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs, {"skin": 2})
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
