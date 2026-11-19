"""Post-hoc check for the RQ2 confirmatory run (not pre-registered; descriptive only).

Why do the label-reading baselines beat PACE's signals? This script measures how
accurate a plain 10-nearest-neighbour classifier is on the exact splits of the
confirmatory run, and how many of the CFMNN's errors it disagrees with. It
rebuilds the splits independently and checks that the CFMNN accuracy matches
the run's runs.csv, which shows the splits are the same.

    python scripts/knn_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_error_detection as red  # noqa: E402
from pace_cfmnn import CFMNN, data  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "error_detection_confirmatory"
UNITS = red.SYNTHETIC + ["digits", "thyroid", "skin", "magic", "letter", "landsat", "page_blocks"]


def splits(name: str, seed: int):
    """(train rows, evaluation X, evaluation y, dataset) as in run_one(protocol='confirmatory')."""
    ds = red.load(name, seed)
    if len(ds.y) < red.SMALL:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for f, (tr_all, ev) in enumerate(outer.split(ds.X, ds.y)):
            inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + f)
            tr_i, _ = next(inner.split(tr_all, ds.y[tr_all]))
            tr = np.random.default_rng(seed + f).permutation(tr_all[tr_i])
            yield ds.X[tr], ds.y[tr], ds.X[ev], ds.y[ev]
        return
    roles = data.role_split(ds.y, seed)
    tr = np.random.default_rng(seed).permutation(roles["train"])
    if name == "S2-shifted":
        sh = data.make_synthetic("S2", seed=seed + 10_000, shift=red.SHIFT)
        yield ds.X[tr], ds.y[tr], sh.X, sh.y
        return
    ev = np.concatenate([roles["pool"], roles["accept"], roles["test"]])
    if len(ev) > red.MAX_EVAL:
        ev = np.sort(np.random.default_rng(seed + 2).choice(ev, red.MAX_EVAL, replace=False))
    yield ds.X[tr], ds.y[tr], ds.X[ev], ds.y[ev]


def main():
    runs = pd.read_csv(OUT / "runs.csv").drop_duplicates(["dataset", "seed"]).set_index(["dataset", "seed"])
    rows = []
    for name in UNITS:
        for seed in red.CONF_SEEDS:
            n = c_ok = k_ok = errs = caught = 0
            for Xtr, ytr, Xev, yev in splits(name, seed):
                sc = data.fit_scaler(Xtr)
                tf = lambda X: np.clip(sc.transform(X), 0.0, 1.0)
                pc = CFMNN().fit(tf(Xtr), ytr).predict(tf(Xev))
                pk = KNeighborsClassifier(n_neighbors=10).fit(tf(Xtr), ytr).predict(tf(Xev))
                e = pc != yev
                n += len(yev); c_ok += (~e).sum(); k_ok += (pk == yev).sum()
                errs += e.sum(); caught += (pk[e] != pc[e]).sum()
            acc = 100 * c_ok / n
            assert abs(acc - runs.loc[(name, seed), "accuracy"]) < 1e-9, (name, seed, acc)
            rows.append({"dataset": name, "seed": seed, "cfmnn": acc, "knn": 100 * k_ok / n,
                         "caught": caught / errs})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "knn_reference.csv", index=False)
    m = df.groupby("dataset", sort=False)[["cfmnn", "knn", "caught"]].mean()
    lines = ["# Post-hoc: CFMNN vs a plain 10-NN classifier on the confirmatory splits", "",
             "Not pre-registered; descriptive only. Seeds 10-19, the same splits as `runs.csv` "
             "(the CFMNN accuracy matches it for every seed). Mean over seeds.", "",
             "| Dataset | CFMNN accuracy (%) | 10-NN accuracy (%) | CFMNN errors where 10-NN predicts another class |",
             "|---|---|---|---|"]
    for ds, r in m.iterrows():
        lines.append(f"| {ds} | {r.cfmnn:.1f} | {r.knn:.1f} | {r.caught:.2f} |")
    (OUT / "knn_reference.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
