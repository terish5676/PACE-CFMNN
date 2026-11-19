"""Speed of PACE-flip: the fast exact explainer against the original loop,
brute-force occlusion and PACE-multi, on the same models, points and
background as ``run_faithfulness.py``.

Also checks, on every dataset and seed, that the fast explainer returns
exactly the same arrays as the original loop (bit for bit).

Usage: python scripts/time_pace_flip.py [--datasets S1 wine] [--seeds 0 1 2] [--n 50]
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
from pace_cfmnn.explain import PaceFlip, contrastive_multi, pace_flip  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from tests.reference_pace_flip import pace_flip_loop  # noqa: E402


def prepare(name, seed, n_explain):
    """Same model, test points and background as run_faithfulness.run."""
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    te = np.random.default_rng(seed).permutation(sp["test"])[:n_explain]
    Xte = sc.transform(ds.X[te])
    model = CFMNN().fit(Xtr, ytr)
    bg = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(200, len(Xtr)), replace=False)]
    return model, Xte, bg


def per_point_ms(fn, n, reps):
    """Median over ``reps`` runs of one batch call, in ms per explained point."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times)) / n


def one_at_a_time_ms(fn, X, reps=3):
    """Median over ``reps`` passes of explaining the points one call each, ms per point."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for p in range(len(X)):
            fn(X[p:p + 1])
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times)) / len(X)


def run(name, seed, n_explain, reps):
    model, X, bg = prepare(name, seed, n_explain)
    n = len(X)
    ref = pace_flip_loop(model, X, bg, return_parts=True)
    ex = PaceFlip(model, bg)
    new = ex.explain(X, return_parts=True)
    identical = all(np.array_equal(u, v) for u, v in zip(ref, new))
    row = {
        "dataset": name, "seed": seed, "n": n, "n_boxes": model.n_boxes_,
        "identical_to_loop": identical,
        "boxes_kept_per_feature": ex.stats_["boxes_kept_per_feature"],
        "share_of_boxes_skipped": ex.stats_["share_of_boxes_skipped"],
        "tie_break_cells_share": ex.stats_["tie_break_cells"] / ex.stats_["cells"],
        "flip_old_loop": per_point_ms(lambda: pace_flip_loop(model, X, bg), n, 1),
        "flip_new": per_point_ms(lambda: pace_flip(model, X, bg), n, reps),
        "flip_new_prebuilt": per_point_ms(lambda: ex.explain(X), n, reps),
        "multi": per_point_ms(lambda: contrastive_multi(model, X), n, reps),
        "occlusion": per_point_ms(lambda: xb.occlusion(model, X, bg), n, 1),
        "single_flip_new": one_at_a_time_ms(ex.explain, X),
        "single_multi": one_at_a_time_ms(lambda x: contrastive_multi(model, x), X),
    }
    print(f"{name} seed {seed}: identical {identical}, new {row['flip_new']:.3f} ms, "
          f"old {row['flip_old_loop']:.2f} ms, multi {row['multi']:.3f} ms, "
          f"occlusion {row['occlusion']:.2f} ms", flush=True)
    return row


def fmt(v):
    return f"{v:.3f}" if v < 1 else (f"{v:.2f}" if v < 10 else f"{v:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["S1", "S2", "S3", "wine", "breast_cancer",
                                                      "thyroid", "shuttle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "speed"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([run(ds, s, a.n, a.reps) for ds in a.datasets for s in a.seeds])
    df.to_csv(out / "runs.csv", index=False)
    g = df.groupby("dataset", sort=False).mean(numeric_only=True)
    same = df.groupby("dataset", sort=False)["identical_to_loop"].all()

    lines = [
        "# PACE-flip speed", "",
        f"Same models, test points and background as `run_faithfulness.py` (main model: fixed "
        f"centroid + search expansion; {a.n} test points per seed; seeds {a.seeds}; 32 replacement "
        "values per feature). Milliseconds per explained point, mean over seeds of the median of "
        f"{a.reps} runs (1 run for the old loop and occlusion, which are slow). All methods use the "
        "model's cache-friendly chunking. Times depend on the machine; compare within a row.", "",
        "## Batch of points (one call explains all of them)", "",
        "| Dataset | Boxes | PACE-flip, original loop | PACE-flip, new | New, tables prebuilt | "
        "PACE-multi | Occlusion (brute force) | New vs loop | New vs occlusion |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dsn, r in g.iterrows():
        lines.append(
            f"| {dsn} | {r.n_boxes:.0f} | {fmt(r.flip_old_loop)} | {fmt(r.flip_new)} | "
            f"{fmt(r.flip_new_prebuilt)} | {fmt(r.multi)} | {fmt(r.occlusion)} | "
            f"{r.flip_old_loop / r.flip_new:.0f}× | {r.occlusion / r.flip_new:.0f}× |")
    lines += ["", "## One point per call (tables prebuilt for PACE-flip)", "",
              "| Dataset | PACE-multi | PACE-flip, new |", "|---|---|---|"]
    for dsn, r in g.iterrows():
        lines.append(f"| {dsn} | {fmt(r.single_multi)} | {fmt(r.single_flip_new)} |")
    lines += ["", "## Work skipped and exactness", "",
              "Boxes kept: per point and feature, boxes the exact bound could not rule out. "
              "Tie-break cells: share of (point, feature, value) cells where boxes of two classes "
              "tied at the top and the nearest centroid decided.", "",
              "| Dataset | Boxes kept per feature | Boxes skipped | Tie-break cells | "
              "Identical to the original loop (all seeds) |", "|---|---|---|---|---|"]
    for dsn, r in g.iterrows():
        lines.append(f"| {dsn} | {r.boxes_kept_per_feature:.1f} of {r.n_boxes:.0f} | "
                     f"{100 * r.share_of_boxes_skipped:.0f}% | {100 * r.tie_break_cells_share:.1f}% | "
                     f"{'yes' if same[dsn] else 'NO'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
