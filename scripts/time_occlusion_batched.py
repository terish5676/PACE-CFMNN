"""PACE-flip's speed against brute-force occlusion run in batches (review check, post hoc).

The paper's occlusion baseline, ``xai_baselines.occlusion``, calls the model
once per point and feature. ``xai_baselines.occlusion_batched`` sends the
modified inputs of all points in one call, which removes that per-call
overhead. This script times PACE-flip and both occlusions on the same models,
test points and background as the paper's timing (``run_faithfulness.py``:
50 test points, 36 for Wine; 200 background rows; 32 values per feature) and
checks that all three give the same scores. A descriptive check added after
the pre-submission review, not planned in advance.

Writes results/review_v1/occlusion_batched/runs.csv and summary.md.

    python -B scripts/time_occlusion_batched.py [--datasets S1 wine] [--seeds 0 1 2] [--reps 3]
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
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import pace_flip  # noqa: E402
from time_pace_flip import fmt, per_point_ms, prepare  # noqa: E402

DATASETS = ["S1", "S2", "S3", "wine", "breast_cancer", "thyroid", "shuttle", "magic", "letter",
            "landsat", "page_blocks"]
NAMES = {"wine": "Wine", "breast_cancer": "Breast Cancer", "thyroid": "Thyroid", "shuttle": "Shuttle",
         "magic": "MAGIC", "letter": "Letter", "landsat": "Landsat", "page_blocks": "Page Blocks"}


def run(name, seed, n_explain, reps):
    model, X, bg = prepare(name, seed, n_explain)
    n = len(X)
    flip = pace_flip(model, X, bg)                      # first calls also warm up
    batched = xb.occlusion_batched(model, X, bg)
    t0 = time.perf_counter()
    loop = xb.occlusion(model, X, bg)                   # slow: one run, as in the paper's timing
    loop_ms = 1000 * (time.perf_counter() - t0) / n
    row = {
        "dataset": name, "seed": seed, "n": n, "d": X.shape[1],
        "flip_ms": per_point_ms(lambda: pace_flip(model, X, bg), n, reps),
        "occlusion_loop_ms": loop_ms,
        "occlusion_batched_ms": per_point_ms(lambda: xb.occlusion_batched(model, X, bg), n, reps),
        "max_diff_batched_vs_loop": float(np.abs(batched - loop).max()),
        "max_diff_batched_vs_flip": float(np.abs(batched - flip).max()),
    }
    print(f"{name} seed {seed}: PACE-flip {row['flip_ms']:.3f} ms, occlusion point by point "
          f"{loop_ms:.2f} ms, batched {row['occlusion_batched_ms']:.2f} ms, "
          f"largest difference {max(row['max_diff_batched_vs_loop'], row['max_diff_batched_vs_flip']):.1e}",
          flush=True)
    return row


def summarise(df, seeds, reps, n_explain):
    g = df.groupby("dataset", sort=False).agg(
        flip=("flip_ms", "mean"), loop=("occlusion_loop_ms", "mean"),
        batched=("occlusion_batched_ms", "mean"),
        diff=("max_diff_batched_vs_loop", "max"), diff_flip=("max_diff_batched_vs_flip", "max"))
    g["loop_x"] = g.loop / g.flip
    g["batched_x"] = g.batched / g.flip
    g["gain"] = g.loop / g.batched
    lines = [
        "# PACE-flip against batched occlusion", "",
        f"Same models, test points and background as the paper's timing (`run_faithfulness.py`; "
        f"{n_explain} test points per seed, 36 for Wine; seeds {seeds}; 32 values per feature). "
        "Milliseconds per explained point, mean over seeds. PACE-flip and batched occlusion: median "
        f"of {reps} runs; point-by-point occlusion: one run, as in the paper's timing table. Times "
        "depend on the machine; compare within a row. A descriptive check added after the "
        "pre-submission review, not planned in advance.", "",
        "| Dataset | PACE-flip | Occlusion, point by point | Occlusion, batched | "
        "Point by point vs PACE-flip | Batched vs PACE-flip | Batching gain | Largest difference |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds, r in g.iterrows():
        lines.append(f"| {NAMES.get(ds, ds)} | {fmt(r.flip)} | {fmt(r.loop)} | {fmt(r.batched)} | "
                     f"{r.loop_x:.0f}× | {r.batched_x:.0f}× | {r.gain:.1f}× | "
                     f"{max(r['diff'], r.diff_flip):.1e} |")
    lines += [
        "",
        f"- **PACE-flip vs batched occlusion:** {g.batched_x.min():.0f}–{g.batched_x.max():.0f}× faster.",
        f"- **PACE-flip vs point-by-point occlusion (this run):** "
        f"{g.loop_x.min():.0f}–{g.loop_x.max():.0f}× faster.",
        f"- **Batching** makes occlusion {g.gain.min():.1f}–{g.gain.max():.1f}× faster.",
        f"- **Same scores:** batched occlusion differs from point-by-point occlusion and from "
        f"PACE-flip by at most {max(g['diff'].max(), g.diff_flip.max()):.1e} (floating-point "
        "rounding).",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "review_v1" / "occlusion_batched"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in a.datasets:
        for s in a.seeds:
            rows.append(run(ds, s, a.n, a.reps))
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)   # saved as it goes
    text = summarise(pd.DataFrame(rows), a.seeds, a.reps, a.n)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
