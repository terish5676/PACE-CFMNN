"""Run P7 (tough review S4 / B8): exact single-feature flip sets against dense occlusion.

Rules: results/flip_intervals/PREREGISTRATION.md (planned in advance).

PACE-flip tries 32 replacement values per feature (quantiles of the explainer's background rows)
and reports the share that change the class. ``flip_exact.flip_sets`` gives the exact set of
values of one feature that change the class, as a union of intervals, from the breakpoints of the
piecewise-linear memberships (Proposition B8). This run checks the exact sets against the model
itself on a dense grid, and measures what the 32-value grid misses.

For each (point, feature):

* the exact flip set on the domain [min(0, x_i), max(1, x_i)] (``flip_sets``);
* the model's class at 10,001 evenly spaced values of the feature on the same domain (dense
  occlusion, ``model.predict_details``), for the first ``N_DENSE`` points of each seed;
* the model's class at PACE-flip's 32 values (``PaceFlip(model, background).values_``).

Check (pass/fail): every dense-grid value that changes the class lies in an exact interval (within
1e-9), and every exact interval wider than the grid step contains a dense-grid value that changes
the class. Secondary: the share of flipping (point, feature) pairs where none of the 32 values
flips; the exact smallest single-feature change against the 32-value grid's.

Setup: the second explanation run's models (``run_faithfulness.setup``, ``backgrounds_v3``, 200
explainer rows), the 11 explanation datasets, seeds 340-349, 50 test points per seed.

    python -B scripts/run_flip_intervals.py --jobs 4
    python -B scripts/run_flip_intervals.py --jobs 4 --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn.explain import PaceFlip  # noqa: E402
from pace_cfmnn.flip_exact import flip_sets  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

DATASETS = list(rf.DATASETS)
SEEDS = list(range(340, 350))
N_POINTS = 50
N_DENSE = 10
GRID = 10_001
TOL = 1e-9
OUT = ROOT / "results" / "flip_intervals"


def _in_set(fs, u):
    """The exact class lookup: True where value u of the feature changes the class."""
    k = np.searchsorted(fs.breaks, u, side="right") - 1
    on = np.isclose(u[:, None], fs.breaks[None, :], rtol=0, atol=1e-12)
    k = np.clip(k, 0, len(fs.open_cls) - 1)
    cls = fs.open_cls[k].copy()
    hit = on.any(axis=1)
    cls[hit] = fs.point_cls[on[hit].argmax(axis=1)]
    return cls != fs.cls_x


def _inside(iv, u, tol=TOL):
    if not iv:
        return np.zeros(len(u), bool)
    lo = np.array([a for a, _ in iv]) - tol
    hi = np.array([b for _, b in iv]) + tol
    return ((u[:, None] >= lo[None, :]) & (u[:, None] <= hi[None, :])).any(axis=1)


def check_pair(model, x, fs, grid=GRID):
    """Dense-occlusion check of one feature's exact flip set.

    Returns (grid flips outside every interval, intervals wider than the step with no grid flip,
    grid flips, intervals)."""
    i = fs.feature
    lo, hi = min(0.0, float(x[i])), max(1.0, float(x[i]))
    u = np.linspace(lo, hi, grid)
    P = np.repeat(x[None, :], grid, axis=0)
    P[:, i] = u
    flips = model.predict_details(P).cls != fs.cls_x
    iv = fs.flip_intervals()
    outside = int((flips & ~_inside(iv, u)).sum())
    step = (hi - lo) / (grid - 1)
    missed = 0
    for a, b in iv:
        if b - a > step and not flips[(u > a) & (u < b)].any():
            missed += 1
    return outside, missed, int(flips.sum()), len(iv)


def run_one(name, seed, n):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    _, bex = rf.backgrounds_v3(Xtr, seed)
    Q = PaceFlip(model, background=bex).values_                 # (32, d)
    cls0 = model.predict_details(Xte).cls
    rows = []
    for p, x in enumerate(Xte):
        sets = flip_sets(model, x)
        best_exact, best_grid = np.inf, np.inf
        for fs in sets:
            i = fs.feature
            iv = fs.flip_intervals()
            q = Q[:, i]
            P = np.repeat(x[None, :], len(q), axis=0)
            P[:, i] = q
            qflip = model.predict_details(P).cls != cls0[p]
            # a value within 1e-9 of a breakpoint sits on a tie up to rounding: compared apart
            gap = np.abs(q[:, None] - fs.breaks[None, :]).min(axis=1)
            at_break = (gap < TOL) & (gap > 0)          # a value exactly on a breakpoint is compared
            dist_exact, _ = fs.smallest_change(float(x[i]))
            dist_grid = float(np.abs(q[qflip] - x[i]).min()) if qflip.any() else np.inf
            best_exact, best_grid = min(best_exact, dist_exact), min(best_grid, dist_grid)
            row = {"dataset": name, "seed": seed, "point": p, "feature": i, "d": len(x),
                   "boxes": int(model.n_boxes_), "n_intervals": len(iv),
                   "exact_measure": float(sum(b - a for a, b in iv)),
                   "exact_flips": bool(iv), "grid32_flips": bool(qflip.any()),
                   "grid32_share": float(qflip.mean()),
                   "grid32_agrees": bool((qflip == _in_set(fs, q))[~at_break].all()),
                   "grid32_at_break": int(at_break.sum()),
                   "dist_exact": dist_exact, "dist_grid32": dist_grid,
                   "point_dist_exact": np.nan, "point_dist_grid32": np.nan,
                   "dense": p < N_DENSE}
            if p < N_DENSE:
                o, mi, nf, _ = check_pair(model, x, fs)
                row.update(dense_outside=o, dense_missed=mi, dense_flips=nf)
            else:
                row.update(dense_outside=np.nan, dense_missed=np.nan, dense_flips=np.nan)
            rows.append(row)
        rows[-1]["point_dist_exact"] = best_exact
        rows[-1]["point_dist_grid32"] = best_grid
    print(f"{name} seed {seed}: {sum(r['exact_flips'] for r in rows)} flipping pairs of {len(rows)}; "
          f"32-value grid misses {sum(r['exact_flips'] and not r['grid32_flips'] for r in rows)}", flush=True)
    return rows


def check(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """PASS if no dense-grid flip lies outside the exact set, no wide interval lacks a grid flip,
    and the class at PACE-flip's 32 values agrees with the exact set everywhere."""
    bad = []
    g = df[df.dense]
    for col, what in (("dense_outside", "dense-grid flips outside the exact set"),
                      ("dense_missed", "exact intervals with no dense-grid flip")):
        s = g.groupby("dataset")[col].sum()
        bad += [f"{ds}: {int(v)} {what}" for ds, v in s.items() if v > 0]
    s = (~df.grid32_agrees).groupby(df.dataset).sum()
    bad += [f"{ds}: {int(v)} pairs where the 32 values disagree with the exact set" for ds, v in s.items() if v > 0]
    return not bad, bad


def summarise(df, out: Path) -> str:
    ok, bad = check(df)
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# Run P7: exact flip sets against dense occlusion", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and explainer rows, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)} seeds). Written by `scripts/run_flip_intervals.py`.", "",
         f"**Check: {'PASS' if ok else 'FAIL'}.** On {int(df.dense.sum()):,} (point, feature) pairs, every "
         f"value of a {GRID:,}-value grid that changes the class lies in the exact flip set (within {TOL:g}), "
         "every exact interval wider than the grid step contains a grid value that changes the class, and "
         f"on all {len(df):,} pairs the class at PACE-flip's 32 values agrees with the exact set "
         f"({int(df.grid32_at_break.sum()):,} values within {TOL:g} of a breakpoint, a tie up to rounding, "
         "left out of that comparison)."]
    L += [f"- {b}" for b in bad]
    L += ["", "| Dataset | d | Boxes | Pairs | Pairs with a flip | Missed by the 32 values | "
          "Missed share | Intervals per flipping pair | Median smallest change, exact | "
          "Median smallest change, 32 values | Points the 32 values call unflippable |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot_f = tot_m = 0
    for ds in [d for d in DATASETS if d in set(df.dataset)]:
        g = df[df.dataset == ds]
        fl = g[g.exact_flips]
        miss = int((fl.grid32_flips == False).sum())  # noqa: E712
        tot_f, tot_m = tot_f + len(fl), tot_m + miss
        pt = g.dropna(subset=["point_dist_exact"])
        fin = pt[np.isfinite(pt.point_dist_exact)]
        none32 = int((np.isfinite(fin.point_dist_exact) & ~np.isfinite(fin.point_dist_grid32)).sum())
        med_e = f"{fin.point_dist_exact.median():.3f}" if len(fin) else "–"
        g32 = fin.point_dist_grid32[np.isfinite(fin.point_dist_grid32)]
        med_g = f"{g32.median():.3f}" if len(g32) else "–"
        L.append(f"| {ds} | {g.d.iloc[0]} | {g.boxes.mean():.0f} | {len(g):,} | {len(fl):,} | {miss:,} | "
                 f"{miss / max(1, len(fl)):.1%} | {fl.n_intervals.mean() if len(fl) else 0:.2f} | {med_e} | "
                 f"{med_g} | {none32} of {len(fin)} |")
    L += ["", f"Over all datasets the 32 values miss {tot_m:,} of {tot_f:,} flipping pairs "
          f"({tot_m / max(1, tot_f):.1%}). Smallest change: the distance from x_i to the nearest value of one "
          "feature that changes the class, minimised over features (points with a flip only)."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P7: exact flip sets")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, help="smoke runs only")
    ap.add_argument("--n", type=int, help="points per seed (smoke runs only)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(d, s, a.n or N_POINTS) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=a.jobs)
    df = pd.read_csv(csv, float_precision="round_trip")
    print(summarise(df, out))
    sys.exit(0 if check(df)[0] else 1)


if __name__ == "__main__":
    main()
