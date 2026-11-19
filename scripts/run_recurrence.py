"""Run P8 (tough review B9): a faster exact mode, and a fair brute-force baseline.

Rules: results/recurrence/PREREGISTRATION.md (planned in advance).

PACE-SHAP's exact mode builds every coalition's memberships with one matrix product
(``exact_algo="matmul"``, K d 2^d multiply-adds for K kept (row, box) pairs). Building the subset
sums one feature at a time (``exact_algo="recurrence"``) costs K 2^d additions. This run checks
that the recurrence gives the same values and times four ways of computing them:

* PACE-SHAP, matmul (the current exact mode);
* PACE-SHAP, recurrence;
* brute force, vectorised (``explain.brute_force_shap``): the model on every mixed input in large
  numpy blocks, with no Proposition 2 and no pruning;
* shap's ExactExplainer (Wine and Letter only, where it is the reference for the check).

Setup: the second explanation run's models (``run_faithfulness.setup``, ``backgrounds_v3``, 200
explainer rows), seeds 350-352, 10 test points per seed. On Wine (13 features) and Letter (16)
both PACE-SHAP modes run with ``exact_max_d=16``. To keep the run near 3 hours, brute force and
ExactExplainer explain one Letter point in total (the first point of seed 350): each takes tens
of minutes there.

It measures time, so run it alone:
    python -B scripts/run_recurrence.py
    python -B scripts/run_recurrence.py --resume
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

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap, brute_force_shap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

SMALL = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
LARGE = ["wine", "letter"]
DATASETS = SMALL + LARGE
SEEDS = [350, 351, 352]
N_POINTS = 10
FEW = {"letter": 1}      # Letter: brute force and ExactExplainer on 1 point of the first seed only
MATMUL, RECUR, BRUTE, EXACT = ("PACE-SHAP, matmul", "PACE-SHAP, recurrence", "Brute force, vectorised",
                               "shap ExactExplainer")
ARMS = [MATMUL, RECUR, BRUTE, EXACT]
TOL_SAME, TOL_EXACT = 1e-12, 1e-9
OUT = ROOT / "results" / "recurrence"


def timed(fn):
    t0 = time.perf_counter()
    v = fn()
    return v, time.perf_counter() - t0


def run_one(name, seed, n):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    _, bex = rf.backgrounds_v3(Xtr, seed)
    B = bex[:rf.SHAP_V3_R]
    d = Xte.shape[1]
    if d > 16:
        raise ValueError(f"{name} has {d} features; this run stops at 16")
    emax = max(12, d)
    PaceShap(model, B, exact_max_d=emax).explain(Xte[:1])          # warm-up, not timed
    few = Xte[:FEW.get(name, n)] if (name not in FEW or seed == SEEDS[0]) else Xte[:0]
    res = {}
    res[MATMUL] = timed(lambda: PaceShap(model, B, exact_max_d=emax).explain(Xte)) + (len(Xte),)
    res[RECUR] = timed(lambda: PaceShap(model, B, exact_max_d=emax,
                                        exact_algo="recurrence").explain(Xte)) + (len(Xte),)
    if len(few):
        res[BRUTE] = timed(lambda: brute_force_shap(model, few, B)) + (len(few),)
        if name in LARGE:
            res[EXACT] = timed(lambda: xb.exact_shap(model, few, B)) + (len(few),)
    ref = res[MATMUL][0]
    rows = []
    for arm, (phi, secs, k) in res.items():
        rows.append({"dataset": name, "seed": seed, "arm": arm, "d": d, "boxes": int(model.n_boxes_),
                     "background_rows": len(B), "n": k, "ms_per_explanation": 1000 * secs / k,
                     "max_abs_diff_matmul": float(np.abs(phi - ref[:k]).max()),
                     # only the points both arms explained (ExactExplainer may explain fewer)
                     "max_abs_diff_exact": (float(np.abs(phi[:min(k, res[EXACT][2])]
                                                         - res[EXACT][0][:min(k, res[EXACT][2])]).max())
                                            if EXACT in res else np.nan),
                     "max_abs_value": float(np.abs(ref).max())})
    print(f"{name} seed {seed}: " + ", ".join(f"{r['arm']} {r['ms_per_explanation']:.0f} ms" for r in rows)
          + f"; largest difference from matmul {max(r['max_abs_diff_matmul'] for r in rows):.1e}", flush=True)
    return rows


def check(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """PASS if the recurrence and brute force match matmul within 1e-12 on every point, and on Wine
    and Letter both PACE-SHAP modes match ExactExplainer within 1e-9."""
    bad = []
    for arm in (RECUR, BRUTE):
        g = df[(df.arm == arm) & (df.max_abs_diff_matmul > TOL_SAME)]
        bad += [f"{r.dataset} seed {r.seed}: {arm} differs from matmul by {r.max_abs_diff_matmul:.1e}"
                for _, r in g.iterrows()]
    g = df[df.arm.isin([MATMUL, RECUR]) & (df.max_abs_diff_exact > TOL_EXACT)]
    bad += [f"{r.dataset} seed {r.seed}: {r.arm} differs from ExactExplainer by {r.max_abs_diff_exact:.1e}"
            for _, r in g.iterrows()]
    return not bad, bad


def summarise(df, out: Path) -> str:
    ok, bad = check(df)
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# Run P8: a faster exact mode, and a fair brute-force baseline", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and 200 explainer rows, seeds "
         f"{', '.join(map(str, seeds))}, one task at a time. Milliseconds per explanation (mean over seeds), "
         "each including its explainer's set-up. Written by `scripts/run_recurrence.py`.", "",
         f"**Check: {'PASS' if ok else 'FAIL'}.** The recurrence and the vectorised brute force match the "
         f"current exact mode within {TOL_SAME:g} on every point; on Wine and Letter both PACE-SHAP modes "
         f"match shap's ExactExplainer within {TOL_EXACT:g}."]
    L += [f"- {b}" for b in bad]
    L += ["", "| Dataset | d | Boxes | ms matmul | ms recurrence | ms brute force | ms ExactExplainer | "
          "matmul ÷ recurrence | brute force ÷ recurrence | Largest difference from matmul |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in [d for d in DATASETS if d in set(df.dataset)]:
        g = df[df.dataset == ds]
        ms = g.groupby("arm").ms_per_explanation.mean()
        f = lambda a: f"{ms[a]:.1f}" if a in ms else "–"   # noqa: E731
        br = f"{ms[BRUTE] / ms[RECUR]:.1f}×" if BRUTE in ms else "–"
        L.append(f"| {ds} | {g.d.iloc[0]} | {g.boxes.mean():.0f} | {f(MATMUL)} | {f(RECUR)} | {f(BRUTE)} | "
                 f"{f(EXACT)} | {ms[MATMUL] / ms[RECUR]:.2f}× | {br} | "
                 f"{g.max_abs_diff_matmul.max():.1e} |")
    let = df[(df.dataset == "letter") & (df.arm == RECUR)]
    if len(let):
        s = let.ms_per_explanation.mean() / 1000
        L += ["", f"Letter, recurrence: {s:.1f} s per explanation. The plan raises the exact limit to 16 "
              f"features if this is at most 5 s: **{'yes' if s <= 5 else 'no'}**."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P8: a faster exact mode")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, help="smoke runs only")
    ap.add_argument("--n", type=int, help="points per seed (smoke runs only)")
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
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=1)
    df = pd.read_csv(csv, float_precision="round_trip")
    print(summarise(df, out))
    sys.exit(0 if check(df)[0] else 1)


if __name__ == "__main__":
    main()
