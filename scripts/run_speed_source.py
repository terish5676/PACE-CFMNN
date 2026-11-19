"""Run P (descriptive): where PACE-SHAP's speed over a general exact explainer comes from.

Rules: results/speed_source/PREREGISTRATION.md (planned in advance).

PACE-SHAP's exact mode returns the same Shapley values of the margin as shap's ExactExplainer,
16-30 times faster (results/review_v1/exact/). This run splits that speed-up in two, by timing
three ways of computing the same values on the same model and rows:

* shap ExactExplainer (``xai_baselines.exact_shap``): calls the model on every coalition and
  every background row, with no use of CFMNN's structure;
* PACE-SHAP without pruning (``PaceShap(prune=False)``): every box's membership under every
  coalition from per-feature terms computed once (Proposition 2), for every box;
* PACE-SHAP (``PaceShap(prune=True)``, the default): the same, after dropping the (row, box)
  pairs that can never be their class's strongest box (Proposition 3).

ExactExplainer / no pruning is the share of the speed-up from the per-feature structure;
no pruning / PACE-SHAP is the share from pruning. Setup: the second explanation run's models
(``run_faithfulness`` v3: the paper's split, the seed's row order, 200 explainer rows), the 6
explanation datasets with at most 12 features, seeds 240-242, 2 test points per seed.

It measures time, so run it alone, with nothing else heavy running:
    python -B scripts/run_speed_source.py
    python -B scripts/run_speed_source.py --resume        # continue after a stop
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
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

DATASETS = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
SEEDS = [240, 241, 242]
N_POINTS = 2
EXACT_ARM, NOPRUNE_ARM, PACE_ARM = "shap ExactExplainer", "PACE-SHAP, no pruning", "PACE-SHAP"
ARMS = [EXACT_ARM, NOPRUNE_ARM, PACE_ARM]
TOL_NOPRUNE, TOL_EXACT = 1e-12, 1e-9
OUT = ROOT / "results" / "speed_source"


def timed(fn):
    t0 = time.perf_counter()
    v = fn()
    return v, time.perf_counter() - t0


def run_one(name, seed, n):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    _, bex = rf.backgrounds_v3(Xtr, seed)
    B = bex[:rf.SHAP_V3_R]
    d = Xte.shape[1]
    if d > 12:
        raise ValueError(f"{name} has {d} features; this run is for exact mode (d <= 12)")
    PaceShap(model, B).explain(Xte[:1])                      # warm-up, not timed
    kept = {}

    def pace(prune):
        ps = PaceShap(model, B, prune=prune)                 # build time included, as in review_v1
        kept[prune] = ps
        return ps.explain(Xte)

    ref, t_pace = timed(lambda: pace(True))
    nopr, t_nopr = timed(lambda: pace(False))
    exact, t_exact = timed(lambda: xb.exact_shap(model, Xte, B))
    skipped = kept[True].stats_["share_of_boxes_skipped"]
    rows = []
    for arm, phi, secs in ((PACE_ARM, ref, t_pace), (NOPRUNE_ARM, nopr, t_nopr),
                           (EXACT_ARM, exact, t_exact)):
        rows.append({"dataset": name, "seed": seed, "arm": arm, "d": d,
                     "boxes": int(model.V_.shape[0]), "background_rows": len(B), "n": len(Xte),
                     "ms_per_explanation": 1000 * secs / len(Xte),
                     "max_abs_diff": float(np.abs(phi - ref).max()),
                     "max_abs_value": float(np.abs(ref).max()),
                     "share_pairs_skipped": skipped if arm == PACE_ARM else np.nan})
    print(f"{name} seed {seed}: ms " + ", ".join(f"{r['arm']} {r['ms_per_explanation']:.0f}"
                                                 for r in rows)
          + f"; pairs skipped {skipped:.3f}; largest difference "
          + f"{max(r['max_abs_diff'] for r in rows):.1e}", flush=True)
    return rows


def check(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """PASS if no pruning matches PACE-SHAP within 1e-12 and ExactExplainer within 1e-9."""
    bad = []
    for arm, tol in ((NOPRUNE_ARM, TOL_NOPRUNE), (EXACT_ARM, TOL_EXACT)):
        g = df[df.arm == arm]
        bad += [f"{r.dataset} seed {r.seed}: {arm} differs by {r.max_abs_diff:.1e} (limit {tol:g})"
                for _, r in g[g.max_abs_diff > tol].iterrows()]
    return not bad, bad


def summarise(df, out: Path) -> str:
    ok, bad = check(df)
    seeds = sorted(int(s) for s in df.seed.unique())
    n = int(df.n.max())
    L = ["# Run P (descriptive): where PACE-SHAP's speed comes from", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and 200 explainer rows, "
         f"seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))}, {n} point"
         f"{'s' if n != 1 else ''} per seed, one task at a time. "
         "Milliseconds per explanation (mean over seeds), each including its explainer's set-up. "
         "Written by `scripts/run_speed_source.py`.", "",
         f"**Check: {'PASS' if ok else 'FAIL'}.** PACE-SHAP without pruning agrees with PACE-SHAP "
         f"within {TOL_NOPRUNE:g}, and shap's ExactExplainer within {TOL_EXACT:g}, on every point."]
    L += [f"- {b}" for b in bad]
    L += ["", "| Dataset | d | Boxes | ms ExactExplainer | ms no pruning | ms PACE-SHAP | "
          "Structure (ExactExplainer ÷ no pruning) | Pruning (no pruning ÷ PACE-SHAP) | "
          "Total (ExactExplainer ÷ PACE-SHAP) | (Row, box) pairs skipped | Largest difference: "
          "no pruning | Largest difference: ExactExplainer |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in [d for d in DATASETS if d in set(df.dataset)] + sorted(set(df.dataset) - set(DATASETS)):
        g = df[df.dataset == ds]
        ms = g.groupby("arm").ms_per_explanation.mean()
        dif = g.groupby("arm").max_abs_diff.max()
        L.append(f"| {ds} | {g.d.iloc[0]} | {g.boxes.mean():.0f} | {ms[EXACT_ARM]:.1f} | "
                 f"{ms[NOPRUNE_ARM]:.1f} | {ms[PACE_ARM]:.1f} | {ms[EXACT_ARM] / ms[NOPRUNE_ARM]:.1f}× | "
                 f"{ms[NOPRUNE_ARM] / ms[PACE_ARM]:.1f}× | {ms[EXACT_ARM] / ms[PACE_ARM]:.1f}× | "
                 f"{g.share_pairs_skipped.mean():.3f} | {dif[NOPRUNE_ARM]:.1e} | {dif[EXACT_ARM]:.1e} |")
    L += ["", "Boxes: mean over seeds. (Row, box) pairs skipped: PACE-SHAP's own count "
          "(`stats_['share_of_boxes_skipped']`), the share of explainer-row and box pairs that "
          "pruning removed. Largest difference: the largest absolute difference from PACE-SHAP over "
          "every point, feature and seed."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P: where PACE-SHAP's speed comes from")
    ap.add_argument("--datasets", nargs="+", help="only these datasets")
    ap.add_argument("--seeds", nargs="+", type=int, help="these seeds instead (smoke runs only)")
    ap.add_argument("--n", type=int, help="points per dataset and seed (smoke runs only)")
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(d, s, a.n or N_POINTS) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=1)   # alone: it is timed
    text = summarise(pd.read_csv(csv, float_precision="round_trip"), out)
    print(text)
    ok, _ = check(pd.read_csv(csv, float_precision="round_trip"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
