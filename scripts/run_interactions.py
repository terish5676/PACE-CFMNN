"""RQ1 extension B4: which features matter together? PACE-SHAP interaction values.

Pre-registered in results/extensions_v1/PREREGISTRATION.md (seeds 40-49).
Same models, test points and explainer rows as the cond protocol of
``run_faithfulness.py`` (50 test points; the first 200 explainer rows). For
every point, ``PaceShap.interactions`` gives exact SHAP interaction values
(d <= 12 only). The global strength of a pair is the mean over points of
|interaction|.

Reported per dataset and seed: every pair's strength; the top pair; on the
synthetic sets, the share of all pair strength among the truly relevant
features 0-2. Primary check (S3, class 1 iff (x0 > 0.5 and x1 < 0.4) or
x2 > 0.8): is (x0, x1), the AND pair, the top pair?

Usage (from the project folder):
    python scripts/run_interactions.py --jobs 6
    python scripts/run_interactions.py --datasets S3 --seeds 0
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn.explain import PaceShap  # noqa: E402
from run_faithfulness import SHAP_V3_R, backgrounds_v3, setup  # noqa: E402

DATASETS = ["S1", "S2", "S3", "S1c", "S2c", "shuttle", "magic", "page_blocks"]   # d <= 12
SYNTHETIC = {"S1", "S2", "S3", "S1c", "S2c"}


def run_one(name, seed, n_explain=50):
    ds, Xtr, Xte, model, _ = setup(name, seed, n_explain)
    _, bex = backgrounds_v3(Xtr, seed)
    t0 = time.perf_counter()
    inter = PaceShap(model, bex[:SHAP_V3_R]).interactions(Xte)
    secs = time.perf_counter() - t0
    strength = np.abs(inter).mean(axis=0)                       # (d, d)
    d = strength.shape[0]
    return [{"dataset": name, "seed": seed, "n": len(Xte), "i": i, "j": j,
             "strength": float(strength[i, j]), "seconds": secs}
            for i in range(d) for j in range(i + 1, d)]


def summarise(df: pd.DataFrame, out: Path) -> str:
    from run_error_detection import ci95
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# RQ1 extension B4: PACE-SHAP interaction values", "",
         f"Exact SHAP interaction values of the predicted class's margin (PaceShap.interactions), "
         f"{SHAP_V3_R} explainer rows, {int(df.n.max())} test points per dataset and seed (fewer if the "
         f"test split is smaller); seeds {seeds}. Pair "
         "strength = mean over points of |interaction value|. Mean ± 95% CI over seeds.", "",
         "| Dataset | Most frequent top pair | Seeds where it is top | Its strength | "
         "Share of pair strength among x0-x2 |", "|---|---|---|---|---|"]
    for ds, g in df.groupby("dataset", sort=False):
        top = g.loc[g.groupby("seed").strength.idxmax()]
        pair = top.groupby(["i", "j"]).size().idxmax()
        n_top = int(((top.i == pair[0]) & (top.j == pair[1])).sum())
        s = g[(g.i == pair[0]) & (g.j == pair[1])].strength
        share = "–"
        if ds in SYNTHETIC:
            rel = g[(g.i < 3) & (g.j < 3)].groupby("seed").strength.sum() / g.groupby("seed").strength.sum()
            share = f"{rel.mean():.2f} ± {ci95(rel):.2f}"
        L.append(f"| {ds} | (x{pair[0]}, x{pair[1]}) | {n_top} of {top.seed.nunique()} | "
                 f"{s.mean():.4f} ± {ci95(s):.4f} | {share} |")
    s3 = df[df.dataset == "S3"]
    if len(s3):
        top = s3.loc[s3.groupby("seed").strength.idxmax()]
        hit = int(((top.i == 0) & (top.j == 1)).sum())
        L += ["", "## S3: is the AND pair (x0, x1) the top pair? (primary)", "",
              f"(x0, x1) is the top pair in {hit} of {len(top)} seeds (pre-registered: supported if at "
              "least 8 of 10).", "", "| Pair | Strength |", "|---|---|"]
        for (i, j), g in s3.groupby(["i", "j"]):
            if i < 3 and j < 3:
                L.append(f"| (x{i}, x{j}) | {g.strength.mean():.4f} ± {ci95(g.strength):.4f} |")
        other = s3[~((s3.i < 3) & (s3.j < 3))].groupby("seed").strength.max()
        L.append(f"| strongest pair with an irrelevant feature | {other.mean():.4f} ± {ci95(other):.4f} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(40, 50)))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or ROOT / "results" / "extensions_v1" / "interactions")
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        tasks = [(n, s) for n in a.datasets for s in a.seeds]
        done = {}
        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                fut = {ex.submit(run_one, n, s, a.n): (n, s) for n, s in tasks}
                for f in as_completed(fut):
                    done[fut[f]] = f.result()
        else:
            for t in tasks:
                done[t] = run_one(*t, a.n)
        pd.DataFrame([r for t in tasks for r in done[t]]).to_csv(out / "runs.csv", index=False)
    print(summarise(pd.read_csv(out / "runs.csv"), out))


if __name__ == "__main__":
    main()
