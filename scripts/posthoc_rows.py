"""Post hoc: B3's conditional game at sigma = 0.1 with more explainer rows (not pre-registered).

The sigma check (``posthoc_sigma.py``) gave the conditional game more rows
by widening its weights, which also changes the game itself. This check
keeps B3's sigma = 0.1 and weight function (the per-feature scale of B3's
200 explainer rows) and gives the game more rows instead. The explainer
rows are B3's 200 rows followed by further training rows, never the
evaluation rows. For each number of rows it computes PACE-SHAP-cond and
PACE-SHAP on the same rows, and reports:

* the effective number of rows of the conditional weights (Kish, median
  over points and over coalitions with half the features or more);
* how often each game's inputs are off the data (B3's realism measure);
* faithfulness on B3's conditional and interventional deletion and
  insertion tests;
* ground truth (top 3, on S1) and the distance between the two games'
  attributions.

With 200 rows it repeats B3's computations; the summary checks that.

Usage (from the project folder):
    python scripts/posthoc_rows.py --jobs 6
    python scripts/posthoc_rows.py --summary-only
Writes results/extensions_v1/conditional/posthoc_rows.csv and .md.
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

from pace_cfmnn import realism  # noqa: E402
from pace_cfmnn.explain import PaceShap, cond_scale  # noqa: E402
from posthoc_sigma import B3, NAMES, TESTS, effective_rows, game_off_share  # noqa: E402
from run_faithfulness import (COND_SIGMA, SHAP_V3_P, SHAP_V3_R, SYNTHETIC, backgrounds_v3,  # noqa: E402
                              cond_curves, deletion_auc, insertion_auc, setup, top3_random_ties)

DATASETS = ["S1", "shuttle", "magic", "page_blocks"]   # exact (d <= 12); S1's features are independent
SEEDS = [40, 41, 42, 43, 44]
ROWS = (SHAP_V3_R, 800)
N_EXPLAIN = 50                                          # B3's test points per dataset and seed


def explainer_rows(Xtr, seed, rows):
    """B3's explainer rows followed by further training rows (never B3's evaluation rows).

    ``backgrounds_v3`` takes the evaluation rows and then the explainer rows
    from one permutation of the training set; this continues that order."""
    perm = np.random.default_rng(seed + 500).permutation(len(Xtr))
    k = min(SHAP_V3_R, len(Xtr) // 2)
    if k + rows > len(Xtr):
        raise ValueError(f"only {len(Xtr) - k} training rows are free for the explainer")
    return Xtr[perm[k:k + rows]]


def run_job(name, seed, n_explain=N_EXPLAIN, rows=ROWS, sigma=COND_SIGMA):
    t0 = time.perf_counter()
    ds, Xtr, Xte, model, _ = setup(name, seed, n_explain)
    bg, bex = backgrounds_v3(Xtr, seed)
    B3_rows = bex[:SHAP_V3_R]
    scale = cond_scale(B3_rows)                 # B3's weight function, kept for every number of rows
    relevant = set(ds.info.get("relevant", [])) if name in SYNTHETIC else None
    man = realism.Manifold(Xtr, seed=seed)
    out = []
    for R in rows:
        B = explainer_rows(Xtr, seed, R)
        m = min(R, len(B3_rows))
        if not np.array_equal(B[:m], B3_rows[:m]):
            raise AssertionError("the explainer rows do not start with B3's")
        ps = PaceShap(model, B, n_permutations=SHAP_V3_P, seed=seed, scale=scale)
        if not ps.exact:
            raise ValueError(f"{name}: PACE-SHAP is not exact here (d > 12)")
        t1 = time.perf_counter()
        cond, inter = ps.explain_games(Xte, (sigma, None))
        secs = time.perf_counter() - t1
        ref = np.abs(inter).sum(axis=1)
        k_large, k_small = effective_rows(Xte, B, scale, sigma)
        for method, s, attr in (("PACE-SHAP-cond", sigma, cond), ("PACE-SHAP", None, inter)):
            _, del_int = deletion_auc(model, Xte, attr, bg, seed=seed, signed=True)
            _, ins_int = insertion_auc(model, Xte, attr, bg, seed=seed)
            _, del_cond = cond_curves(model, Xte, attr, bg, seed=seed)
            _, ins_cond = cond_curves(model, Xte, attr, bg, seed=seed, insertion=True)
            t3 = top3_random_ties(attr, np.random.default_rng(seed + 7))
            dist = np.divide(np.abs(attr - inter).sum(axis=1), ref, out=np.zeros(len(attr)),
                             where=ref > 0)
            out.append({"dataset": name, "seed": seed, "rows": R, "method": method,
                        "eff_rows_large": k_large if s is not None else float(R),
                        "eff_rows_small": k_small if s is not None else float(R),
                        "off_share": game_off_share(man, Xte, B, scale, seed, s),
                        "margin_auc": del_int.mean(), "insertion_margin_auc": ins_int.mean(),
                        "cond_margin_auc": del_cond.mean(),
                        "cond_insertion_margin_auc": ins_cond.mean(),
                        "gt_top3_precision": (np.mean([len(set(t) & relevant) / 3 for t in t3])
                                              if relevant else np.nan),
                        "dist_to_pace_shap": float(dist.mean()),
                        "explain_seconds_both_games": secs, "n": len(Xte)})
    print(f"{name} seed {seed} done in {time.perf_counter() - t0:.0f} s; conditional deletion "
          + ", ".join(f"{r['method']} {r['rows']} rows: {r['cond_margin_auc']:.3f}" for r in out),
          flush=True)
    return out


# ------------------------------------------------------------------- summary
def summarise(df: pd.DataFrame, out: Path) -> str:
    b3 = pd.read_csv(B3 / "runs.csv")
    real = pd.read_csv(B3 / "realism.csv")
    rows = sorted(int(r) for r in df["rows"].unique())
    dsets = [d for d in DATASETS if d in set(df.dataset)] + sorted(set(df.dataset) - set(DATASETS))
    seeds = sorted(int(s) for s in df.seed.unique())
    sec = iter(range(1, 20))
    get = lambda d, m, R, col: df[(df.dataset == d) & (df.method == m) & (df["rows"] == R)].set_index("seed")[col]
    L = ["# Post hoc: B3's conditional game at σ = 0.1 with more explainer rows", "",
         "**Post hoc, not pre-registered.** B3 fixed σ = 0.1 and 200 explainer rows in advance, and its "
         "results stand as run. The σ check (`posthoc_sigma.md`) gave the conditional game more rows by "
         "widening its weights, which also changes the game. This check keeps σ = 0.1 and B3's weight "
         "function (the per-feature scale of B3's 200 rows) and adds rows instead: B3's 200 explainer "
         "rows followed by further training rows, never the evaluation rows.", "",
         f"Datasets {', '.join(NAMES.get(d, d) for d in dsets)}; seeds {seeds}; B3's models, "
         f"{int(df.n.max())} test points, evaluation rows and tests. For each number of rows "
         f"({', '.join(map(str, rows))}), PACE-SHAP-cond and PACE-SHAP are computed on the same rows "
         "from one shared table of margins.", "",
         f"Descriptive only: with {len(seeds)} seeds a two-sided Wilcoxon test cannot reach p < 0.05, "
         "so no tests are run. Cells give the mean over seeds.", ""]

    # ---- reproduction of B3 at 200 rows
    cols = [m for m, _, _ in TESTS] + ["gt_top3_precision"]
    L += ["## Check: 200 rows repeat B3", "",
          "Largest absolute difference from B3's own results, over all datasets and seeds here.", "",
          "| Method | Faithfulness and ground truth | Inputs off the data |", "|---|---|---|"]
    for method, scheme in (("PACE-SHAP-cond", "shap_cond"), ("PACE-SHAP", "shap_int")):
        mine = df[(df.method == method) & (df["rows"] == SHAP_V3_R)].set_index(["dataset", "seed"])
        if mine.empty:
            continue
        ref = b3[b3.method == method].set_index(["dataset", "seed"]).reindex(mine.index)
        rr = real[real.method == scheme].set_index(["dataset", "seed"]).reindex(mine.index)
        da = np.nanmax((mine[cols] - ref[cols]).abs().to_numpy())
        dr = np.nanmax((mine.off_share - rr.off_share).abs().to_numpy())
        L.append(f"| {method} | {da:.1e} | {dr:.1e} |")

    head = "| Dataset | " + " | ".join(f"{R} rows" for R in rows) + " |"
    rule = "|---|" + "---|" * len(rows)
    L += ["", f"## {next(sec)}. Effective explainer rows of the conditional game", "",
          "Kish effective rows (Σw)² / Σw², median over points and over coalitions with half the "
          "features or more (not all); then the median over seeds.", "", head, rule]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{get(d, 'PACE-SHAP-cond', R, 'eff_rows_large').median():.1f}" for R in rows) + " |")
    L += ["", "Smaller coalitions (1 feature up to just under half), same summary:", "", head, rule]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{get(d, 'PACE-SHAP-cond', R, 'eff_rows_small').median():.1f}" for R in rows) + " |")

    pts = real[(real.method == "points") & real.seed.isin(seeds)].groupby("dataset").off_share.mean()
    L += ["", f"## {next(sec)}. Inputs off the data (%)", "",
          "B3's measure, 200 inputs per point. Last column: the explained points themselves.", "",
          "| Dataset | " + " | ".join(f"Conditional, {R} rows" for R in rows) + " | "
          + " | ".join(f"Interventional, {R} rows" for R in rows) + " | Points |",
          "|---|" + "---|" * (2 * len(rows) + 1)]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{100 * get(d, m, R, 'off_share').mean():.1f}" for m in ("PACE-SHAP-cond", "PACE-SHAP")
            for R in rows) + f" | {100 * pts.get(d, np.nan):.1f} |")

    for m, lower, title in TESTS:
        L += ["", f"## {next(sec)}. {title}", "",
              "In brackets after PACE-SHAP-cond: seeds where it is more faithful than PACE-SHAP on the "
              f"same rows. Last column: seeds where PACE-SHAP-cond with {rows[-1]} rows is more "
              f"faithful than with {rows[0]}.", "",
              "| Dataset | " + " | ".join(f"PACE-SHAP-cond, {R} rows" for R in rows) + " | "
              + " | ".join(f"PACE-SHAP, {R} rows" for R in rows) + f" | {rows[-1]} vs {rows[0]} rows |",
              "|---|" + "---|" * (2 * len(rows) + 1)]
        better = (lambda a, b: a < b) if lower else (lambda a, b: a > b)
        for d in dsets:
            cells = []
            for R in rows:
                c, i = get(d, "PACE-SHAP-cond", R, m), get(d, "PACE-SHAP", R, m)
                cells.append(f"{c.mean():.3f} ({int(better(c, i.reindex(c.index)).sum())}/{len(c)})")
            cells += [f"{get(d, 'PACE-SHAP', R, m).mean():.3f}" for R in rows]
            hi, lo = get(d, "PACE-SHAP-cond", rows[-1], m), get(d, "PACE-SHAP-cond", rows[0], m)
            cells.append(f"{int(better(hi, lo.reindex(hi.index)).sum())}/{len(hi)}")
            L.append(f"| {NAMES.get(d, d)} | " + " | ".join(cells) + " |")

    syn = [d for d in dsets if d in SYNTHETIC]
    if syn:
        L += ["", f"## {next(sec)}. Ground truth: share of the true features x0–x2 in the top 3", "",
              "| Dataset | " + " | ".join(f"PACE-SHAP-cond, {R} rows" for R in rows) + " | "
              + " | ".join(f"PACE-SHAP, {R} rows" for R in rows) + " |",
              "|---|" + "---|" * (2 * len(rows))]
        for d in syn:
            L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
                f"{get(d, mm, R, 'gt_top3_precision').mean():.2f}" for mm in ("PACE-SHAP-cond", "PACE-SHAP")
                for R in rows) + " |")

    L += ["", f"## {next(sec)}. Distance between the two games' attributions", "",
          "Per point Σ|φ_cond − φ_PACE-SHAP| / Σ|φ_PACE-SHAP| on the same rows, mean over points and "
          "seeds.", "", head, rule]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{get(d, 'PACE-SHAP-cond', R, 'dist_to_pace_shap').mean():.2f}" for R in rows) + " |")
    text = "\n".join(L) + "\n"
    (out / "posthoc_rows.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--rows", nargs="+", type=int, default=list(ROWS))
    ap.add_argument("--n", type=int, default=N_EXPLAIN, help="test points (B3: 50; other values "
                    "are for quick trials and do not repeat B3)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(B3))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "posthoc_rows.csv"
    if not a.summary_only:
        tasks = [(d, s) for d in a.datasets for s in a.seeds]
        slow_first = ["S1", "magic", "page_blocks", "shuttle"]
        tasks.sort(key=lambda t: slow_first.index(t[0]) if t[0] in slow_first else -1)
        done = {}

        def save():
            pd.DataFrame([r for t in tasks if t in done for r in done[t]]).to_csv(csv, index=False)

        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                fut = {ex.submit(run_job, d, s, a.n, tuple(a.rows)): (d, s) for d, s in tasks}
                for f in as_completed(fut):
                    done[fut[f]] = f.result()
                    save()
        else:
            for t in tasks:
                done[t] = run_job(*t, a.n, tuple(a.rows))
                save()
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
