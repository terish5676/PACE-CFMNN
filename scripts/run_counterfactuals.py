"""Counterfactual benchmark (blueprint 5.3, 10): PACE box-directed vs DiCE vs nearest unlike neighbour.

Same models and splits as the RQ1 benchmark (paper's 80/20 split, main model:
fixed centroid + search). For each explained test point the target is the
runner-up class B. Every counterfactual is checked on the model itself.

Reported per method: found (a candidate was returned), valid (the model really
predicts the target), L1 distance, features changed, plausibility (L1 distance
to the nearest training point of the target class), locked features kept
(Thyroid only: features 0 and 1, age and sex in the standard attribute order,
are locked) and time per counterfactual.

DiCE settings (``--dice``):
* ``v2`` (default, fairer): 5 candidates, DiCE's own post-hoc sparsity output,
  and the nearest candidate (L1) that the model predicts as the target.
* ``v1``: the original setting (one candidate, pre-sparsity output), kept so
  the earlier results in ``results/counterfactuals_v1`` reproduce.

``--extended`` (RQ1 extension B1, pre-registered in
results/extensions_v1/PREREGISTRATION.md; seeds 40-49) adds PACE-sparse
(``counterfactual.sparse_counterfactual``: the fewest changed features, scored
exactly from the model's terms) and seed-level paired tests (per-seed means
over points where both counterfactuals are valid; Wilcoxon over seeds, Holm
across datasets). ``--jobs`` runs dataset x seed pairs in parallel (times then
include CPU sharing).

Usage (from the project folder):
    python scripts/run_counterfactuals.py
    python scripts/run_counterfactuals.py --datasets wine thyroid --seeds 0 --n 10
    python scripts/run_counterfactuals.py --extended --jobs 6
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.counterfactual import counterfactual, sparse_counterfactual  # noqa: E402

DATASETS = ["S1", "S2", "S3", "wine", "breast_cancer", "thyroid", "shuttle",
            "magic", "letter", "landsat", "page_blocks"]
LOCKED = {"thyroid": [0, 1]}
METHODS = ["PACE", "DiCE", "Nearest unlike neighbour"]
METHODS_EXT = ["PACE", "PACE-sparse", "DiCE", "Nearest unlike neighbour"]
# (focus, other, column); the first is the pre-registered primary comparison.
EXT_TESTS = [("PACE-sparse", "DiCE", "n_changed"), ("PACE-sparse", "DiCE", "l1"),
             ("PACE-sparse", "DiCE", "plausibility"), ("PACE-sparse", "PACE", "n_changed"),
             ("PACE-sparse", "PACE", "l1"), ("PACE-sparse", "DiCE", "valid")]


DICE = {"v1": dict(n_cfs=1, sparse=False, pick="first"),
        "v2": dict(n_cfs=5, sparse=True, pick="nearest_valid")}


def run_one(name: str, seed: int, n: int, dice: str = "v2", extended: bool = False) -> list[dict]:
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    te = np.random.default_rng(seed).permutation(sp["test"])[:n]
    Xte = np.clip(sc.transform(ds.X[te]), 0.0, 1.0)
    model = CFMNN().fit(Xtr, ytr)
    d = Xtr.shape[1]
    lock_idx = LOCKED.get(name, [])
    lock = np.zeros(d, bool)
    lock[lock_idx] = True
    det = model.predict_details(Xte)
    pred_tr = model.predict(Xtr)          # for the nearest unlike neighbour, computed once
    rows = []
    for r in range(len(Xte)):
        if det.runner_up[r] < 0:
            continue
        x, target = Xte[r], model.classes_[det.runner_up[r]]
        same = Xtr[ytr == target]
        out = {}
        t0 = time.perf_counter()
        c = counterfactual(model, x, target, locked=lock)
        out["PACE"] = (c.x_cf if c.found else None, time.perf_counter() - t0, c.model_calls)
        how = ""
        if extended:
            t0 = time.perf_counter()
            c2 = sparse_counterfactual(model, x, target, locked=lock)
            out["PACE-sparse"] = (c2.x_cf if c2.found else None, time.perf_counter() - t0,
                                  c2.model_calls)
            how = c2.how
        t0 = time.perf_counter()
        with contextlib.redirect_stderr(io.StringIO()):
            cf = xb.dice_counterfactual(model, x, Xtr, ytr, target, seed=seed, locked=tuple(lock_idx),
                                        **DICE[dice])
        out["DiCE"] = (cf, time.perf_counter() - t0, np.nan)
        t0 = time.perf_counter()
        cand = Xtr[pred_tr == target]
        nun = cand[np.abs(cand - x).sum(axis=1).argmin()] if len(cand) else None
        out["Nearest unlike neighbour"] = (nun, time.perf_counter() - t0, np.nan)
        for m, (z, secs, calls) in out.items():
            row = {"dataset": name, "seed": seed, "point": int(te[r]), "method": m,
                   "found": z is not None, "ms": 1000 * secs, "model_calls": calls}
            if extended:
                row["how"] = how if m == "PACE-sparse" else ""
            if z is not None:
                z = np.asarray(z, float)
                row.update(valid=bool(model.predict(z[None])[0] == target),
                           l1=float(np.abs(z - x).sum()),
                           n_changed=int((np.abs(z - x) > 1e-9).sum()),
                           plausibility=float(np.abs(same - z).sum(axis=1).min()) if len(same) else np.nan,
                           locked_kept=bool(np.allclose(z[lock], x[lock])) if lock.any() else np.nan)
            else:
                row.update(valid=False, l1=np.nan, n_changed=np.nan, plausibility=np.nan,
                           locked_kept=np.nan)
            rows.append(row)
    return rows


def ci95(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


CF_MEASURES = [("l1", "L1 distance"), ("n_changed", "Features changed"), ("plausibility", "Plausibility")]


def paired_points(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """PACE minus DiCE per point where both are valid (seeds pooled), Holm across datasets."""
    from run_error_detection import holm
    ok = df[df.valid.astype(str) == "True"]
    rows = []
    for ds in df.dataset.unique():
        g = ok[ok.dataset == ds]
        a = g[g.method == "PACE"].set_index(["seed", "point"])[col].astype(float)
        b = g[g.method == "DiCE"].set_index(["seed", "point"])[col].astype(float)
        d = (a - b.reindex(a.index)).dropna()
        p = stats.wilcoxon(d).pvalue if len(d) > 1 and (d != 0).any() else 1.0
        rows.append({"dataset": ds, "diff": d.mean(), "lower": int((d < 0).sum()), "n": len(d), "p": p})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p.to_numpy())
    return res


def paired_seeds(df: pd.DataFrame, a: str, b: str, col: str) -> pd.DataFrame:
    """a minus b per seed, then a Wilcoxon test over seeds (Holm across datasets).

    For L1, features changed and plausibility the per-seed value is the mean
    difference over points where both counterfactuals are valid; for
    ``col="valid"`` it is the difference in the share of valid answers over
    all points."""
    from run_error_detection import holm
    valid = df.valid.astype(str) == "True"
    rows = []
    for ds in df.dataset.unique():
        if col == "valid":
            g = df[df.dataset == ds].assign(v=valid[df.dataset == ds].astype(float))
            share = g.groupby(["method", "seed"]).v.mean()
            per = (share[a] - share[b]).dropna()
            pts = int((g.method == a).sum())
        else:
            g = df[valid & (df.dataset == ds)]
            A = g[g.method == a].set_index(["seed", "point"])[col].astype(float)
            B = g[g.method == b].set_index(["seed", "point"])[col].astype(float)
            dd = (A - B.reindex(A.index)).dropna()
            per = dd.groupby(level="seed").mean()
            pts = len(dd)
        p = stats.wilcoxon(per).pvalue if len(per) > 1 and (per != 0).any() else 1.0
        rows.append({"dataset": ds, "diff": per.mean(), "ci": ci95(per), "lower": int((per < 0).sum()),
                     "higher": int((per > 0).sum()), "n": len(per), "points": pts, "p": p})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p.to_numpy())
    return res


def summarise(df: pd.DataFrame, out: Path, dice: str = "v2") -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    extended = "PACE-sparse" in set(df.method)
    methods = METHODS_EXT if extended else METHODS
    per = []
    for (ds, m, s), g in df.groupby(["dataset", "method", "seed"], sort=False):
        v = g[g.valid]
        per.append({"dataset": ds, "method": m, "seed": s, "found": 100 * g.found.mean(),
                    "valid": 100 * g.valid.mean(), "l1": v.l1.mean(), "n_changed": v.n_changed.mean(),
                    "plausibility": v.plausibility.mean(),
                    "locked_kept": 100 * g.locked_kept.dropna().astype(float).mean()
                    if g.locked_kept.notna().any() else np.nan,
                    "ms": g.ms.median(), "calls": g.model_calls.mean()})
    p = pd.DataFrame(per)
    L = ["# Counterfactuals: PACE" + (", PACE-sparse" if extended else "")
         + " vs DiCE vs nearest unlike neighbour", "",
         f"Paper's 80/20 split, main model (fixed centroid + search); up to {df.groupby(['dataset', 'seed']).point.nunique().max()} "
         f"test points per dataset and seed; seeds {seeds}. Target = the runner-up class. "
         "Mean ± 95% CI over seeds of per-seed means. L1, features changed and plausibility are over "
         "valid counterfactuals only; lower is better for all three. Plausibility = L1 distance to the "
         "nearest training point of the target class. Time = median per counterfactual. Thyroid locks "
         "features 0 and 1 (age and sex). DiCE: " + ("5 candidates, its own sparse output, the nearest "
         "one the model predicts as the target." if dice == "v2" else "one candidate, pre-sparsity output "
         "(original setting)."), "",
         "| Dataset | Method | Valid (%) | L1 distance | Features changed | Plausibility | "
         "Locked kept (%) | ms | Model calls |",
         "|---|---|---|---|---|---|---|---|---|"]
    f = lambda c, k=3: (f"{c.mean():.{k}f} ± {ci95(c):.{k}f}" if c.notna().any() else "–")
    if extended:
        L[4:4] = ["PACE-sparse: the fewest changed features that make the model predict the target, "
                  "scored exactly from the model's own terms and verified on the model (falls back "
                  "to PACE when PACE changes fewer features). Times are medians with several runs "
                  "sharing the CPU, so compare them within a dataset only.", ""]
    for ds in df.dataset.unique():
        for m in methods:
            g = p[(p.dataset == ds) & (p.method == m)]
            if g.empty:
                continue
            L.append(f"| {ds} | {m} | {f(g.valid, 1)} | {f(g.l1)} | {f(g.n_changed, 2)} | "
                     f"{f(g.plausibility)} | {f(g.locked_kept, 1)} | {f(g.ms, 2)} | {f(g.calls, 1)} |")
    if extended:
        names = dict(CF_MEASURES + [("valid", "Valid share")])
        for k, (a, b, col) in enumerate(EXT_TESTS):
            res = paired_seeds(df, a, b, col)
            better = ("higher" if col == "valid" else "lower")
            L += ["", f"## {a} minus {b}: {names[col].lower()}"
                  f"{' (primary)' if k == 0 else ' (secondary)'}", "",
                  "Per-seed mean difference (points where both are valid"
                  + ("; for validity, the share of valid answers over all points" if col == "valid" else "")
                  + "), mean ± 95% CI over seeds; two-sided Wilcoxon signed-rank test over seeds, "
                  f"Holm-corrected across datasets. Better for {a}: {better}.", "",
                  f"| Dataset | Difference | Seeds where {a} is lower | Seeds where it is higher | "
                  "Points | p | p (Holm) |",
                  "|---|---|---|---|---|---|---|"]
            for _, r in res.iterrows():
                L.append(f"| {r.dataset} | {r['diff']:+.3f} ± {r.ci:.3f} | {r.lower} of {r.n} | "
                         f"{r.higher} of {r.n} | {r.points} | {r.p:.3g} | {r.p_holm:.3g} |")
        if "how" in df.columns:
            g = df[df.method == "PACE-sparse"]
            cols = ["exact", "greedy", "pace", "none"]
            share = (g.groupby("dataset", sort=False).how.value_counts(normalize=True)
                     .unstack(fill_value=0).reindex(columns=cols, fill_value=0))
            L += ["", "## How PACE-sparse found its answer", "",
                  "Share of points: exact subset search, greedy search, PACE's own answer "
                  "(fewer or equally many changes), none found.", "",
                  "| Dataset | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for ds, r in share.iterrows():
                L.append(f"| {ds} | " + " | ".join(f"{100 * r[c]:.0f}%" for c in cols) + " |")
    elif dice == "v2":
        L += ["", "## PACE minus DiCE, per point", "",
              "Points where both counterfactuals are valid, seeds pooled; two-sided Wilcoxon "
              "signed-rank test, Holm-corrected across datasets for each measure. Negative = PACE "
              "lower (better for all three).", "",
              "| Dataset | Measure | Difference | Points where PACE is lower | p | p (Holm) |",
              "|---|---|---|---|---|---|"]
        tests = {name: paired_points(df, col) for col, name in CF_MEASURES}
        for ds in df.dataset.unique():
            for name, res in tests.items():
                r = res[res.dataset == ds].iloc[0]
                L.append(f"| {ds} | {name} | {r['diff']:+.3f} | {r.lower} of {r.n} | {r.p:.3g} | "
                         f"{r.p_holm:.3g} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dice", choices=["v1", "v2"], default="v2")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--extended", action="store_true",
                    help="RQ1 extension B1: add PACE-sparse and seed-level paired tests")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.extended and a.dice != "v2":
        raise SystemExit("--extended needs --dice v2")
    seeds = a.seeds or (list(range(40, 50)) if a.extended else [0, 1, 2])
    out = Path(a.out or ROOT / "results" / ("extensions_v1/counterfactuals" if a.extended else
                                            "counterfactuals" if a.dice == "v2" else "counterfactuals_v1"))
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        tasks = [(n, s) for n in a.datasets for s in seeds]
        done = {}
        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                fut = {ex.submit(run_one, n, s, a.n, a.dice, a.extended): (n, s) for n, s in tasks}
                for f in as_completed(fut):
                    done[fut[f]] = f.result()
                    print(f"{fut[f][0]} seed {fut[f][1]} done", flush=True)
        else:
            for t in tasks:
                t0 = time.perf_counter()
                done[t] = run_one(*t, a.n, a.dice, a.extended)
                print(f"{t[0]} seed {t[1]}: {time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame([r for t in tasks for r in done[t]]).to_csv(out / "runs.csv", index=False)
    print(summarise(pd.read_csv(out / "runs.csv"), out, a.dice))


if __name__ == "__main__":
    main()
