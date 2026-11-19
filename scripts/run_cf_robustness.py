"""Run B: more counterfactual baselines, and four robustness measures.

Rules: results/cf_robustness/PREREGISTRATION.md. Plan: research_audit/LIMITATIONS_PLAN.md, run B.

B1's design (`scripts/run_counterfactuals.py`): 11 datasets, 30 test points per seed, the target
is the runner-up class, Thyroid's features 0 and 1 stay locked. Methods:

* PACE and PACE-sparse;
* DiCE random in the paper's v2 setting (5 candidates, DiCE's sparsity output, nearest valid),
  and DiCE genetic and DiCE kdtree in the same setting;
* the nearest unlike neighbour;
* a brute-force search over 1-2 features, using training-data values (`brute_force_counterfactual`).

Measures: validity, features changed, L1, plausibility and milliseconds, as in B1, and

* r1, input sensitivity: for each of 5 perturbed copies x' of x (`Perturber`, kappa = 0.05, copies
  the model predicts like x), recompute the counterfactual; r1 is the median over copies of
  L1(cf', cf) / L1(x', x). Lower is steadier. DiCE genetic and DiCE kdtree are left out (cost);
* r2: the share of 20 noisy copies of the answer (kappa = 0.05) still predicted as the target;
* r3: the share of 4 CFMNNs, fitted on the training rows in other orders, that predict the target
  for the answer;
* r4: the target margin at the answer (`xai_baselines.margin_score`).

    python -B scripts/run_cf_robustness.py --jobs 12
    python -B scripts/run_cf_robustness.py --jobs 12 --resume
    python -B scripts/run_cf_robustness.py --summary-only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.counterfactual import (brute_force_counterfactual, counterfactual,  # noqa: E402
                                       replacement_values, sparse_counterfactual)
from pace_cfmnn.neighborhood import Perturber  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_counterfactuals import DATASETS, DICE, LOCKED  # noqa: E402
from run_readable_baselines import holm  # noqa: E402

OUT = ROOT / "results" / "cf_robustness"
SEEDS = list(range(110, 120))
N_POINTS = 30
METHODS = ["PACE", "PACE-sparse", "DiCE random", "DiCE genetic", "DiCE kdtree",
           "Nearest unlike neighbour", "Brute force"]
# DiCE genetic and DiCE kdtree are left out of r1 because of their cost (kdtree: about 25 s a call)
R1_METHODS = [m for m in METHODS if m not in ("DiCE genetic", "DiCE kdtree")]
KAPPA, N_R1, N_R2, N_R3 = 0.05, 5, 20, 4
# (focus, other, column, lower_is_better): the pre-registered primaries
PRIMARIES = [("PACE", "DiCE random", "r1", True), ("PACE-sparse", "DiCE random", "r1", True),
             ("PACE", "DiCE random", "r3", False), ("PACE-sparse", "DiCE random", "r3", False)]
COLUMNS = ["dataset", "seed", "point", "method", "found", "ms", "model_calls", "how", "valid", "l1",
           "n_changed", "plausibility", "locked_kept", "r1", "r1_copies", "r2", "r3", "r4"]
# Slowest first: this only orders the tasks.
COST = {"letter": 6, "shuttle": 5, "magic": 4, "landsat": 3, "page_blocks": 2}


# ------------------------------------------------------------------ the measures
def input_sensitivity(x, cf, copies, regenerate) -> tuple[float, int]:
    """r1: the median over ``copies`` x' of L1(cf', cf) / L1(x', x), and how many copies counted.

    ``regenerate(x')`` returns a counterfactual or None. A copy counts if it is not x itself and
    its counterfactual exists.
    """
    x, cf = np.asarray(x, float), np.asarray(cf, float)
    ratios = []
    for z in copies:
        step = float(np.abs(z - x).sum())
        if step <= 0:
            continue
        cf2 = regenerate(z)
        if cf2 is not None:
            ratios.append(float(np.abs(np.asarray(cf2, float) - cf).sum()) / step)
    return (float(np.median(ratios)) if ratios else np.nan), len(ratios)


def answer_noise(model, cf, target, pert: Perturber, seed: int) -> float:
    """r2: the share of the perturbed copies of the answer that the model predicts as the target."""
    Z = pert.sample(np.asarray(cf, float)[None, :], seed=seed)[0]
    return float(np.mean(model.predict(Z) == target))


def refit_validity(models, cf, target) -> float:
    """r3: the share of the refitted models that predict the target for the answer."""
    cf = np.asarray(cf, float)[None, :]
    return float(np.mean([m.predict(cf)[0] == target for m in models]))


# ------------------------------------------------------------------ one dataset and seed
def make_generators(model, Xtr, ytr, pred_tr, lock, lock_idx, seed):
    values = replacement_values(Xtr)

    def dice(method):
        def f(x, target):
            with contextlib.redirect_stderr(io.StringIO()):
                return xb.dice_counterfactual(model, x, Xtr, ytr, target, seed=seed,
                                              locked=tuple(lock_idx), method=method, **DICE["v2"])
        return f

    def pace(x, target):
        c = counterfactual(model, x, target, locked=lock)
        return (c.x_cf if c.found else None), c.model_calls, ""

    def pace_sparse(x, target):
        c = sparse_counterfactual(model, x, target, locked=lock)
        return (c.x_cf if c.found else None), c.model_calls, c.how

    def nun(x, target):
        cand = Xtr[pred_tr == target]
        return (cand[np.abs(cand - x).sum(axis=1).argmin()] if len(cand) else None), np.nan, ""

    def brute(x, target):
        c = brute_force_counterfactual(model, x, target, values, locked=lock)
        return (c.x_cf if c.found else None), c.model_calls, ""

    def wrap(f):
        return lambda x, target: (f(x, target), np.nan, "")

    return {"PACE": pace, "PACE-sparse": pace_sparse, "DiCE random": wrap(dice("random")),
            "DiCE genetic": wrap(dice("genetic")), "DiCE kdtree": wrap(dice("kdtree")),
            "Nearest unlike neighbour": nun, "Brute force": brute}


def run_one(name: str, seed: int, n: int = N_POINTS) -> list[dict]:
    warnings.filterwarnings("ignore")
    t_start = time.perf_counter()
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    te = np.random.default_rng(seed).permutation(sp["test"])[:n]
    Xte = np.clip(sc.transform(ds.X[te]), 0.0, 1.0)
    model = CFMNN().fit(Xtr, ytr)
    others = [CFMNN().fit(Xtr[p], ytr[p]) for p in
              (np.random.default_rng(10_000 * seed + j).permutation(len(ytr)) for j in range(1, N_R3 + 1))]
    d = Xtr.shape[1]
    lock_idx = LOCKED.get(name, [])
    lock = np.zeros(d, bool)
    lock[lock_idx] = True
    det = model.predict_details(Xte)
    pred_tr = model.predict(Xtr)
    gens = make_generators(model, Xtr, ytr, pred_tr, lock, lock_idx, seed)
    pert1 = Perturber(kappa=KAPPA, k=N_R1).fit(Xtr)
    pert2 = Perturber(kappa=KAPPA, k=N_R2).fit(Xtr)
    rows = []
    for r in range(len(Xte)):
        if det.runner_up[r] < 0:
            continue
        x, t_idx = Xte[r], int(det.runner_up[r])
        target = model.classes_[t_idx]
        same = Xtr[ytr == target]
        pred_x = model.predict(x[None])[0]
        rng_seed = 1_000_003 * seed + int(te[r])
        copies = pert1.sample(x, seed=rng_seed)[0]
        copies = copies[model.predict(copies) == pred_x]      # copies with the same prediction as x
        for m in METHODS:
            t0 = time.perf_counter()
            z, calls, how = gens[m](x, target)
            ms = 1000 * (time.perf_counter() - t0)
            row = dict.fromkeys(COLUMNS, np.nan)
            row.update(dataset=name, seed=seed, point=int(te[r]), method=m, found=z is not None,
                       ms=ms, model_calls=calls, how=how, valid=False, locked_kept=np.nan)
            if z is not None:
                z = np.asarray(z, float)
                row.update(valid=bool(model.predict(z[None])[0] == target),
                           l1=float(np.abs(z - x).sum()), n_changed=int((np.abs(z - x) > 1e-9).sum()),
                           plausibility=float(np.abs(same - z).sum(axis=1).min()) if len(same) else np.nan,
                           locked_kept=bool(np.allclose(z[lock], x[lock])) if lock.any() else np.nan,
                           r2=answer_noise(model, z, target, pert2, rng_seed),
                           r3=refit_validity(others, z, target),
                           r4=float(xb.margin_score(model, z[None], t_idx)[0]))
                if m in R1_METHODS:
                    row["r1"], row["r1_copies"] = input_sensitivity(
                        x, z, copies, lambda xp, m=m: gens[m](xp, target)[0])
            rows.append(row)
    print(f"{name} seed {seed}: {len(rows) // len(METHODS)} points, "
          f"{time.perf_counter() - t_start:.0f} s", flush=True)
    return rows


# ------------------------------------------------------------------ summary
def point_means(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per dataset and seed, the mean over points of ``col`` for each method."""
    return df.groupby(["dataset", "seed", "method"])[col].mean().unstack("method")


def compare(df: pd.DataFrame, a: str, b: str, col: str, lower_is_better: bool = False) -> pd.DataFrame:
    """Paired two-sided Wilcoxon over seeds, Holm across datasets: a vs b."""
    pm, out = point_means(df, col), []
    for ds in [x for x in DATASETS if x in set(pm.index.get_level_values(0))]:
        g = pm.loc[ds]
        if a not in g or b not in g:
            continue
        diff = (g[a] - g[b]).dropna().to_numpy()
        if len(diff) == 0:
            continue
        p = wilcoxon(diff).pvalue if len(diff) > 1 and np.any(diff != 0) else 1.0
        out.append({"dataset": ds, "diff": float(diff.mean()), "p": float(p), "n": len(diff)})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = holm(res.p.to_numpy())
    good = res["diff"] < 0 if lower_is_better else res["diff"] > 0
    res["verdict"] = np.where(res.p_holm >= 0.05, "level", np.where(good, "better", "worse"))
    return res


def tally(res: pd.DataFrame) -> str:
    if res.empty:
        return "no data"
    c = res.verdict.value_counts()
    return f"better {c.get('better', 0)}, level {c.get('level', 0)}, worse {c.get('worse', 0)}"


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [x for x in DATASETS if x in set(df.dataset)]
    methods = [m for m in METHODS if m in set(df.method)]
    df = df.copy()
    df["valid_all"] = (df.found.astype(bool) & df.valid.astype(bool)).astype(float)
    L = ["# Run B: counterfactual baselines and robustness", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). B1's design, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}), {N_POINTS} points per seed. Each comparison: "
         "paired two-sided Wilcoxon over seeds on the per-seed mean over points, Holm across "
         "datasets; better / level / worse for the first method (for r1 lower is better; for r3 "
         "higher). Written by `scripts/run_cf_robustness.py`.", "",
         "## Primary comparisons", ""]
    for a, b, col, low in PRIMARIES:
        res = compare(df, a, b, col, low)
        if res.empty:
            continue
        L += [f"**{a} vs {b}, {col}** ({'lower' if low else 'higher'} is better): {tally(res)}", "",
              "| Dataset | Mean difference | p | Holm p | Verdict |", "|---|---|---|---|---|"]
        L += [f"| {r.dataset} | {r['diff']:+.4f} | {r.p:.4f} | {r.p_holm:.4f} | {r.verdict} |"
              for _, r in res.iterrows()]
        L.append("")
    for title, col, fmt in (("Validity: found and confirmed by the model (share of points)", "valid_all", "{:.3f}"),
                            ("Features changed (of the found answers)", "n_changed", "{:.2f}"),
                            ("L1 distance (of the found answers)", "l1", "{:.3f}"),
                            ("Plausibility: L1 to the nearest training point of the target class", "plausibility", "{:.3f}"),
                            ("Milliseconds per point", "ms", "{:.0f}"),
                            ("r1, input sensitivity (lower is steadier)", "r1", "{:.3f}"),
                            ("r2, share of 20 noisy copies of the answer still the target", "r2", "{:.3f}"),
                            ("r3, share of 4 refitted models that confirm the answer", "r3", "{:.3f}"),
                            ("r4, target margin at the answer", "r4", "{:.3f}")):
        piv = df.groupby(["dataset", "method"])[col].mean().unstack("method").reindex(index=ds, columns=methods)
        L += [f"## {title}, mean over points and seeds", "", "| Dataset | " + " | ".join(methods) + " |",
              "|---|" + "---|" * len(methods)]
        L += [f"| {d} | " + " | ".join("" if np.isnan(v) else fmt.format(v) for v in r) + " |"
              for d, r in piv.iterrows()]
        L.append("")
    cnt = df.groupby("method").agg(points=("point", "size"), found=("found", "sum"),
                                   valid=("valid_all", "sum")).reindex(methods)
    L += ["## Counts over all datasets and seeds", "", "```", cnt.to_string(), "```", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run B: counterfactual baselines and robustness")
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
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs, COST)
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
