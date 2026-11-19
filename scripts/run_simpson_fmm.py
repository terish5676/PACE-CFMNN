"""Run M: Simpson's fuzzy min-max classifier trained from scratch, and PACE on it.

Rules: results/simpson_fmm/PREREGISTRATION.md. The model is ``pace_cfmnn.simpson_fmm.SimpsonFMM``:
CFMNN's membership, size limit and expansion, with Simpson's overlap test and contraction after
every expansion, and ties to the lowest box index (no centroid). As Run I's Part A, on this model:

* ``exact``: on the explanation datasets with at most 12 features (S1, S2, S3, Shuttle, MAGIC,
  Page Blocks), PACE-SHAP's exact values against shap's ExactExplainer (``xai_baselines.exact_shap``)
  on the first 5 explained points of each seed, with the second explanation run's split, points and
  200 explainer rows. Check: largest difference at most 1e-9 (PASS or FAIL);
* ``models``: on the 13 accuracy datasets, the accuracy table's split (``run_generality.prepare``),
  SimpsonFMM and CFMNN (theta = 0.3, gamma = 4) side by side (``run_generality.describe``), with
  seconds to train, contractions, and the pairs of boxes of different classes that Simpson's test
  still finds overlapping after training.

    python -B scripts/run_simpson_fmm.py --jobs 12
    python -B scripts/run_simpson_fmm.py --jobs 12 --resume
    python -B scripts/run_simpson_fmm.py --summary-only
    python -B scripts/run_simpson_fmm.py --datasets skin --seeds 192 --out <folder> --fresh   # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from accuracy_table import DATASETS  # noqa: E402  (the 13 accuracy datasets)
from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_cfmnn.simpson_fmm import SimpsonFMM  # noqa: E402
from run_generality import describe, prepare  # noqa: E402

OUT = ROOT / "results" / "simpson_fmm"
SEEDS = list(range(220, 230))
THETA, GAMMA = 0.3, 4.0
MODELS = ["CFMNN", "Simpson-scratch"]
EXACT_DATASETS = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
N_EXACT = 5
EXACT_TOLERANCE = 1e-9
COST = {"skin": 6, "letter": 5, "S1": 4, "S2": 4, "S3": 4, "shuttle": 3, "magic": 2}
COLUMNS = ["dataset", "seed", "n_train", "n_test", "d", "model", "boxes", "train_errors",
           "train_error_pct", "train_errors_tied", "train_errors_tied_inside_both",
           "train_tie_share_pct", "train_tie_inside_both_share_pct", "test_accuracy", "test_macro_f1",
           "test_errors", "test_centroid_decided", "test_centroid_decided_pct",
           "test_errors_centroid_decided", "test_errors_centroid_decided_pct", "contractions",
           "case_1", "case_2", "case_3", "case_4", "overlapping_pairs", "seconds"]


def run_models(name: str, seed: int):
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = prepare(name, seed)
    base = {"dataset": name, "seed": seed, "n_train": len(ytr), "n_test": len(yte), "d": Xtr.shape[1]}
    rows = []
    for model, make in (("CFMNN", lambda: CFMNN(theta=THETA, gamma=GAMMA)),
                        ("Simpson-scratch", lambda: SimpsonFMM(theta=THETA, gamma=GAMMA))):
        t0 = time.perf_counter()
        m = make().fit(Xtr, ytr)
        secs = time.perf_counter() - t0
        st = m.fit_stats_
        rec = {c: np.nan for c in COLUMNS}
        rec.update({**base, "model": model, **describe(m, Xtr, ytr, Xte, yte), "seconds": secs,
                    "overlapping_pairs": SimpsonFMM.overlapping_pairs(m)})
        if model == "Simpson-scratch":
            rec.update({"contractions": st["n_contractions"],
                        **{f"case_{i}": st[f"case_{i}"] for i in range(1, 5)}})
        rows.append({c: rec[c] for c in COLUMNS})
    print(f"{name} seed {seed}: " + ", ".join(
        f"{r['model']} {r['test_accuracy']:.1f}% ({r['boxes']} boxes, {r['train_errors_tied']}/"
        f"{r['train_errors']} ties, {r['seconds']:.0f} s)" for r in rows), flush=True)
    return rows


def exact_inputs(name: str, seed: int):
    """The second explanation run's training rows (with their labels), points and explainer rows."""
    ds, Xtr, Xte, _, _ = rf.setup(name, seed, N_EXACT)
    sp = data.paper_split(ds.y, seed)
    ytr = ds.y[np.random.default_rng(seed).permutation(sp["train"])]     # rf.setup's row order
    if len(ytr) != len(Xtr):
        raise RuntimeError("labels and training rows do not match")
    _, bex = rf.backgrounds_v3(Xtr, seed)
    return Xtr, ytr, Xte, bex[:rf.SHAP_V3_R]


def run_exact(name: str, seed: int):
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, B = exact_inputs(name, seed)
    if Xte.shape[1] > 12:
        raise ValueError(f"{name} has {Xte.shape[1]} features; the exact check is for at most 12")
    sim = SimpsonFMM(theta=THETA, gamma=GAMMA).fit(Xtr, ytr)
    t0 = time.perf_counter()
    ex = PaceShap(sim, B)
    phi = ex.explain(Xte)
    t_pace = time.perf_counter() - t0
    if not ex.exact:
        raise RuntimeError("PACE-SHAP did not run in exact mode")
    t0 = time.perf_counter()
    ref = xb.exact_shap(sim, Xte, B)
    t_shap = time.perf_counter() - t0
    diff = float(np.abs(phi - ref).max())
    print(f"{name} seed {seed}: {sim.n_boxes_} boxes, {sim.fit_stats_['n_contractions']} contractions, "
          f"largest difference {diff:.1e}", flush=True)
    return [{"dataset": name, "seed": seed, "d": Xte.shape[1], "n": len(Xte), "background_rows": len(B),
             "boxes": sim.n_boxes_, "contractions": sim.fit_stats_["n_contractions"],
             "max_abs_diff": diff, "max_abs_value": float(np.abs(ref).max()),
             "pace_ms_per_explanation": 1000 * t_pace / len(Xte),
             "shap_exact_ms_per_explanation": 1000 * t_shap / len(Xte)}]


# ------------------------------------------------------------------ summary
def _pool(g, num, den):
    d = g[den].sum()
    return 100 * g[num].sum() / d if d else np.nan


def summarise(out: Path) -> str:
    L = ["# Run M: Simpson's fuzzy min-max classifier trained from scratch", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The exact check is a check; everything "
         "else is descriptive (no test). θ = 0.3, γ = 4 for both models. Means over seeds; shares "
         "pooled over seeds. Written by `scripts/run_simpson_fmm.py`.", ""]
    ecsv = out / "exact.csv"
    if ecsv.exists():
        e = pd.read_csv(ecsv)
        worst = float(e.max_abs_diff.max())
        verdict = "PASS" if worst <= EXACT_TOLERANCE else "FAIL"
        L += ["## PACE-SHAP on SimpsonFMM against shap's ExactExplainer", "",
              f"**Check: {verdict}** (largest difference {worst:.1e}; the plan's limit is {EXACT_TOLERANCE:.0e}).", "",
              "| Dataset | Seeds | Points | Boxes (mean) | Contractions (mean) | Largest difference | "
              "ms PACE-SHAP | ms ExactExplainer |", "|---|---|---|---|---|---|---|---|"]
        for ds in [d for d in EXACT_DATASETS if d in set(e.dataset)]:
            g = e[e.dataset == ds]
            L.append(f"| {ds} | {g.seed.nunique()} | {int(g.n.sum())} | {g.boxes.mean():,.1f} | "
                     f"{g.contractions.mean():,.1f} | {g.max_abs_diff.max():.1e} | "
                     f"{g.pace_ms_per_explanation.mean():,.1f} | {g.shap_exact_ms_per_explanation.mean():,.1f} |")
        L.append("")
    csv = out / "runs.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        ds_all = [d for d in DATASETS if d in set(df.dataset)]
        piv = lambda col: df.pivot_table(index="dataset", columns="model", values=col)   # noqa: E731
        L += ["## SimpsonFMM beside CFMNN (13 accuracy datasets)", "",
              "| Dataset | Accuracy CFMNN | Accuracy Simpson | Macro-F1 CFMNN | Macro-F1 Simpson | Boxes CFMNN | "
              "Boxes Simpson | Training error % CFMNN | Training error % Simpson | Ties % CFMNN | Ties % Simpson | "
              "Tie-rule-decided test % CFMNN | Tie-rule-decided test % Simpson | Contractions | Pairs still "
              "overlapping | Seconds CFMNN | Seconds Simpson |", "|" + "---|" * 17]
        acc, f1, box, te, sec = (piv(c) for c in ("test_accuracy", "test_macro_f1", "boxes",
                                                  "train_error_pct", "seconds"))
        con, ov = piv("contractions"), piv("overlapping_pairs")
        for ds in ds_all:
            g = df[df.dataset == ds]
            gc, gs = g[g.model == "CFMNN"], g[g.model == "Simpson-scratch"]
            L.append(f"| {ds} | {acc.loc[ds, 'CFMNN']:.2f} | {acc.loc[ds, 'Simpson-scratch']:.2f} | "
                     f"{f1.loc[ds, 'CFMNN']:.2f} | {f1.loc[ds, 'Simpson-scratch']:.2f} | "
                     f"{box.loc[ds, 'CFMNN']:,.1f} | {box.loc[ds, 'Simpson-scratch']:,.1f} | "
                     f"{te.loc[ds, 'CFMNN']:.2f} | {te.loc[ds, 'Simpson-scratch']:.2f} | "
                     f"{_pool(gc, 'train_errors_tied_inside_both', 'train_errors'):.1f} | "
                     f"{_pool(gs, 'train_errors_tied_inside_both', 'train_errors'):.1f} | "
                     f"{_pool(gc, 'test_centroid_decided', 'n_test'):.1f} | "
                     f"{_pool(gs, 'test_centroid_decided', 'n_test'):.1f} | {con.loc[ds, 'Simpson-scratch']:,.1f} | "
                     f"{ov.loc[ds, 'Simpson-scratch']:,.1f} | {sec.loc[ds, 'CFMNN']:,.1f} | "
                     f"{sec.loc[ds, 'Simpson-scratch']:,.1f} |")
        L += ["", "Ties: training errors inside a box of both classes (Proposition 1's form), pooled. "
              "Tie-rule-decided: test predictions whose class the tie rule decides (CFMNN: nearest "
              "centroid; Simpson: the lowest box index), pooled. Pairs still overlapping: pairs of boxes "
              "of different classes that Simpson's test still finds overlapping after training (a box "
              "made as a new point is never tested).", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run M: Simpson's FMM trained from scratch")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--part", choices=["models", "exact", "both"], default="both")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    if (a.datasets or a.seeds) and out.resolve() == OUT.resolve():
        raise SystemExit("a smoke slice needs --out outside results/simpson_fmm")
    out.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds or SEEDS
    if not a.summary_only:
        if a.part in ("models", "both"):
            names = [d for d in DATASETS if not a.datasets or d in a.datasets]
            todo = pending(out / "runs.csv", [(d, s) for d in names for s in seeds], a.resume, a.fresh)
            run_all(run_models, todo, out / "runs.csv", a.jobs, COST)
        if a.part in ("exact", "both"):
            names = [d for d in EXACT_DATASETS if not a.datasets or d in a.datasets]
            if names:
                todo = pending(out / "exact.csv", [(d, s) for d in names for s in seeds], a.resume, a.fresh)
                run_all(run_exact, todo, out / "exact.csv", a.jobs, COST)
    print(summarise(out))


if __name__ == "__main__":
    main()
