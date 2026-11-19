"""Run I: generality (Simpson's contraction) and sensitivity (gamma).

Rules: results/generality/PREREGISTRATION.md.

Two parts, both on the paper's 80/20 split with the training rows in the seed's order, as the
accuracy table and the second explanation run (seeds 150-159):

* ``models`` (runs.csv), on the 13 accuracy datasets. Four models per dataset and seed:
  CFMNN at theta = 0.3 with gamma = 2, 4 and 8, and "Simpson": the gamma = 4 model after
  Simpson's overlap test and contraction on every pair of boxes of different classes
  (``contraction.contract_all``, Run F's SC-all arm). For each: boxes, training error, how
  many training errors are ties, how many test predictions the nearest centroid decides, test
  accuracy and macro-F1.
* ``exact`` (exact.csv), on the six explanation datasets with at most 12 features. On the
  Simpson model, PACE-SHAP's exact values against shap's ExactExplainer on the same model,
  points and rows (the first 5 explained points, the 200 explainer rows of v3).

Primary: test accuracy at gamma = 2 against gamma = 4, and at gamma = 8 against gamma = 4;
paired two-sided Wilcoxon over seeds, Holm across the 13 datasets. Everything else is
descriptive.

    python -B scripts/run_generality.py --jobs 12
    python -B scripts/run_generality.py --jobs 12 --resume     # continue after a stop
    python -B scripts/run_generality.py --summary-only
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
from pace_cfmnn.contraction import contract_all, rows_without_box  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_cfmnn.ties import centroid_decided, training_error_ties  # noqa: E402
from run_readable_baselines import compare, tally  # noqa: E402

OUT = ROOT / "results" / "generality"
SEEDS = list(range(150, 160))
THETA = 0.3
GAMMAS = [2.0, 4.0, 8.0]
BASE_GAMMA = 4.0                       # the paper's gamma; Simpson's contraction starts from it
SIMPSON = "Simpson"
MODELS = [f"gamma={g:g}" for g in GAMMAS] + [SIMPSON]
BASE = f"gamma={BASE_GAMMA:g}"
PRIMARY = [("gamma=2", BASE), ("gamma=8", BASE)]
# Part A1: RQ1's explanation datasets with at most 12 features (PACE-SHAP's exact mode)
EXACT_DATASETS = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
N_EXACT = 5                            # explained points per seed
EXACT_TOLERANCE = 1e-9                 # "the same values": largest difference at most this
# Slowest first: this only orders the tasks.
COST = {"skin": 6, "letter": 5, "S1": 4, "S2": 4, "S3": 4, "shuttle": 3, "magic": 2}


def prepare(name: str, seed: int):
    """The accuracy table's split: 80/20, training rows in the seed's order, scaled on them."""
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    return (sc.transform(ds.X[order]), ds.y[order],
            sc.transform(ds.X[sp["test"]]), ds.y[sp["test"]])


def describe(model, Xtr, ytr, Xte, yte) -> dict:
    """Boxes, training-error ties, centroid-decided test rows and test scores of one model."""
    tr = training_error_ties(model, Xtr, ytr)
    te = centroid_decided(model, Xte, yte)
    met = classification_metrics(yte, model.predict(Xte))
    pct = lambda a, b: 100 * a / b if b else np.nan   # noqa: E731
    return {"boxes": model.n_boxes_, **tr,
            "train_tie_share_pct": pct(tr["train_errors_tied"], tr["train_errors"]),
            "train_tie_inside_both_share_pct": pct(tr["train_errors_tied_inside_both"],
                                                   tr["train_errors"]),
            "test_accuracy": met["accuracy"], "test_macro_f1": met["macro_f1"],
            "test_errors": te["n_errors"], "test_centroid_decided": te["n_centroid_decided"],
            "test_centroid_decided_pct": pct(te["n_centroid_decided"], te["n"]),
            "test_errors_centroid_decided": te["n_errors_centroid_decided"],
            "test_errors_centroid_decided_pct": pct(te["n_errors_centroid_decided"], te["n_errors"])}


def run_models(name: str, seed: int):
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = prepare(name, seed)
    base = {"dataset": name, "seed": seed, "n_train": len(ytr), "n_test": len(yte),
            "d": Xtr.shape[1]}
    blank = {"contractions": np.nan, "sweeps": np.nan, "rows_evicted": np.nan,
             "boxes_removed": np.nan}
    rows, start = [], None
    for g in GAMMAS:
        t0 = time.perf_counter()
        m = CFMNN(theta=THETA, gamma=g).fit(Xtr, ytr)
        secs = time.perf_counter() - t0
        if g == BASE_GAMMA:
            start = m
        rows.append({**base, "model": f"gamma={g:g}", "gamma": g,
                     **describe(m, Xtr, ytr, Xte, yte), **blank, "seconds": secs})
    t0 = time.perf_counter()
    sim, st, _ = contract_all(start, Xtr)
    secs = time.perf_counter() - t0
    if st["rows_evicted"] != rows_without_box(sim, len(ytr)):
        raise RuntimeError(f"{name} seed {seed}: the evicted rows do not add up")
    rows.append({**base, "model": SIMPSON, "gamma": BASE_GAMMA, **describe(sim, Xtr, ytr, Xte, yte),
                 **{k: st[k] for k in blank}, "seconds": secs})
    print(f"{name} seed {seed}: " + ", ".join(
        f"{r['model']} {r['test_accuracy']:.1f}% ({r['train_errors_tied']}/{r['train_errors']} "
        "ties)" for r in rows), flush=True)
    return rows


def run_exact(name: str, seed: int):
    """PACE-SHAP's exact values on the Simpson model against shap's ExactExplainer."""
    warnings.filterwarnings("ignore")
    _, Xtr, Xte, model, _ = rf.setup(name, seed, N_EXACT)
    if Xte.shape[1] > 12:
        raise ValueError(f"{name} has {Xte.shape[1]} features; the exact check is for at most 12")
    _, bex = rf.backgrounds_v3(Xtr, seed)
    B = bex[:rf.SHAP_V3_R]
    sim, st, _ = contract_all(model, Xtr)
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
    print(f"{name} seed {seed}: {st['contractions']} contractions, largest difference {diff:.1e}",
          flush=True)
    return [{"dataset": name, "seed": seed, "d": Xte.shape[1], "n": len(Xte),
             "background_rows": len(B), "boxes_start": st["boxes_start"],
             "boxes": sim.n_boxes_, "contractions": st["contractions"],
             "rows_evicted": st["rows_evicted"], "max_abs_diff": diff,
             "max_abs_value": float(np.abs(ref).max()),
             "pace_ms_per_explanation": 1000 * t_pace / len(Xte),
             "shap_exact_ms_per_explanation": 1000 * t_shap / len(Xte)}]


# ------------------------------------------------------------------ summary
def _pooled(df, num, den):
    """Per dataset and model: 100 * sum over seeds of ``num`` / sum of ``den`` (NaN if 0)."""
    g = df.groupby(["dataset", "model"])[[num, den]].sum()
    return (100 * g[num] / g[den].where(g[den] > 0)).unstack("model")


def _table(L, piv, ds, models, fmt):
    piv = piv.reindex(index=ds, columns=models)
    L += ["| Dataset | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    L += [f"| {d} | " + " | ".join("–" if pd.isna(v) else fmt.format(v) for v in r) + " |"
          for d, r in piv.iterrows()]
    L.append("")


def summarise(out: Path) -> str:
    L = ["# Run I: generality (Simpson's contraction) and sensitivity (γ)", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The paper's 80/20 split, θ = 0.3. "
         "Written by `scripts/run_generality.py`.", ""]
    runs, exact = out / "runs.csv", out / "exact.csv"
    if runs.exists():
        df = pd.read_csv(runs)
        seeds = sorted(int(s) for s in df.seed.unique())
        ds = [d for d in DATASETS if d in set(df.dataset)]
        models = [m for m in MODELS if m in set(df.model)]
        gm = [m for m in models if m != SIMPSON]
        L += [f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}), {len(ds)} datasets. Each comparison: paired "
              "two-sided Wilcoxon over seeds, Holm across datasets; better / level / worse for the "
              "first model.", "", "## Primary: test accuracy at γ = 2 and γ = 8 against γ = 4", ""]
        for a, b in PRIMARY:
            r = compare(df, a, b, "test_accuracy")
            if r.empty:
                continue
            L += [f"**{a} vs {b}**: {tally(r)}", "",
                  "| Dataset | Mean difference (points) | p | Holm p | Verdict |", "|---|---|---|---|---|"]
            L += [f"| {x.dataset} | {x['diff']:+.2f} | {x.p:.4f} | {x.p_holm:.4f} | {x.verdict} |"
                  for _, x in r.iterrows()]
            L.append("")
        mean = lambda col: df.pivot_table(index="dataset", columns="model", values=col)  # noqa: E731
        L += ["## Part B: CFMNN at γ = 2, 4 and 8 (means over seeds)", "", "Test accuracy (%):", ""]
        _table(L, mean("test_accuracy"), ds, gm, "{:.2f}")
        L += ["Test macro-F1 (%):", ""]
        _table(L, mean("test_macro_f1"), ds, gm, "{:.2f}")
        L += ["Boxes:", ""]
        _table(L, mean("boxes"), ds, gm, "{:,.1f}")
        L += ["Training error (%):", ""]
        _table(L, mean("train_error_pct"), ds, gm, "{:.2f}")
        L += ["Share of training errors that are ties (%), all seeds pooled; – = no training error. "
              "Proposition 1 predicts 100 at every γ:", ""]
        _table(L, _pooled(df, "train_errors_tied_inside_both", "train_errors"), ds, gm, "{:.1f}")
        bad = df[(df.model != SIMPSON) & (df.train_errors_tied_inside_both != df.train_errors)]
        L += [f"Training errors that are not ties, over every dataset, seed and γ: "
              f"**{int((bad.train_errors - bad.train_errors_tied_inside_both).sum())}** of "
              f"{int(df[df.model != SIMPSON].train_errors.sum()):,}.", "",
              "Share of test predictions decided by the nearest centroid (%), all seeds pooled:", ""]
        _table(L, _pooled(df, "test_centroid_decided", "n_test"), ds, gm, "{:.1f}")
        L += ["Share of test errors that are among those predictions (%), all seeds pooled:", ""]
        _table(L, _pooled(df, "test_errors_centroid_decided", "test_errors"), ds, gm, "{:.1f}")
        if SIMPSON in models:
            pair = [BASE, SIMPSON]
            s = df[df.model == SIMPSON].groupby("dataset")
            L += ["## Part A (2): Simpson's contraction next to CFMNN (γ = 4)", "",
                  "What the contraction did (means over seeds):", "",
                  "| Dataset | Boxes before | Contractions | Seeds with no contraction | Boxes removed | "
                  "Training rows evicted | Boxes after |", "|---|---|---|---|---|---|---|"]
            b0 = df[df.model == BASE].groupby("dataset").boxes.mean()
            for d in ds:
                if d not in s.groups:
                    continue
                g = s.get_group(d)
                L.append(f"| {d} | {b0[d]:,.1f} | {g.contractions.mean():,.1f} | "
                         f"{int((g.contractions == 0).sum())} of {len(g)} | {g.boxes_removed.mean():.1f} | "
                         f"{g.rows_evicted.mean():,.1f} | {g.boxes.mean():,.1f} |")
            L += ["", "Training error (%):", ""]
            _table(L, mean("train_error_pct"), ds, pair, "{:.2f}")
            L += ["Share of training errors that are ties (%), all seeds pooled; – = no training "
                  "error. A tie: the row's own class reaches the winner's membership:", ""]
            _table(L, _pooled(df, "train_errors_tied", "train_errors"), ds, pair, "{:.1f}")
            L += ["... of which inside a box of both classes (both memberships are 1), the form "
                  "Proposition 1 gives for CFMNN (%):", ""]
            _table(L, _pooled(df, "train_errors_tied_inside_both", "train_errors"), ds, pair, "{:.1f}")
            L += ["Test accuracy (%):", ""]
            _table(L, mean("test_accuracy"), ds, pair, "{:.2f}")
            L += ["Test macro-F1 (%):", ""]
            _table(L, mean("test_macro_f1"), ds, pair, "{:.2f}")
            L += ["Share of test predictions decided by the nearest centroid (%), all seeds pooled:", ""]
            _table(L, _pooled(df, "test_centroid_decided", "n_test"), ds, pair, "{:.1f}")
    if exact.exists():
        e = pd.read_csv(exact)
        worst = float(e.max_abs_diff.max())
        ok = worst <= EXACT_TOLERANCE
        L += ["## Part A (1): PACE-SHAP on the Simpson model against shap's ExactExplainer", "",
              f"**Check: {'PASS' if ok else 'FAIL'}** (largest difference {worst:.1e}; the plan's "
              f"limit is {EXACT_TOLERANCE:.0e}).", "",
              "| Dataset | Features | Seeds | Points | Contractions (mean) | Seeds with no contraction | "
              "Boxes (mean) | Largest difference | Largest value | ms PACE-SHAP | ms ExactExplainer |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in [x for x in EXACT_DATASETS if x in set(e.dataset)]:
            g = e[e.dataset == d]
            L.append(f"| {d} | {int(g.d.iloc[0])} | {len(g)} | {int(g.n.sum())} | "
                     f"{g.contractions.mean():,.1f} | {int((g.contractions == 0).sum())} of {len(g)} | "
                     f"{g.boxes.mean():,.1f} | {g.max_abs_diff.max():.1e} | {g.max_abs_value.max():.3f} | "
                     f"{g.pace_ms_per_explanation.mean():,.1f} | "
                     f"{g.shap_exact_ms_per_explanation.mean():,.1f} |")
        L += ["", "Largest difference: the largest absolute difference between the two, over every "
              "point, feature and seed. Where a seed has no contraction, the Simpson model is the "
              "CFMNN itself.", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run I: generality and sensitivity")
    ap.add_argument("--part", choices=["all", "models", "exact"], default="all")
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
    if not a.summary_only:
        seeds = a.seeds or SEEDS
        for part, fn, names, csv in (("models", run_models, DATASETS, out / "runs.csv"),
                                     ("exact", run_exact, EXACT_DATASETS, out / "exact.csv")):
            if a.part not in ("all", part):
                continue
            chosen = [d for d in names if a.datasets is None or d in a.datasets]
            tasks = [(d, s) for d in chosen for s in seeds]
            print(f"--- part {part}: {len(tasks)} tasks", flush=True)
            run_all(fn, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs, COST)
    print(summarise(out))


if __name__ == "__main__":
    main()
