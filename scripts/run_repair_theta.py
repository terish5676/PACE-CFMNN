"""Self-repair at the theta chosen by validation (planned in advance).

Rules: results/repair_theta/PREREGISTRATION.md. Theta chosen by validation is more accurate
than theta = 0.3 on 7 of 13 datasets, but with many more boxes (results/readable_baselines/).
Self-repair at theta = 0.3 also gains accuracy (RQ3). This run puts the two side by side on the
same rows: why repair at theta = 0.3 rather than choose a smaller theta?

Four models per dataset and seed, on the readable-baselines split (the paper's 80/20 split, the
seed's row order, min-max scaling fitted on the training rows):

* B0-0.3: CFMNN at theta = 0.3 (the explained model);
* SR-0.3: B0-0.3, then self-repair;
* B0-val: CFMNN at the theta chosen by validation. For a study seed (70-89) this theta is read
  from results/readable_baselines/runs.csv; for any other seed it is chosen the same way;
* SR-val: B0-val, then self-repair.

Self-repair is RQ3's SR arm: up to 300 of the model's own training errors, the smallest cut,
10 rounds, gated on the training rows, so no label beyond the training rows is used. For a
study seed, B0-0.3 and B0-val must reproduce the readable-baselines run exactly, or the task
fails.

    python -B scripts/run_repair_theta.py --jobs 6
    python -B scripts/run_repair_theta.py --jobs 6 --resume     # continue after a stop
    python -B scripts/run_repair_theta.py --summary-only
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.audit import prediction_churn  # noqa: E402
from pace_cfmnn.refine import contains  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_readable_baselines import (DATASETS, GAMMA, GRIDS, SEEDS, box_size, compare,  # noqa: E402
                                    cv_folds, prepare, tally, tune)

READABLE_RUNS = ROOT / "results" / "readable_baselines" / "runs.csv"
OUT = ROOT / "results" / "repair_theta"
MODELS = ["B0-0.3", "SR-0.3", "B0-val", "SR-val"]
# RQ3's self-repair (scripts/run_refinement_v2.py, arm SR): budget, rounds, the smallest cut.
BUDGET, N_ROUNDS, CHOOSER = 300, 10, "volume"
FIXED_THETA = 0.3
CHECK_TOL = 1e-9
# Slowest first: seconds per seed on one core in the timing check (Skin ~5,000, Letter ~1,700,
# MAGIC ~400, Landsat ~250). This only orders the tasks.
COST = {"skin": 6, "letter": 5, "magic": 4, "landsat": 3, "S2": 2, "S1": 1}


# ------------------------------------------------------------------ theta chosen by validation
@lru_cache(maxsize=1)
def readable_runs() -> pd.DataFrame:
    """The readable-baselines rows of CFMNN-0.3 and CFMNN-val (read only)."""
    if not READABLE_RUNS.exists():
        return pd.DataFrame(columns=["dataset", "seed", "model", "setting", "accuracy", "macro_f1"])
    r = pd.read_csv(READABLE_RUNS)
    return r[r.model.isin(["CFMNN-0.3", "CFMNN-val", "kNN", "Random Forest"])].copy()


def readable_row(name: str, seed: int, model: str):
    r = readable_runs()
    hit = r[(r.dataset == name) & (r.seed == seed) & (r.model == model)]
    return None if hit.empty else hit.iloc[0]


def theta_val(name: str, seed: int, Xtr, ytr) -> tuple[float, str]:
    """The theta chosen by validation, and where it came from."""
    row = readable_row(name, seed, "CFMNN-val")
    if row is not None:
        return float(row.setting), "readable_baselines"
    setting, _ = tune("CFMNN-val", GRIDS["CFMNN-val"], Xtr, ytr, seed, 1, cv_folds(ytr, seed))
    return float(setting), "tuned here"


# ------------------------------------------------------------------ training errors
def training_errors(model: CFMNN, X, y) -> dict:
    """Training errors, and how many of them are ties self-repair can cut.

    ``in_winning_box``: the winning box is of a wrong class and contains the row (membership 1),
    so the shrink can move one of its faces past the row. ``own_box_full``: the row also has
    membership 1 in a box of its own class.
    """
    det = model.predict_details(X)
    t = np.searchsorted(model.classes_, y)
    err = np.flatnonzero(det.cls != t)
    in_win = sum(contains(model, int(det.winner[p]), X[p]) for p in err)
    own = int(np.sum(det.support[err, t[err]] >= 1 - 1e-12)) if len(err) else 0
    return {"train_errors": len(err), "train_error_pct": 100 * len(err) / len(y),
            "train_errors_in_winning_box": int(in_win), "train_errors_own_box_full": own}


# ------------------------------------------------------------------ one dataset and seed
def run_one(name: str, seed: int, tags=("0.3", "val")):
    """``tags`` picks the models: "0.3" gives B0-0.3 and SR-0.3, "val" gives B0-val and SR-val
    (the default, both, is this run's protocol)."""
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = prepare(name, seed)
    tv, source = theta_val(name, seed, Xtr, ytr) if "val" in tags else (np.nan, "not run")
    rows = []
    for tag, theta in (t for t in (("0.3", FIXED_THETA), ("val", tv)) if t[0] in tags):
        t0 = time.perf_counter()
        b0 = CFMNN(theta=theta, gamma=GAMMA).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        p0 = np.asarray(b0.predict(Xte))
        t0 = time.perf_counter()
        sr, _, edits = run_self_repair(b0, Xtr, ytr, Xtr, ytr, chooser=CHOOSER, budget=BUDGET,
                                       n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False)
        repair_s = time.perf_counter() - t0
        p1 = np.asarray(sr.predict(Xte))
        for model, m, pred, secs in ((f"B0-{tag}", b0, p0, fit_s), (f"SR-{tag}", sr, p1, repair_s)):
            boxes, cond = box_size(m.V_, m.W_)
            churn = prediction_churn(yte, p0, pred)
            rec = {"dataset": name, "seed": seed, "model": model, "theta": theta,
                   "theta_source": source if tag == "val" else "fixed",
                   "accuracy": 100 * float(np.mean(pred == yte)),
                   "macro_f1": 100 * float(f1_score(yte, pred, average="macro")),
                   "boxes": boxes, "conditions_per_box": cond, **training_errors(m, Xtr, ytr),
                   "edits_logged": np.nan, "edits_already_right": np.nan, "edits_accepted": np.nan,
                   "test_changed": churn["changed"], "test_fixed": churn["fixed"],
                   "test_broken": churn["broken"], "seconds": secs,
                   "n_train": len(ytr), "n_test": len(yte), "readable_accuracy": np.nan,
                   "readable_macro_f1": np.nan}
            if model.startswith("SR"):
                rec.update({"edits_logged": len(edits),
                            "edits_already_right": sum(e["reason"] == "already right" for e in edits),
                            "edits_accepted": sum(bool(e["accepted"]) for e in edits)})
            else:
                ref = readable_row(name, seed, "CFMNN-0.3" if tag == "0.3" else "CFMNN-val")
                if ref is not None:
                    rec["readable_accuracy"], rec["readable_macro_f1"] = ref.accuracy, ref.macro_f1
                    if (abs(rec["accuracy"] - ref.accuracy) > CHECK_TOL
                            or abs(rec["macro_f1"] - ref.macro_f1) > CHECK_TOL):
                        raise RuntimeError(f"{name} seed {seed} {model} does not reproduce the "
                                           f"readable-baselines run: accuracy {rec['accuracy']} vs "
                                           f"{ref.accuracy}, macro-F1 {rec['macro_f1']} vs {ref.macro_f1}")
            rows.append(rec)
    print(f"{name} seed {seed} (theta-val {tv:g}): " + ", ".join(
        f"{r['model']} {r['accuracy']:.1f}" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------ summary
COMPARISONS = [("P1 (primary)", "SR-0.3", "B0-val", "Does repair at theta = 0.3 match choosing theta by validation?"),
               ("P2 (primary)", "SR-val", "B0-val", "Does repair still help at the theta validation chose?"),
               ("S1", "SR-0.3", "B0-0.3", "Does repair help on this split?"),
               ("S2", "SR-val", "SR-0.3", "With repair on both, which theta is better?")]


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    models = [m for m in MODELS if m in set(df.model)]
    L = ["# Self-repair at the theta chosen by validation", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The readable-baselines split, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}). Each comparison: paired two-sided Wilcoxon over "
         "seeds, Holm across datasets; better / level / worse for the first model. Written by "
         "`scripts/run_repair_theta.py`.", ""]
    b0 = df[df.model.str.startswith("B0") & df.readable_accuracy.notna()]
    if len(b0):
        bad = ((b0.accuracy - b0.readable_accuracy).abs() > CHECK_TOL) | \
              ((b0.macro_f1 - b0.readable_macro_f1).abs() > CHECK_TOL)
        L += [f"**Check:** {len(b0) - int(bad.sum())} of {len(b0)} B0 rows reproduce the "
              "readable-baselines accuracy and macro-F1 exactly.", ""]
    for title, col in (("Test accuracy (%)", "accuracy"), ("Test macro-F1 (%)", "macro_f1")):
        piv = df.pivot_table(index="dataset", columns="model", values=col).reindex(ds)[models]
        L += [f"## {title}, mean over seeds", "", "| Dataset | " + " | ".join(models) + " |",
              "|---|" + "---|" * len(models)]
        L += [f"| {d} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for d, r in piv.iterrows()]
        L.append("")
    for col, label in (("accuracy", "accuracy"), ("macro_f1", "macro-F1 (S3)")):
        L += [f"## Comparisons by {label}", ""]
        for code, a, b, q in COMPARISONS:
            res = compare(df, a, b, col)
            if res.empty:
                continue
            L += [f"**{code if col == 'accuracy' else 'S3, ' + code.split()[0]}: {a} vs {b}** "
                  f"({col}). {q} {tally(res)}", "",
                  "| Dataset | Mean difference | p | Holm p | Verdict |", "|---|---|---|---|---|"]
            L += [f"| {r.dataset} | {r['diff']:+.2f} | {r.p:.4f} | {r.p_holm:.4f} | {r.verdict} |"
                  for _, r in res.iterrows()]
            L.append("")
    L += ["## Descriptive", ""]
    val = df[df.model == "B0-val"]
    L += ["Theta chosen by validation (count over seeds):", ""]
    L += [f"- {d}: " + ", ".join(f"{k:g} ×{v}" for k, v in val[val.dataset == d].theta.value_counts()
                                 .sort_index().items()) for d in ds]
    L.append("")
    cols = ["boxes", "conditions_per_box", "train_errors", "train_errors_in_winning_box",
            "train_errors_own_box_full", "edits_accepted", "test_fixed", "test_broken", "seconds"]
    means = df.groupby(["dataset", "model"], sort=False)[cols].mean().reindex(
        pd.MultiIndex.from_product([ds, models]))
    L += ["Means over seeds (boxes; conditions per box; training errors; of them, inside the "
          "winning wrong-class box; of them, with a full box of their own class; edits accepted; "
          "test predictions fixed and broken against that theta's B0; seconds):", "",
          "```", means.round(2).to_string(), "```", ""]
    ref = readable_runs()
    ref = ref[ref.model.isin(["kNN", "Random Forest"]) & ref.seed.isin(seeds)]
    if len(ref):
        best = ref.pivot_table(index="dataset", columns="model", values="accuracy").max(axis=1)
        acc = df.pivot_table(index="dataset", columns="model", values="accuracy").reindex(ds)
        gap = acc[models].sub(best.reindex(ds), axis=0)
        L += ["Gap to the better of the tuned kNN and the Random Forest (readable-baselines run, "
              "same seeds; accuracy points):", "", "```", gap.round(2).to_string(), "```", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Self-repair at the theta chosen by validation")
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
