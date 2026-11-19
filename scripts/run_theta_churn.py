"""Run O: re-tune theta or repair? What each update does to the model's other answers.

Rules: results/theta_churn/PREREGISTRATION.md (planned in advance).

Choosing theta by validation beats self-repair at theta = 0.3 on 7 of 13 datasets, with 6-23
times as many boxes (results/repair_theta/). This run asks what else each update changes. On the
repair-at-theta split (the readable-baselines split: the paper's 80/20 split, the seed's row
order, min-max scaling fitted on the training rows; seeds 70-89, 13 datasets, gamma = 4) it fits
CFMNN (B0) at every theta of the validation grid (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05) and
records, for each fit:

* test accuracy and macro-F1, boxes, training errors, seconds;
* against B0-0.3 (the explained model), the test predictions changed, fixed (wrong -> right) and
  broken (right -> wrong). The negative flip rate (NFR) is broken / test rows, the share of test
  points the old model got right and the new one gets wrong (Yan et al., 2021).

Self-repair's own flips are not recomputed: SR-0.3 against B0-0.3, and SR-val against B0-val,
are read from results/repair_theta/runs.csv (columns test_changed, test_fixed, test_broken).
B0 is deterministic, so B0 at theta = 0.3 and at the theta chosen by validation must reproduce
results/readable_baselines/runs.csv exactly, or the task fails.

    python -B scripts/run_theta_churn.py --jobs 12
    python -B scripts/run_theta_churn.py --jobs 12 --resume     # continue after a stop
    python -B scripts/run_theta_churn.py --summary-only
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.audit import prediction_churn  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_readable_baselines import (DATASETS, GAMMA, SEEDS, THETAS, box_size, compare,  # noqa: E402
                                    prepare, tally)
from run_repair_theta import CHECK_TOL, FIXED_THETA, readable_row, theta_val  # noqa: E402

OUT = ROOT / "results" / "theta_churn"
REPAIR_RUNS = ROOT / "results" / "repair_theta" / "runs.csv"
# Slowest first (seconds per seed at theta = 0.05 in the readable-baselines run). Only orders tasks.
COST = {"skin": 6, "letter": 5, "magic": 4, "landsat": 3, "S3": 2, "S2": 2, "S1": 2, "shuttle": 1}


# ------------------------------------------------------------------ flips
def flip_rates(y_true, pred_before, pred_after) -> dict:
    """Counts from ``audit.prediction_churn``, and the same as percentages of the test rows."""
    c = prediction_churn(y_true, pred_before, pred_after)
    n = len(np.asarray(y_true))
    return {"test_changed": c["changed"], "test_fixed": c["fixed"], "test_broken": c["broken"],
            "changed_pct": 100 * c["changed"] / n, "pfr": 100 * c["fixed"] / n,
            "nfr": 100 * c["broken"] / n,
            "fix_share": 100 * c["fixed"] / c["changed"] if c["changed"] else np.nan}


def grid_theta(theta: float) -> float:
    """The grid value equal to ``theta`` (the readable run stores the chosen theta as a float)."""
    hit = [t for t in THETAS if np.isclose(t, theta, rtol=0, atol=1e-12)]
    if not hit:
        raise ValueError(f"theta {theta} is not in the validation grid {THETAS}")
    return hit[0]


def check_reproduces(name: str, seed: int, model: str, acc: float, f1: float):
    """Compare with the readable-baselines row; raise if it differs. Returns the row (or None)."""
    ref = readable_row(name, seed, model)
    if ref is not None and (abs(acc - ref.accuracy) > CHECK_TOL or abs(f1 - ref.macro_f1) > CHECK_TOL):
        raise RuntimeError(f"{name} seed {seed} {model} does not reproduce the readable-baselines "
                           f"run: accuracy {acc} vs {ref.accuracy}, macro-F1 {f1} vs {ref.macro_f1}")
    return ref


# ------------------------------------------------------------------ one dataset and seed
def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = prepare(name, seed)
    tv, source = theta_val(name, seed, Xtr, ytr)
    tv = grid_theta(tv)
    t_task = time.perf_counter()          # the 7 fits and their predictions (not the tuning)
    fits = {}
    for theta in THETAS:
        t0 = time.perf_counter()
        m = CFMNN(theta=theta, gamma=GAMMA).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.asarray(m.predict(Xte))
        predict_s = time.perf_counter() - t0
        train_errors = int(np.sum(np.asarray(m.predict(Xtr)) != ytr))
        boxes, cond = box_size(m.V_, m.W_)
        fits[theta] = (pred, fit_s, predict_s, train_errors, boxes, cond)
    base = fits[FIXED_THETA][0]
    task_s = time.perf_counter() - t_task
    rows = []
    for theta in THETAS:
        pred, fit_s, predict_s, train_errors, boxes, cond = fits[theta]
        acc = 100 * float(np.mean(pred == yte))
        f1 = 100 * float(f1_score(yte, pred, average="macro"))
        rec = {"dataset": name, "seed": seed, "theta": theta, "theta_val": tv,
               "theta_val_source": source, "is_fixed": theta == FIXED_THETA, "is_val": theta == tv,
               "accuracy": acc, "macro_f1": f1, "boxes": boxes, "conditions_per_box": cond,
               "train_errors": train_errors, "train_error_pct": 100 * train_errors / len(ytr),
               **flip_rates(yte, base, pred), "fit_s": fit_s, "predict_s": predict_s,
               "task_s": task_s, "n_train": len(ytr), "n_test": len(yte), "readable_accuracy": np.nan,
               "readable_macro_f1": np.nan}
        for flag, model in ((rec["is_fixed"], "CFMNN-0.3"), (rec["is_val"], "CFMNN-val")):
            if flag:
                ref = check_reproduces(name, seed, model, acc, f1)
                if ref is not None:
                    rec["readable_accuracy"], rec["readable_macro_f1"] = ref.accuracy, ref.macro_f1
        rows.append(rec)
    print(f"{name} seed {seed} (theta-val {tv:g}): " + ", ".join(
        f"{r['theta']:g} {r['accuracy']:.1f}" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------ summary helpers
def compare_lower(df: pd.DataFrame, a: str, b: str, col: str) -> pd.DataFrame:
    """``compare`` (paired two-sided Wilcoxon over seeds, Holm across datasets) for a measure
    where lower is better: the verdict is for ``a``, "better" when its value is lower."""
    res = compare(df, a, b, col)
    if res.empty:
        return res
    res["verdict"] = np.where(res.p_holm >= 0.05, "level", np.where(res["diff"] < 0, "better", "worse"))
    return res


def repair_flips(repair: pd.DataFrame, seeds) -> pd.DataFrame:
    """SR-0.3 and SR-val rows of the repair-at-theta run, with the flip rates in percent."""
    need = {"dataset", "seed", "model", "accuracy", "macro_f1", "boxes", "test_changed",
            "test_fixed", "test_broken", "n_test"}
    if not need <= set(repair.columns):
        return pd.DataFrame(columns=["dataset", "seed", "model", "accuracy", "macro_f1", "boxes",
                                     "changed_pct", "pfr", "nfr", "fix_share"])
    r = repair[repair.model.isin(["SR-0.3", "SR-val"]) & repair.seed.isin(seeds)].copy()
    for col, src in (("changed_pct", "test_changed"), ("pfr", "test_fixed"), ("nfr", "test_broken")):
        r[col] = 100 * r[src] / r.n_test
    r["fix_share"] = 100 * r.test_fixed / r.test_changed.where(r.test_changed > 0)
    return r


def p1_frame(df: pd.DataFrame, repair: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset, seed and update: repair (SR-0.3) and re-tune (B0 at theta-val)."""
    retune = df[df.is_val].assign(model="B0-val")
    sr = repair_flips(repair, sorted(df.seed.unique()))
    sr = sr[sr.model == "SR-0.3"]
    cols = ["dataset", "seed", "model", "accuracy", "changed_pct", "pfr", "nfr", "fix_share"]
    return pd.concat([sr[cols], retune[cols]], ignore_index=True)


def boxes_to_match(df: pd.DataFrame, repair: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the fewest mean boxes at which B0's mean accuracy reaches SR-0.3's."""
    sr = repair_flips(repair, sorted(df.seed.unique()))
    sr = sr[sr.model == "SR-0.3"].groupby("dataset")[["accuracy", "boxes"]].mean()
    g = df.groupby(["dataset", "theta"])[["accuracy", "boxes"]].mean()
    out = []
    for ds in [d for d in DATASETS if d in set(df.dataset) and d in sr.index]:
        b03 = g.loc[(ds, FIXED_THETA), "boxes"]
        ok = g.loc[ds][g.loc[ds].accuracy >= sr.loc[ds, "accuracy"]]
        if ok.empty:
            out.append({"dataset": ds, "sr_accuracy": sr.loc[ds, "accuracy"], "theta": np.nan,
                        "b0_accuracy": np.nan, "boxes": np.nan, "boxes_vs_0.3": np.nan})
            continue
        th = ok.boxes.idxmin()
        out.append({"dataset": ds, "sr_accuracy": sr.loc[ds, "accuracy"], "theta": th,
                    "b0_accuracy": ok.loc[th, "accuracy"], "boxes": ok.loc[th, "boxes"],
                    "boxes_vs_0.3": ok.loc[th, "boxes"] / b03})
    return pd.DataFrame(out)


def _grid_table(df, col, fmt, ds, title):
    piv = df.pivot_table(index="dataset", columns="theta", values=col).reindex(index=ds, columns=THETAS)
    L = [title, "", "| Dataset | " + " | ".join(f"θ = {t:g}" for t in THETAS) + " |",
         "|---|" + "---|" * len(THETAS)]
    L += [f"| {d} | " + " | ".join(fmt.format(v) for v in r) + " |" for d, r in piv.iterrows()]
    return L + [""]


# ------------------------------------------------------------------ summary
def summarise(df: pd.DataFrame, out: Path, repair: pd.DataFrame | None = None) -> str:
    if repair is None:
        repair = pd.read_csv(REPAIR_RUNS, float_precision="round_trip") if REPAIR_RUNS.exists() \
            else pd.DataFrame(columns=["dataset", "seed", "model"])
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    L = ["# Run O: re-tune θ or repair?", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The repair-at-θ split, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}), γ = 4. Flips are counted against B0 at θ = 0.3 "
         "on the same test rows: fixed = wrong → right, broken = right → wrong. NFR (negative "
         "flip rate) = broken / test rows; PFR = fixed / test rows; both in % of the test rows. "
         "Self-repair's flips come from `results/repair_theta/runs.csv`. Written by "
         "`scripts/run_theta_churn.py`.", ""]
    chk = df[df.readable_accuracy.notna()]
    if len(chk):
        bad = ((chk.accuracy - chk.readable_accuracy).abs() > CHECK_TOL) | \
              ((chk.macro_f1 - chk.readable_macro_f1).abs() > CHECK_TOL)
        L += [f"**Check:** {len(chk) - int(bad.sum())} of {len(chk)} B0 rows at θ = 0.3 or θ-val "
              "reproduce the readable-baselines accuracy and macro-F1 exactly.", ""]
    p1 = p1_frame(df, repair)
    if set(p1.model) == {"SR-0.3", "B0-val"}:
        res = compare_lower(p1, "SR-0.3", "B0-val", "nfr")
        m = p1.pivot_table(index="dataset", columns="model", values=["nfr", "pfr", "changed_pct",
                                                                    "accuracy", "fix_share"],
                           dropna=False)
        L += ["## P1 (primary): negative flip rate, repair (SR-0.3) vs re-tuning (B0 at θ-val)", "",
              "Both updates start from B0 at θ = 0.3. Lower is better; the verdict is for "
              f"repair. {tally(res)}", "",
              "| Dataset | NFR repair | NFR re-tune | Mean difference | p | Holm p | Verdict |",
              "|---|---|---|---|---|---|---|"]
        for _, r in res.iterrows():
            L.append(f"| {r.dataset} | {m.loc[r.dataset, ('nfr', 'SR-0.3')]:.3f} | "
                     f"{m.loc[r.dataset, ('nfr', 'B0-val')]:.3f} | {r['diff']:+.3f} | {r.p:.4f} | "
                     f"{r.p_holm:.4f} | {r.verdict} |")
        L += ["", "Descriptive, the same two updates (means over seeds, % of test rows; accuracy "
              "after the update):", "",
              "| Dataset | Changed, repair | Changed, re-tune | PFR repair | PFR re-tune | "
              "Fix share repair | Fix share re-tune | Accuracy, repair | Accuracy, re-tune |",
              "|---|---|---|---|---|---|---|---|---|"]
        for d in [d for d in ds if d in m.index]:
            L.append(f"| {d} | {m.loc[d, ('changed_pct', 'SR-0.3')]:.2f} | "
                     f"{m.loc[d, ('changed_pct', 'B0-val')]:.2f} | {m.loc[d, ('pfr', 'SR-0.3')]:.2f} | "
                     f"{m.loc[d, ('pfr', 'B0-val')]:.2f} | {m.loc[d, ('fix_share', 'SR-0.3')]:.1f} | "
                     f"{m.loc[d, ('fix_share', 'B0-val')]:.1f} | {m.loc[d, ('accuracy', 'SR-0.3')]:.2f} | "
                     f"{m.loc[d, ('accuracy', 'B0-val')]:.2f} |")
        L += ["", "Fix share: fixed ÷ changed, in % (as in RQ4), the mean over seeds that changed "
              "something.", ""]
    else:
        L += ["## P1 (primary)", "", "No self-repair rows for these seeds in "
              "`results/repair_theta/runs.csv`, so P1 is not computed.", ""]
    L += ["## Descriptive: B0 across the θ grid (means over seeds)", ""]
    L += _grid_table(df, "accuracy", "{:.2f}", ds, "Test accuracy (%):")
    L += _grid_table(df, "boxes", "{:.1f}", ds, "Boxes:")
    L += _grid_table(df, "nfr", "{:.3f}", ds, "NFR against B0 at θ = 0.3 (% of test rows):")
    L += _grid_table(df, "pfr", "{:.3f}", ds, "PFR against B0 at θ = 0.3 (% of test rows):")
    L += _grid_table(df, "train_errors", "{:.1f}", ds, "Training errors:")
    L += _grid_table(df, "macro_f1", "{:.2f}", ds, "Test macro-F1 (%):")
    sr = repair_flips(repair, seeds)
    if len(sr):
        g = sr.groupby(["dataset", "model"])[["accuracy", "macro_f1", "boxes", "changed_pct", "pfr",
                                              "nfr"]].mean()
        L += ["Self-repair, from `results/repair_theta/runs.csv` (means over the same seeds; flips "
              "against that θ's B0):", "", "```", g.round(3).to_string(), "```", ""]
        bm = boxes_to_match(df, repair)
        L += ["Boxes needed to match repair: the fewest mean boxes at which B0's mean accuracy "
              "reaches SR-0.3's mean accuracy (blank: no θ in the grid reaches it):", "",
              "| Dataset | SR-0.3 accuracy | θ | B0 accuracy | Boxes | Boxes ÷ boxes at θ = 0.3 |",
              "|---|---|---|---|---|---|"]
        for _, r in bm.iterrows():
            if np.isnan(r.theta):
                L.append(f"| {r.dataset} | {r.sr_accuracy:.2f} | | | | |")
            else:
                L.append(f"| {r.dataset} | {r.sr_accuracy:.2f} | {r.theta:g} | {r.b0_accuracy:.2f} | "
                         f"{r.boxes:.1f} | {r['boxes_vs_0.3']:.2f} |")
        L.append("")
    val = df[df.is_val]
    L += ["θ chosen by validation (count over seeds):", ""]
    L += [f"- {d}: " + ", ".join(f"{k:g} ×{v}" for k, v in val[val.dataset == d].theta.value_counts()
                                 .sort_index().items()) for d in ds]
    secs = df.groupby("dataset")[["fit_s", "predict_s"]].sum().reindex(ds) / len(seeds)
    if "task_s" in df:
        secs["task_s"] = df.groupby(["dataset", "seed"]).task_s.first().groupby("dataset").mean()
    L += ["", "Seconds per seed (means; fit and test prediction summed over the 7 fits; task: the "
          "whole task, with the training-row predictions, without the tuning; tasks shared the "
          "CPU):", "", "```", secs.round(2).to_string(), "```", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run O: re-tune theta or repair?")
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
    print(summarise(pd.read_csv(csv, float_precision="round_trip"), out))


if __name__ == "__main__":
    main()
