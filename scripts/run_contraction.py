"""Run F: self-repair against contraction (planned in advance).

Rules: results/contraction/PREREGISTRATION.md. CFMNN trains without Simpson's contraction, so
boxes of different classes overlap, and every training error is a tie inside such an overlap.
Self-repair (RQ3) shrinks the wrong-class box at up to 300 training errors and keeps an edit
only if a gate passes. The standard remedy for overlap in fuzzy min-max networks is the overlap
test and contraction (Simpson 1992; GFMM). This run puts them side by side on the same rows.

Arms, for each dataset and seed, on RQ3's split (``run_refinement.split``: 50/20/10/10/10, the
training rows in the seed's order, min-max scaling fitted on them). Only the training rows and
the test rows are used:

* B0: CFMNN, theta = 0.3, gamma = 4 (the starting model).
* SR: B0, then self-repair as in RQ3 (``active.run_self_repair``: up to 300 training errors,
  the smallest cut, 10 rounds, gated on the training rows).
* SC-all: B0, then Simpson's contraction on every overlap between boxes of different classes
  (``contraction.contract_all``).
* SC-err: B0, then Simpson's contraction only at training errors, the same up-to-300 rows in
  the same order as SR, with no gate (``contraction.contract_at_errors``).
* GC-all, GC-err: as SC-all and SC-err, with GFMM's overlap test and contraction
  (hyperbox-brain's ``overlap_resolving_num_data``) in place of Simpson's. Simpson's test uses
  strict inequalities, so it misses boxes that share a face; GFMM's counts them.
* GFMM: hyperbox-brain's ``OnlineGFMM`` at theta = 0.3, gamma = 4, trained on the same rows in
  the same order. It runs its own overlap test and contraction while it trains.

    python -B scripts/run_contraction.py --jobs 12
    python -B scripts/run_contraction.py --jobs 12 --resume     # continue after a stop
    python -B scripts/run_contraction.py --summary-only
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

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.audit import prediction_churn  # noqa: E402
from pace_cfmnn.contraction import (contract_all, contract_at_errors,  # noqa: E402
                                    rows_in_wrong_class_box, rows_without_box)
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_refinement import split  # noqa: E402
from run_refinement_v2 import DATASETS, NAMES, mean_table, test_endpoint  # noqa: E402

SEEDS = list(range(120, 130))
THETA, GAMMA = 0.3, 4.0
# RQ3's self-repair (scripts/run_refinement_v2.py, arm SR): budget, rounds, the smallest cut.
BUDGET, N_ROUNDS, CHOOSER = 300, 10, "volume"
ARMS = ["B0", "SR", "SC-all", "SC-err", "GC-all", "GC-err", "GFMM"]
OUT = ROOT / "results" / "contraction"
# Slowest first (self-repair's seconds per seed in RQ3). This only orders the tasks.
COST = {"letter": 6, "magic": 5, "landsat": 4, "skin": 3, "page_blocks": 2, "thyroid": 1}


def make_gfmm():
    """GFMM as in scripts/run_readable_baselines.py: one pass at theta, no shrinking loop."""
    from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
    return OnlineGFMM(theta=THETA, theta_min=THETA, gamma=GAMMA)


# ------------------------------------------------------------------ one dataset and seed
def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    P = split(name, seed)
    (Xtr, ytr), (Xt, yt) = P["train"], P["test"]
    t0 = time.perf_counter()
    b0 = CFMNN(theta=THETA, gamma=GAMMA).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    p0 = np.asarray(b0.predict(Xt))
    err0 = 100 * float(np.mean(b0.predict(Xtr) != ytr))
    rows = []

    def record(arm, pred, pred_train, secs, model=None, boxes_end=None, **extra):
        m = classification_metrics(yt, pred)
        churn = prediction_churn(yt, p0, pred)
        rec = {"dataset": name, "seed": seed, "arm": arm, "n_train": len(ytr), "n_test": len(yt),
               "test_macro_f1": m["macro_f1"], "test_accuracy": m["accuracy"],
               "test_min_class_recall": m["min_class_recall"],
               "boxes_start": b0.n_boxes_ if model is not None else np.nan,
               "boxes_end": model.n_boxes_ if model is not None else boxes_end,
               "rows_evicted": rows_without_box(model, len(ytr)) if model is not None else np.nan,
               "rows_evicted_pct": np.nan,
               "train_error_start": err0 if model is not None else np.nan,
               "train_error_end": 100 * float(np.mean(np.asarray(pred_train) != ytr)),
               "train_rows_in_wrong_box": (rows_in_wrong_class_box(model, Xtr, ytr)
                                           if model is not None else np.nan),
               "test_changed": churn["changed"], "test_fixed": churn["fixed"],
               "test_broken": churn["broken"],
               "test_fixed_pct": 100 * churn["fixed"] / len(yt),
               "test_broken_pct": 100 * churn["broken"] / len(yt),
               "edits_tried": np.nan, "edits_kept": np.nan, "case_1": np.nan, "case_2": np.nan,
               "case_3": np.nan, "case_4": np.nan, "sweeps": np.nan, "boxes_removed": np.nan,
               "skipped_already_right": np.nan, "skipped_no_overlap": np.nan,
               "skipped_other": np.nan, "seconds": secs}
        if model is not None:
            rec["rows_evicted_pct"] = 100 * rec["rows_evicted"] / len(ytr)
        rec.update(extra)
        rows.append(rec)

    record("B0", p0, b0.predict(Xtr), fit_s, model=b0)

    t0 = time.perf_counter()
    sr, _, edits = run_self_repair(b0, Xtr, ytr, Xtr, ytr, chooser=CHOOSER, budget=BUDGET,
                                   n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False)
    secs = time.perf_counter() - t0
    kept = [e for e in edits if e["accepted"]]
    record("SR", sr.predict(Xt), sr.predict(Xtr), secs, model=sr, edits_tried=len(edits),
           edits_kept=len(kept), boxes_removed=sum(bool(e["shrink_removed"]) for e in kept),
           skipped_already_right=sum(e["reason"] == "already right" for e in edits),
           skipped_other=sum(e["reason"] == "not inside the winning box" for e in edits))

    def contraction_arm(arm, fn):
        t0 = time.perf_counter()
        m, st, _ = fn()
        secs = time.perf_counter() - t0
        if st["rows_evicted"] != rows_without_box(m, len(ytr)):
            raise RuntimeError(f"{name} seed {seed} {arm}: the evicted rows do not add up")
        record(arm, m.predict(Xt), m.predict(Xtr), secs, model=m,
               edits_tried=st.get("errors_tried", st["contractions"]),
               edits_kept=st["contractions"], boxes_removed=st["boxes_removed"],
               **{k: st[k] for k in ("case_1", "case_2", "case_3", "case_4")},
               **{k: st[k] for k in ("sweeps", "skipped_already_right", "skipped_no_overlap",
                                     "skipped_other") if k in st})

    contraction_arm("SC-all", lambda: contract_all(b0, Xtr))
    contraction_arm("SC-err", lambda: contract_at_errors(b0, Xtr, ytr, budget=BUDGET, seed=seed))
    contraction_arm("GC-all", lambda: contract_all(b0, Xtr, rule="gfmm"))
    contraction_arm("GC-err", lambda: contract_at_errors(b0, Xtr, ytr, budget=BUDGET, seed=seed,
                                                         rule="gfmm"))

    t0 = time.perf_counter()
    g = make_gfmm().fit(Xtr, ytr)
    secs = time.perf_counter() - t0
    record("GFMM", np.asarray(g.predict(Xt)), np.asarray(g.predict(Xtr)), secs,
           boxes_end=len(np.asarray(g.V)))
    print(f"{name} seed {seed}: " + ", ".join(
        f"{r['arm']} {r['test_macro_f1']:.1f}" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------ summary
# (code, question, value, first arm, second arm); every test is two-sided.
ENDPOINTS = [
    ("P1 (primary)", "Self-repair or Simpson's contraction on every overlap: which is more accurate?",
     "test_macro_f1", "SR", "SC-all"),
    ("S1", "Self-repair or ungated contraction at the same training errors?",
     "test_macro_f1", "SR", "SC-err"),
    ("S2a", "Self-repair or GFMM's contraction rule on every overlap?", "test_macro_f1", "SR", "GC-all"),
    ("S2b", "Self-repair or GFMM's contraction rule at the same training errors?",
     "test_macro_f1", "SR", "GC-err"),
    ("S3", "Self-repair or GFMM, which contracts while it trains?", "test_macro_f1", "SR", "GFMM"),
    ("S4a", "Does self-repair help on these seeds?", "test_macro_f1", "SR", "B0"),
    ("S4b", "Does contraction on every overlap help?", "test_macro_f1", "SC-all", "B0"),
    ("S4c", "Does contraction at 300 training errors help?", "test_macro_f1", "SC-err", "B0"),
    ("S4d", "Does GFMM's rule on every overlap help?", "test_macro_f1", "GC-all", "B0"),
    ("S4f", "Does GFMM's rule at 300 training errors help?", "test_macro_f1", "GC-err", "B0"),
    ("S4e", "Is GFMM more accurate than the starting model?", "test_macro_f1", "GFMM", "B0"),
    ("S5", "Lowest class recall: self-repair or contraction on every overlap?",
     "test_min_class_recall", "SR", "SC-all"),
    ("S6", "Boxes at the end: self-repair or contraction on every overlap?",
     "boxes_end", "SR", "SC-all"),
    ("S7", "Test predictions broken (right to wrong): self-repair or contraction on every overlap?",
     "test_broken_pct", "SR", "SC-all"),
    ("S8", "Training rows evicted from their boxes: self-repair or contraction on every overlap?",
     "rows_evicted_pct", "SR", "SC-all"),
    ("S9", "P1 by accuracy", "test_accuracy", "SR", "SC-all"),
]
TABLES = [("Test macro-F1 (%)", "test_macro_f1", "{:.1f}"),
          ("Test accuracy (%)", "test_accuracy", "{:.1f}"),
          ("Lowest class recall (%)", "test_min_class_recall", "{:.1f}"),
          ("Boxes at the end", "boxes_end", "{:.1f}"),
          ("Training rows evicted from their boxes (% of training rows)", "rows_evicted_pct", "{:.2f}"),
          ("Training rows evicted from their boxes (count)", "rows_evicted", "{:,.1f}"),
          ("Training rows inside a box of another class, at the end (count)",
           "train_rows_in_wrong_box", "{:,.1f}"),
          ("Training error at the end (%)", "train_error_end", "{:.2f}"),
          ("Test predictions fixed, wrong → right, against B0 (% of test rows)", "test_fixed_pct", "{:.2f}"),
          ("Test predictions broken, right → wrong, against B0 (% of test rows)", "test_broken_pct", "{:.2f}"),
          ("Edits or contractions kept", "edits_kept", "{:,.1f}"),
          ("Boxes removed because no training row was left in them", "boxes_removed", "{:.1f}"),
          ("Sweeps until no overlap was found (SC-all, GC-all)", "sweeps", "{:.1f}"),
          ("Training errors skipped: the overlap test found no overlap", "skipped_no_overlap", "{:.1f}"),
          ("Seconds (fit for B0 and GFMM; the repair step for the others)", "seconds", "{:.1f}")]


def verdict(diff: float, p_holm: float) -> str:
    """For the first arm's value: higher, lower, or no significant difference."""
    if not p_holm < 0.05:
        return "no significant difference"
    return "higher" if diff > 0 else "lower"


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    arms = [a for a in ARMS if a in set(df.arm)]
    L = ["# Run F: self-repair against contraction", "",
         f"Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). RQ3's split, seeds {seeds[0]}–{seeds[-1]} "
         f"({len(seeds)}), {len(ds)} datasets. Each comparison: paired two-sided Wilcoxon over seeds, "
         "Holm across the datasets; first arm − second arm. `*` = Holm-adjusted p < 0.05. Written by "
         "`scripts/run_contraction.py`.", "",
         "Arms: B0 = the starting CFMNN (θ = 0.3, γ = 4); SR = self-repair; SC-all = Simpson's "
         "contraction on every overlap; SC-err = Simpson's contraction at the same up-to-300 "
         "training errors as SR, no gate; GC-all, GC-err = the same with GFMM's overlap test and "
         "contraction rule; GFMM = online GFMM, which contracts while it trains.", "",
         "## Comparisons", "", "Mean difference over seeds (first − second):", "",
         "| Code | Question | Value | First − second | " + " | ".join(NAMES[d] for d in ds)
         + " | Higher / no significant difference / lower |",
         "|---|---|---|---|" + "---|" * (len(ds) + 1)]
    detail = []
    for code, q, value, a, b in ENDPOINTS:
        if not {a, b} <= set(arms):
            continue
        t = test_endpoint(df, value, a, b, "two-sided").set_index("dataset")
        v = [verdict(t.loc[d, "diff"], t.loc[d, "p_holm"]) for d in t.index]
        L.append(f"| {code} | {q} | {value} | {a} − {b} | " + " | ".join(
            f"{t.loc[d, 'diff']:+.2f}{'*' if t.loc[d, 'p_holm'] < 0.05 else ''}" if d in t.index
            else "–" for d in ds) + f" | {v.count('higher')} / "
            f"{v.count('no significant difference')} / {v.count('lower')} |")
        detail.append(f"| {code} | " + " | ".join(
            f"{t.loc[d, 'wins']}–{t.loc[d, 'losses']} (n={t.loc[d, 'n']}), p={t.loc[d, 'p']:.3g}, "
            f"Holm {t.loc[d, 'p_holm']:.3g}" if d in t.index else "not tested" for d in ds) + " |")
    L += ["", "Wins–losses over seeds (n = seeds tested), raw p and Holm-adjusted p:", "",
          "| Code | " + " | ".join(NAMES[d] for d in ds) + " |", "|---|" + "---|" * len(ds)]
    L += detail + ["", "Means over seeds follow. A blank cell means the value does not apply to "
                   "that arm.", ""]
    for title, value, fmt in TABLES:
        L += [f"## {title}", ""] + mean_table(df, arms, value, fmt)
    cases = df[df.arm.isin(["SC-all", "SC-err"])]
    if len(cases):
        tot = cases.groupby("arm")[["case_1", "case_2", "case_3", "case_4"]].sum().reindex(
            [a for a in arms if a.startswith("SC")])
        tot = tot.fillna(0)
        L += ["## Simpson's cases used (contractions, summed over datasets and seeds)", "",
              "Cases 1 and 2: the boxes overlap partly on the contracted feature. Cases 3 and 4: "
              "one box lies inside the other there.", "",
              "| Arm | Case 1 | Case 2 | Case 3 | Case 4 |", "|---|---|---|---|---|"]
        L += [f"| {a} | " + " | ".join(f"{int(x):,}" for x in r) + " |" for a, r in tot.iterrows()]
        L.append("")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run F: self-repair against contraction")
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
