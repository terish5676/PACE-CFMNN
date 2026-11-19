"""Run H: a smarter order of training errors for self-repair.

Rules: results/repair_order/PREREGISTRATION.md. Runbook: research_audit/HOME_PC_RUNBOOK.md, run H.

Run A's split and SR arm (RQ3's split, theta = 0.3, the volume cut, the default gate 0.5 / 1.0
against the last model, box cap 1.1 times the start, gated on the training rows, the cached
gate). Every arm starts from the same fitted model B0 and the same seeded random order of
training errors; only the order in which errors are tried, and how many, changes
(`run_self_repair(..., order=...)`):

* random-300, random-1000, random-all: today's seeded random order (random-300 is today's SR);
* random-300-real (amendment 1): the same order, but tries on rows that are already right, or
  not inside the winning box, do not count, so 300 edits reach the gate;
* box-300, box-1000: errors grouped by the wrong-class box that wins them, biggest group first;
* greedy-300, greedy-1000: each error scored by a trial edit (training rows fixed minus rows
  broken, through the cached gate), the best tried first, re-scored lazily.

"300" is RQ3's default budget, min(300, pool / 4): 273 on Page Blocks, 300 elsewhere.

Primary: greedy-300 vs random-300, test macro-F1, paired two-sided Wilcoxon over seeds, Holm
across the 6 datasets. Amendment 1 (before the run) adds greedy-300 vs random-300-real, and two
efficiency tests of greedy-300 vs random-all, on seconds and on edits that reached the gate.

    python -B scripts/run_repair_order.py --jobs 12
    python -B scripts/run_repair_order.py --jobs 12 --resume
    python -B scripts/run_repair_order.py --summary-only
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
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_readable_baselines import compare, tally  # noqa: E402
from run_refinement import MAX_BUDGET, N_ROUNDS, split  # noqa: E402
from run_repair_sensitivity import use_fast_gate  # noqa: E402

OUT = ROOT / "results" / "repair_order"
DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = list(range(130, 140))
# (arm, order, budget, count_skips); budget "default" is RQ3's min(300, pool // 4), "all" every
# training error; count_skips=False counts only the tries that reach the gate (amendment 1)
ARMS = [("random-300", "random", "default", True),
        ("random-1000", "random", 1000, True),
        ("random-all", "random", "all", True),
        ("random-300-real", "random", "default", False),
        ("box-300", "box", "default", True),
        ("box-1000", "box", 1000, True),
        ("greedy-300", "greedy", "default", True),
        ("greedy-1000", "greedy", 1000, True)]
MODELS = ["B0"] + [a[0] for a in ARMS]
PRIMARY = ("greedy-300", "random-300")
SECONDARY = [("box-300", "random-300"), ("greedy-300", "random-all"), ("greedy-1000", "random-1000"),
             ("box-1000", "random-1000"), ("greedy-300", "box-300")]
# Amendment 1, written before the run (PREREGISTRATION.md, "Amendment 1").
AMENDED = [("greedy-300", "random-300-real")]
# (arm, against, column, word if the first arm's mean is higher, word if lower); lower is less work
EFFICIENCY = [("greedy-300", "random-all", "seconds", "slower", "faster"),
              ("greedy-300", "random-all", "edits_gated", "more", "fewer")]
REASONS = {"macro-F1": "rej_macro_f1", "class recall": "rej_class_recall", "box cap": "rej_box_cap",
           "already right": "skip_already_right", "not inside the winning box": "skip_not_inside"}
# Slowest first: this only orders the tasks.
COST = {"skin": 6, "letter": 5, "magic": 4, "landsat": 3, "page_blocks": 2, "thyroid": 1}
COLUMNS = (["dataset", "seed", "arm", "order", "budget", "gate_impl", "n_train", "n_pool", "n_test",
            "train_errors_start", "boxes_start", "boxes_end", "train_error_start", "train_error_end",
            "test_accuracy", "test_macro_f1", "test_min_class_recall", "test_balanced_accuracy",
            "edits_logged", "edits_gated", "edits_accepted"] + list(REASONS.values())
           + ["churn_changed", "churn_fixed", "churn_broken", "trial_edits", "seconds"])
METR = {"accuracy": "test_accuracy", "macro_f1": "test_macro_f1",
        "min_class_recall": "test_min_class_recall", "balanced_accuracy": "test_balanced_accuracy"}


def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    if not use_fast_gate():
        raise RuntimeError("run E's summary does not say PASS: the cached gate may not be used")
    P = split(name, seed)
    (Xtr, ytr), (Xp, _), (Xt, yt) = P["train"], P["pool"], P["test"]
    model = CFMNN().fit(Xtr, ytr)
    p0 = np.asarray(model.predict(Xt))
    wrong0 = np.asarray(model.predict(Xtr)) != ytr
    base = {c: np.nan for c in COLUMNS}
    base.update({"dataset": name, "seed": seed, "gate_impl": "cached", "n_train": len(ytr),
                 "n_pool": len(Xp), "n_test": len(yt), "train_errors_start": int(wrong0.sum()),
                 "boxes_start": model.n_boxes_, "train_error_start": 100 * float(wrong0.mean())})

    def record(arm, order, budget, m, secs, edits=None, rounds=None):
        pred = np.asarray(m.predict(Xt))
        met = classification_metrics(yt, pred)
        churn = prediction_churn(yt, p0, pred)
        rec = {**base, "arm": arm, "order": order, "budget": budget, "boxes_end": m.n_boxes_,
               "train_error_end": 100 * float(np.mean(m.predict(Xtr) != ytr)),
               **{col: met[k] for k, col in METR.items()},
               "churn_changed": churn["changed"], "churn_fixed": churn["fixed"],
               "churn_broken": churn["broken"], "seconds": secs}
        if edits is not None:
            rec.update({"edits_logged": len(edits),
                        "edits_gated": sum(e["edit"] == "shrink" for e in edits),
                        "edits_accepted": sum(bool(e["accepted"]) for e in edits),
                        **{col: sum(e["reason"] == r for e in edits) for r, col in REASONS.items()},
                        "trial_edits": rounds[-1].get("trial_edits", 0)})
        print(f"  {name} seed {seed} {arm}: macro-F1 {rec['test_macro_f1']:.2f}, boxes "
              f"{rec['boxes_end']}, accepted {rec['edits_accepted']}, {secs:.0f}s", flush=True)
        return rec

    rows = [record("B0", "", 0, model, 0.0)]
    default_budget = min(MAX_BUDGET, len(Xp) // 4)
    for arm, order, budget, count_skips in ARMS:
        n = default_budget if budget == "default" else len(ytr) if budget == "all" else budget
        t0 = time.perf_counter()
        m, rounds, edits = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=n,
                                           n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False,
                                           fast_gate=True, order=order, count_skips=count_skips)
        rows.append(record(arm, order, n, m, time.perf_counter() - t0, edits, rounds))
    return rows


# ------------------------------------------------------------------ summary
def comparisons(df: pd.DataFrame, col: str = "test_macro_f1") -> dict:
    """Primary, secondary and amendment-1 pairs: paired two-sided Wilcoxon over seeds, Holm
    across datasets."""
    d = df.rename(columns={"arm": "model"})
    return {(a, b): compare(d, a, b, col) for a, b in [PRIMARY] + SECONDARY + AMENDED}


def efficiency(df: pd.DataFrame) -> dict:
    """Amendment 1's efficiency tests, the same test; verdicts say which arm did more work."""
    d = df.rename(columns={"arm": "model"})
    out = {}
    for a, b, col, higher, lower in EFFICIENCY:
        r = compare(d, a, b, col)
        if not r.empty:
            r["verdict"] = r.verdict.map({"better": higher, "worse": lower, "level": "level"})
        out[(a, b, col)] = r
    return out


def reading(primary: str, real: str) -> str:
    """How amendment 1 reads the primary verdict on one dataset (against random-300-real)."""
    if primary == "better":
        return {"better": "a better order", "level": "the gain comes from not wasting tries",
                "worse": "the gain comes from not wasting tries (worse than real tries)"}[real]
    return f"{primary} (against real tries: {real})"


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    models = [m for m in MODELS if m in set(df.arm)]
    res = comparisons(df)
    L = ["# Run H: the order of training errors in self-repair", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Run A's split and SR arm, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}), cached gate. Each comparison: paired two-sided "
         "Wilcoxon over seeds, Holm across datasets; better / level / worse for the first arm. "
         "Written by `scripts/run_repair_order.py`.", ""]
    for i, (pair, r) in enumerate(res.items()):
        if r.empty:
            continue
        if i == 0:
            L += ["## Primary: greedy-300 vs random-300, test macro-F1", ""]
        elif i == 1:
            L += ["## Secondary comparisons, test macro-F1", ""]
        elif pair == AMENDED[0]:
            L += ["## Amendment 1: greedy-300 vs random-300-real, test macro-F1", ""]
        L += [f"**{pair[0]} vs {pair[1]}**: {tally(r)}", "",
              "| Dataset | Mean difference (points) | p | Holm p | Verdict |", "|---|---|---|---|---|"]
        L += [f"| {x.dataset} | {x['diff']:+.2f} | {x.p:.4f} | {x.p_holm:.4f} | {x.verdict} |"
              for _, x in r.iterrows()]
        L.append("")
    prim, real = res[PRIMARY], res[AMENDED[0]]
    if not prim.empty and not real.empty:
        both = prim.merge(real, on="dataset", suffixes=("", "_real"))
        L += ["## Amendment 1: how to read the primary verdict", "",
              "| Dataset | Primary | Against random-300-real | Reading |", "|---|---|---|---|"]
        L += [f"| {x.dataset} | {x.verdict} | {x.verdict_real} | {reading(x.verdict, x.verdict_real)} |"
              for _, x in both.iterrows()]
        L.append("")
    for (a, b, col), r in efficiency(df).items():
        if r.empty:
            continue
        c = r.verdict.value_counts()
        L += [f"## Amendment 1, efficiency: {a} vs {b}, {col}", "",
              f"**{a} vs {b}** ({col}): " + ", ".join(f"{k} {v}" for k, v in c.items()), "",
              "| Dataset | Mean difference | p | Holm p | Verdict |", "|---|---|---|---|---|"]
        L += [f"| {x.dataset} | {x['diff']:+.1f} | {x.p:.4f} | {x.p_holm:.4f} | {x.verdict} |"
              for _, x in r.iterrows()]
        L.append("")
    for title, col, fmt in (("Test macro-F1 (%)", "test_macro_f1", "{:.2f}"),
                            ("Test accuracy (%)", "test_accuracy", "{:.2f}"),
                            ("Lowest class recall (%)", "test_min_class_recall", "{:.2f}"),
                            ("Training error at the end (%)", "train_error_end", "{:.2f}"),
                            ("Edits that reached the gate", "edits_gated", "{:.0f}"),
                            ("Edits accepted", "edits_accepted", "{:.0f}"),
                            ("Tries skipped: already right", "skip_already_right", "{:.0f}"),
                            ("Test predictions fixed against B0", "churn_fixed", "{:.1f}"),
                            ("Test predictions broken against B0", "churn_broken", "{:.1f}"),
                            ("Trial edits scored (greedy only)", "trial_edits", "{:.0f}"),
                            ("Seconds", "seconds", "{:.1f}")):
        piv = df.pivot_table(index="dataset", columns="arm", values=col).reindex(index=ds, columns=models)
        L += [f"## {title}, mean over seeds", "", "| Dataset | " + " | ".join(models) + " |",
              "|---|" + "---|" * len(models)]
        L += [f"| {d} | " + " | ".join("" if np.isnan(v) else fmt.format(v) for v in r) + " |"
              for d, r in piv.iterrows()]
        L.append("")
    start = df[df.arm == "B0"].groupby("dataset").train_errors_start.mean().reindex(ds)
    L += ["## Training errors at the start (mean over seeds)", "",
          "| " + " | ".join(ds) + " |", "|" + "---|" * len(ds),
          "| " + " | ".join(f"{v:.0f}" for v in start) + " |", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run H: the order of training errors in self-repair")
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
