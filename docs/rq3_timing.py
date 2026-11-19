"""How long self-repair itself takes, and how much of it is the gate (descriptive check).

Made after the RQ3 run, for docs/RQ3_self_repair.md. It is not part of any
pre-registered study and tests nothing.

Why: the committed SR times (results/refinement_v2_confirmatory/runs.csv,
``time_s``) include the certificate audit, which runs on 2,000 test rows after
every accepted edit, and they were measured with 4 jobs sharing 4 cores. A
retrain's time there is a bare fit. This script times SR exactly as RQ3 calls
it (scripts/run_refinement_v2.py), with no audit hook, one dataset at a time.

For seed 50 of each RQ3 dataset it records the time to fit CFMNN, the time
of one prediction pass over the training rows, the SR time, and how much of
the SR time is spent in the gate (``Gate.measure``, wrapped here with a timer;
the library is not changed). SR's test macro-F1 must equal the committed value
for seed 50, or the script stops.

    python -B docs/rq3_timing.py
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, refine  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from run_refinement import MAX_BUDGET, N_ROUNDS, split  # noqa: E402

DATASETS = ["thyroid", "page_blocks", "landsat", "skin", "magic", "letter"]
SEED = 50
OUT = ROOT / "results" / "rq3_explainer_timing"
COMMITTED = ROOT / "results" / "refinement_v2_confirmatory" / "runs.csv"

# A timer around the gate's measurement (one prediction pass over the gate rows).
_GATE = {"seconds": 0.0, "calls": 0}
_measure = refine.Gate.measure


def _timed_measure(self, model):
    t = time.perf_counter()
    try:
        return _measure(self, model)
    finally:
        _GATE["seconds"] += time.perf_counter() - t
        _GATE["calls"] += 1


refine.Gate.measure = _timed_measure


def run_one(name: str, committed: float) -> dict:
    P = split(name, SEED)
    (Xtr, ytr), (Xp, _), (Xt, yt) = P["train"], P["pool"], P["test"]
    budget = min(MAX_BUDGET, len(Xp) // 4)
    t = time.perf_counter()
    model = CFMNN().fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    model.predict(Xtr)
    pass_s = time.perf_counter() - t
    _GATE.update(seconds=0.0, calls=0)
    t = time.perf_counter()
    sr, _, edits = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=budget,
                                   n_rounds=N_ROUNDS, seed=SEED, gate_labels_are_new=False)
    sr_s = time.perf_counter() - t
    f1 = classification_metrics(yt, sr.predict(Xt))["macro_f1"]
    if abs(f1 - committed) > 1e-9:
        raise SystemExit(f"{name}: SR macro-F1 {f1!r} differs from the committed {committed!r}")
    e = pd.DataFrame(edits)
    shrinks = e[e["edit"] == "shrink"]
    rec = {"dataset": name, "seed": SEED, "n_train": len(Xtr), "d": Xtr.shape[1],
           "boxes_start": model.n_boxes_, "boxes_end": sr.n_boxes_, "budget": budget,
           "errors_tried": len(e), "shrinks_tried": len(shrinks),
           "shrinks_accepted": int(shrinks["accepted"].sum()),
           "fit_s": fit_s, "train_pass_s": pass_s, "sr_s": sr_s,
           "gate_s": _GATE["seconds"], "gate_calls": _GATE["calls"],
           "gate_share": 100 * _GATE["seconds"] / sr_s if sr_s else np.nan,
           "sr_over_fit": sr_s / fit_s, "test_macro_f1": f1, "committed_macro_f1": committed}
    print(f"{name}: fit {fit_s:.2f}s, SR {sr_s:.1f}s, gate {rec['gate_share']:.0f}% of it "
          f"({_GATE['calls']} passes), macro-F1 matches", flush=True)
    return rec


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(COMMITTED)
    c = c[(c["arm"] == "SR") & (c["seed"] == SEED)].set_index("dataset")
    t0 = time.perf_counter()
    rows = [run_one(n, float(c.loc[n, "test_macro_f1"])) for n in DATASETS]
    df = pd.DataFrame(rows)
    df["committed_time_s"] = [float(c.loc[n, "time_s"]) for n in df["dataset"]]
    df.to_csv(OUT / "runs.csv", index=False)
    lines = [
        "# Self-repair timing, seed 50 (descriptive check, made after the RQ3 run)",
        "",
        "Written by `docs/rq3_timing.py`. Not part of any pre-registered study; nothing is tested.",
        "SR is called exactly as in `scripts/run_refinement_v2.py`, without the certificate audit,",
        "one dataset at a time. SR's test macro-F1 equals the committed value on every dataset.",
        "",
        f"Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}, "
        f"{platform.system()} {platform.release()}. Total {time.perf_counter() - t0:.0f} s.",
        "",
        "| Dataset | Training rows | Features | Boxes | Shrinks tried | Accepted | Fit (s) "
        "| One pass over training rows (s) | SR (s) | Gate share of SR (%) | SR ÷ fit "
        "| Committed SR time, with audit, 4 jobs (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(f"| {r.dataset} | {r.n_train:,} | {r.d} | {r.boxes_start} | {r.shrinks_tried} "
                     f"| {r.shrinks_accepted} | {r.fit_s:.2f} | {r.train_pass_s:.3f} | {r.sr_s:.1f} "
                     f"| {r.gate_share:.0f} | {r.sr_over_fit:.0f} | {r.committed_time_s:.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
