"""Run T: how long each repair setup takes on one machine (a descriptive timing check).

Made after the RQ3 results, for the RQ3 time table; it is not pre-registered and tests nothing.
Runbook: research_audit/HOME_PC_RUNBOOK.md, run T.

For seed 50 of RQ3's split, on the 6 RQ3 datasets, one task at a time in one process, it times:

* train CFMNN: ``CFMNN().fit`` on the training rows (the median of 3 fits);
* SR, old gate: ``run_self_repair(..., fast_gate=False)``, as RQ3 ran it;
* SR, cached gate: the same with ``fast_gate=True``;
* AL: ``run_refinement`` with the order-ensemble selector and the smallest-cut shrink, gated on
  the acceptance rows (the old gate: AL has no cached gate);
* SR+AL: SR with the cached gate, then the same AL; the total and the AL stage alone.

Every call is the one in ``scripts/run_refinement_v2.py``, without the certificate audit, so
each row times only the work itself. Each setup's test macro-F1 must equal the committed RQ3
value (results/refinement_v2_confirmatory/runs.csv), and both SR gates must give the same model;
the summary says **Verdict: PASS** only if all of them do.

    python -B scripts/time_repair_arms.py
    python -B scripts/time_repair_arms.py --resume
    python -B scripts/time_repair_arms.py --summary-only
"""

from __future__ import annotations

import argparse
import os
import platform
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
from pace_cfmnn.active import LabelOracle, run_refinement, run_self_repair  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from check_fast_gate import same_model  # noqa: E402
from run_refinement import MAX_BUDGET, N_ROUNDS, split  # noqa: E402
from run_refinement_v2 import SELECTOR, VOL  # noqa: E402

OUT = ROOT / "results" / "repair_timing"
COMMITTED = ROOT / "results" / "refinement_v2_confirmatory" / "runs.csv"
DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEED = 50
N_FITS = 3
F1_TOL = 1e-12
# (setup, the committed RQ3 arm its test macro-F1 must equal)
SETUPS = [("Train CFMNN", "B0"), ("SR, old gate", "SR"), ("SR, cached gate", "SR"),
          ("AL", "AL"), ("SR+AL", "SR+AL")]
COLUMNS = ["dataset", "seed", "setup", "seconds", "stage_seconds", "labels", "test_macro_f1",
           "committed_macro_f1", "matches_committed", "machine", "python", "cpu_count"]
_REF = {}


def committed(name: str, seed: int) -> dict:
    """RQ3's test macro-F1 for each arm, one dataset and seed (read only)."""
    if "runs" not in _REF:
        _REF["runs"] = pd.read_csv(COMMITTED)
    r = _REF["runs"]
    r = r[(r.dataset == name) & (r.seed == seed)]
    return dict(zip(r.arm, r.test_macro_f1))


def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    P = split(name, seed)
    (Xtr, ytr), (Xp, yp), (Xi, _), (Xa, ya), (Xt, yt) = (P[r] for r in
                                                       ("train", "pool", "inner", "accept", "test"))
    ref = committed(name, seed)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    machine = {"machine": platform.processor() or platform.machine(),
               "python": platform.python_version(), "cpu_count": os.cpu_count()}
    rows = []

    def record(setup, arm, model, secs, labels, stage=np.nan):
        f1 = classification_metrics(yt, model.predict(Xt))["macro_f1"]
        ok = arm in ref and abs(f1 - ref[arm]) <= F1_TOL
        rows.append({"dataset": name, "seed": seed, "setup": setup, "seconds": secs,
                     "stage_seconds": stage, "labels": labels, "test_macro_f1": f1,
                     "committed_macro_f1": ref.get(arm, np.nan), "matches_committed": bool(ok),
                     **machine})
        print(f"  {name} seed {seed} {setup}: {secs:.2f}s, macro-F1 {f1:.2f} "
              f"({'matches' if ok else 'DIFFERS FROM'} RQ3)", flush=True)

    fits = []
    for _ in range(N_FITS):
        t0 = time.perf_counter()
        model = CFMNN().fit(Xtr, ytr)
        fits.append(time.perf_counter() - t0)
    record("Train CFMNN", "B0", model, float(np.median(fits)), 0)

    def sr(fast):
        t0 = time.perf_counter()
        m, _, _ = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=budget,
                                  n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False,
                                  fast_gate=fast)
        return m, time.perf_counter() - t0

    sr_old, s_old = sr(False)
    record("SR, old gate", "SR", sr_old, s_old, 0)
    sr_new, s_new = sr(True)
    record("SR, cached gate", "SR", sr_new, s_new, 0)
    if not same_model(sr_old, sr_new):
        rows[-1]["matches_committed"] = False
        print(f"  {name} seed {seed}: the two SR gates gave different models", flush=True)

    def al(start):
        oracle = LabelOracle(yp)
        t0 = time.perf_counter()
        m, _, _ = run_refinement(start, Xtr, ytr, Xp, oracle, Xi, Xa, ya, SELECTOR, VOL,
                                 budget=budget, n_rounds=N_ROUNDS, seed=seed)
        return m, time.perf_counter() - t0, len(oracle.revealed) + len(Xa)

    m, secs, labels = al(model)
    record("AL", "AL", m, secs, labels)
    m, stage, labels = al(sr_new)
    record("SR+AL", "SR+AL", m, s_new + stage, labels, stage)
    return rows


def summarise(df: pd.DataFrame, out: Path) -> str:
    ds = [d for d in DATASETS if d in set(df.dataset)]
    setups = [s for s, _ in SETUPS]
    bad = df[~df.matches_committed.astype(bool)]
    complete = len(df) == len(DATASETS) * len(SETUPS)
    verdict = "PASS" if bad.empty and complete else "FAIL"
    m = df.iloc[0]
    L = ["# Run T: time of each repair setup on one machine (descriptive check)", "",
         "Made after the RQ3 results; not pre-registered, nothing is tested. Seed "
         f"{SEED}, RQ3's split, one task at a time in one process, no certificate audit. "
         "Written by `scripts/time_repair_arms.py`.", "",
         f"Machine: {m.machine}, {int(m.cpu_count)} logical CPUs, Python {m.python}.", "",
         f"**Verdict: {verdict}**" + ("" if verdict == "PASS" else
                                       f" ({len(bad)} setups differ from RQ3"
                                       + ("" if complete else ", or some are missing") + ")"), ""]
    piv = df.pivot_table(index="setup", columns="dataset", values="seconds").reindex(
        index=setups, columns=ds)
    stage = df[df.setup == "SR+AL"].set_index("dataset").stage_seconds
    L += ["## Seconds", "", "SR+AL is SR with the cached gate, then AL; the AL stage alone is in "
          "brackets. AL has no cached gate.", "",
          "| Setup | " + " | ".join(ds) + " |", "|---|" + "---|" * len(ds)]
    for s in setups:
        cells = []
        for d in ds:
            v = piv.loc[s, d]
            cell = "" if np.isnan(v) else f"{v:.2f}"
            if s == "SR+AL" and d in stage.index and not np.isnan(v):
                cell += f" ({stage[d]:.2f})"
            cells.append(cell)
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    lab = df[df.setup == "AL"].set_index("dataset").labels.reindex(ds)
    L += ["", "Labels used beyond the training rows (AL and SR+AL; SR uses none): "
          + ", ".join(f"{d} {int(v):,}" for d, v in lab.items() if not np.isnan(v)) + ".", ""]
    base = piv.loc["SR, cached gate"]
    L += ["## Times as a multiple of SR with the cached gate", "",
          "| Setup | " + " | ".join(ds) + " |", "|---|" + "---|" * len(ds)]
    for s in ("SR, old gate", "AL", "SR+AL"):
        L.append(f"| {s} | " + " | ".join("" if np.isnan(piv.loc[s, d] / base[d]) else
                                          f"{piv.loc[s, d] / base[d]:.1f}×" for d in ds) + " |")
    L += ["", "## Test macro-F1 against RQ3 (seed 50)", "",
          "| Dataset | Setup | Macro-F1 | RQ3 | Same |", "|---|---|---|---|---|"]
    L += [f"| {r.dataset} | {r.setup} | {r.test_macro_f1:.4f} | {r.committed_macro_f1:.4f} | "
          f"{'yes' if r.matches_committed else 'NO'} |" for _, r in df.iterrows()]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run T: time of each repair setup (descriptive)")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(d, SEED) for d in a.datasets or DATASETS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, 1)   # one task at a time
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
