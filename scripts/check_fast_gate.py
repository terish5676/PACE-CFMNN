"""Run E: does the cached gate make exactly the standard gate's decisions?

Rules: results/fast_gate_check/PREREGISTRATION.md. Plan: research_audit/LIMITATIONS_PLAN.md, run E.

The cached gate (`refine.CachedGate`, used by `run_self_repair(..., fast_gate=True)`) keeps the
membership and distance tables of the gate rows and re-scores only the rows a one-box edit can
change. It must give the same accept/reject decisions, so the same repaired model, as the standard
gate. This script replays RQ3's SR arm exactly as `scripts/run_refinement_v2.py` ran it, with the
cached gate, and compares with the committed RQ3 results (`results/refinement_v2_confirmatory/`):

* each edit's `train_index`, `accepted`, `reason` and `x_correct_after`, against edits.csv.gz (arm SR);
* the final test macro-F1, against runs.csv (arm SR), to 1e-12.

On seed 50 only, the standard gate is also run on the same machine. Its edit log, model and test
predictions must equal the cached gate's, and the two times give the speed ratio.

    python -B scripts/check_fast_gate.py --jobs 12
    python -B scripts/check_fast_gate.py --jobs 12 --resume
    python -B scripts/check_fast_gate.py --summary-only
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
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_refinement import MAX_BUDGET, N_ROUNDS, split  # noqa: E402

REF = ROOT / "results" / "refinement_v2_confirmatory"
OUT = ROOT / "results" / "fast_gate_check"
DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = list(range(50, 60))
STANDARD_SEED = 50          # the seed that is also run with the standard gate
F1_TOL = 1e-12
EDIT_KEYS = ("train_index", "accepted", "reason", "x_correct_after")
# Slowest first: this only orders the tasks.
COST = {"skin": 6, "letter": 5, "magic": 4, "landsat": 3, "page_blocks": 2, "thyroid": 1}
_REF = {}


def committed(name: str, seed: int):
    """RQ3's SR-arm edit log and test macro-F1 for one dataset and seed (read only)."""
    if not _REF:
        _REF["edits"] = pd.read_csv(REF / "edits.csv.gz")
        _REF["runs"] = pd.read_csv(REF / "runs.csv")
    e, r = _REF["edits"], _REF["runs"]
    e = e[(e.dataset == name) & (e.seed == seed) & (e.arm == "SR")].reset_index(drop=True)
    r = r[(r.dataset == name) & (r.seed == seed) & (r.arm == "SR")]
    if len(r) != 1 or e.empty:
        raise RuntimeError(f"no committed SR rows for {name} seed {seed}")
    return e, float(r.iloc[0].test_macro_f1)


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def edit_mismatches(edits: list[dict], ref: pd.DataFrame) -> int:
    """How many edits differ from the committed log (a different length counts every extra edit)."""
    bad = abs(len(edits) - len(ref))
    for mine, (_, theirs) in zip(edits, ref.iterrows()):
        diff = False
        for key in EDIT_KEYS:
            a, b = mine.get(key, np.nan), theirs[key]
            if key == "reason":
                diff |= str("" if _missing(a) else a) != str("" if _missing(b) else b)
            elif _missing(a) or _missing(b):
                diff |= not (_missing(a) and _missing(b))
            else:
                diff |= bool(a) != bool(b) if key != "train_index" else int(a) != int(b)
        bad += diff
    return bad


def same_model(a: CFMNN, b: CFMNN) -> bool:
    return (all(np.array_equal(getattr(a, k), getattr(b, k))
                for k in ("V_", "W_", "sum_", "count_", "box_class_", "centroid_"))
            and len(a.members_) == len(b.members_)
            and all(np.array_equal(p, q) for p, q in zip(a.members_, b.members_)))


def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    P = split(name, seed)
    (Xtr, ytr), (Xp, _), (Xt, yt) = P["train"], P["pool"], P["test"]
    model = CFMNN().fit(Xtr, ytr)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    ref_edits, ref_f1 = committed(name, seed)

    def replay(fast_gate: bool):
        t0 = time.perf_counter()
        sr, _, edits = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=budget,
                                       n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False,
                                       fast_gate=fast_gate)
        return sr, edits, time.perf_counter() - t0

    sr, edits, secs = replay(True)
    f1 = classification_metrics(yt, sr.predict(Xt))["macro_f1"]
    row = {"dataset": name, "seed": seed, "budget": budget, "n_edits": len(edits),
           "n_edits_committed": len(ref_edits),
           "edits_accepted": sum(bool(e["accepted"]) for e in edits),
           "edit_mismatches": edit_mismatches(edits, ref_edits),
           "macro_f1": f1, "macro_f1_committed": ref_f1, "macro_f1_abs_diff": abs(f1 - ref_f1),
           "boxes_end": sr.n_boxes_, "cached_s": secs,
           "standard_run": False, "standard_equals_cached": np.nan, "standard_s": np.nan,
           "standard_over_cached": np.nan}
    row["matches_committed"] = bool(row["edit_mismatches"] == 0 and row["macro_f1_abs_diff"] <= F1_TOL)
    if seed == STANDARD_SEED:
        std, std_edits, std_secs = replay(False)
        same = bool(std_edits == edits and same_model(std, sr)
                    and np.array_equal(std.predict(Xt), sr.predict(Xt)))
        row.update({"standard_run": True, "standard_equals_cached": same, "standard_s": std_secs,
                    "standard_over_cached": std_secs / secs if secs > 0 else np.nan})
    print(f"{name} seed {seed}: {len(edits)} edits, mismatches {row['edit_mismatches']}, "
          f"macro-F1 diff {row['macro_f1_abs_diff']:.2e}, cached {secs:.1f}s"
          + (f", standard {row['standard_s']:.1f}s, equal {same}" if seed == STANDARD_SEED else ""),
          flush=True)
    return [row]


def summarise(df: pd.DataFrame, out: Path) -> str:
    expected = len(DATASETS) * len(SEEDS)
    ok = df[df.matches_committed.astype(bool)]
    std = df[df.standard_run.astype(bool)]
    std_ok = std[std.standard_equals_cached.astype(bool)]
    complete = len(df) == expected and set(df.dataset) == set(DATASETS) and set(df.seed) == set(SEEDS)
    passed = bool(complete and len(ok) == expected and len(std) == len(DATASETS)
                  and len(std_ok) == len(std))
    L = ["# Run E: the cached gate", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). RQ3's SR arm replayed with "
         "`fast_gate=True`, seeds 50–59, and compared with `results/refinement_v2_confirmatory/`. "
         "Written by `scripts/check_fast_gate.py`.", "",
         f"**Verdict: {'PASS' if passed else 'FAIL'}**", "",
         f"- Replayed runs equal to the committed edit log and macro-F1 (to {F1_TOL:g}): "
         f"{len(ok)} of {expected} expected ({len(df)} present).",
         f"- Seed {STANDARD_SEED}, standard gate equal to the cached gate (edit log, model, test "
         f"predictions): {len(std_ok)} of {len(DATASETS)} datasets ({len(std)} run).",
         f"- Largest macro-F1 difference: {df.macro_f1_abs_diff.max():.2e}; edits that differ: "
         f"{int(df.edit_mismatches.sum())}.", ""]
    if not passed:
        L += ["The cached gate is not used anywhere.", ""]
    L += ["## Per dataset", "", "| Dataset | Seeds matching | Edits (total) | Mismatched edits | "
          "Max macro-F1 diff |", "|---|---|---|---|---|"]
    for d in DATASETS:
        g = df[df.dataset == d]
        if g.empty:
            continue
        L.append(f"| {d} | {int(g.matches_committed.astype(bool).sum())} of {len(g)} | "
                 f"{int(g.n_edits.sum())} | {int(g.edit_mismatches.sum())} | "
                 f"{g.macro_f1_abs_diff.max():.2e} |")
    L += ["", f"## Speed on seed {STANDARD_SEED} (not part of the pass rule)", "",
          "Both gates ran in the same process on the same machine; several tasks shared the CPU, "
          "so read the ratio, not the seconds.", "",
          "| Dataset | Standard gate (s) | Cached gate (s) | Standard ÷ cached |", "|---|---|---|---|"]
    for _, r in std.sort_values("dataset").iterrows():
        L.append(f"| {r.dataset} | {r.standard_s:.1f} | {r.cached_s:.1f} | {r.standard_over_cached:.2f} |")
    if len(std):
        L += ["", f"Total over these datasets: standard {std.standard_s.sum():.0f} s, cached "
              f"{std.cached_s.sum():.0f} s, ratio {std.standard_s.sum() / std.cached_s.sum():.2f}."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run E: check the cached gate against RQ3")
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
