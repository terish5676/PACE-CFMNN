"""Run P3 (tough review D5): Skin without its duplicate rows.

Rules: results/skin_dedup/PREREGISTRATION.md (planned in advance).

Skin has 245,057 rows but only 51,444 distinct ones, so a test row is often an exact copy of a
training row. The paper's largest self-repair gain is on Skin (SR - B0 = +5.66 macro-F1 points,
seeds 50-59). This run repeats the Skin parts of the paper on ``skin_dedup`` (``data.load``: the
first copy of each row kept), with the same code, on new seeds 260-269:

* ``refinement``: ``run_refinement_v2.run_one`` (B0, B9, AL, SR, SR+AL, the retrains, kNN, RF).
  The primary test is SR against B0 on macro-F1 (paired two-sided Wilcoxon over the 10 seeds;
  one test, so no Holm correction).
* ``reference``: original Skin on the same seeds, B0 and SR only (the same calls as
  ``run_refinement_v2.run_one``), so the gain with and without duplicates is compared on the same
  seeds.
* ``accuracy``: ``accuracy_table.run_one`` (CFMNN, 10-NN, Random Forest on the paper's split).
* ``theta``: ``run_theta_churn.run_one`` (CFMNN at 7 values of theta, theta by validation, the
  flip rates against theta = 0.3).
* ``repair``: ``run_repair_theta.run_one(tags=("0.3",))`` (B0-0.3 and SR-0.3 on the paper's
  split; SR at theta by validation took 6,401 s per seed on Skin, so it is not run).

    python -B scripts/run_skin_dedup.py --jobs 4
    python -B scripts/run_skin_dedup.py --jobs 4 --resume
"""

from __future__ import annotations

import argparse
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

import accuracy_table as at  # noqa: E402
import run_refinement_v2 as rv2  # noqa: E402
import run_repair_theta as rt  # noqa: E402
import run_theta_churn as tc  # noqa: E402
from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_refinement import MAX_BUDGET, N_ROUNDS, split  # noqa: E402

NAME, ORIGINAL = "skin_dedup", "skin"
SEEDS = list(range(260, 270))
PAPER_GAIN = 5.66          # SR - B0 macro-F1 on Skin, seeds 50-59 (results/refinement_v2_confirmatory)
THRESHOLD = PAPER_GAIN / 2
PARTS = ["refinement", "reference", "accuracy", "theta", "repair"]
COST = {"reference": 5, "refinement": 4, "theta": 3, "accuracy": 2, "repair": 1}
OUT = ROOT / "results" / "skin_dedup"


# ------------------------------------------------------------------ the parts (top level, for jobs > 1)
def part_refinement(name, seed):
    return rv2.run_one(name, seed)[0]


def part_reference(name, seed):
    """B0 and SR exactly as ``run_refinement_v2.run_one`` makes them (the certificates it records
    along the way do not change the model)."""
    warnings.filterwarnings("ignore")
    P = split(name, seed)
    (Xtr, ytr), (Xp, _), (Xt, yt) = P["train"], P["pool"], P["test"]
    rows = []
    t = time.perf_counter()
    model = CFMNN().fit(Xtr, ytr)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    base = {"dataset": name, "seed": seed, "n_train": len(Xtr), "n_test": len(Xt), "budget": budget}
    rows.append({**base, "arm": "B0", **{f"test_{k}": v for k, v in classification_metrics(yt, model.predict(Xt)).items()},
                 "boxes_end": model.n_boxes_, "time_s": time.perf_counter() - t})
    t = time.perf_counter()
    sr, _, _ = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=budget, n_rounds=N_ROUNDS,
                               seed=seed, gate_labels_are_new=False)
    rows.append({**base, "arm": "SR", **{f"test_{k}": v for k, v in classification_metrics(yt, sr.predict(Xt)).items()},
                 "boxes_end": sr.n_boxes_, "time_s": time.perf_counter() - t})
    print(f"{name} seed {seed} reference: B0 {rows[0]['test_macro_f1']:.2f}, SR {rows[1]['test_macro_f1']:.2f}",
          flush=True)
    return rows


def part_accuracy(name, seed):
    return at.run_one(name, seed)


def part_theta(name, seed):
    return tc.run_one(name, seed)


def part_repair(name, seed):
    return rt.run_one(name, seed, tags=("0.3",))


FUNCS = {"refinement": part_refinement, "reference": part_reference, "accuracy": part_accuracy,
         "theta": part_theta, "repair": part_repair}


def dataset_for(part):
    return ORIGINAL if part == "reference" else NAME


# ------------------------------------------------------------------ the primary test and its reading
def paired_test(piv: pd.DataFrame, a: str, b: str) -> dict:
    d = (piv[a] - piv[b]).dropna().to_numpy()
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"diff": float(d.mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d),
            "p": p}


def reading(t: dict) -> str:
    """The pre-registered reading of the primary result."""
    if t["p"] < 0.05 and t["diff"] > 0:
        if t["diff"] >= THRESHOLD:
            return (f"Self-repair still helps without the duplicates, by at least half the paper's gain "
                    f"({t['diff']:+.2f} ≥ {THRESHOLD:.2f}): the Skin result stands, reported with this number.")
        return (f"Self-repair still helps, but by less than half the paper's gain ({t['diff']:+.2f} < "
                f"{THRESHOLD:.2f}): the paper reports this gain and says the duplicates inflated Skin's.")
    if t["p"] < 0.05:
        return "Self-repair hurts without the duplicates: Skin leaves the list of datasets where it helps."
    return ("No significant difference without the duplicates: the paper attributes Skin's gain to the "
            "duplicate rows and no longer counts Skin as a dataset where self-repair helps.")


def summarise(frames: dict, out: Path) -> str:
    ds = data.load(NAME, seed=SEEDS[0])
    L = ["# Run P3: Skin without its duplicate rows", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Written by `scripts/run_skin_dedup.py`.", "",
         f"`skin_dedup` keeps the first copy of each row: {ds.info['dedup_rows_after']:,} of "
         f"{ds.info['dedup_rows_before']:,} rows.", ""]
    ref = frames.get("refinement")
    if ref is not None and len(ref):
        piv = ref.pivot_table(index="seed", columns="arm", values="test_macro_f1")
        t = paired_test(piv, "SR", "B0")
        L += ["## Primary: does self-repair still help?", "",
              f"SR − B0, macro-F1 on `skin_dedup`, seeds {piv.index.min()}–{piv.index.max()} ({t['n']} seeds): "
              f"**{t['diff']:+.2f} points** (SR better on {t['wins']}, worse on {t['losses']}), Wilcoxon "
              f"p = {t['p']:.4f}. The paper's gain on Skin with duplicates: {PAPER_GAIN:+.2f}; half of it: "
              f"{THRESHOLD:.2f}.", "", f"**Reading:** {reading(t)}", ""]
        rr = frames.get("reference")
        if rr is not None and len(rr):
            rp = rr.pivot_table(index="seed", columns="arm", values="test_macro_f1")
            tr = paired_test(rp, "SR", "B0")
            L += [f"Reference, original Skin on the same seeds: SR − B0 = {tr['diff']:+.2f} (better on "
                  f"{tr['wins']}, worse on {tr['losses']}, p = {tr['p']:.4f}).", ""]
        L += ["### Every arm (mean over seeds)", "",
              "| Arm | Macro-F1 | Accuracy | Lowest class recall | Labels | Broken predictions (%) |",
              "|---|---|---|---|---|---|"]
        for arm, g in ref.groupby("arm", sort=False):
            L.append(f"| {arm} | {g.test_macro_f1.mean():.2f} | {g.test_accuracy.mean():.2f} | "
                     f"{g.test_min_class_recall.mean():.2f} | {g.total_labels.mean():.0f} | "
                     f"{g.churn_broken_share.mean():.3f} |")
        t2 = paired_test(piv, "SR+AL", "SR")
        L += ["", f"SR+AL − SR (do bought labels add anything?): {t2['diff']:+.2f}, better on {t2['wins']}, "
              f"worse on {t2['losses']}, p = {t2['p']:.4f} (secondary).", ""]
    acc = frames.get("accuracy")
    if acc is not None and len(acc):
        L += ["## Accuracy table (secondary)", "", "| Model | Accuracy, `skin_dedup` |", "|---|---|"]
        for m, g in acc.groupby("model", sort=False):
            L.append(f"| {m} | {g.accuracy.mean():.2f} |")
        L += [""]
    th, rep = frames.get("theta"), frames.get("repair")
    if th is not None and len(th):
        L += ["## CFMNN across theta (secondary)", "", "| theta | Accuracy | Boxes | NFR (%) |", "|---|---|---|---|"]
        for theta, g in th.groupby("theta"):
            L.append(f"| {theta:g} | {g.accuracy.mean():.2f} | {g.boxes.mean():.0f} | {g.nfr.mean():.3f} |")
        L += [f"", f"theta by validation: {', '.join(f'{v:g}' for v in sorted(th[th.is_val].theta.unique()))}.", ""]
        if rep is not None and len(rep):
            fr = tc.p1_frame(th, rep)
            piv = fr.pivot_table(index="seed", columns="model", values="nfr")
            tn = paired_test(-piv, "SR-0.3", "B0-val")      # positive: repair breaks fewer
            L += [f"Broken correct predictions (NFR), repair (SR-0.3) against re-tuning (B0 at theta by "
                  f"validation): repair breaks {tn['diff']:.3f} points fewer on average (fewer on {tn['wins']} "
                  f"seeds, more on {tn['losses']}, p = {tn['p']:.4f}).", ""]
    if rep is not None and len(rep):
        piv = rep.pivot_table(index="seed", columns="model", values="accuracy")
        if {"SR-0.3", "B0-0.3"} <= set(piv.columns):
            ta = paired_test(piv, "SR-0.3", "B0-0.3")
            L += [f"Repair on the paper's split (SR-0.3 − B0-0.3, accuracy): {ta['diff']:+.2f}, better on "
                  f"{ta['wins']}, worse on {ta['losses']}, p = {ta['p']:.4f}.", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P3: Skin without duplicates")
    ap.add_argument("--parts", nargs="+", default=PARTS, choices=PARTS)
    ap.add_argument("--seeds", nargs="+", type=int, help="smoke runs only")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for part in a.parts:
        csv = out / f"{part}.csv"
        if not a.summary_only:
            tasks = [(dataset_for(part), s) for s in a.seeds or SEEDS]
            run_all(FUNCS[part], pending(csv, tasks, a.resume, a.fresh), csv, jobs=a.jobs)
        if csv.exists():
            frames[part] = pd.read_csv(csv, float_precision="round_trip")
    print(summarise(frames, out))


if __name__ == "__main__":
    main()
