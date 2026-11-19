"""Run L: Covertype with theta chosen by validation, then self-repair. Descriptive.

Rules: results/covertype_theta/PREREGISTRATION.md. No comparison is tested (one dataset).

For each seed (210-214), on a stratified 10% of Covertype's rows (``SUBSAMPLE``) and Run K's split (``run_covertype.prepare``: the paper's 80/20 split,
the training rows in the seed's order, min-max scaling fitted on them), the four models of the
repair-at-theta run (``scripts/run_repair_theta.py``):

* B0-0.3: CFMNN at theta = 0.3, gamma = 4 (Run K's model);
* SR-0.3: B0-0.3, then RQ3's self-repair (budget 300, 10 rounds, the smallest cut, the gate on
  the training rows, ``fast_gate=True``);
* B0-val: CFMNN at the theta chosen by 3-fold stratified CV accuracy over
  ``run_readable_baselines.GRIDS["CFMNN-val"]`` (0.5 down to 0.05; on an exact tie the larger);
* SR-val: B0-val, then the same self-repair.

Recorded per seed and model: theta, boxes, training error and its ties, the share of test
predictions decided by the nearest centroid, test accuracy and macro-F1, edits tried and
accepted, seconds (fit, repair, tuning), PACE-flip and PACE-SHAP time per explanation on 50 test
points, and peak memory. 10-NN and the Random Forest are quoted from Run K, not rerun.

``TUNE_ROWS`` and ``GATE_ROWS`` cap the rows used for tuning and for the repair gate (a
stratified subsample, seeded by the study seed). Their values were fixed from the smoke slice
before any study seed ran (PREREGISTRATION.md); ``None`` means all training rows.

One task at a time, with nothing else running: the times are measures.

    python -B scripts/run_covertype_theta.py
    python -B scripts/run_covertype_theta.py --resume
    python -B scripts/run_covertype_theta.py --summary-only
    python -B scripts/run_covertype_theta.py --seeds 192 --out <folder> --fresh   # smoke slice
    python -B scripts/run_covertype_theta.py --check-k 190 --out <folder>        # B0-0.3 reproduces Run K
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.explain import PaceFlip, PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_cfmnn.ties import centroid_decided, training_error_ties  # noqa: E402
from run_covertype import NAME, peak_memory_mb, prepare, timed  # noqa: E402
from run_readable_baselines import GRIDS, cv_folds, tune  # noqa: E402

OUT = ROOT / "results" / "covertype_theta"
SEEDS = list(range(210, 215))
GAMMA, FIXED_THETA = 4.0, 0.3
BUDGET, N_ROUNDS, CHOOSER = 300, 10, "volume"          # RQ3's self-repair
# The study runs on a stratified 10% of Covertype's rows (seeded by the study seed): on all rows the
# gate's table would need about 58 GB and a seed about 10 hours (smoke slice, PREREGISTRATION.md).
SUBSAMPLE = 0.1
TUNE_ROWS: int | None = None                           # fixed before the run (PREREGISTRATION.md)
GATE_ROWS: int | None = None                           # fixed before the run (PREREGISTRATION.md)
MODELS = ["B0-0.3", "SR-0.3", "B0-val", "SR-val"]
RUN_K = ROOT / "results" / "covertype" / "runs.csv"
# Run K's smoke slice on seed 190, full data (Amendment 1 of results/covertype/PREREGISTRATION.md).
RUN_K_SMOKE = {190: 55.97}
COLUMNS = ["dataset", "seed", "model", "theta", "theta_source", "subsample", "n_train", "n_test",
           "tune_rows", "gate_rows", "cv_scores", "boxes", "train_errors", "train_error_pct",
           "train_errors_tied", "train_errors_tied_inside_both", "train_tie_share_pct",
           "test_accuracy", "test_macro_f1", "test_errors", "test_centroid_decided",
           "test_centroid_decided_pct", "test_errors_centroid_decided",
           "test_errors_centroid_decided_pct", "edits_logged", "edits_already_right",
           "edits_accepted", "fit_seconds", "repair_seconds", "tune_seconds", "n_explained",
           "pace_flip_ms", "pace_shap_ms", "pace_shap_mode", "peak_memory_mb", "task_seconds"]


def stratified_rows(y, n: int | None, seed: int) -> np.ndarray:
    """Indices of a stratified subsample of n rows (all rows if n is None or too large), in order."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    keep, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(keep)


def choose_theta(Xtr, ytr, seed: int, tune_rows: int | None):
    """The theta of the readable-baselines run's CFMNN-val, tuned on (a subsample of) the rows."""
    idx = stratified_rows(ytr, tune_rows, seed)
    X, y = Xtr[idx], ytr[idx]
    (theta, scores), secs = timed(lambda: tune("CFMNN-val", GRIDS["CFMNN-val"], X, y, seed, 1,
                                               cv_folds(y, seed)))
    return float(theta), scores, len(idx), secs


def describe(model: CFMNN, Xtr, ytr, Xte, yte) -> dict:
    tr = training_error_ties(model, Xtr, ytr)
    te = centroid_decided(model, Xte, yte)
    met = classification_metrics(yte, model.predict(Xte))
    pct = lambda a, b: 100 * a / b if b else np.nan   # noqa: E731
    return {"boxes": model.n_boxes_, **tr,
            "train_tie_share_pct": pct(tr["train_errors_tied_inside_both"], tr["train_errors"]),
            "test_accuracy": met["accuracy"], "test_macro_f1": met["macro_f1"],
            "test_errors": te["n_errors"], "test_centroid_decided": te["n_centroid_decided"],
            "test_centroid_decided_pct": pct(te["n_centroid_decided"], te["n"]),
            "test_errors_centroid_decided": te["n_errors_centroid_decided"],
            "test_errors_centroid_decided_pct": pct(te["n_errors_centroid_decided"], te["n_errors"])}


def explain_times(model: CFMNN, Xtr, Xex, seed: int) -> dict:
    """PACE-flip and PACE-SHAP on the explained points, as Run K times them."""
    _, bex = rf.backgrounds_v3(Xtr, seed)
    _, flip_s = timed(lambda: PaceFlip(model, background=bex).explain(Xex))
    ex = PaceShap(model, bex[:rf.SHAP_V3_R], n_permutations=rf.SHAP_V3_P, seed=seed)
    phi, shap_s = timed(lambda: ex.explain(Xex))
    if phi.shape != Xex.shape:
        raise RuntimeError("PACE-SHAP returned the wrong shape")
    return {"n_explained": len(Xex), "pace_flip_ms": 1000 * flip_s / len(Xex),
            "pace_shap_ms": 1000 * shap_s / len(Xex), "pace_shap_mode": ex.stats_["mode"]}


def repair(b0: CFMNN, Xtr, ytr, seed: int, gate_rows: int | None):
    g = stratified_rows(ytr, gate_rows, seed)
    (sr, _, edits), secs = timed(lambda: run_self_repair(
        b0, Xtr, ytr, Xtr[g], ytr[g], chooser=CHOOSER, budget=BUDGET, n_rounds=N_ROUNDS, seed=seed,
        gate_labels_are_new=False, fast_gate=True))
    return sr, edits, len(g), secs


def run_one(name: str, seed: int, subsample: float = 1.0, ds=None, tune_rows=TUNE_ROWS,
            gate_rows=GATE_ROWS):
    warnings.filterwarnings("ignore")
    t_task = time.perf_counter()
    Xtr, ytr, Xte, yte, Xex = prepare(seed, subsample, ds)
    say = lambda msg: print(f"  seed {seed}: {msg}", flush=True)   # noqa: E731
    tv, scores, n_tune, tune_s = choose_theta(Xtr, ytr, seed, tune_rows)
    say(f"theta by validation {tv:g} on {n_tune:,} rows ({tune_s:.0f} s); CV accuracy "
        + ", ".join(f"{t:g}: {100 * s:.2f}" for t, s in zip(GRIDS['CFMNN-val'], scores)))
    rows = []
    for tag, theta in (("0.3", FIXED_THETA), ("val", tv)):
        b0, fit_s = timed(lambda: CFMNN(theta=theta, gamma=GAMMA).fit(Xtr, ytr))
        say(f"B0-{tag} (theta {theta:g}) trained in {fit_s:.0f} s, {b0.n_boxes_} boxes")
        sr, edits, n_gate, rep_s = repair(b0, Xtr, ytr, seed, gate_rows)
        say(f"SR-{tag}: {sum(bool(e['accepted']) for e in edits)} of {len(edits)} edits accepted "
            f"({rep_s:.0f} s, gate on {n_gate:,} rows)")
        for model, m in ((f"B0-{tag}", b0), (f"SR-{tag}", sr)):
            rec = {c: np.nan for c in COLUMNS}
            rec.update({"dataset": name, "seed": seed, "model": model, "theta": theta,
                        "theta_source": "fixed" if tag == "0.3" else "validation",
                        "subsample": subsample, "n_train": len(ytr), "n_test": len(yte),
                        "tune_rows": n_tune, "gate_rows": n_gate,
                        "cv_scores": ";".join(f"{s:.6f}" for s in scores), "tune_seconds": tune_s,
                        **describe(m, Xtr, ytr, Xte, yte), **explain_times(m, Xtr, Xex, seed)})
            if model.startswith("SR"):
                rec.update({"edits_logged": len(edits), "repair_seconds": rep_s,
                            "edits_already_right": sum(e["reason"] == "already right" for e in edits),
                            "edits_accepted": sum(bool(e["accepted"]) for e in edits)})
            else:
                rec["fit_seconds"] = fit_s
            rows.append(rec)
    total = time.perf_counter() - t_task
    for r in rows:
        r["peak_memory_mb"] = peak_memory_mb()
        r["task_seconds"] = total
    print(f"{name} seed {seed} (theta-val {tv:g}): " + ", ".join(
        f"{r['model']} {r['test_accuracy']:.2f}%" for r in rows)
        + f"; {total / 60:.1f} min, peak memory {rows[0]['peak_memory_mb']:.0f} MB", flush=True)
    return rows


def check_k(seed: int, ds=None) -> float:
    """B0-0.3 on a Run K seed must give Run K's test accuracy: the setup is Run K's."""
    Xtr, ytr, Xte, yte, _ = prepare(seed, 1.0, ds)
    acc = classification_metrics(yte, CFMNN(theta=FIXED_THETA, gamma=GAMMA).fit(Xtr, ytr).predict(Xte))["accuracy"]
    if seed in RUN_K_SMOKE:
        want, ok = RUN_K_SMOKE[seed], abs(round(acc, 2) - RUN_K_SMOKE[seed]) < 1e-9
    else:
        k = pd.read_csv(RUN_K)
        want = float(k[(k.seed == seed) & (k.model == "CFMNN")].test_accuracy.iloc[0])
        ok = abs(acc - want) < 1e-9
    print(f"Run K check, seed {seed}: B0-0.3 {acc:.4f}% against Run K's {want}: {'PASS' if ok else 'FAIL'}",
          flush=True)
    if not ok:
        raise SystemExit(1)
    return acc


# ------------------------------------------------------------------ summary
def _ms(v) -> str:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "–"
    return f"{v.mean():,.2f} ± {v.std(ddof=1):,.2f}" if len(v) > 1 else f"{v.mean():,.2f}"


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    share = sorted(set(df.subsample))
    L = ["# Run L: Covertype with theta chosen by validation, then self-repair", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Descriptive: no comparison is tested. "
         f"Run K's split, seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}), γ = 4; "
         f"{int(df.n_train.iloc[0]):,} training rows, {int(df.n_test.iloc[0]):,} test rows. "
         "Mean ± standard deviation over seeds. Written by `scripts/run_covertype_theta.py`.", ""]
    if share != [1.0]:
        L += [f"**Not the full data:** share of the rows used = {share}.", ""]
    th = df[df.model == "B0-val"]
    L += ["## θ chosen by validation", "",
          "| Seed | θ | Rows used for tuning | Seconds to tune |", "|---|---|---|---|"]
    L += [f"| {int(r.seed)} | {r.theta:g} | {int(r.tune_rows):,} | {r.tune_seconds:,.0f} |" for r in th.itertuples()]
    L += ["", "## The four models", "",
          "| Model | Test accuracy (%) | Test macro-F1 (%) | Boxes | Training error (%) | "
          "Training errors that are ties (%) | Test predictions decided by the nearest centroid (%) | "
          "Edits accepted | Seconds (fit or repair) | PACE-flip ms | PACE-SHAP ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in [m for m in MODELS if m in set(df.model)]:
        g = df[df.model == m]
        secs = g.repair_seconds if m.startswith("SR") else g.fit_seconds
        tied = 100 * g.train_errors_tied_inside_both.sum() / g.train_errors.sum() if g.train_errors.sum() else np.nan
        dec = 100 * g.test_centroid_decided.sum() / g.n_test.sum()
        L.append(f"| {m} | {_ms(g.test_accuracy)} | {_ms(g.test_macro_f1)} | {_ms(g.boxes)} | "
                 f"{_ms(g.train_error_pct)} | {tied:.2f} | {dec:.2f} | {_ms(g.edits_accepted)} | "
                 f"{_ms(secs)} | {_ms(g.pace_flip_ms)} | {_ms(g.pace_shap_ms)} |")
    L += ["", "Ties: training errors inside a box of both classes (Proposition 1's form), pooled over "
          "seeds. Centroid-decided: pooled over seeds.", ""]
    k = pd.read_csv(RUN_K) if RUN_K.exists() else None
    if k is not None:
        L += ["## For reference: Run K (seeds 170–174, not rerun)", "", "| Model | Test accuracy (%) | Test macro-F1 (%) |",
              "|---|---|---|"]
        for m in ["CFMNN", "10-NN", "Random Forest"]:
            g = k[k.model == m]
            L.append(f"| {m} (Run K) | {_ms(g.test_accuracy)} | {_ms(g.test_macro_f1)} |")
        L.append("")
    L += ["## Cost of the run", "",
          f"Minutes per seed: {_ms(df.groupby('seed').task_seconds.first() / 60)}. Peak memory of the "
          f"process (MB): {_ms(df.groupby('seed').peak_memory_mb.first())}. Gate rows: "
          f"{sorted(int(v) for v in df.gate_rows.dropna().unique())}.", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run L: Covertype, theta by validation and self-repair")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--subsample", type=float, default=SUBSAMPLE, help="other values: smoke slices only")
    ap.add_argument("--tune-rows", type=int, default=TUNE_ROWS, help="smoke slices only")
    ap.add_argument("--gate-rows", type=int, default=GATE_ROWS, help="smoke slices only")
    ap.add_argument("--out")
    ap.add_argument("--check-k", type=int, help="only check that B0-0.3 reproduces Run K on this seed")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.check_k is not None:
        check_k(a.check_k)
        return
    out = Path(a.out) if a.out else OUT
    smoke = a.subsample != SUBSAMPLE or a.tune_rows != TUNE_ROWS or a.gate_rows != GATE_ROWS or a.seeds
    if smoke and out.resolve() == OUT.resolve():
        raise SystemExit("a smoke slice (other seeds, rows or limits) needs --out outside results/covertype_theta")
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(NAME, s, a.subsample) for s in a.seeds or SEEDS]
        todo = pending(csv, tasks, a.resume, a.fresh)
        if todo:
            ds = data.load(NAME)
            run_all(lambda n, s, sub: run_one(n, s, sub, ds, a.tune_rows, a.gate_rows), todo, csv, jobs=1)
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
