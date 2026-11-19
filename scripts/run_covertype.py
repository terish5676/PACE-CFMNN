"""Run K: one large dataset (Covertype: 581,012 rows, 54 features, 7 classes). Descriptive.

Rules: results/covertype/PREREGISTRATION.md. No comparison is tested.

For each seed (170-174), on the paper's 80/20 split with the training rows in the seed's order
and min-max scaling fitted on them (as the accuracy table):

* CFMNN (theta = 0.3, gamma = 4): seconds to train, boxes, training error, how many training
  errors are ties, how many test predictions the nearest centroid decides and their share of
  the test errors, test accuracy and macro-F1;
* 10-nearest neighbours and a Random Forest with 200 trees: test accuracy and macro-F1;
* PACE-flip and PACE-SHAP (sampled mode, 32 orders per row, 200 rows) on 50 test points, with
  the explainer rows of the second explanation run: milliseconds per explanation.

One seed at a time, with nothing else running: the times are measures.

    python -B scripts/run_covertype.py
    python -B scripts/run_covertype.py --resume            # continue after a stop
    python -B scripts/run_covertype.py --summary-only
    python -B scripts/run_covertype.py --seeds 190 --subsample 0.1 --out <folder>   # smoke slice
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.explain import PaceFlip, PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_cfmnn.ties import centroid_decided, training_error_ties  # noqa: E402

OUT = ROOT / "results" / "covertype"
NAME = "covertype"
SEEDS = list(range(170, 175))
THETA, GAMMA = 0.3, 4.0
N_EXPLAIN = 50
MODELS = ["CFMNN", "10-NN", "Random Forest"]
COLUMNS = ["dataset", "seed", "model", "subsample", "n_train", "n_test", "d", "classes",
           "test_accuracy", "test_macro_f1", "fit_seconds", "predict_seconds", "boxes",
           "train_errors", "train_error_pct", "train_errors_tied", "train_errors_tied_inside_both",
           "train_tie_share_pct", "tie_count_seconds", "test_errors", "test_centroid_decided",
           "test_centroid_decided_pct", "test_errors_centroid_decided",
           "test_errors_centroid_decided_pct", "n_explained", "pace_flip_ms", "pace_shap_ms",
           "pace_shap_mode", "pace_shap_boxes_skipped_pct", "peak_memory_mb", "task_seconds"]


def peak_memory_mb() -> float:
    """Peak memory of this process in MB (NaN if the platform gives none)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

            c = Counters()
            c.cb = ctypes.sizeof(c)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                                   wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return float("nan")
            return c.PeakWorkingSetSize / 2 ** 20
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024
    except Exception:  # noqa: BLE001 - a missing number must not stop the run
        return float("nan")


def prepare(seed: int, subsample: float = 1.0, ds=None):
    """The accuracy table's split, optionally on a stratified share of the rows (smoke only)."""
    ds = ds or data.load(NAME)
    X, y = ds.X, ds.y
    if subsample < 1.0:
        keep, _ = train_test_split(np.arange(len(y)), train_size=subsample, stratify=y,
                                   random_state=seed)
        keep = np.sort(keep)
        X, y = X[keep], y[keep]
    sp = data.paper_split(y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(X[order])
    te = sp["test"]
    explain = np.random.default_rng(seed).permutation(te)[:N_EXPLAIN]   # as run_faithfulness.setup
    return (sc.transform(X[order]), y[order], sc.transform(X[te]), y[te], sc.transform(X[explain]))


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_one(name: str, seed: int, subsample: float = 1.0, ds=None):
    warnings.filterwarnings("ignore")
    t_task = time.perf_counter()
    Xtr, ytr, Xte, yte, Xex = prepare(seed, subsample, ds)
    base = {c: np.nan for c in COLUMNS}
    base.update({"dataset": name, "seed": seed, "subsample": subsample, "n_train": len(ytr),
                 "n_test": len(yte), "d": Xtr.shape[1], "classes": len(np.unique(ytr))})
    say = lambda msg: print(f"  seed {seed}: {msg}", flush=True)   # noqa: E731

    model, fit_s = timed(lambda: CFMNN(theta=THETA, gamma=GAMMA).fit(Xtr, ytr))
    say(f"CFMNN trained in {fit_s:.0f} s, {model.n_boxes_} boxes")
    tr, tie_s = timed(lambda: training_error_ties(model, Xtr, ytr))
    say(f"training errors {tr['train_errors']}, ties {tr['train_errors_tied_inside_both']} "
        f"({tie_s:.0f} s)")
    pred, pred_s = timed(lambda: model.predict(Xte))
    te = centroid_decided(model, Xte, yte)
    met = classification_metrics(yte, pred)
    pct = lambda a, b: 100 * a / b if b else np.nan   # noqa: E731

    _, bex = rf.backgrounds_v3(Xtr, seed)
    flip, flip_s = timed(lambda: PaceFlip(model, background=bex).explain(Xex))
    shap_ex = PaceShap(model, bex[:rf.SHAP_V3_R], n_permutations=rf.SHAP_V3_P, seed=seed)
    phi, shap_s = timed(lambda: shap_ex.explain(Xex))
    if flip.shape != Xex.shape or phi.shape != Xex.shape:
        raise RuntimeError("an explainer returned the wrong shape")
    say(f"PACE-flip {1000 * flip_s / len(Xex):.1f} ms, PACE-SHAP {1000 * shap_s / len(Xex):.1f} ms "
        "per explanation")

    rows = [{**base, "model": "CFMNN", "test_accuracy": met["accuracy"],
             "test_macro_f1": met["macro_f1"], "fit_seconds": fit_s, "predict_seconds": pred_s,
             "boxes": model.n_boxes_, **tr,
             "train_tie_share_pct": pct(tr["train_errors_tied_inside_both"], tr["train_errors"]),
             "tie_count_seconds": tie_s, "test_errors": te["n_errors"],
             "test_centroid_decided": te["n_centroid_decided"],
             "test_centroid_decided_pct": pct(te["n_centroid_decided"], te["n"]),
             "test_errors_centroid_decided": te["n_errors_centroid_decided"],
             "test_errors_centroid_decided_pct": pct(te["n_errors_centroid_decided"], te["n_errors"]),
             "n_explained": len(Xex), "pace_flip_ms": 1000 * flip_s / len(Xex),
             "pace_shap_ms": 1000 * shap_s / len(Xex), "pace_shap_mode": shap_ex.stats_["mode"],
             "pace_shap_boxes_skipped_pct": 100 * shap_ex.stats_["share_of_boxes_skipped"]}]
    del model, shap_ex
    make = {"10-NN": lambda: KNeighborsClassifier(n_neighbors=10),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=seed,
                                                            n_jobs=-1)}
    for m in MODELS[1:]:
        clf, f_s = timed(lambda: make[m]().fit(Xtr, ytr))
        p, p_s = timed(lambda: clf.predict(Xte))
        mm = classification_metrics(yte, p)
        rows.append({**base, "model": m, "test_accuracy": mm["accuracy"],
                     "test_macro_f1": mm["macro_f1"], "fit_seconds": f_s, "predict_seconds": p_s})
        say(f"{m} {mm['accuracy']:.2f}% ({f_s + p_s:.0f} s)")
    total = time.perf_counter() - t_task
    for r in rows:
        r["peak_memory_mb"] = peak_memory_mb()
        r["task_seconds"] = total
    print(f"{name} seed {seed}: " + ", ".join(f"{r['model']} {r['test_accuracy']:.2f}%" for r in rows)
          + f"; {total / 60:.1f} min, peak memory {rows[0]['peak_memory_mb']:.0f} MB", flush=True)
    return rows


# ------------------------------------------------------------------ summary
def _ms(v) -> str:
    """Mean ± standard deviation over seeds (the mean alone for one seed)."""
    v = np.asarray(v, float)
    return f"{v.mean():,.2f} ± {v.std(ddof=1):,.2f}" if len(v) > 1 else f"{v.mean():,.2f}"


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    c = df[df.model == "CFMNN"]
    share = sorted(set(df.subsample))
    L = ["# Run K: one large dataset (Covertype)", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Descriptive: no comparison is tested. "
         f"The paper's 80/20 split, seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}), θ = 0.3, γ = 4; "
         f"{int(df.n_train.iloc[0]):,} training rows, {int(df.n_test.iloc[0]):,} test rows, "
         f"{int(df.d.iloc[0])} features, {int(df.classes.iloc[0])} classes. Mean ± standard deviation "
         "over seeds. Written by `scripts/run_covertype.py`.", ""]
    if share != [1.0]:
        L += [f"**Not the full data:** share of the rows used = {share}.", ""]
    L += ["## Accuracy", "",
          "| Model | Test accuracy (%) | Test macro-F1 (%) | Seconds to train | Seconds to predict |",
          "|---|---|---|---|---|"]
    for m in [m for m in MODELS if m in set(df.model)]:
        g = df[df.model == m]
        L.append(f"| {m} | {_ms(g.test_accuracy)} | {_ms(g.test_macro_f1)} | {_ms(g.fit_seconds)} | "
                 f"{_ms(g.predict_seconds)} |")
    tied, errs = int(c.train_errors_tied_inside_both.sum()), int(c.train_errors.sum())
    any_tie = int(c.train_errors_tied.sum())
    dec, n = int(c.test_centroid_decided.sum()), int(c.n_test.sum())
    edec, eall = int(c.test_errors_centroid_decided.sum()), int(c.test_errors.sum())
    share_of = lambda a, b: "–" if b == 0 else f"{100 * a / b:.2f}%"   # noqa: E731
    L += ["", "## CFMNN: size, training errors and ties", "", "| Measure | Value |", "|---|---|",
          f"| Seconds to train | {_ms(c.fit_seconds)} |",
          f"| Boxes | {_ms(c.boxes)} |",
          f"| Training rows per box | {_ms(c.n_train / c.boxes)} |",
          f"| Training error (%) | {_ms(c.train_error_pct)} |",
          f"| Training errors, all seeds | {errs:,} |",
          f"| ... that are ties inside a box of both classes (Proposition 1's form) | {tied:,} "
          f"({share_of(tied, errs)}) |",
          f"| ... whose own class reaches the winner's membership | {any_tie:,} "
          f"({share_of(any_tie, errs)}) |",
          f"| Test predictions decided by the nearest centroid, all seeds | {dec:,} of {n:,} "
          f"({share_of(dec, n)}) |",
          f"| Test errors among those predictions, all seeds | {edec:,} of {eall:,} "
          f"({share_of(edec, eall)}) |",
          "", f"## Explanation time ({int(c.n_explained.iloc[0])} test points per seed)", "",
          "| Method | Milliseconds per explanation |", "|---|---|",
          f"| PACE-flip | {_ms(c.pace_flip_ms)} |",
          f"| PACE-SHAP ({c.pace_shap_mode.iloc[0]} mode, 32 orders per row, 200 rows) | "
          f"{_ms(c.pace_shap_ms)} |", "",
          f"PACE-SHAP skipped {_ms(c.pace_shap_boxes_skipped_pct)}% of the (row, box) pairs as unable "
          "to change the margin.", "", "## Cost of the run", "",
          f"Minutes per seed: {_ms(c.task_seconds / 60)}. Peak memory of the process (MB): "
          f"{_ms(c.peak_memory_mb)}.", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run K: one large dataset (Covertype)")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="share of the rows to use (smoke slices only; the study uses 1.0)")
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not 0 < a.subsample <= 1:
        raise SystemExit("--subsample must be in (0, 1]")
    out = Path(a.out) if a.out else OUT
    if a.subsample < 1 and out.resolve() == OUT.resolve():
        raise SystemExit("a subsample is a smoke slice: give --out a folder outside results/covertype")
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(NAME, s, a.subsample) for s in a.seeds or SEEDS]
        todo = pending(csv, tasks, a.resume, a.fresh)
        if todo:
            ds = data.load(NAME)                      # read the file once for every seed
            run_all(lambda n, s, sub: run_one(n, s, sub, ds), todo, csv, jobs=1)
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
