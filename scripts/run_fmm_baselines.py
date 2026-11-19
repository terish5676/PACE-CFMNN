"""Run P2 (tough review D4): CFMNN with self-repair against tuned fuzzy min-max networks.

Rules: results/fmm_baselines/PREREGISTRATION.md (planned in advance).

Question: is CFMNN with self-repair as accurate as the fuzzy min-max networks that handle overlaps
by design, when every model is tuned the same way?

Models, per dataset and seed, all on the paper's 80/20 split (``run_readable_baselines.prepare``:
the seed's row order, min-max scaling fitted on the training rows):

* CFMNN-SR-val (the method): CFMNN at the theta chosen by 3-fold cross-validation, then self-repair
  (RQ2's SR arm as in ``run_repair_theta.py``: up to 300 of the model's own training errors, the
  smallest cut, 10 rounds, gated on the training rows, no label beyond them);
* EFMNN-val, KNEFMNN-val, RFMNN-val, IOL-GFMM-val: hyperbox-brain 0.1.6, each at the theta chosen
  by the same cross-validation (``pace_cfmnn.fmm_baselines``);
* the same five at a fixed theta = 0.3 (secondary), and CFMNN-val and CFMNN-0.3 without repair
  (descriptive).

Every model chooses theta over {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5} by mean validation accuracy
on the same stratified folds, an exact tie going to the largest theta; gamma = 4 throughout.

Primary: test macro-F1 of CFMNN-SR-val against each tuned baseline: paired two-sided Wilcoxon over
the 10 seeds, Holm across the 6 datasets x 4 baselines = 24 tests.

Skin amendment (``--skin-share``): if the 1-seed pilot projects more than 24 hours for the study,
Skin is replaced, before any study seed runs, by a stratified sample of its rows drawn with the
seed (share fixed in the pre-registration), then split as above.

    python -B scripts/run_fmm_baselines.py --jobs 6
    python -B scripts/run_fmm_baselines.py --jobs 6 --resume
    python -B scripts/run_fmm_baselines.py --seeds 194 --out results/fmm_baselines/pilot   # pilot
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
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import data  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.fmm_baselines import FMM_MODELS, GAMMA, THETAS, cv_folds, make_fmm, n_boxes, tune_theta  # noqa: E402
from pace_cfmnn.resume import run_all  # noqa: E402
from run_error_detection import holm  # noqa: E402
from run_readable_baselines import prepare as paper_prepare  # noqa: E402

DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = list(range(250, 260))
FAMILIES = ["CFMNN"] + FMM_MODELS
METHOD = "CFMNN-SR-val"
BASELINES = [f"{m}-val" for m in FMM_MODELS]
BUDGET, N_ROUNDS, CHOOSER = 300, 10, "volume"         # run_repair_theta.py's self-repair
FIXED_THETA = 0.3
# Slowest first (laptop timing check, PREREGISTRATION.md); this only orders the tasks.
COST = {"skin": 6, "letter": 5, "magic": 4, "landsat": 3, "thyroid": 2, "page_blocks": 1}
OUT = ROOT / "results" / "fmm_baselines"


# ------------------------------------------------------------------ data
def prepare(name: str, seed: int, skin_share: float | None = None):
    """The paper split; for Skin with ``skin_share``, of a stratified sample of its rows first."""
    if name != "skin" or not skin_share:
        return paper_prepare(name, seed)
    ds = data.load(name, seed=seed)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=skin_share, random_state=seed)
    keep = np.sort(next(sss.split(np.zeros(len(ds.y)), ds.y))[0])
    X, y = ds.X[keep], ds.y[keep]
    sp = data.paper_split(y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])          # as paper_prepare
    sc = data.fit_scaler(X[order])
    return sc.transform(X[order]), y[order], sc.transform(X[sp["test"]]), y[sp["test"]]


# ------------------------------------------------------------------ one dataset, seed and family
def _fit(name, theta, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_fmm(name, theta).fit(X, y)


def _scores(pred, yte):
    pred = np.asarray(pred)
    return {"accuracy": 100 * float(np.mean(pred == yte)),
            "macro_f1": 100 * float(f1_score(yte, pred, average="macro"))}


def run_one(name: str, seed: int, family: str, skin_share: float | None = None):
    """Rows for one model family: CFMNN gives CFMNN-val, CFMNN-SR-val, CFMNN-0.3 and CFMNN-SR-0.3;
    a baseline gives <name>-val and <name>-0.3."""
    warnings.filterwarnings("ignore")
    t_task = time.perf_counter()
    Xtr, ytr, Xte, yte = prepare(name, seed, skin_share)
    t0 = time.perf_counter()
    theta, cv = tune_theta(family, Xtr, ytr, seed, THETAS, folds=cv_folds(ytr, seed))
    tune_s = time.perf_counter() - t0
    base = {"dataset": name, "seed": seed, "family": family, "skin_share": skin_share or 1.0,
            "n_train": len(ytr), "n_test": len(yte)}
    rows, done = [], {}
    for setting, th in (("val", theta), ("0.3", FIXED_THETA)):
        if th in done:                                     # theta-val = 0.3: the same models
            rows += [{**r, "model": r["model"].replace("-val", "-0.3"), "setting": setting, "cv_scores": "",
                      "tune_s": 0.0} for r in done[th]]
            continue
        t0 = time.perf_counter()
        m = _fit(family, th, Xtr, ytr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = m.predict(Xte)
        pred_s = time.perf_counter() - t0
        rec = {**base, "model": f"{family}-{setting}", "setting": setting, "theta": th,
               "cv_scores": ";".join(f"{s:.6f}" for s in cv) if setting == "val" else "",
               **_scores(pred, yte), "boxes": n_boxes(m), "tune_s": tune_s if setting == "val" else 0.0,
               "fit_s": fit_s, "repair_s": 0.0, "predict_s": pred_s, "edits_accepted": np.nan}
        new = [rec]
        if family == "CFMNN":
            t0 = time.perf_counter()
            sr, _, edits = run_self_repair(m, Xtr, ytr, Xtr, ytr, chooser=CHOOSER, budget=BUDGET,
                                           n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False)
            repair_s = time.perf_counter() - t0
            new.append({**rec, "model": f"CFMNN-SR-{setting}", **_scores(sr.predict(Xte), yte),
                        "boxes": n_boxes(sr), "repair_s": repair_s,
                        "edits_accepted": sum(bool(e["accepted"]) for e in edits)})
        done[th] = new
        rows += new
    task_s = time.perf_counter() - t_task
    for r in rows:
        r["task_s"] = task_s
    print(f"{name} seed {seed} {family} (theta-val {theta:g}): " + ", ".join(
        f"{r['model']} F1 {r['macro_f1']:.1f}" for r in rows) + f" ({task_s:.0f}s)", flush=True)
    return rows


# ------------------------------------------------------------------ the pilot's projection
PILOT_SEED, LIMIT_HOURS, SKIN_SHARE = 194, 24.0, 0.2


def projection_hours(df, n_seeds=len(SEEDS), jobs=6, running=0, elapsed_s=0.0) -> float:
    """Study hours projected from a one-seed pilot: the pilot's task seconds x seeds / jobs, never
    below the longest task. ``running`` tasks still unfinished after ``elapsed_s`` seconds count at
    that time, so a pilot stopped early gives a lower bound."""
    task = df.drop_duplicates(["dataset", "seed", "family"]).task_s if len(df) else pd.Series(dtype=float)
    total = float(task.sum()) + running * elapsed_s
    return max(total * n_seeds / jobs, float(task.max()) if len(task) else 0.0, elapsed_s) / 3600


def projection_text(df, jobs, running=0, elapsed_s=0.0) -> str:
    h = projection_hours(df, jobs=jobs, running=running, elapsed_s=elapsed_s)
    per = (df.drop_duplicates(["dataset", "seed", "family"]).groupby("dataset").task_s.sum() / 3600).round(2)
    over = h > LIMIT_HOURS
    return "\n".join([
        f"Pilot tasks finished: {df.drop_duplicates(['dataset', 'seed', 'family']).shape[0]}"
        + (f"; still running: {running} (counted at {elapsed_s / 3600:.1f} h each)" if running else ""),
        "Pilot hours per dataset (all families, one seed): " + ", ".join(f"{k} {v}" for k, v in per.items()),
        f"Projected study: {h:.1f} h with {jobs} jobs ({'a lower bound' if running else 'estimate'}).",
        (f"Rule (PREREGISTRATION.md): more than {LIMIT_HOURS:g} h, so the study runs with "
         f"--skin-share {SKIN_SHARE:g}." if over else
         f"Rule (PREREGISTRATION.md): at most {LIMIT_HOURS:g} h"
         + (" so far; the pilot must finish before deciding." if running else ", so Skin runs in full."))])


def pending(csv_path, tasks, resume=False, fresh=False):
    """As ``resume.pending``, but a task is (dataset, seed, family), so a stopped run keeps the
    families it finished."""
    p = Path(csv_path)
    has_rows = p.exists() and p.stat().st_size > 0
    if has_rows and fresh:
        p.unlink()
        has_rows = False
    if has_rows and not resume:
        raise SystemExit(f"{p} already has results. Run the same command with --resume to "
                         "continue, or with --fresh to delete them and start over.")
    done = set()
    if has_rows:
        d = pd.read_csv(p, usecols=["dataset", "seed", "family"])
        done = set(zip(d.dataset.astype(str), d.seed.astype(int), d.family.astype(str)))
    todo = [t for t in tasks if (str(t[0]), int(t[1]), str(t[2])) not in done]
    print(f"{len(tasks) - len(todo)} of {len(tasks)} dataset-seed-family tasks already done; "
          f"{len(todo)} to run.", flush=True)
    return todo


# ------------------------------------------------------------------ comparisons
def compare(df, a, baselines, col="macro_f1"):
    """``a`` against each baseline on each dataset: paired two-sided Wilcoxon over seeds, Holm
    across every (dataset, baseline) test together."""
    from scipy.stats import wilcoxon
    out = []
    for b in baselines:
        for ds in [d for d in DATASETS if d in set(df.dataset)]:
            g = df[df.dataset == ds].pivot_table(index="seed", columns="model", values=col)
            if a not in g or b not in g:
                continue
            dlt = (g[a] - g[b]).dropna().to_numpy()
            p = float(wilcoxon(dlt).pvalue) if len(dlt) > 1 and np.any(dlt != 0) else 1.0
            out.append({"baseline": b, "dataset": ds, "n": len(dlt), "method": float(g[a].mean()),
                        "other": float(g[b].mean()), "diff": float(dlt.mean()),
                        "wins": int((dlt > 0).sum()), "losses": int((dlt < 0).sum()), "p": p})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = holm(res.p.to_numpy())
    res["verdict"] = np.where(res.p_holm >= 0.05, "no significant difference",
                              np.where(res["diff"] > 0, "better", "worse"))
    return res


def tally(res) -> str:
    c = res.verdict.value_counts()
    return (f"better {c.get('better', 0)}, no significant difference "
            f"{c.get('no significant difference', 0)}, worse {c.get('worse', 0)}")


def reading(res) -> str:
    """The pre-registered reading of the primary result."""
    c = res.verdict.value_counts()
    worse, n = int(c.get("worse", 0)), len(res)
    if worse > n / 2:
        return ("Most tuned baselines beat CFMNN with repair: RQ2 narrows to repairing a CFMNN "
                "already in use, and 'beats Simpson's contraction' stops being the main context.")
    if worse == 0:
        return "CFMNN with repair is never significantly worse than a tuned fuzzy min-max baseline."
    return (f"Mixed: CFMNN with repair is significantly worse in {worse} of {n} comparisons; the paper "
            "names them.")


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    share = df[df.dataset == "skin"].skin_share.unique()
    models = [m for m in ["CFMNN-SR-val", "CFMNN-val"] + BASELINES +
              ["CFMNN-SR-0.3", "CFMNN-0.3"] + [f"{m}-0.3" for m in FMM_MODELS] if m in set(df.model)]
    L = ["# Run P2: CFMNN with self-repair against tuned fuzzy min-max networks", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The paper split, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}); theta by 3-fold cross-validation for every model. "
         "Written by `scripts/run_fmm_baselines.py`."]
    if len(share) and float(share[0]) < 1:
        L += ["", f"Skin: the stratified {float(share[0]):.0%} sample of the amendment."]
    prim = compare(df, METHOD, BASELINES)
    if not prim.empty:
        L += ["", "## Primary: test macro-F1, CFMNN-SR-val against each tuned baseline", "",
              f"Holm across all {len(prim)} tests. {tally(prim)}. {reading(prim)}", "",
              "| Baseline | Dataset | Seeds | CFMNN-SR-val | Baseline | Difference | Seeds better / worse | "
              "p | p (Holm) | Verdict for CFMNN-SR-val |", "|---|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {r.baseline} | {r.dataset} | {r.n} | {r.method:.2f} | {r.other:.2f} | {r['diff']:+.2f} | "
              f"{r.wins} / {r.losses} | {r.p:.4f} | {r.p_holm:.4f} | {r.verdict} |" for _, r in prim.iterrows()]
    for title, a, bs, col in (("Secondary: test accuracy, tuned", METHOD, BASELINES, "accuracy"),
                              ("Secondary: test macro-F1 at a fixed theta = 0.3", "CFMNN-SR-0.3",
                               [f"{m}-0.3" for m in FMM_MODELS], "macro_f1"),
                              ("Descriptive: test macro-F1, tuned CFMNN without repair", "CFMNN-val",
                               BASELINES, "macro_f1")):
        res = compare(df, a, bs, col)
        if res.empty:
            continue
        L += ["", f"## {title}", "", f"{a} against each; Holm across {len(res)} tests. {tally(res)}.", "",
              "| Baseline | Dataset | Difference | Seeds better / worse | p (Holm) | Verdict |",
              "|---|---|---|---|---|---|"]
        L += [f"| {r.baseline} | {r.dataset} | {r['diff']:+.2f} | {r.wins} / {r.losses} | {r.p_holm:.4f} | "
              f"{r.verdict} |" for _, r in res.iterrows()]
    for title, col, fmt in (("Test macro-F1 (%)", "macro_f1", "{:.2f}"), ("Boxes", "boxes", "{:.0f}"),
                            ("Seconds (tuning + fit + repair)", "seconds", "{:.0f}")):
        d = df.assign(seconds=df.tune_s + df.fit_s + df.repair_s)
        piv = d.pivot_table(index="dataset", columns="model", values=col).reindex(ds)[models]
        L += ["", f"## {title}, mean over seeds", "", "| Dataset | " + " | ".join(models) + " |",
              "|---|" + "---|" * len(models)]
        L += [f"| {k} | " + " | ".join(fmt.format(v) for v in r) + " |" for k, r in piv.iterrows()]
    val = df[df.setting == "val"].drop_duplicates(["dataset", "seed", "family"])
    L += ["", "## Theta chosen by cross-validation (count over seeds)", ""]
    for fam in FAMILIES:
        g = val[val.family == fam]
        L.append(f"- {fam}: " + "; ".join(
            f"{d} " + ", ".join(f"{k:g} ×{v}" for k, v in g[g.dataset == d].theta.value_counts().sort_index().items())
            for d in ds if (g.dataset == d).any()))
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P2: tuned fuzzy min-max baselines")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--families", nargs="+", choices=FAMILIES)
    ap.add_argument("--seeds", nargs="+", type=int, help="the pilot only")
    ap.add_argument("--skin-share", type=float, help="the Skin amendment (PREREGISTRATION.md)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--project", action="store_true", help="print the pilot's projection and the rule's verdict")
    ap.add_argument("--running", type=int, default=0, help="with --project: pilot tasks still running")
    ap.add_argument("--elapsed-hours", type=float, default=0.0, help="with --project: the pilot's wall-clock hours")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if a.project:
        df = pd.read_csv(csv) if csv.exists() and csv.stat().st_size else pd.DataFrame(
            columns=["dataset", "seed", "family", "task_s"])
        print(projection_text(df, a.jobs if a.jobs > 1 else 6, a.running, 3600 * a.elapsed_hours))
        return
    if not a.summary_only:
        tasks = [(d, s, f, a.skin_share) for d in a.datasets or DATASETS for s in a.seeds or SEEDS
                 for f in a.families or FAMILIES]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=a.jobs, cost=COST)
    df = pd.read_csv(csv, float_precision="round_trip", keep_default_na=False, na_values=[""])
    print(summarise(df, out))


if __name__ == "__main__":
    main()
