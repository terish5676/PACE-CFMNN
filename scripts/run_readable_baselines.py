"""Readable baselines and theta chosen by validation (planned in advance).

Rules: results/readable_baselines/PREREGISTRATION.md. Is CFMNN as accurate as other
models people can read, when every model is tuned the same way, and was theta = 0.3
a poor choice?

Every model gets the paper's 80/20 split, the seed's order of the training rows and
min-max scaling fitted on the training rows (as scripts/accuracy_table.py). A tuned
model picks its setting by stratified 3-fold cross-validation on the training rows
(the folds keep the row order), taking the simplest setting on an exact tie, and is
then refitted on all training rows and scored once on the test rows.

    python -B scripts/run_readable_baselines.py --jobs 6
    python -B scripts/run_readable_baselines.py --jobs 6 --resume     # continue after a stop
    python -B scripts/run_readable_baselines.py --summary-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

DATASETS = ["S1", "S2", "S3", "wine", "breast_cancer", "digits", "thyroid", "shuttle", "skin",
            "magic", "letter", "landsat", "page_blocks"]
SEEDS = list(range(70, 90))
FOLDS = 3
GAMMA = 4.0

# Settings tried by each tuned model, simplest first: on an exact tie the first one wins.
THETAS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]                 # larger theta = fewer boxes
GRIDS = {
    "CFMNN-val": THETAS,
    "Tree": [4, 8, 16, 32, 64, 128, 256, 512, 1024],            # max leaves
    "LogReg": [0.01, 0.1, 1.0, 10.0, 100.0],                    # C (smaller = more regularised)
    "kNN": [20, 10, 5, 3, 1],                                   # k (larger = smoother)
}
# Not tuned. EBM: main effects only (no pair terms), library defaults otherwise; its default
# with pair terms took 26 min for one fit on Wine. GFMM: theta = 0.3, as the explained model;
# tuning it took 36 min per seed on Skin. Both decided before the run (PREREGISTRATION.md).
FIXED = {"CFMNN-0.3": 0.3, "GFMM-0.3": 0.3}
EBM_KW: dict = {"interactions": 0}
MODELS = ["CFMNN-0.3", "CFMNN-val", "Tree", "LogReg", "EBM", "GFMM-0.3", "kNN", "Random Forest"]
READABLE = ["Tree", "LogReg", "EBM"]                            # compared with CFMNN-val
OUT = ROOT / "results" / "readable_baselines"


# ------------------------------------------------------------------ models
def make(model: str, setting, seed: int, inner_jobs: int):
    """A fresh, unfitted model with the given setting."""
    if model in ("CFMNN-0.3", "CFMNN-val"):
        return CFMNN(theta=setting, gamma=GAMMA)
    if model == "Tree":
        return DecisionTreeClassifier(max_leaf_nodes=setting, random_state=seed)
    if model == "LogReg":
        return LogisticRegression(C=setting, max_iter=5000)
    if model == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier
        # n_jobs = 1 always: EBM's own worker processes crash on this machine (joblib/loky).
        return ExplainableBoostingClassifier(random_state=seed, n_jobs=1, **EBM_KW)
    if model == "GFMM-0.3":
        # theta_min = theta: one pass at this theta, no shrinking loop. GFMM's theta caps each
        # box side; CFMNN's caps their sum (d * theta), so GFMM's boxes are at most as large.
        from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
        return OnlineGFMM(theta=setting, theta_min=setting, gamma=GAMMA)
    if model == "kNN":
        return KNeighborsClassifier(n_neighbors=setting, n_jobs=inner_jobs)
    if model == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=inner_jobs)
    raise ValueError(model)


def fixed_setting(model: str):
    return FIXED.get(model)


def cv_folds(y, seed: int, k: int = FOLDS):
    """Stratified folds whose index lists are sorted, so each fold keeps the row order."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(np.sort(tr), np.sort(va)) for tr, va in skf.split(np.zeros(len(y)), y)]


def pick(scores) -> int:
    """Index of the best mean CV accuracy; on an exact tie the first (simplest) setting."""
    scores = np.round(np.asarray(scores, float), 12)
    return int(np.flatnonzero(scores == scores.max())[0])


def tune(model, grid, X, y, seed, inner_jobs, folds=None):
    """Mean CV accuracy of each setting, and the chosen one."""
    folds = folds if folds is not None else cv_folds(y, seed)
    scores = []
    for s in grid:
        acc = [np.mean(make(model, s, seed, inner_jobs).fit(X[tr], y[tr]).predict(X[va]) == y[va])
               for tr, va in folds]
        scores.append(float(np.mean(acc)))
    return grid[pick(scores)], scores


# ------------------------------------------------------------------ size of a model
def box_size(V, W, tol: float = 1e-12):
    """Rules = boxes; conditions of a box = features whose interval is narrower than [0, 1]."""
    V, W = np.asarray(V, float), np.asarray(W, float)
    if V.size == 0:
        return 0, float("nan")
    cond = ((V > tol) | (W < 1 - tol)).sum(axis=1)
    return len(V), float(cond.mean())


def tree_size(tree):
    """Rules = leaves; conditions of a leaf = distinct features tested on its path."""
    t = tree.tree_
    counts, stack = [], [(0, frozenset())]
    while stack:
        node, feats = stack.pop()
        if t.children_left[node] == -1:
            counts.append(len(feats))
            continue
        f = feats | {int(t.feature[node])}
        stack += [(t.children_left[node], f), (t.children_right[node], f)]
    return len(counts), float(np.mean(counts))


def model_size(model: str, m):
    if model.startswith("CFMNN"):
        return box_size(m.V_, m.W_)
    if model == "GFMM-0.3":
        return box_size(m.V, m.W)
    if model == "Tree":
        return tree_size(m)
    if model == "EBM":
        return len(m.term_features_), float(np.mean([len(f) for f in m.term_features_]))
    if model == "LogReg":
        return int(np.count_nonzero(m.coef_)), float("nan")
    return float("nan"), float("nan")


# ------------------------------------------------------------------ one dataset and seed
def prepare(name: str, seed: int):
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])        # as accuracy_table.py
    sc = data.fit_scaler(ds.X[order])
    return (sc.transform(ds.X[order]), ds.y[order],
            sc.transform(ds.X[sp["test"]]), ds.y[sp["test"]])


def run_one(name: str, seed: int, inner_jobs: int = 1, models=None):
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = prepare(name, seed)
    folds = cv_folds(ytr, seed)
    rows = []
    for model in models or MODELS:
        t0 = time.perf_counter()
        if model in GRIDS:
            setting, cv = tune(model, GRIDS[model], Xtr, ytr, seed, inner_jobs, folds)
        else:
            setting, cv = fixed_setting(model), []
        t1 = time.perf_counter()
        m = make(model, setting, seed, inner_jobs).fit(Xtr, ytr)
        t2 = time.perf_counter()
        pred = np.asarray(m.predict(Xte))
        t3 = time.perf_counter()
        rules, cond = model_size(model, m)
        rows.append({"dataset": name, "seed": seed, "model": model,
                     "setting": setting, "cv_scores": ";".join(f"{s:.6f}" for s in cv),
                     "accuracy": 100 * float(np.mean(pred == yte)),
                     "macro_f1": 100 * float(f1_score(yte, pred, average="macro")),
                     "rules": rules, "conditions_per_rule": cond,
                     "tune_s": t1 - t0, "fit_s": t2 - t1, "predict_s": t3 - t2,
                     "n_train": len(ytr), "n_test": len(yte)})
    print(f"{name} seed {seed}: " + ", ".join(f"{r['model']} {r['accuracy']:.1f}" for r in rows),
          flush=True)
    return rows


# ------------------------------------------------------------------ summary
def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(df, a, b, col="accuracy"):
    """Paired two-sided Wilcoxon over seeds, Holm across datasets: a vs b."""
    out = []
    for ds in [d for d in DATASETS if d in set(df.dataset)]:
        g = df[df.dataset == ds].pivot_table(index="seed", columns="model", values=col)
        if a not in g or b not in g:
            continue
        d = (g[a] - g[b]).dropna().to_numpy()
        p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else 1.0
        out.append({"dataset": ds, "diff": float(d.mean()), "p": float(p), "n": len(d)})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = holm(res.p.to_numpy())
    res["verdict"] = np.where(res.p_holm >= 0.05, "level", np.where(res["diff"] > 0, "better", "worse"))
    return res


def tally(res) -> str:
    if res.empty:
        return "no data"
    c = res.verdict.value_counts()
    return f"better {c.get('better', 0)}, level {c.get('level', 0)}, worse {c.get('worse', 0)}"


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# Readable baselines and theta chosen by validation", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The paper's 80/20 split, "
         f"seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}). Test accuracy (%), mean over seeds. "
         "Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; "
         "better / level / worse for the first model.", ""]
    models = [m for m in MODELS if m in set(df.model)]
    acc = df.pivot_table(index="dataset", columns="model", values="accuracy").reindex(
        [d for d in DATASETS if d in set(df.dataset)])[models]
    L += ["## Accuracy", "", "| Dataset | " + " | ".join(models) + " |",
          "|---|" + "---|" * len(models)]
    L += [f"| {ds} | " + " | ".join(f"{v:.1f}" for v in r) + " |" for ds, r in acc.iterrows()]

    def block(title, pairs, col):
        L.extend(["", f"## {title}", ""])
        for a, b in pairs:
            res = compare(df, a, b, col)
            if res.empty:
                continue
            L.append(f"**{a} vs {b}** ({col}): {tally(res)}")
            L.append("")
            L.append("| Dataset | Mean difference | p | Holm p | Verdict |")
            L.append("|---|---|---|---|---|")
            L.extend(f"| {r.dataset} | {r['diff']:+.2f} | {r.p:.4f} | {r.p_holm:.4f} | {r.verdict} |"
                     for _, r in res.iterrows())
            L.append("")

    primary = [("CFMNN-val", b) for b in READABLE] + [("CFMNN-0.3", "GFMM-0.3")]
    block("Primary: CFMNN-val against the readable models, and CFMNN against GFMM at the same "
          "theta (accuracy)", primary, "accuracy")
    block("Secondary: the explained model, CFMNN-0.3, against the readable models (accuracy)",
          [("CFMNN-0.3", b) for b in READABLE], "accuracy")
    block("Secondary: theta by validation against theta = 0.3 (accuracy)",
          [("CFMNN-val", "CFMNN-0.3")], "accuracy")
    block("Secondary: the primary comparisons by macro-F1", primary, "macro_f1")

    L += ["", "## Descriptive", ""]
    if {"kNN", "Random Forest"} <= set(models):
        ref = acc[["kNN", "Random Forest"]].max(axis=1)
        L.append("Gap to the better of the tuned kNN and the Random Forest (points): "
                 + ", ".join(f"{m} {(acc[m] - ref).min():+.1f} to {(acc[m] - ref).max():+.1f}"
                             for m in ["CFMNN-0.3", "CFMNN-val"] if m in acc))
        L.append("")
    tuned = df[df.model.isin(GRIDS)]
    L += ["Settings chosen (count over seeds):", ""]
    for (m, ds), g in tuned.groupby(["model", "dataset"], sort=False):
        L.append(f"- {m}, {ds}: " + ", ".join(f"{k} ×{v}" for k, v in g.setting.value_counts().items()))
    size = df.groupby(["dataset", "model"])[["rules", "conditions_per_rule", "fit_s", "predict_s"]].mean()
    L += ["", "Mean size and time per dataset (rules; conditions per rule; fit s; predict s):", "",
          size.round(2).to_string(), ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Readable baselines and theta by validation")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--models", nargs="+", choices=MODELS)
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
        inner = 1 if a.jobs > 1 else -1
        tasks = [(d, s, inner, a.models) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        # Slowest first (seconds per seed in the smoke runs: Letter ~2,400, Skin ~2,200,
        # Shuttle ~1,300, S1-S3 ~400). This only orders the tasks.
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs,
                {"letter": 6, "skin": 5, "shuttle": 4, "S1": 2, "S2": 2, "S3": 2,
                 "magic": 1, "landsat": 1})
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
