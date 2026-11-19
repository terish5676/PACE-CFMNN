"""Refinement v2: the confirmatory run for the reframed RQ3 and RQ4 (seeds 50-59).

Rules: ``results/refinement_v2_confirmatory/PREREGISTRATION.md`` (committed
before this run). The design follows the exploratory checks
(``results/refinement_v2_explore/``, seeds 0-2).

For each dataset and seed, with v1's split, budget, rounds, gate and box cap
(``run_refinement.split``):

* B0: no refinement.
* B9: CFMNN continued on every pool label (v1's reference).
* SR: self-repair, the smallest-cut shrink tried on ``budget`` of the
  model's own training errors (the same number as the labels AL buys), gated
  on the training rows. It uses no label beyond the training rows (the
  exploratory arm SR-300-train).
* AL: active learning alone, ``SELECTOR`` with the smallest-cut shrink
  (A+B+C'-volume), ``budget`` pool labels, gated on the acceptance rows.
* SR+AL: self-repair, then the same active learning on top.
* Retraining with every label SR+AL used (training rows, its bought labels
  and the acceptance rows): CFMNN in a fresh random order, CFMNN with the new
  rows appended to the training order, kNN (k = 10) and a Random Forest.

Measured: test metrics; prediction churn against B0 on the whole test set
(changed, fixed, broken and the fix share, fixed / changed); for the
edit-based arms, the box-level and tight certificates of every accepted edit
on a fixed sample of 2,000 test rows (``audit.certify``); PACE-flip top-3
churn on 100 of those rows.

    python scripts/run_refinement_v2.py --jobs 4
    python scripts/run_refinement_v2.py --jobs 4 --resume
    python scripts/run_refinement_v2.py --summary-only
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import LabelOracle, run_refinement, run_self_repair  # noqa: E402
from pace_cfmnn.audit import explanation_churn, prediction_churn  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.explain import pace_flip  # noqa: E402
from run_refinement import COST, MAX_BUDGET, N_ROUNDS, split  # noqa: E402
from run_refinement_explore import Certificates, edit_stats  # noqa: E402

DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = list(range(50, 60))
SELECTOR = "order_ensemble"
VOL = "A+B+C'-volume"
N_AUDIT, N_EXPLAIN, N_FLIP_BG = 2000, 100, 200
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "min_class_recall"]
F1_MARGIN = 1.0      # non-inferiority margin for macro-F1, in points
RECALL_MARGIN = 2.0  # non-inferiority margin for the lowest class recall, in points


def run_one(name: str, seed: int):
    t0 = time.perf_counter()
    P = split(name, seed)
    (Xtr, ytr), (Xp, yp), (Xi, _), (Xa, ya), (Xt, yt) = (P[r] for r in
                                                       ("train", "pool", "inner", "accept", "test"))
    model = CFMNN().fit(Xtr, ytr)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    au = np.random.default_rng(seed + 3).permutation(len(Xt))[:min(N_AUDIT, len(Xt))]
    Xau = Xt[au]
    Xex, bg = Xau[:N_EXPLAIN], Xtr[:N_FLIP_BG]
    p0 = model.predict(Xt)
    e0 = pace_flip(model, Xex, bg)
    base = {"dataset": name, "seed": seed, "n_train": len(Xtr), "n_pool": len(Xp),
            "n_accept": len(Xa), "n_test": len(Xt), "budget": budget, "boxes_start": model.n_boxes_,
            "train_error_start": 100 * np.mean(model.predict(Xtr) != ytr)}
    runs, edits_all = [], []

    def record(arm, pred, pool_labels, acc_labels, t, m=None, cert=None, edits=None, own=None):
        rec = {**base, "arm": arm, "pool_labels": pool_labels, "accept_labels": acc_labels,
               "total_labels": pool_labels + acc_labels,
               **{f"test_{k}": v for k, v in classification_metrics(yt, pred).items() if k in METRICS},
               **{f"churn_{k}": v for k, v in prediction_churn(yt, p0, pred).items()},
               "time_s": time.perf_counter() - t}
        rec["churn_changed_share"] *= 100
        rec["churn_broken_share"] = 100 * rec["churn_broken"] / len(yt)
        rec["churn_fixed_share"] = 100 * rec["churn_fixed"] / len(yt)
        moved = rec["churn_fixed"] + rec["churn_broken"]   # changes between two wrong classes excluded
        rec["churn_fix_share"] = 100 * rec["churn_fixed"] / moved if moved else np.nan
        if m is not None:
            e1 = pace_flip(m, Xex, bg)
            rec.update({"boxes_end": m.n_boxes_, "train_error_end": 100 * np.mean(m.predict(Xtr) != ytr),
                        **{f"expl_{k}": v for k, v in explanation_churn(e0, e1).items()},
                        "expl_top1_change_share": explanation_churn(e0, e1, k=1)["topk_change_share"]})
            if cert is not None:
                rec.update(cert.summary())
                for kind in ("loose", "tight"):
                    out = ~cert.union[kind][:N_EXPLAIN]
                    c = explanation_churn(e0[out], e1[out])
                    rec.update({f"expl_{kind}_out_n": c["n"], f"expl_{kind}_out_top3": c["topk_change_share"]})
        if own is not None:
            c = prediction_churn(yt, own, pred)
            rec.update({"own_churn_changed_share": 100 * c["changed_share"],
                        "own_churn_broken_share": 100 * c["broken"] / len(yt)})
        if edits is not None:
            rec.update(edit_stats(edits, budget))
            edits_all.extend({"dataset": name, "seed": seed, "arm": arm, **e} for e in edits)
        runs.append(rec)
        print(f"  {name} seed {seed} {arm}: macro-F1 {rec['test_macro_f1']:.2f}, labels "
              f"{rec['total_labels']}, changed {rec['churn_changed_share']:.2f}%, broken "
              f"{rec['churn_broken_share']:.2f}% ({rec['time_s']:.0f}s)", flush=True)

    t = time.perf_counter()
    record("B0", p0, 0, 0, t, m=model)
    t = time.perf_counter()
    order = np.random.default_rng(seed + 1).permutation(len(Xp))
    b9 = CFMNN().fit(np.vstack([Xtr, Xp[order]]), np.concatenate([ytr, yp[order]]))
    record("B9", b9.predict(Xt), len(Xp), 0, t, m=b9)

    t = time.perf_counter()
    cert = Certificates(Xau)
    m, _, el = run_refinement(model, Xtr, ytr, Xp, LabelOracle(yp), Xi, Xa, ya, SELECTOR, VOL,
                              budget=budget, n_rounds=N_ROUNDS, seed=seed, on_accept=cert)
    record("AL", m.predict(Xt), budget, len(Xa), t, m=m, cert=cert, edits=el)

    # Self-repair gated on the training rows: no label beyond them.
    t = time.perf_counter()
    cert = Certificates(Xau)
    sr, _, el = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=budget,
                                n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False,
                                on_accept=cert)
    record("SR", sr.predict(Xt), 0, 0, t, m=sr, cert=cert, edits=el)

    t = time.perf_counter()
    oracle = LabelOracle(yp)
    m, _, el = run_refinement(sr, Xtr, ytr, Xp, oracle, Xi, Xa, ya, SELECTOR, VOL, budget=budget,
                              n_rounds=N_ROUNDS, seed=seed, on_accept=cert)
    assert len(oracle.revealed) == budget == len(set(oracle.revealed))
    record("SR+AL", m.predict(Xt), budget, len(Xa), t, m=m, cert=cert, edits=el)

    # Retraining with every label SR+AL used.
    bought = oracle.revealed
    Xn, yn = np.vstack([Xp[bought], Xa]), np.concatenate([yp[bought], ya])
    t = time.perf_counter()
    m = CFMNN().fit(np.vstack([Xtr, Xn]), np.concatenate([ytr, yn]))
    record("Retrain, appended", m.predict(Xt), budget, len(Xa), t, m=m)
    t = time.perf_counter()
    X_all, y_all = np.vstack([Xtr, Xn]), np.concatenate([ytr, yn])
    shuf = np.random.default_rng(seed + 5).permutation(len(y_all))
    m = CFMNN().fit(X_all[shuf], y_all[shuf])
    record("Retrain, shuffled", m.predict(Xt), budget, len(Xa), t, m=m)
    for kind, make in (("kNN", lambda: KNeighborsClassifier(n_neighbors=10)),
                       ("RF", lambda: RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1))):
        t = time.perf_counter()
        start = make().fit(Xtr, ytr).predict(Xt)
        record(f"{kind}, training rows", start, 0, 0, t)
        t = time.perf_counter()
        record(f"{kind}, every label", make().fit(X_all, y_all).predict(Xt), budget, len(Xa), t, own=start)
    print(f"{name} seed {seed} done in {time.perf_counter() - t0:.0f}s", flush=True)
    return runs, edits_all


# --------------------------------------------------------------------- summary
NAMES = {"thyroid": "Thyroid", "landsat": "Landsat", "page_blocks": "Page Blocks",
         "magic": "MAGIC", "letter": "Letter", "skin": "Skin"}
# (name, question, value, first arm, second arm, kind, margin); kind "two-sided" or "ni"
# (one-sided non-inferiority: first − second > −margin).
ENDPOINTS = [
    ("P3", "Do bought labels add anything after self-repair?", "test_macro_f1", "SR+AL", "SR",
     "two-sided", 0.0),
    ("S3a", "Does self-repair (no new labels) help?", "test_macro_f1", "SR", "B0", "two-sided", 0.0),
    ("S3b", "Is self-repair no worse than bought labels alone, within the margin?", "test_macro_f1",
     "SR", "AL", "ni", F1_MARGIN),
    ("S3c", "Is SR+AL's lowest class recall no worse than B0's, within the margin?",
     "test_min_class_recall", "SR+AL", "B0", "ni", RECALL_MARGIN),
    ("S3d", "Is SR's lowest class recall no worse than B0's, within the margin?",
     "test_min_class_recall", "SR", "B0", "ni", RECALL_MARGIN),
    ("P4", "Of the predictions an update changes, are more of them fixes with repair than with retraining?",
     "churn_fix_share", "SR+AL", "Retrain, shuffled", "two-sided", 0.0),
    ("S4a", "Does repair break fewer correct predictions than retraining?", "churn_broken_share",
     "Retrain, shuffled", "SR+AL", "two-sided", 0.0),
    ("S4b", "Does repair change fewer explanations (PACE-flip top 3) than retraining?",
     "expl_topk_change_share", "Retrain, shuffled", "SR+AL", "two-sided", 0.0),
    ("S4c", "Is repair more accurate than retraining with the same labels?", "test_macro_f1", "SR+AL",
     "Retrain, shuffled", "two-sided", 0.0),
]
EDIT_ARMS = ["AL", "SR", "SR+AL"]
# Skin has 3 features, so its top-3 set is every feature and never changes.
EXCLUDE = {"S4b": ("skin",)}


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def test_endpoint(runs, value, a, b, kind, margin=0.0, exclude=()):
    piv = runs.pivot_table(index=["dataset", "seed"], columns="arm", values=value, dropna=False)
    rows = []
    for ds in [d for d in DATASETS if d in piv.index.get_level_values(0) and d not in exclude]:
        d = (piv.loc[ds, a] - piv.loc[ds, b]).dropna().to_numpy()   # a seed with no change: dropped
        if kind == "ni":
            x = d + margin
            p = wilcoxon(x, alternative="greater").pvalue if np.any(x != 0) else 1.0
        else:
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append({"dataset": ds, "diff": d.mean(), "wins": int((d > 0).sum()),
                     "losses": int((d < 0).sum()), "n": len(d), "p": p})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    return t


def mean_table(runs, arms, value, fmt="{:.1f}") -> list[str]:
    ds = [d for d in DATASETS if d in set(runs.dataset)]
    piv = runs.pivot_table(index="arm", columns="dataset", values=value, aggfunc="mean")
    lines = ["| Arm | " + " | ".join(NAMES[d] for d in ds) + " |", "|---|" + "---|" * len(ds)]
    for a in arms:
        if a in piv.index:
            lines.append(f"| {a} | " + " | ".join(
                "" if pd.isna(piv.loc[a, d]) else fmt.format(piv.loc[a, d]) for d in ds) + " |")
    return lines + [""]


def summarise(out: Path) -> str:
    runs = pd.read_csv(out / "runs.csv")
    seeds = sorted(runs.seed.unique())
    ds = [d for d in DATASETS if d in set(runs.dataset)]
    arms = list(dict.fromkeys(runs.arm))
    lines = ["# Refinement v2: confirmatory run", "",
             f"Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Seeds {seeds[0]}–{seeds[-1]} "
             f"({len(seeds)} seeds), {len(ds)} datasets. Paired Wilcoxon over seeds, Holm across the "
             "datasets of each endpoint; `*` = Holm-adjusted p < 0.05. Two-sided, except the "
             "non-inferiority endpoints (S3b, S3c, S3d), which are one-sided tests of first − second "
             f"> −margin (macro-F1 margin {F1_MARGIN:g} point, lowest class recall margin "
             f"{RECALL_MARGIN:g} points). Written by `scripts/run_refinement_v2.py`.", "",
             "## Endpoints", "", "Mean difference (first arm − second arm) over seeds:", "",
             "| Endpoint | Question | Value | First − second | " + " | ".join(NAMES[d] for d in ds) + " |",
             "|---|---|---|---|" + "---|" * len(ds)]
    detail = []
    for name, q, value, a, b, kind, margin in ENDPOINTS:
        t = test_endpoint(runs, value, a, b, kind, margin, EXCLUDE.get(name, ())).set_index("dataset")
        cells = [f"{t.loc[d, 'diff']:+.2f}{'*' if t.loc[d, 'p_holm'] < 0.05 else ''}"
                 if d in t.index else "–" for d in ds]
        lines.append(f"| {name} | {q} | {value} | {a} − {b} | " + " | ".join(cells) + " |")
        detail.append(f"| {name} | " + " | ".join(
            f"{t.loc[d, 'wins']}–{t.loc[d, 'losses']} (n={t.loc[d, 'n']}), p={t.loc[d, 'p']:.3g}, "
            f"Holm {t.loc[d, 'p_holm']:.3g}" if d in t.index else "not tested" for d in ds) + " |")
    lines += ["", "Wins–losses over seeds (n = seeds tested), raw p and Holm-adjusted p:", "",
              "| Endpoint | " + " | ".join(NAMES[d] for d in ds) + " |", "|---|" + "---|" * len(ds)]
    lines += detail + [""]
    edit_runs = runs[runs.arm.isin(EDIT_ARMS)]
    lines += ["## Certificate check (C4)", "",
              f"Test-row changes outside the tight certificate, summed over {len(edit_runs)} edit-based "
              f"runs: **{int(edit_runs.cert_tight_misses.sum())}** (must be 0); outside the box-level "
              f"certificate: **{int(edit_runs.cert_loose_misses.sum())}** (must be 0). Accepted edits "
              f"checked: {int(edit_runs.cert_edits.sum()):,}.", ""]
    means = runs.pivot_table(index="arm", columns="dataset", values="test_macro_f1", aggfunc="mean")
    if {"B0", "RF, every label"} <= set(means.index):
        gap = means.loc["RF, every label"] - means.loc["B0"]
        lines += ["## Share of the gap to RF closed (macro-F1, %)", "",
                  "100 × (arm − B0) / (RF with every label − B0), from the means over seeds.", "",
                  "| Arm | " + " | ".join(NAMES[d] for d in ds) + " |", "|---|" + "---|" * len(ds)]
        for a in ["SR", "AL", "SR+AL", "Retrain, shuffled", "Retrain, appended", "B9"]:
            if a in means.index:
                lines.append(f"| {a} | " + " | ".join(
                    f"{100 * (means.loc[a, d] - means.loc['B0', d]) / gap[d]:.0f}" if gap[d] > 0 else "–"
                    for d in ds) + " |")
        lines += [""]
    for title, value, fmt in (("Test macro-F1 (%)", "test_macro_f1", "{:.1f}"),
                              ("Lowest class recall (%)", "test_min_class_recall", "{:.1f}"),
                              ("Test accuracy (%)", "test_accuracy", "{:.1f}"),
                              ("Labels used beyond the training rows", "total_labels", "{:,.0f}"),
                              ("Test predictions changed against B0 (%)", "churn_changed_share", "{:.2f}"),
                              ("Test predictions broken, right → wrong, against B0 (% of test rows)",
                               "churn_broken_share", "{:.2f}"),
                              ("Test predictions fixed, wrong → right, against B0 (% of test rows)",
                               "churn_fixed_share", "{:.2f}"),
                              ("Fix share: fixed / (fixed + broken), against B0 (%)",
                               "churn_fix_share", "{:.1f}"),
                              ("PACE-flip top-3 changed against B0, 100 test rows (share)",
                               "expl_topk_change_share", "{:.2f}"),
                              ("PACE-flip top feature changed against B0, 100 test rows (share)",
                               "expl_top1_change_share", "{:.2f}"),
                              ("Training error at the end (%)", "train_error_end", "{:.1f}"),
                              ("Run time (s)", "time_s", "{:.0f}")):
        lines += [f"## {title}", ""] + mean_table(runs, arms, value, fmt)
    edit_arms = EDIT_ARMS
    lines += ["## Certificates (edit-based arms; 2,000 test rows)", "",
              "SR+AL's certificates cover all its edits, self-repair included.", ""]
    for kind in ("tight", "loose"):
        for title, value, fmt in ((f"{kind}: changes outside it (must be 0)", f"cert_{kind}_misses", "{:.0f}"),
                                  (f"{kind}: size per edit (% of rows)", f"cert_{kind}_mean_size_share", "{:.2f}"),
                                  (f"{kind}: precision, changed / certified (%)", f"cert_{kind}_precision", "{:.1f}"),
                                  (f"{kind}: rows some edit could change (%)", f"cert_{kind}_union_share", "{:.1f}"),
                                  (f"{kind}: PACE-flip top-3 changed outside it (share)", f"expl_{kind}_out_top3", "{:.2f}")):
            lines += [f"### {title}", ""] + mean_table(runs, edit_arms, value, fmt)
    lines += ["## kNN and RF: predictions changed against the same learner on the training rows (%)", ""]
    lines += mean_table(runs, ["kNN, every label", "RF, every label"], "own_churn_changed_share", "{:.2f}")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default=str(ROOT / "results" / "refinement_v2_confirmatory"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        tasks = [(n, s) for n in a.datasets for s in a.seeds]
        path = lambda t: parts / f"{t[0]}__{t[1]}.pkl"
        todo = [t for t in tasks if not (a.resume and path(t).exists())]

        def save(t, res):
            with open(path(t), "wb") as f:
                pickle.dump(res, f)
            res_all = []
            for d in tasks:
                if path(d).exists():
                    with open(path(d), "rb") as f:
                        res_all.append(pickle.load(f))
            pd.DataFrame([r for rs, _ in res_all for r in rs]).to_csv(out / "runs.csv", index=False)
            pd.DataFrame([e for _, es in res_all for e in es]).to_csv(
                out / "edits.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                futures = {ex.submit(run_one, *t): t for t in
                           sorted(todo, key=lambda t: -COST.get(t[0], 0))}
                for f in as_completed(futures):
                    save(futures[f], f.result())
        else:
            for t in todo:
                save(t, run_one(*t))
    print(summarise(out))


if __name__ == "__main__":
    main()
