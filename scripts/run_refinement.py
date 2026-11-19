"""Explanation-guided selective refinement (blueprint 7, 9.3; RQ3-RQ4).

Three protocols:

* ``pilot`` (seeds 0-2, exploratory): checks that the loop runs, measures
  cost, and counts how often each edit applies and passes the gate.
* ``eligibility`` (pilot seeds only): counts, per dataset, the pool points the
  starting model gets wrong while they lie inside a wrong-class box, i.e. what
  a shrink could repair. A dataset enters the confirmatory run if every pilot
  seed has at least ``MIN_REPAIRABLE`` of them.
* ``confirmatory`` (seeds 10-19): the pre-registered run
  (``results/refinement_confirmatory/PREREGISTRATION.md``). Repair arms as in
  the pilot, plus random selection with A+B+C for the selection test.

For each dataset and seed:

1. Role split 50/20/10/10/10 (``data.role_split``). Training rows are
   shuffled with the seed; the scaler is fitted on them; rows are clipped to
   [0, 1]. The main model (fixed centroid + search expansion) is trained.
2. Arms (all with the same label budget, rounds, gate and box cap):
   * repair test (RQ4): order-ensemble selection with A+B, A+B+C,
     A+B+C'-random and A+B+C'-volume;
   * selection test (RQ3): A+B with random, margin, PD + U* and the order
     ensemble (shared with the repair test);
   * references: B0, no refinement; B9, refit on training + the whole pool.
3. Test rows are measured after every round; nothing reads them for a
   decision. Every arm asserts the oracle audit: exactly the budgeted pool
   labels were revealed, each once.

Usage (from the project folder):
    python scripts/run_refinement.py
    python scripts/run_refinement.py --protocol eligibility
    python scripts/run_refinement.py --protocol confirmatory --jobs 4
    python scripts/run_refinement.py --datasets page_blocks --seeds 0
    python scripts/run_refinement.py --summary-only
    python scripts/run_refinement.py --jobs 4      # 4 dataset x seed runs at a time

Every run is seeded and independent, so ``--jobs`` changes only the speed:
the CSV files are identical (rows are written in a fixed order) apart from
the timing columns, which grow when runs share the CPU. Each worker uses one
BLAS thread.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.active import LabelOracle, run_refinement  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.neighborhood import Perturber  # noqa: E402
from run_error_detection import ci95, load, paired  # noqa: E402

PILOT_DATASETS = ["S2", "thyroid", "landsat", "page_blocks", "magic"]
PILOT_SEEDS = [0, 1, 2]
N_ROUNDS = 10
MAX_BUDGET = 300
REPAIR = ["A+B", "A+B+C", "A+B+C'-random", "A+B+C'-volume"]
SELECT = ["random", "margin", "pd_u_star"]
# (arm name, selector, operation); the order-ensemble A+B arm belongs to both tests.
ARMS = [(f"order_ensemble | {op}", "order_ensemble", op) for op in REPAIR] + \
       [(f"{sel} | A+B", sel, "A+B") for sel in SELECT]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "min_class_recall"]

# Confirmatory run (pre-registered in results/refinement_confirmatory/PREREGISTRATION.md).
CONF_DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
CONF_SEEDS = list(range(10, 20))
CONF_ARMS = [(f"order_ensemble | {op}", "order_ensemble", op) for op in REPAIR] + \
            [("random | A+B+C", "random", "A+B+C")]
# Eligibility (pilot seeds only): candidates of at least 2,000 rows, so the role split applies.
ELIG_CANDIDATES = ["S1", "S2", "S3", "S2-imbalanced", "S2-label-noise", "thyroid", "shuttle",
                   "landsat", "page_blocks", "magic", "letter", "skin"]
MIN_REPAIRABLE = 25
# Larger first, so parallel workers finish together (does not change any result).
COST = {"skin": 6, "letter": 5, "magic": 4, "S2": 3, "landsat": 2, "thyroid": 1}
# Pre-registered comparisons: (name, first arm, second arm).
ENDPOINTS = [("R1 (primary): A+B+C minus A+B+C'-volume", "order_ensemble | A+B+C",
              "order_ensemble | A+B+C'-volume"),
             ("R2: A+B+C minus A+B", "order_ensemble | A+B+C", "order_ensemble | A+B"),
             ("R3: A+B+C minus A+B+C'-random", "order_ensemble | A+B+C",
              "order_ensemble | A+B+C'-random"),
             ("S1: order ensemble minus random selection, both A+B+C", "order_ensemble | A+B+C",
              "random | A+B+C")]


def split(name: str, seed: int):
    ds = load(name, seed)
    roles = data.role_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(roles["train"])
    scaler = data.fit_scaler(ds.X[order])
    tf = lambda X: np.clip(scaler.transform(X), 0.0, 1.0)
    part = {r: (tf(ds.X[i]), ds.y[i]) for r, i in roles.items() if r != "train"}
    part["train"] = (tf(ds.X[order]), ds.y[order])
    return part


def run_one(name: str, seed: int, protocol: str = "pilot"):
    t0 = time.perf_counter()
    P = split(name, seed)
    (Xtr, ytr), (Xp, yp), (Xi, _), (Xa, ya), (Xt, yt) = (P[r] for r in
                                                       ("train", "pool", "inner", "accept", "test"))
    model = CFMNN().fit(Xtr, ytr)
    pert = Perturber(kappa=0.05, k=50).fit(Xtr)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    base = {"dataset": name, "seed": seed, "n_train": len(Xtr), "n_pool": len(Xp),
            "n_accept": len(Xa), "n_test": len(Xt), "budget": budget, "n_rounds": N_ROUNDS,
            "boxes_start": model.n_boxes_}
    runs, rounds, edits = [], [], []
    start = classification_metrics(yt, model.predict(Xt))
    runs.append({**base, "arm": "B0 no refinement", "selector": "", "operation": "",
                 "boxes_end": model.n_boxes_, **{f"test_{k}": start[k] for k in METRICS},
                 "time_s": 0.0})
    order = np.random.default_rng(seed + 1).permutation(len(Xp))
    t = time.perf_counter()
    full = CFMNN().fit(np.vstack([Xtr, Xp[order]]), np.concatenate([ytr, yp[order]]))
    m9 = classification_metrics(yt, full.predict(Xt))
    runs.append({**base, "arm": "B9 refit on all pool labels", "selector": "", "operation": "",
                 "boxes_end": full.n_boxes_, **{f"test_{k}": m9[k] for k in METRICS},
                 "time_s": time.perf_counter() - t})
    for arm, sel, op in (CONF_ARMS if protocol == "confirmatory" else ARMS):
        t = time.perf_counter()
        oracle = LabelOracle(yp)
        m, rl, el = run_refinement(model, Xtr, ytr, Xp, oracle, Xi, Xa, ya, sel, op,
                                   budget=budget, n_rounds=N_ROUNDS, seed=seed, perturber=pert,
                                   X_test=Xt, y_test=yt)
        # Audit: exactly the budgeted labels were revealed, each once, all of them logged.
        assert len(oracle.revealed) == budget == len(set(oracle.revealed))
        assert set(oracle.revealed) == {e["pool_index"] for e in el}
        e = pd.DataFrame(el)
        first = e.drop_duplicates("pool_index")
        ok = e[e.accepted]
        sh = e[e.edit == "shrink"]
        runs.append({**base, "arm": arm, "selector": sel, "operation": op, "boxes_end": m.n_boxes_,
                     **{f"test_{k}": rl[-1][f"test_{k}"] for k in METRICS},
                     "accept_macro_f1": rl[-1]["accept_macro_f1"],
                     "n_wrong_selected": int(first.was_wrong.sum()),
                     "n_shrink_possible": int(first.shrink_possible.sum()),
                     "n_shrink_tried": len(sh), "n_shrink_accepted": int(sh.accepted.sum()),
                     "n_add_accepted": int((ok.edit == "add").sum()),
                     "n_points_left_out": int(budget - len(ok)),
                     "n_rejected_box_cap": int((e.reason == "box cap").sum()),
                     "n_rejected_f1_or_recall": int(e.reason.isin(["macro-F1", "class recall"]).sum()),
                     "mean_share_cut": float(sh.shrink_share_cut.mean()) if len(sh) else np.nan,
                     "mean_rows_cut": float(sh.shrink_n_cut.mean()) if len(sh) else np.nan,
                     "time_s": time.perf_counter() - t})
        for rec in rl:
            rounds.append({"dataset": name, "seed": seed, "arm": arm, **rec})
        for rec in el:
            edits.append({"dataset": name, "seed": seed, "arm": arm, **rec})
        print(f"  {name} seed {seed} {arm}: test macro-F1 {start['macro_f1']:.2f} -> "
              f"{rl[-1]['test_macro_f1']:.2f}, boxes {model.n_boxes_} -> {m.n_boxes_}, "
              f"wrong {runs[-1]['n_wrong_selected']}, shrink possible "
              f"{runs[-1]['n_shrink_possible']} accepted {runs[-1]['n_shrink_accepted']} "
              f"({runs[-1]['time_s']:.0f}s)", flush=True)
    print(f"{name} seed {seed} done in {time.perf_counter() - t0:.0f}s", flush=True)
    return runs, rounds, edits


# --------------------------------------------------------------------- summary
def eligibility(out: Path) -> str:
    """Repairable pool errors per dataset on the pilot seeds (no refinement is run)."""
    rows = []
    for name in ELIG_CANDIDATES:
        for seed in PILOT_SEEDS:
            P = split(name, seed)
            (Xtr, ytr), (Xp, yp) = P["train"], P["pool"]
            m = CFMNN().fit(Xtr, ytr)
            det = m.predict_details(Xp)
            wrong = det.cls != np.searchsorted(m.classes_, yp)
            w = det.winner
            inside = ((m.V_[w] <= Xp) & (Xp <= m.W_[w])).all(axis=1)
            rows.append({"dataset": name, "seed": seed, "pool": len(Xp),
                         "pool_errors": int(wrong.sum()), "repairable": int((wrong & inside).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "eligibility.csv", index=False)
    lines = ["# Which datasets can the shrink repair? (pilot seeds 0-2)", "",
             "Repairable = pool points the starting model gets wrong while they lie inside the "
             f"winning wrong-class box. Eligible: at least {MIN_REPAIRABLE} in every pilot seed. "
             "No refinement result is used.", "",
             "| Dataset | Pool rows | Pool errors (per seed) | Repairable (per seed) | Eligible |",
             "|---|---|---|---|---|"]
    for ds, g in df.groupby("dataset", sort=False):
        ok = (g.repairable >= MIN_REPAIRABLE).all()
        lines.append(f"| {ds} | {g.pool.iloc[0]} | {', '.join(map(str, g.pool_errors))} | "
                     f"{', '.join(map(str, g.repairable))} | {'yes' if ok else 'no'} |")
    text = "\n".join(lines) + "\n"
    (out / "eligibility.md").write_text(text, encoding="utf-8")
    return text


def endpoints(runs: pd.DataFrame) -> str:
    """The pre-registered paired tests: Wilcoxon over seeds, Holm across datasets."""
    df = runs.rename(columns={"arm": "method"})
    show = lambda arm: arm.replace(" | ", ", ")      # the names as summarise displays them
    lines = []
    for title, a, b in ENDPOINTS:
        res = paired(df, show(a), show(b), metric="test_macro_f1")
        lines += ["", f"## {title} (test macro-F1, points)", "",
                  "Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across the "
                  "datasets in this table; d_z = mean difference / its standard deviation.", "",
                  "| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |",
                  "|---|---|---|---|---|---|"]
        for _, r in res.iterrows():
            lines.append(f"| {r.dataset} | {r['diff']:+.2f} ± {r.ci:.2f} | {r.wins} of {r.n} | "
                         f"{r.p:.3g} | {r.p_holm:.3g} | {r.dz:.2f} |")
    return "\n".join(lines) + "\n"


def summarise(runs: pd.DataFrame, out: Path, protocol: str = "pilot") -> str:
    runs = runs.copy()
    runs["selector"] = runs.selector.fillna("")
    runs["arm"] = runs.arm.str.replace(" | ", ", ", regex=False)   # "|" would break the tables
    seeds = [int(s) for s in sorted(runs.seed.unique())]
    conf = protocol == "confirmatory"
    title = ("# Refinement confirmatory run (pre-registered)" if conf else
             "# Phase 5 pilot: selective refinement (exploratory)")
    what = ("Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). " if conf else
            "Not a test of any hypothesis: it sets up the confirmatory run. ")
    lines = [title, "",
             f"Seeds {seeds}. " + what +
             "Mean ± 95% CI over seeds, test rows. Budget = min(300, a quarter of the pool) labels "
             f"in {N_ROUNDS} rounds; gate on the acceptance rows (macro-F1 may fall at most 0.5 "
             "points, a class's recall at most 1 point; boxes at most 1.1 × the start). "
             "Times are wall-clock per arm and grow if the machine sleeps or runs share the CPU.",
             ""]
    arms = list(dict.fromkeys(runs.arm))
    lines += ["## Test macro-F1 (%)", "", "| Dataset | " + " | ".join(arms) + " |",
              "|---|" + "---|" * len(arms)]
    for ds, g in runs.groupby("dataset", sort=False):
        cells = []
        for arm in arms:
            v = g[g.arm == arm].test_macro_f1
            cells.append(f"{v.mean():.2f} ± {ci95(v):.2f}" if len(v) else "")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "## Change in test macro-F1 vs B0 (points), and box growth", "",
              "| Dataset | Arm | Δ macro-F1 | Δ min class recall | Boxes start → end |",
              "|---|---|---|---|---|"]
    for ds, g in runs.groupby("dataset", sort=False):
        b0 = g[g.arm == "B0 no refinement"].set_index("seed")
        for arm in arms[1:]:
            a = g[g.arm == arm].set_index("seed")
            d = a.test_macro_f1 - b0.test_macro_f1.reindex(a.index)
            r = a.test_min_class_recall - b0.test_min_class_recall.reindex(a.index)
            lines.append(f"| {ds} | {arm} | {d.mean():+.2f} ± {ci95(d):.2f} | {r.mean():+.2f} | "
                         f"{a.boxes_start.mean():.0f} → {a.boxes_end.mean():.0f} |")
    act = runs[runs.selector != ""]
    cols = ["n_wrong_selected", "n_shrink_possible", "n_shrink_tried", "n_shrink_accepted",
            "n_add_accepted", "n_points_left_out", "n_rejected_box_cap", "n_rejected_f1_or_recall",
            "mean_share_cut", "mean_rows_cut", "time_s"]
    lines += ["", "## What the loop did (mean per seed)", "",
              "| Dataset | Arm | Budget | " + " | ".join(cols) + " |",
              "|---|---|---|" + "---|" * len(cols)]
    for (ds, arm), g in act.groupby(["dataset", "arm"], sort=False):
        lines.append(f"| {ds} | {arm} | {g.budget.mean():.0f} | " +
                     " | ".join(f"{g[c].mean():.2f}" if "share" in c or "rows" in c else f"{g[c].mean():.1f}"
                                for c in cols) + " |")
    text = "\n".join(lines) + "\n"
    if conf:
        text += endpoints(runs[runs.selector != ""])
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")     # the summary has ± and Δ
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["pilot", "eligibility", "confirmatory"],
                    default="pilot")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="dataset x seed runs in parallel processes (results are identical)")
    a = ap.parse_args()
    conf = a.protocol == "confirmatory"
    out = Path(a.out or ROOT / "results" / ("refinement_confirmatory" if conf else
                                            "refinement_pilot"))
    out.mkdir(parents=True, exist_ok=True)
    if a.protocol == "eligibility":
        print(eligibility(out))
        return
    if not a.summary_only:
        tasks = [(n, s) for n in a.datasets or (CONF_DATASETS if conf else PILOT_DATASETS)
                 for s in a.seeds or (CONF_SEEDS if conf else PILOT_SEEDS)]
        done = {}

        def save():   # completed runs, always in task order
            parts = [done[t] for t in tasks if t in done]
            for i, name in enumerate(("runs", "rounds", "edits")):
                pd.DataFrame([row for p in parts for row in p[i]]).to_csv(out / f"{name}.csv",
                                                                          index=False)

        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"            # inherited by the worker processes
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                todo = sorted(tasks, key=lambda t: -COST.get(t[0], 0))
                futures = {ex.submit(run_one, n, s, a.protocol): (n, s) for n, s in todo}
                for f in as_completed(futures):
                    done[futures[f]] = f.result()
                    save()
        else:
            for t in tasks:
                done[t] = run_one(*t, a.protocol)
                save()
    print(summarise(pd.read_csv(out / "runs.csv"), out, a.protocol))


if __name__ == "__main__":
    main()
