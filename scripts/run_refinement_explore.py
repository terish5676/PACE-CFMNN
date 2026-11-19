"""Refinement v2, exploratory checks E1-E4 (seeds 0-2; descriptive only, no p-values).

v1 (``results/refinement_confirmatory/``) left five questions open. This run
looks at each one on the 6 v1 datasets with v1's split, budget, gate and box
cap (``run_refinement.split``), on the pilot seeds 0-2:

* References: B0 (no refinement); B9 (refit on every pool label); order
  ensemble (OE) with A+B and with A+B+C'-volume (v1's best rule); CFMNN, kNN
  (k = 10) and a Random Forest trained on the training rows plus the labels the
  OE A+B+C'-volume arm bought, with and without the acceptance rows (which the
  gated arms read too).
* E1, self-repair: the shrink-and-gate step on the model's own training errors
  (``active.run_self_repair``), which buys no labels. SR-300 (300 errors, gate
  on the acceptance rows), SR-300-train (gate on the training rows: no labels
  beyond training at all), SR-all (up to 2,000 errors) and SR-all followed by
  OE A+B+C'-volume (do bought labels add anything after self-repair?).
* E2, which feature to cut (OE selection): the explanation cut (C), C-fewest
  (fewest training members lost), C-PACE-SHAP and C-oracle (every feature
  through the gate; an upper bound).
* E3, which points to buy (all with the volume cut): random, margin, OE, kNN
  disagreement and oracle-errors (the pool's hidden labels; an upper bound).
  The hit rate is the share of bought points the model got wrong.
* E4, auditability, on a fixed sample of at most 2,000 test rows: prediction
  churn against B0 (changed, fixed, broken); for edit-based arms, the
  certificates of every accepted edit (``audit.certify``: the box-level one
  and the tight one), checked against the changes the edit actually made, and
  their size; PACE-flip explanation churn
  (top-3 features) on 100 of those rows, inside and outside the certificate.

Every label is counted: pool labels bought, acceptance rows read by the gate
or used for training, and none for the training rows themselves.

Usage (from the project folder):
    python scripts/run_refinement_explore.py --jobs 4
    python scripts/run_refinement_explore.py --datasets page_blocks --seeds 0
    python scripts/run_refinement_explore.py --jobs 4 --resume     # skip finished runs
    python scripts/run_refinement_explore.py --summary-only

Each finished (dataset, seed) is saved at once under ``parts/``; the CSV files
are rebuilt from the parts, in a fixed order, after every run.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import LabelOracle, run_refinement, run_self_repair  # noqa: E402
from pace_cfmnn.audit import certify, explanation_churn, prediction_churn, top_groups  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.explain import pace_flip  # noqa: E402
from run_refinement import COST, MAX_BUDGET, N_ROUNDS, split  # noqa: E402

DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = [0, 1, 2]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "min_class_recall"]
N_AUDIT, N_EXPLAIN, N_FLIP_BG, N_SHAP_BG = 2000, 100, 200, 20
SR_ALL = 2000
VOL = "A+B+C'-volume"
# (arm, group, selector, operation) for the active-learning loop.
LOOP_ARMS = [
    ("OE | A+B", "reference", "order_ensemble", "A+B"),
    (f"OE | {VOL}", "reference", "order_ensemble", VOL),
    ("OE | A+B+C", "E2 cut", "order_ensemble", "A+B+C"),
    ("OE | A+B+C-fewest", "E2 cut", "order_ensemble", "A+B+C-fewest"),
    ("OE | A+B+C-paceshap", "E2 cut", "order_ensemble", "A+B+C-paceshap"),
    ("OE | A+B+C-oracle", "E2 cut", "order_ensemble", "A+B+C-oracle"),
    (f"random | {VOL}", "E3 selection", "random", VOL),
    (f"margin | {VOL}", "E3 selection", "margin", VOL),
    (f"knn | {VOL}", "E3 selection", "knn", VOL),
    (f"oracle_errors | {VOL}", "E3 selection", "oracle_errors", VOL),
]


class Certificates:
    """``on_accept`` hook: both certificates of every accepted edit, on fixed audit rows.

    The top groups of the model after one edit are reused as those before the
    next one when the model has not changed in between (rejected edits are
    rolled back), which halves the cost.
    """

    def __init__(self, X):
        self.X = X
        self.union = {"loose": np.zeros(len(X), dtype=bool), "tight": np.zeros(len(X), dtype=bool)}
        self.sizes = {"loose": [], "tight": []}
        self.misses = {"loose": 0, "tight": 0}
        self.changes = []
        self._last = None

    def _groups(self, model):
        key = (model.V_, model.W_, model.centroid_, model.box_class_)
        if self._last is not None and all(
                a.shape == b.shape and np.array_equal(a, b) for a, b in zip(key, self._last[0])):
            return self._last[1]
        return top_groups(model, self.X)

    def __call__(self, before, after, removed):
        tb, pb = self._groups(before)
        ta, pa = top_groups(after, self.X)
        self._last = ((after.V_.copy(), after.W_.copy(), after.centroid_.copy(),
                       after.box_class_.copy()), (ta, pa))
        changed = pb != pa
        self.changes.append(changed.sum())
        for tag, flagged in zip(("loose", "tight"), certify(before, after, tb, ta, removed)):
            self.misses[tag] += int((changed & ~flagged).sum())
            self.union[tag] |= flagged
            self.sizes[tag].append(flagged.sum())

    def summary(self) -> dict:
        n, changes = len(self.X), np.asarray(self.changes, float)
        out = {"cert_edits": len(changes), "cert_mean_changes": changes.mean() if len(changes) else np.nan}
        for tag in ("loose", "tight"):
            sizes = np.asarray(self.sizes[tag], float)
            out.update({f"cert_{tag}_misses": self.misses[tag],
                        f"cert_{tag}_union_share": 100 * self.union[tag].mean(),
                        f"cert_{tag}_mean_size_share": 100 * sizes.mean() / n if len(sizes) else np.nan,
                        f"cert_{tag}_precision": 100 * changes.sum() / sizes.sum() if sizes.sum() else np.nan})
        return out


def edit_stats(edits: list[dict], budget: int | None) -> dict:
    if not edits:
        return {}
    e = pd.DataFrame(edits)
    sh = e[e.edit == "shrink"]
    out = {"n_shrink_tried": len(sh), "n_shrink_accepted": int(sh.accepted.sum()),
           "mean_share_cut": float(sh[sh.accepted].shrink_share_cut.mean()) if sh.accepted.any() else np.nan,
           "mean_rows_cut": float(sh[sh.accepted].shrink_n_cut.mean()) if sh.accepted.any() else np.nan}
    if "pool_index" in e:
        first = e.drop_duplicates("pool_index")
        out.update({"n_wrong_selected": int(first.was_wrong.sum()),
                    "hit_rate": 100 * first.was_wrong.mean(),
                    "n_add_accepted": int(((e.edit == "add") & e.accepted).sum())})
    else:
        out.update({"n_errors_tried": len(e), "n_skipped": int((e.edit == "none").sum())})
    return out


def run_one(name: str, seed: int):
    t0 = time.perf_counter()
    P = split(name, seed)
    (Xtr, ytr), (Xp, yp), (Xi, _), (Xa, ya), (Xt, yt) = (P[r] for r in
                                                       ("train", "pool", "inner", "accept", "test"))
    model = CFMNN().fit(Xtr, ytr)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    rng = np.random.default_rng(seed + 3)
    au = rng.permutation(len(Xt))[:min(N_AUDIT, len(Xt))]   # explanation rows = the first 100
    Xau, yau = Xt[au], yt[au]
    Xex = Xau[:N_EXPLAIN]
    bg_flip, bg_shap = Xtr[:N_FLIP_BG], Xtr[:N_SHAP_BG]
    p0 = model.predict(Xau)
    e0 = pace_flip(model, Xex, bg_flip)
    train_err0 = 100 * np.mean(model.predict(Xtr) != ytr)
    base = {"dataset": name, "seed": seed, "n_train": len(Xtr), "n_pool": len(Xp),
            "n_accept": len(Xa), "n_test": len(Xt), "n_audit": len(Xau), "budget": budget,
            "boxes_start": model.n_boxes_, "train_error_start": train_err0}
    runs, edits_all = [], []

    def record(arm, group, pred_test, pred_audit, pool_labels, acc_labels, t, m=None, cert=None,
               edits=None, extra=None):
        rec = {**base, "arm": arm, "group": group, "pool_labels": pool_labels,
               "accept_labels": acc_labels, "total_labels": pool_labels + acc_labels,
               **{f"test_{k}": v for k, v in classification_metrics(yt, pred_test).items()
                  if k in METRICS},
               **{f"churn_{k}": v for k, v in prediction_churn(yau, p0, pred_audit).items()},
               "time_s": time.perf_counter() - t}
        rec["churn_changed_share"] *= 100
        if m is not None:
            e1 = pace_flip(m, Xex, bg_flip)
            rec.update({"boxes_end": m.n_boxes_,
                        "train_error_end": 100 * np.mean(m.predict(Xtr) != ytr),
                        **{f"expl_{k}": v for k, v in explanation_churn(e0, e1).items()}})
            if cert is not None:
                rec.update(cert.summary())
                for kind in ("loose", "tight"):
                    inside = cert.union[kind][:N_EXPLAIN]
                    for tag, mask in (("in", inside), ("out", ~inside)):
                        c = explanation_churn(e0[mask], e1[mask])
                        rec.update({f"expl_{kind}_{tag}_n": c["n"],
                                    f"expl_{kind}_{tag}_any": c["any_change_share"],
                                    f"expl_{kind}_{tag}_top3": c["topk_change_share"]})
        if edits is not None:
            rec.update(edit_stats(edits, budget))
            edits_all.extend({"dataset": name, "seed": seed, "arm": arm, **e} for e in edits)
        rec.update(extra or {})
        runs.append(rec)
        print(f"  {name} seed {seed} {arm}: macro-F1 {rec['test_macro_f1']:.2f}, "
              f"labels {rec['total_labels']}, churn {rec['churn_changed_share']:.2f}% "
              f"({rec['time_s']:.0f}s)", flush=True)

    t = time.perf_counter()
    record("B0", "reference", model.predict(Xt), p0, 0, 0, t, m=model)
    order = np.random.default_rng(seed + 1).permutation(len(Xp))
    t = time.perf_counter()
    full = CFMNN().fit(np.vstack([Xtr, Xp[order]]), np.concatenate([ytr, yp[order]]))
    record("B9 refit, all pool labels", "reference", full.predict(Xt), full.predict(Xau),
           len(Xp), 0, t, m=full)

    # The active-learning arms.
    bought = None
    for arm, group, sel, op in LOOP_ARMS:
        t = time.perf_counter()
        oracle = LabelOracle(yp)
        cert = Certificates(Xau)
        m, _, el = run_refinement(model, Xtr, ytr, Xp, oracle, Xi, Xa, ya, sel, op, budget=budget,
                                   n_rounds=N_ROUNDS, seed=seed, X_test=None, y_test=None,
                                   background=bg_shap, on_accept=cert)
        assert len(oracle.revealed) == budget == len(set(oracle.revealed))
        record(arm, group, m.predict(Xt), m.predict(Xau), budget, len(Xa), t, m=m, cert=cert,
               edits=el)
        if arm == f"OE | {VOL}":
            bought = oracle.revealed.copy()

    # References trained on the same bought labels (those of OE | A+B+C'-volume).
    Xb, yb = Xp[bought], yp[bought]
    own_start = {}
    for tag, Xs, ys, acc in (("train only", Xtr, ytr, 0),
                             ("+bought", np.vstack([Xtr, Xb]), np.concatenate([ytr, yb]), 0),
                             ("+bought +acceptance", np.vstack([Xtr, Xb, Xa]),
                              np.concatenate([ytr, yb, ya]), len(Xa))):
        pool = 0 if tag == "train only" else budget
        if tag != "train only":
            t = time.perf_counter()
            m = CFMNN().fit(Xs, ys)
            record(f"CFMNN refit, {tag}", "reference", m.predict(Xt), m.predict(Xau), pool, acc, t, m=m)
        for kind, make in (("kNN", lambda: KNeighborsClassifier(n_neighbors=10)),
                           ("RF", lambda: RandomForestClassifier(n_estimators=100, random_state=seed,
                                                                 n_jobs=1))):
            t = time.perf_counter()
            clf = make().fit(Xs, ys)
            pa = clf.predict(Xau)
            if tag == "train only":
                own_start[kind] = pa
                record(f"{kind}, {tag}", "reference", clf.predict(Xt), pa, pool, acc, t)
                continue
            c = prediction_churn(yau, own_start[kind], pa)   # against the same learner on train only
            record(f"{kind}, {tag}", "reference", clf.predict(Xt), pa, pool, acc, t,
                   extra={"own_churn_changed_share": 100 * c["changed_share"],
                          "own_churn_fixed": c["fixed"], "own_churn_broken": c["broken"]})

    # E1: self-repair.
    sr_all = None
    for arm, n, gate_X, gate_y, new in (("SR-300", budget, Xa, ya, True),
                                        ("SR-300-train", budget, Xtr, ytr, False),
                                        ("SR-all", SR_ALL, Xa, ya, True)):
        t = time.perf_counter()
        cert = Certificates(Xau)
        m, _, el = run_self_repair(model, Xtr, ytr, gate_X, gate_y, chooser="volume", budget=n,
                                    n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=new,
                                    on_accept=cert)
        record(arm, "E1 self-repair", m.predict(Xt), m.predict(Xau), 0, len(Xa) if new else 0, t,
               m=m, cert=cert, edits=el)
        if arm == "SR-all":
            sr_all, sr_cert = m, cert
    t = time.perf_counter()
    oracle = LabelOracle(yp)
    m, _, el = run_refinement(sr_all, Xtr, ytr, Xp, oracle, Xi, Xa, ya, "order_ensemble", VOL,
                               budget=budget, n_rounds=N_ROUNDS, seed=seed, on_accept=sr_cert)
    assert len(oracle.revealed) == budget
    record(f"SR-all + OE | {VOL}", "E1 self-repair", m.predict(Xt), m.predict(Xau), budget, len(Xa),
           t, m=m, cert=sr_cert, edits=el)
    print(f"{name} seed {seed} done in {time.perf_counter() - t0:.0f}s", flush=True)
    return runs, edits_all


# --------------------------------------------------------------------- summary
NAMES = {"thyroid": "Thyroid", "landsat": "Landsat", "page_blocks": "Page Blocks",
         "magic": "MAGIC", "letter": "Letter", "skin": "Skin"}


def table(runs, arms, value, fmt="{:.1f}", title=None) -> list[str]:
    ds = [d for d in DATASETS if d in set(runs.dataset)]
    piv = runs.pivot_table(index="arm", columns="dataset", values=value, aggfunc="mean")
    lines = ([f"### {title}", ""] if title else []) + \
        ["| Arm | " + " | ".join(NAMES[d] for d in ds) + " |", "|---|" + "---|" * len(ds)]
    for a in arms:
        if a in piv.index:
            lines.append(f"| {a} | " + " | ".join(
                "" if pd.isna(piv.loc[a, d]) else fmt.format(piv.loc[a, d]) for d in ds) + " |")
    return lines + [""]


def summarise(out: Path) -> str:
    runs = pd.read_csv(out / "runs.csv")
    seeds = sorted(runs.seed.unique())
    b0 = runs[runs.arm == "B0"].set_index(["dataset", "seed"])
    runs["d_f1"] = runs.test_macro_f1 - b0.test_macro_f1.reindex(
        pd.MultiIndex.from_frame(runs[["dataset", "seed"]])).to_numpy()
    lines = ["# Refinement v2: exploratory checks E1–E4", "",
             f"Seeds {', '.join(map(str, seeds))}; descriptive only (means over seeds, no p-values). "
             "v1's split, budget (min(300, a quarter of the pool)), 10 rounds, gate and box cap. "
             "Written by `scripts/run_refinement_explore.py` from `runs.csv`.", "",
             "## Labels used beyond the training rows", ""]
    lines += table(runs, list(dict.fromkeys(runs.arm)), "total_labels", "{:,.0f}")
    all_arms = list(dict.fromkeys(runs.arm))
    lines += ["## Test macro-F1 (%)", ""] + table(runs, all_arms, "test_macro_f1")
    lines += ["## Change in test macro-F1 against B0 (points)", ""] + table(runs, all_arms[1:], "d_f1", "{:+.2f}")
    lines += ["## Lowest class recall (%)", ""] + table(runs, all_arms, "test_min_class_recall")
    sr = ["SR-300", "SR-300-train", "SR-all", f"SR-all + OE | {VOL}", f"OE | {VOL}"]
    lines += ["## E1: self-repair", ""]
    lines += table(runs, sr + ["B0"], "train_error_end", title="Training error at the end (%)")
    lines += table(runs, sr[:3], "n_shrink_accepted", "{:.0f}", title="Accepted shrinks")
    e2 = ["OE | A+B+C", f"OE | {VOL}", "OE | A+B+C-fewest", "OE | A+B+C-paceshap", "OE | A+B+C-oracle"]
    lines += ["## E2: which feature to cut", ""]
    lines += table(runs, e2, "n_shrink_accepted", "{:.0f}", title="Accepted shrinks")
    lines += table(runs, e2, "mean_share_cut", "{:.3f}", title="Mean share of the box's width cut")
    lines += table(runs, e2, "mean_rows_cut", "{:.0f}", title="Mean training members cut")
    e3 = [f"{s} | {VOL}" for s in ("random", "margin", "OE", "knn", "oracle_errors")]
    lines += ["## E3: which points to buy", ""]
    lines += table(runs, e3, "hit_rate", title="Share of bought points the model got wrong (%)")
    edit_arms = [a for a in all_arms if runs[runs.arm == a].cert_edits.notna().any()]
    lines += ["## E4: auditability", "",
              "Audit rows: a fixed sample of at most 2,000 test rows. Churn is against B0.", ""]
    lines += table(runs, all_arms[1:], "churn_changed_share", "{:.2f}", title="Predictions changed (%)")
    lines += table(runs, all_arms[1:], "churn_fixed", "{:.0f}", title="Predictions fixed (wrong → right)")
    lines += table(runs, all_arms[1:], "churn_broken", "{:.0f}", title="Predictions broken (right → wrong)")
    for kind, what in (("tight", "tight certificate"), ("loose", "box-level certificate")):
        lines += [f"### E4, {what}", ""]
        lines += table(runs, edit_arms, f"cert_{kind}_misses", "{:.0f}",
                       title="Changes outside it (must be 0)")
        lines += table(runs, edit_arms, f"cert_{kind}_mean_size_share", "{:.2f}",
                       title="Size per edit (% of audit rows)")
        lines += table(runs, edit_arms, f"cert_{kind}_precision", "{:.1f}",
                       title="Precision: changed rows / certified rows, over all edits (%)")
        lines += table(runs, edit_arms, f"cert_{kind}_union_share", "{:.1f}",
                       title="Rows some edit could have changed (% of audit rows)")
        lines += table(runs, edit_arms, f"expl_{kind}_out_n", "{:.0f}",
                       title="Explained rows outside it (of 100)")
        lines += table(runs, edit_arms, f"expl_{kind}_out_top3", "{:.2f}",
                       title="PACE-flip top-3 changed on those rows (share)")
    lines += table(runs, edit_arms, "cert_mean_changes", "{:.2f}",
                   title="Audit rows whose prediction one edit changed (mean)")
    lines += table(runs, all_arms[1:], "expl_topk_change_share", "{:.2f}",
                   title="PACE-flip top-3 changed, all 100 explained rows (share)")
    own = [a for a in all_arms if a.startswith(("kNN, +", "RF, +"))]
    lines += table(runs, own, "own_churn_changed_share", "{:.2f}",
                   title="kNN and RF: predictions changed against the same learner on train only (%)")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default=str(ROOT / "results" / "refinement_v2_explore"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="skip (dataset, seed) runs already saved")
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
            done = [t for t in tasks if path(t).exists()]
            res_all = []
            for d in done:
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
