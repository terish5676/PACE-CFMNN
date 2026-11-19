"""RQ2 v2: does CFMNN's own uncertainty add to standard data-based trust signals?

Same datasets, splits and eligibility rule as the RQ2 confirmatory run
(``run_error_detection.py --protocol confirmatory``):

* datasets under 2,000 rows: stratified 5-fold cross-validation, 20% of each
  training part as the inner rows, pooled out-of-fold scores;
* otherwise the role split 50/20/10/10/10, scoring pool + acceptance + test
  rows (at most 20,000).

Signals (``pace_cfmnn.trust_combine``): U*, margin, tie, box evidence, box
member count, order ensemble (CFMNN-native), and kNN disagreement and Trust
Score (data-based). Learned combinations are logistic regressions fitted on
the **inner rows with their labels** (a labelled calibration set), scored on
the evaluation rows. Every group of signals gets the same combiner.

Protocols:

* ``exploratory`` (seeds 0-9): the design run. It also tries PD as a feature,
  to decide whether PD is worth its cost.
* ``confirmatory`` (seeds 20-29): pre-registered in
  ``results/error_detection_v2/PREREGISTRATION.md``.
* ``ensembles`` (seeds 60-69): Test 3 of ``results/review_v1/PREREGISTRATION.md``.
  Is the order ensemble's gain specific to CFMNN? Adds two generic ensembles:
  5 CFMNNs on bootstrap samples, and 5 one-pass SGD logistic regressions
  trained in different orders. Output in ``results/review_v1/ensembles/``.

Results are saved after every dataset-seed pair. ``--resume`` continues a
stopped run; ``--fresh`` deletes earlier results and starts over.

Usage:
    python scripts/run_error_detection_v2.py --protocol exploratory --jobs 2
    python scripts/run_error_detection_v2.py --protocol confirmatory --jobs 4
    python -B scripts/run_error_detection_v2.py --protocol ensembles --jobs 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.evaluate import error_detection_metrics  # noqa: E402
from pace_cfmnn.neighborhood import Perturber  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_cfmnn.trust_combine import (ALL, DATA, ENSEMBLES, NATIVE, LearnedCombiner,  # noqa: E402
                                      bagged_members, ensemble_features, order_members,
                                      pd_computer, sgd_order_members, trust_features)
from run_error_detection import (CONF_DATASETS, MAX_EVAL, MIN_ERRORS, SHIFT, SMALL,  # noqa: E402
                                 SYNTHETIC, ci95, fmt, load, paired)

SEEDS = {"exploratory": list(range(10)), "confirmatory": list(range(20, 30)),
         "ensembles": list(range(60, 70))}
WITH_PD = {"exploratory": True, "confirmatory": False,   # design rule: PD never added >= 0.01, so out
           "ensembles": False}
SINGLES = ["kNN", "Trust Score", "Order ensemble", "U*", "Margin", "Box evidence", "Box count"]
GROUPS = {"LR: kNN + TS": DATA, "LR: kNN + TS + order ens.": DATA + ("Order ensemble",),
          "LR: native": NATIVE, "LR: all": ALL}
ENS_GROUPS = {"LR: kNN + TS": DATA,
              "LR: kNN + TS + order ens.": DATA + ("Order ensemble",),
              "LR: kNN + TS + bagged ens.": DATA + ("Bagged ensemble",),
              "LR: kNN + TS + order ens. + bagged ens.": DATA + ("Order ensemble", "Bagged ensemble"),
              "LR: kNN + TS + SGD order ens.": DATA + ("SGD order ensemble",)}
ENS_DATASETS = [d for d in CONF_DATASETS if d not in ("wine", "breast_cancer", "shuttle")]
COST = {"skin": 6, "letter": 5, "magic": 4, "shuttle": 3, "S2-shifted": 2}


def groups(with_pd: bool) -> dict:
    g = dict(GROUPS)
    for f in ALL:                                            # drop-one, descriptive
        g[f"LR: all without {f}"] = tuple(x for x in ALL if x != f)
    if with_pd:
        g["LR: all + PD"] = ALL + ("PD",)
        g["LR: native + PD"] = NATIVE + ("PD",)
    return g


def evaluate_split(Xtr, ytr, Xin, yin, Xev, yev, seed: int, with_pd: bool,
                   ensembles: bool = False) -> dict:
    model = CFMNN().fit(Xtr, ytr)
    members = order_members(model, Xtr, ytr, n_models=5, seed=seed + 3)   # as the RQ2 baseline
    pdc = pd_computer(model, Perturber(kappa=0.05, k=50).fit(Xtr)) if with_pd else None
    Fi, pi = trust_features(model, Xtr, ytr, Xin, members, pdc, pd_seed=10_000 + seed)
    Fe, pe = trust_features(model, Xtr, ytr, Xev, members, pdc, pd_seed=seed)
    ei, ee = pi != yin, pe != yev
    singles, grp = SINGLES + (["PD"] if with_pd else []), groups(with_pd)
    if ensembles:                                    # Test 3 of results/review_v1
        bagged = bagged_members(model, Xtr, ytr, n_models=5, seed=seed + 5)
        sgd = sgd_order_members(Xtr, ytr, n_models=5, seed=seed + 7)
        Fi.update(ensemble_features(Xin, pi, bagged, sgd))
        Fe.update(ensemble_features(Xev, pe, bagged, sgd))
        singles, grp = SINGLES + list(ENSEMBLES), ENS_GROUPS
    scores = {s: Fe[s] for s in singles}
    for name, feats in grp.items():
        scores[name] = LearnedCombiner(feats).fit(Fi, ei).score(Fe)
    scores["Random"] = np.random.default_rng(seed).uniform(size=len(ee))
    return {"scores": scores, "err": ee, "inner_errors": int(ei.sum()), "n_inner": len(yin),
            "n_boxes": model.n_boxes_}


def run_one(name: str, seed: int, protocol: str) -> list[dict]:
    t0 = time.perf_counter()
    with_pd = WITH_PD[protocol]
    ds = load(name, seed)
    if len(ds.y) < SMALL:
        parts = []
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for f, (tr_all, ev) in enumerate(outer.split(ds.X, ds.y)):
            inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + f)
            tr_i, in_i = next(inner.split(tr_all, ds.y[tr_all]))
            tr = np.random.default_rng(seed + f).permutation(tr_all[tr_i])
            sc = data.fit_scaler(ds.X[tr])
            tf = lambda X: np.clip(sc.transform(X), 0.0, 1.0)
            parts.append(evaluate_split(tf(ds.X[tr]), ds.y[tr], tf(ds.X[tr_all[in_i]]),
                                        ds.y[tr_all[in_i]], tf(ds.X[ev]), ds.y[ev], seed * 10 + f,
                                        with_pd, ensembles=protocol == "ensembles"))
        r = {"scores": {m: np.concatenate([p["scores"][m] for p in parts]) for m in parts[0]["scores"]},
             "err": np.concatenate([p["err"] for p in parts]),
             "inner_errors": sum(p["inner_errors"] for p in parts),
             "n_inner": sum(p["n_inner"] for p in parts),
             "n_boxes": float(np.mean([p["n_boxes"] for p in parts]))}
        split = "5-fold CV"
    else:
        roles = data.role_split(ds.y, seed)
        order = np.random.default_rng(seed).permutation(roles["train"])
        sc = data.fit_scaler(ds.X[order])
        tf = lambda X: np.clip(sc.transform(X), 0.0, 1.0)
        if name == "S2-shifted":
            sh = data.make_synthetic("S2", seed=seed + 10_000, shift=SHIFT)
            Xev, yev = tf(sh.X), sh.y
        else:
            ev = np.concatenate([roles["pool"], roles["accept"], roles["test"]])
            if len(ev) > MAX_EVAL:
                ev = np.sort(np.random.default_rng(seed + 2).choice(ev, MAX_EVAL, replace=False))
            Xev, yev = tf(ds.X[ev]), ds.y[ev]
        r = evaluate_split(tf(ds.X[order]), ds.y[order], tf(ds.X[roles["inner"]]),
                           ds.y[roles["inner"]], Xev, yev, seed, with_pd,
                           ensembles=protocol == "ensembles")
        split = "role split"
    err = r["err"]
    base = {"dataset": name, "seed": seed, "protocol": protocol, "split": split,
            "n_eval": len(err), "n_errors": int(err.sum()), "accuracy": 100 * float(1 - err.mean()),
            "n_inner": r["n_inner"], "inner_errors": r["inner_errors"], "n_boxes": r["n_boxes"],
            "total_s": time.perf_counter() - t0}
    rows = [{**base, "method": m, **error_detection_metrics(err, s)} for m, s in r["scores"].items()]
    au = {x["method"]: x["auprc"] for x in rows}
    top = "LR: all" if "LR: all" in au else "LR: kNN + TS + order ens."
    print(f"{name} seed {seed}: errors {base['n_errors']} inner errors {base['inner_errors']} "
          f"AUPRC {top} {au[top]:.3f} LR kNN+TS {au['LR: kNN + TS']:.3f} kNN {au['kNN']:.3f} "
          f"({base['total_s']:.0f}s)", flush=True)
    return rows


def summarise(df: pd.DataFrame, out: Path, protocol: str) -> str:
    meta = df.drop_duplicates(["dataset", "seed"])
    elig = meta.groupby("dataset", sort=False)["n_errors"].min() >= MIN_ERRORS
    ok = [d for d in elig.index if elig[d]]
    syn_ok = [d for d in SYNTHETIC if d in ok]
    real_ok = [d for d in ok if d not in SYNTHETIC]
    cols = ["auprc", "auroc", "precision_at", "base_rate"]
    fam = (df[df.dataset.isin(syn_ok)].groupby(["seed", "method"], as_index=False)[cols].mean()
           .assign(dataset="Synthetic family"))
    units = pd.concat([fam, df[df.dataset.isin(real_ok)]], ignore_index=True)
    unit_names = (["Synthetic family"] if syn_ok else []) + real_ok
    seeds = sorted(int(s) for s in df.seed.unique())
    ens = protocol == "ensembles"
    conf = protocol in ("confirmatory", "ensembles")
    rules = "../PREREGISTRATION.md" if ens else "PREREGISTRATION.md"
    L = ["# Review Test 3 (pre-registered): is the order ensemble's gain specific to CFMNN?" if ens else
         f"# RQ2 v2 ({'confirmatory, pre-registered' if conf else 'exploratory design run'}): "
         "CFMNN's own uncertainty vs data-based trust signals", "",
         (f"Rules: [`{rules}`]({rules}). " if conf else "") +
         f"Seeds {seeds}. Same splits and eligibility (at least {MIN_ERRORS} errors in every seed) "
         "as the RQ2 confirmatory run. Learned combinations: logistic regression on standardised "
         "signals, fitted on the inner rows with their labels. Mean ± 95% CI over seeds.", "",
         f"Tested units: the synthetic family ({', '.join(syn_ok)}) and "
         f"{', '.join(real_ok) or 'no real datasets'}.", "",
         "## Datasets", "", "| Dataset | Split | Eligible | Accuracy (%) | Errors (min) | Inner errors (min) |",
         "|---|---|---|---|---|---|"]
    for ds, g in meta.groupby("dataset", sort=False):
        L.append(f"| {ds} | {g.split.iloc[0]} | {'yes' if elig[ds] else 'no'} | "
                 f"{g.accuracy.mean():.1f} | {g.n_errors.min()} | {g.inner_errors.min()} |")
    main = (["kNN", "Trust Score", "Order ensemble", *ENSEMBLES, *ENS_GROUPS] if ens else
            ["kNN", "Trust Score", "Order ensemble", "U*", "Box evidence", "LR: kNN + TS",
             "LR: kNN + TS + order ens.", "LR: native", "LR: all"])
    main = [m for m in main if m in set(df.method)] + (["LR: all + PD"] if "LR: all + PD" in set(df.method) else [])
    for metric, title in [("auprc", "AUPRC (higher is better)"), ("auroc", "AUROC")]:
        L += ["", f"## {title}", "", "| Unit | Base rate | " + " | ".join(main) + " |",
              "|---|---|" + "---|" * len(main)]
        for u in unit_names:
            g = units[units.dataset == u]
            best = max(g[g.method == m][metric].mean() for m in main)
            cells = []
            for m in main:
                v = g[g.method == m][metric]
                c = fmt(v.mean(), ci95(v))
                cells.append(f"**{c}**" if np.isclose(v.mean(), best) else c)
            L.append(f"| {u} | {g.base_rate.mean():.3f} | " + " | ".join(cells) + " |")
    if ens:                                               # Test 3's pre-registered endpoints
        ends = [("LR: kNN + TS + order ens.", "LR: kNN + TS + bagged ens.", "T3a (primary): CFMNN's "
                 "order ensemble vs a generic bagged ensemble of CFMNNs, each added to kNN + TS"),
                ("LR: kNN + TS + order ens. + bagged ens.", "LR: kNN + TS + bagged ens.", "T3b: does the "
                 "order ensemble add beyond the bagged ensemble?"),
                ("LR: kNN + TS + SGD order ens.", "LR: kNN + TS", "T3c: does another learner's order "
                 "ensemble add to kNN + TS?"),
                ("LR: kNN + TS + order ens.", "LR: kNN + TS", "T3d (replication of v2's Q1): does "
                 "CFMNN's order ensemble add to kNN + TS?")]
    elif "LR: kNN + TS + order ens." in set(df.method):   # the pre-registered endpoints
        ends = [("LR: kNN + TS + order ens.", "LR: kNN + TS", "Q1 (primary): does CFMNN's order "
                 "sensitivity add to the data-based baselines? LR(kNN + TS + order ens.) minus LR(kNN + TS)"),
                ("LR: all", "LR: kNN + TS + order ens.", "Q1b: do the geometry signals add beyond that? "
                 "LR(all) minus LR(kNN + TS + order ens.)"),
                ("LR: all", "LR: kNN + TS", "Q1c (the draft's primary): LR(all) minus LR(kNN + TS)"),
                ("LR: kNN + TS + order ens.", "kNN", "Q2a: LR(kNN + TS + order ens.) minus kNN disagreement"),
                ("LR: kNN + TS + order ens.", "Trust Score", "Q2b: LR(kNN + TS + order ens.) minus Trust Score"),
                ("LR: native", "kNN", "Q3: CFMNN-native signals alone, LR(native) minus kNN disagreement")]
    else:                                                 # the design run's comparisons
        ends = [("LR: all", "LR: kNN + TS", "Q1 (draft): do CFMNN-native signals add to the data-based "
                 "baselines? LR(all) minus LR(kNN + TS)"),
                ("LR: all", "kNN", "Q2a: LR(all) minus kNN disagreement"),
                ("LR: all", "Trust Score", "Q2b: LR(all) minus Trust Score"),
                ("LR: native", "kNN", "Q3: CFMNN-native signals alone, LR(native) minus kNN disagreement")]
    if "LR: all + PD" in set(df.method):
        ends.append(("LR: all + PD", "LR: all", "Design check: does PD add? LR(all + PD) minus LR(all)"))
    for a, b, title in ends:
        t = paired(units, a, b)
        L += ["", f"## {title} (AUPRC)", "",
              "Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.", "",
              "| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |",
              "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.dataset} | {fmt(r['diff'], r.ci)} | {r.wins} of {r.n} | {r.p:.3g} | "
                     f"{r.p_holm:.3g} | {r.dz:.2f} |")
    if "LR: all" in set(df.method):
        L += ["", "## Drop-one (descriptive): AUPRC lost when a signal is removed from LR(all)", "",
              "| Unit | " + " | ".join(ALL) + " |", "|---|" + "---|" * len(ALL)]
        for u in unit_names:
            g = units[units.dataset == u].groupby("method").auprc.mean()
            L.append(f"| {u} | " + " | ".join(f"{g['LR: all'] - g[f'LR: all without {f}']:+.3f}"
                                             for f in ALL) + " |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["exploratory", "confirmatory", "ensembles"],
                    default="exploratory")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "results" / {
        "confirmatory": "error_detection_v2", "exploratory": "error_detection_v2_design",
        "ensembles": "review_v1/ensembles"}[a.protocol]
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        # Test 3 skips the three datasets that never had enough errors (Wine, Breast Cancer, Shuttle).
        names = a.datasets or (ENS_DATASETS if a.protocol == "ensembles" else CONF_DATASETS)
        tasks = [(n, s, a.protocol) for n in names for s in a.seeds or SEEDS[a.protocol]]
        run_all(run_one, pending(out / "runs.csv", tasks, a.resume, a.fresh), out / "runs.csv",
                a.jobs, COST)
    print(summarise(pd.read_csv(out / "runs.csv"), out, a.protocol))


if __name__ == "__main__":
    main()
