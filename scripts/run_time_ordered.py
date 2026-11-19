"""Run P9 (tough review B10): a model in use, on time-ordered data.

Rules: results/time_ordered/PREREGISTRATION.md (planned in advance). Data: OpenML's electricity
(id 151; New South Wales electricity market, 1996-1998), 45,312 half-hourly rows in time order.
The study uses the first 134 full weeks (45,024 rows); the last 288 rows, a partial week, are not
used. Features: the day of the week, mapped to (day - 1) / 6, and the six numeric columns as the
file gives them (already scaled to [0, 1] by the file's authors). The date column is dropped.

Each seed (380-389) fixes the order of the rows inside each week; the weeks themselves are fixed
by time. Weeks 0-3 train the first model. Then, for each week w = 4, ..., 132:

1. The week in use: every arm's deployed model predicts week w. For the INC and SR arms, a
   detector flags likely errors (the order ensemble and kNN disagreement, added), and, for SR,
   PACE-SHAP explains 10 decisions.
2. The week's labels arrive, and each arm updates its model with them:
   * INC: the incremental update, ``CFMNN.add_points`` (one-pass training continued);
   * SR: the same update, then self-repair on the training rows (300 errors, 10 rounds, the
     smallest cut, the gate on the training rows, so no label beyond the weeks received);
   * RETUNE: every 4th week, theta re-chosen by 3-fold validation on all rows so far and the
     model refitted on them; in the other weeks, the incremental update;
   * INC-0.3 and SR-0.3: INC and SR at the paper's theta = 0.3.
   INC, SR and RETUNE start at theta_0, chosen by 3-fold validation on weeks 0-3.
3. Week w + 1 (unseen by every model): changed, fixed and broken predictions of the update
   (before against after), and the certificate's churn bound: the certificate of the whole update
   is computed on one random half of week w + 1, its one-sided 95% Clopper-Pearson bound is
   compared with the churn on the other half.

    python -B scripts/run_time_ordered.py --jobs 6
    python -B scripts/run_time_ordered.py --jobs 6 --resume       # continue after a stop
    python -B scripts/run_time_ordered.py --summary-only
    python -B scripts/run_time_ordered.py --smoke --out <dir>     # seed 194, 16 weeks; not a study run
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.audit import churn_bound, impact_set, prediction_churn  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_cfmnn.trust_baselines import knn_disagreement  # noqa: E402
from run_readable_baselines import GAMMA, GRIDS, cv_folds, tune  # noqa: E402

OUT = ROOT / "results" / "time_ordered"
DATASET = "electricity"
SEEDS = list(range(380, 390))
WEEK = 336                      # 7 days of 48 half-hours
N_WEEKS = 134                   # full weeks in the file
START = 4                       # weeks 0-3 train the first model
FIXED_THETA = 0.3
RETUNE_EVERY = 4                # weeks
BUDGET, N_ROUNDS, CHOOSER = 300, 10, "volume"
N_MEMBERS, KNN_K = 5, 10
N_EXPLAIN, N_BACKGROUND, RECENT_WEEKS = 10, 100, 4
TOP_SHARE = 0.1                 # the detector flags this share of each week
ALPHA = 0.05
ARMS = ("INC", "SR", "RETUNE", "INC-0.3", "SR-0.3")
CERTIFIED = ("INC", "SR", "INC-0.3", "SR-0.3")


@lru_cache(maxsize=1)
def stream():
    """(X, y, feature names) in time order: 134 full weeks, the date dropped."""
    ds = data.load(DATASET)
    X = np.c_[(ds.X[:, 1] - 1) / 6, ds.X[:, 2:]]
    n = N_WEEKS * WEEK
    return X[:n], ds.y[:n], ["day"] + list(ds.feature_names[2:])


def week_orders(seed: int, n_weeks: int = N_WEEKS) -> list[np.ndarray]:
    """Row indices of each week, in the seed's order inside the week."""
    rng = np.random.default_rng(seed)
    return [w * WEEK + rng.permutation(WEEK) for w in range(n_weeks)]


def tune_theta(X, y, seed: int) -> float:
    setting, _ = tune("CFMNN-val", GRIDS["CFMNN-val"], X, y, seed, 1, cv_folds(y, seed))
    return float(setting)


def repair(model: CFMNN, X_lab, y_lab, seed: int, X_cert=None):
    """Gated self-repair on the training rows; the union of its edits' certificates on X_cert."""
    cert = None if X_cert is None else np.zeros(len(X_cert), dtype=bool)

    def on_accept(before, after, removed):
        cert[:] |= impact_set(before, after, X_cert, removed=removed, tight=True)

    out, _, edits = run_self_repair(model, X_lab, y_lab, X_lab, y_lab, chooser=CHOOSER, budget=BUDGET,
                                    n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False,
                                    fast_gate=True, on_accept=None if X_cert is None else on_accept)
    return out, cert, edits


def detector(members, X_lab, y_lab, Xw, yw, pred) -> dict:
    """AUROC of each signal for the deployed model's errors, and precision of the top flags."""
    err = pred != yw
    ens = np.mean([m.predict(Xw) != pred for m in members], axis=0)
    knn = knn_disagreement(X_lab, y_lab, Xw, pred, k=KNN_K)
    both = ens + knn
    out = {"error_pct": 100 * float(err.mean())}
    ok = 0 < err.sum() < len(err)
    for name, s in (("ensemble", ens), ("knn", knn), ("detector", both)):
        out[f"auc_{name}"] = float(roc_auc_score(err, s)) if ok else np.nan
    k = int(round(TOP_SHARE * len(err)))
    top = np.argsort(-both, kind="stable")[:k]
    out["flag_precision_pct"] = 100 * float(err[top].mean())
    return out


def run_seed(dataset: str, seed: int, n_weeks: int = N_WEEKS, retune_every: int = RETUNE_EVERY):
    warnings.filterwarnings("ignore")
    if dataset != DATASET:
        raise ValueError(f"this run uses {DATASET} only")
    X, y, _ = stream()
    orders = week_orders(seed, n_weeks)
    lab = np.concatenate(orders[:START])
    t0 = time.perf_counter()
    theta0 = tune_theta(X[lab], y[lab], seed)
    first_tune_s = time.perf_counter() - t0
    Xl, yl = X[lab], y[lab]
    models = {"INC": CFMNN(theta=theta0, gamma=GAMMA).fit(Xl, yl),
              "INC-0.3": CFMNN(theta=FIXED_THETA, gamma=GAMMA).fit(Xl, yl)}
    models["RETUNE"] = copy.deepcopy(models["INC"])
    models["SR"] = repair(models["INC"], Xl, yl, seed * 1000 + START - 1)[0]
    models["SR-0.3"] = repair(models["INC-0.3"], Xl, yl, seed * 1000 + START - 1)[0]
    theta = {"INC": theta0, "SR": theta0, "RETUNE": theta0, "INC-0.3": FIXED_THETA, "SR-0.3": FIXED_THETA}
    # Order-ensemble members at theta_0, each with its own order inside every week.
    member_orders = [week_orders(seed * 100 + k, n_weeks) for k in range(N_MEMBERS)]
    members = [CFMNN(theta=theta0, gamma=GAMMA).fit(X[np.concatenate(o[:START])], y[np.concatenate(o[:START])])
               for o in member_orders]
    rows = []
    for w in range(START, n_weeks - 1):
        idx = orders[w]
        Xw, yw = X[idx], y[idx]
        nxt = np.arange((w + 1) * WEEK, (w + 2) * WEEK)
        Xn, yn = X[nxt], y[nxt]
        rng = np.random.default_rng(seed * 1000 + w)
        half = rng.permutation(WEEK)
        A, B = half[:WEEK // 2], half[WEEK // 2:]

        # 1. The week in use.
        info = {a: {} for a in ARMS}
        for a in ARMS:
            pred = models[a].predict(Xw)
            info[a].update({"accuracy": 100 * float(np.mean(pred == yw)),
                            "macro_f1": 100 * float(f1_score(yw, pred, average="macro"))})
            if a in ("INC", "SR"):
                info[a].update(detector(members, Xl, yl, Xw, yw, pred))
        recent = Xl[-RECENT_WEEKS * WEEK:]
        bg = recent[rng.choice(len(recent), N_BACKGROUND, replace=False)]
        t0 = time.perf_counter()
        ps = PaceShap(models["SR"], bg, seed=seed)
        ps.explain(Xw[:N_EXPLAIN])
        info["SR"].update({"explain_ms": 1000 * (time.perf_counter() - t0) / N_EXPLAIN,
                           "explain_exact": bool(ps.exact)})

        # 2. The week's labels arrive: each arm updates.
        lab = np.r_[lab, idx]
        Xl, yl = X[lab], y[lab]
        new_rows = np.arange(len(lab) - WEEK, len(lab))
        for a in ARMS:
            before = models[a]
            pb = before.predict(Xn)
            after = copy.deepcopy(before)
            cert, cert_s, edits, refit = None, 0.0, None, False
            t0 = time.perf_counter()
            if a == "RETUNE" and (w - START + 1) % retune_every == 0:
                theta[a] = tune_theta(Xl, yl, seed)
                after = CFMNN(theta=theta[a], gamma=GAMMA).fit(Xl, yl)
                refit = True
            else:
                after.add_points(Xw, yw, rows=new_rows)
            update_s = time.perf_counter() - t0
            if a in CERTIFIED:
                t1 = time.perf_counter()
                cert = impact_set(before, after, Xn, tight=True)
                cert_s += time.perf_counter() - t1
            if a in ("SR", "SR-0.3"):
                t1 = time.perf_counter()
                after, sr_cert, edits = repair(after, Xl, yl, seed * 1000 + w, Xn)
                total = time.perf_counter() - t1
                t2 = time.perf_counter()
                cert |= sr_cert
                cert_s += time.perf_counter() - t2
                update_s += total            # includes the per-edit certificates (on_accept)
            pa = after.predict(Xn)
            ch = prediction_churn(yn, pb, pa)
            rec = {"dataset": dataset, "seed": seed, "week": w, "arm": a, "theta": theta[a],
                   "theta_0": theta0, "first_tune_s": first_tune_s, "refit": refit,
                   "boxes": after.n_boxes_, **info[a],
                   "next_changed": ch["changed"], "next_fixed": ch["fixed"], "next_broken": ch["broken"],
                   "nfr_pct": 100 * ch["broken"] / WEEK, "changed_pct": 100 * ch["changed"] / WEEK,
                   "next_accuracy_after": 100 * float(np.mean(pa == yn)), "update_s": update_s,
                   "cert_s": cert_s,
                   "sr_edits": len(edits) if edits is not None else np.nan,
                   "sr_accepted": sum(bool(e["accepted"]) for e in edits) if edits is not None else np.nan}
            if cert is not None:
                changed = pb != pa
                bound = churn_bound(cert[A], ALPHA)
                rec.update({"cert_pct": 100 * float(cert.mean()),
                            "cert_contains_changes": bool(np.all(cert[changed])),
                            "bound_upper_pct": 100 * bound["upper"],
                            "churn_other_half_pct": 100 * float(changed[B].mean()),
                            "bound_holds": bool(changed[B].mean() <= bound["upper"])})
            rows.append(rec)
            models[a] = after
        for m, o in zip(members, member_orders):
            m.add_points(X[o[w]], y[o[w]])
        if w % 10 == 0 or w == n_weeks - 2:
            print(f"seed {seed} week {w}: " + ", ".join(
                f"{r['arm']} acc {r['accuracy']:.1f} NFR {r['nfr_pct']:.2f}%" for r in rows[-len(ARMS):]),
                flush=True)
    return rows


# ------------------------------------------------------------------ tests
def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def differences(df: pd.DataFrame, a: str, b: str, metric: str) -> pd.Series:
    """a minus b on ``metric``, one value per (week, seed)."""
    p = df.pivot_table(index=["week", "seed"], columns="arm", values=metric)
    return (p[a] - p[b]).dropna()


def test(d: pd.Series, unit: str) -> dict:
    """Two-sided Wilcoxon signed-rank test of the differences averaged within ``unit``.

    unit "week": the mean over seeds of each week (the primary); "seed": the mean over weeks of
    each seed; "month": the mean of 4-week blocks of the weekly means.
    """
    if unit == "week":
        v = d.groupby(level="week").mean()
    elif unit == "seed":
        v = d.groupby(level="seed").mean()
    else:
        wk = d.groupby(level="week").mean()
        v = wk.groupby((wk.index.to_numpy() - START) // 4).mean()
    v = v.to_numpy()
    same = bool(len(v) and np.all(v == 0))
    p = 1.0 if same or len(v) < 2 else float(wilcoxon(v).pvalue)
    return {"unit": unit, "n": len(v), "mean_diff": float(v.mean()) if len(v) else np.nan,
            "a_lower": int((v < 0).sum()), "a_higher": int((v > 0).sum()), "p": p, "same": same}


def verdict(r: dict, p_key: str, lower_is_better: bool) -> str:
    if r["same"]:
        return "no change"
    if r[p_key] >= ALPHA:
        return "no significant difference"
    better = r["mean_diff"] < 0 if lower_is_better else r["mean_diff"] > 0
    return "better" if better else "worse"


SECONDARY = [("SR", "INC", "nfr_pct", True, "Robustness: NFR, SR minus INC"),
             ("SR-0.3", "INC-0.3", "nfr_pct", True, "NFR at θ = 0.3, SR minus INC"),
             ("SR", "RETUNE", "nfr_pct", True, "NFR, SR minus RETUNE"),
             ("SR", "INC", "accuracy", False, "Accuracy in use, SR minus INC"),
             ("SR", "RETUNE", "accuracy", False, "Accuracy in use, SR minus RETUNE"),
             ("SR-0.3", "INC-0.3", "accuracy", False, "Accuracy in use at θ = 0.3, SR minus INC")]


def summarise(out: Path) -> str:
    df = pd.read_csv(out / "weeks.csv")
    seeds = sorted(df.seed.unique())
    L = ["# Run P9: a model in use, on time-ordered data (electricity)", "",
         "Rules: results/time_ordered/PREREGISTRATION.md. Seeds " + ", ".join(map(str, seeds))
         + f"; weeks {df.week.min()}–{df.week.max()} (each update is scored on the next week).", ""]
    t0 = df.drop_duplicates("seed").theta_0.value_counts().sort_index()
    L.append("θ₀ chosen on weeks 0–3 (seeds): " + ", ".join(f"{t:g} ({c})" for t, c in t0.items()) + ".")

    L += ["", "## Primary: broken predictions in the next week (NFR), SR minus INC", "",
          "Mean over seeds for each week; two-sided Wilcoxon signed-rank test over the weeks.", ""]
    prim = test(differences(df, "SR", "INC", "nfr_pct"), "week")
    pv = verdict(prim, "p", True)
    L += ["| Weeks | Mean difference (% of the week) | Weeks SR lower / higher | p | Verdict for SR |",
          "|---|---|---|---|---|",
          f"| {prim['n']} | {prim['mean_diff']:+.4f} | {prim['a_lower']} / {prim['a_higher']} | "
          f"{prim['p']:.3g} | {pv} |"]

    res = []
    for a, b, metric, low, title in SECONDARY:
        d = differences(df, a, b, metric)
        units = ("seed", "month") if title.startswith("Robustness") else ("week",)
        for u in units:
            r = test(d, u)
            r.update({"title": title + ("" if u == "week" else f" (by {u})"), "low": low})
            res.append(r)
    adj = holm([r["p"] for r in res])
    L += ["", "## Secondary (Holm across this table)", "",
          "| Comparison | Unit | n | Mean difference | Lower / higher | p | p (Holm) | Verdict for the first arm |",
          "|---|---|---|---|---|---|---|---|"]
    for r, ph in zip(res, adj):
        r["p_holm"] = ph
        L.append(f"| {r['title']} | {r['unit']} | {r['n']} | {r['mean_diff']:+.4f} | {r['a_lower']} / "
                 f"{r['a_higher']} | {r['p']:.3g} | {ph:.3g} | {verdict(r, 'p_holm', r['low'])} |")

    L += ["", "## Check: the certificate bounds the next week's churn", "",
          "Bound from one random half of the week (one-sided 95% Clopper–Pearson on the certificate's "
          "share); churn measured on the other half. PASS if it holds in at least 95% of weeks × seeds.", "",
          "| Arm | Containment (changes inside the certificate) | Bound holds | Median certificate % | "
          "Median bound % | Median churn % | Check |", "|---|---|---|---|---|---|---|"]
    contained = True
    for a in CERTIFIED:
        g = df[df.arm == a]
        holds = g.bound_holds.astype(bool).mean()
        contain = g.cert_contains_changes.astype(bool).mean()
        contained &= bool(contain == 1)
        L.append(f"| {a} | {100 * contain:.1f}% | {100 * holds:.1f}% | {g.cert_pct.median():.1f} | "
                 f"{g.bound_upper_pct.median():.1f} | {g.churn_other_half_pct.median():.1f} | "
                 f"{'PASS' if holds >= 0.95 and contain == 1 else 'FAIL'} |")
    L += ["", f"**Containment: {'PASS' if contained else 'FAIL'}** (Proposition 4: every changed "
          "prediction lies inside the certificate, in every week, seed and certified arm)."]

    L += ["", "## Descriptive", "", "| Arm | Accuracy in use | Macro-F1 | NFR % | Changed % | Boxes at the end | "
          "Update seconds (median) |", "|---|---|---|---|---|---|---|"]
    last = df.week.max()
    for a in ARMS:
        g = df[df.arm == a]
        L.append(f"| {a} | {g.accuracy.mean():.2f} | {g.macro_f1.mean():.2f} | {g.nfr_pct.mean():.3f} | "
                 f"{g.changed_pct.mean():.2f} | {g[g.week == last].boxes.mean():.0f} | {g.update_s.median():.2f} |")
    L += ["", "| Arm | Error % | AUROC: ensemble, kNN, detector | Precision of the top 10% flags |",
          "|---|---|---|---|"]
    for a in ("INC", "SR"):
        g = df[df.arm == a]
        L.append(f"| {a} | {g.error_pct.mean():.1f} | {g.auc_ensemble.mean():.3f}, {g.auc_knn.mean():.3f}, "
                 f"{g.auc_detector.mean():.3f} | {g.flag_precision_pct.mean():.1f}% |")
    g = df[df.arm == "SR"]
    L += ["", f"PACE-SHAP on SR's decisions: {g.explain_ms.median():.1f} ms per explanation (median), "
          f"exact in {100 * g.explain_exact.astype(bool).mean():.0f}% of weeks."]
    rt = df[(df.arm == "RETUNE") & df.refit.astype(bool)]
    if len(rt):
        L.append("RETUNE's θ at its refits: " + ", ".join(
            f"{t:g} ({c})" for t, c in rt.theta.value_counts().sort_index().items()) + ".")

    L += ["", "## Reading (fixed in the pre-registration)", "",
          f"- Primary: SR {'broke fewer' if pv == 'better' else 'broke more' if pv == 'worse' else 'showed no significant difference in broken'} "
          "predictions than the incremental update."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="seed 194, 16 weeks: a timing check")
    ap.add_argument("--weeks", type=int, help="with --smoke: how many weeks")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    if a.smoke and out.resolve() == OUT.resolve():
        raise SystemExit("--smoke writes outside results/time_ordered: pass --out <dir>")
    if a.weeks and not a.smoke:
        raise SystemExit("--weeks is for --smoke only; the study uses every week")
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        seeds = [194] if a.smoke else SEEDS
        n_weeks = (a.weeks or 16) if a.smoke else N_WEEKS
        tasks = pending(out / "weeks.csv", [(DATASET, s, n_weeks) for s in seeds], a.resume, a.fresh)
        run_all(run_seed, tasks, out / "weeks.csv", a.jobs)
    text = summarise(out)
    print(text)
    if "**Containment: FAIL**" in text:
        sys.exit(1)


if __name__ == "__main__":
    main()
