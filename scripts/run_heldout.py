"""Run P10 (tough review B11): RQ1 and RQ2 again, on held-out datasets chosen by a rule.

Rules: results/heldout/PREREGISTRATION.md (planned in advance). The datasets: the rule in
results/heldout/RULE.md, frozen before any download: every OpenML-CC18 dataset with numeric
features only, at most 12 features, at least 2,000 rows, no missing values, and not already in
the paper (``data.HELDOUT``: phoneme, wilt, mfeat_morphological, jungle_chess).

RQ1, seeds 360-369, 50 test points per seed: the paper's two explanation protocols, unchanged.
* Like for like (``run_faithfulness.run_cond``): PACE-SHAP and KernelSHAP on the same 200
  explainer rows, KernelSHAP at 200 coalitions. Each task also records the largest difference
  between the two explainers' values, since with d <= 6 features KernelSHAP enumerates every
  coalition (2^d - 2 <= 62 < 200) and so is exact too.
* The second explanation run (``run_faithfulness.run``, protocol "v3"): PACE-SHAP against
  KernelSHAP at its usual setting (20 background rows) and the other explainers.
RQ2, seeds 370-379, on the repair-at-theta split (the paper's 80/20 split, the seed's row order,
min-max scaling fitted on the training rows; ``run_readable_baselines.prepare``):
* B0-0.3: CFMNN at theta = 0.3;
* SR-0.3: B0-0.3, then self-repair (300 errors, 10 rounds, the smallest cut, gated on the
  training rows: no label beyond them), exactly as ``run_repair_theta.run_one``;
* B0-val: CFMNN at the theta chosen by 3-fold validation (re-tuning).
Every model's test predictions are compared with B0-0.3's: changed, fixed and broken.

    python -B scripts/run_heldout.py --jobs 6
    python -B scripts/run_heldout.py --jobs 6 --resume       # continue after a stop
    python -B scripts/run_heldout.py --summary-only
    python -B scripts/run_heldout.py --smoke --out <dir>     # seed 194, 10 points; not a study run
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
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.audit import prediction_churn  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
import run_faithfulness as rf  # noqa: E402
from run_readable_baselines import GAMMA, box_size, prepare  # noqa: E402
from run_repair_theta import BUDGET, CHOOSER, FIXED_THETA, N_ROUNDS, theta_val, training_errors  # noqa: E402

OUT = ROOT / "results" / "heldout"
DATASETS = list(data.HELDOUT)
RQ1_SEEDS = list(range(360, 370))
RQ2_SEEDS = list(range(370, 380))
N_EXPLAIN = 50
# Slowest first (Jungle Chess has 44,819 rows); this only orders the tasks.
COST = {"jungle_chess": 4, "phoneme": 2, "wilt": 1, "mfeat_morphological": 1}
ALPHA = 0.05
CHECK_TOL = 1e-9


# ------------------------------------------------------------------ RQ1
def like_for_like_check(name: str, seed: int, n_explain: int) -> dict:
    """Largest |PACE-SHAP - KernelSHAP| on run_cond's explainer rows and points."""
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n_explain)
    _, bex = rf.backgrounds_v3(Xtr, seed)
    B = bex[:rf.SHAP_V3_R]
    pace = PaceShap(model, B, n_permutations=rf.SHAP_V3_P, seed=seed).explain(Xte)
    kern = xb.kernel_shap_game(model, Xte, B, nsamples=rf.COND_NSAMPLES, seed=seed)
    d = Xte.shape[1]
    return {"d": d, "kernel_enumerates": rf.COND_NSAMPLES >= (1 << d) - 2,
            "like_for_like_max_abs_diff": float(np.max(np.abs(pace - kern)))}


def run_rq1(name: str, seed: int, n_explain: int = N_EXPLAIN):
    warnings.filterwarnings("ignore")
    check = like_for_like_check(name, seed, n_explain)
    cond_rows, _, _ = rf.run_cond(name, seed, n_explain)
    v3_rows, _ = rf.run(name, seed, n_explain, protocol="v3")
    return [{**r, **check} for r in cond_rows + v3_rows]


# ------------------------------------------------------------------ RQ2
def run_rq2(name: str, seed: int):
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = prepare(name, seed)
    tv, source = theta_val(name, seed, Xtr, ytr)
    t0 = time.perf_counter()
    b0 = CFMNN(theta=FIXED_THETA, gamma=GAMMA).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    p0 = np.asarray(b0.predict(Xte))
    t0 = time.perf_counter()
    sr, _, edits = run_self_repair(b0, Xtr, ytr, Xtr, ytr, chooser=CHOOSER, budget=BUDGET,
                                   n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False)
    repair_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    bv = CFMNN(theta=tv, gamma=GAMMA).fit(Xtr, ytr)
    val_s = time.perf_counter() - t0
    rows = []
    for model, m, theta, secs in (("B0-0.3", b0, FIXED_THETA, fit_s), ("SR-0.3", sr, FIXED_THETA, repair_s),
                                  ("B0-val", bv, tv, val_s)):
        pred = np.asarray(m.predict(Xte))
        boxes, cond = box_size(m.V_, m.W_)
        churn = prediction_churn(yte, p0, pred)
        rec = {"dataset": name, "seed": seed, "model": model, "theta": theta,
               "theta_source": source if model == "B0-val" else "fixed",
               "accuracy": 100 * float(np.mean(pred == yte)),
               "macro_f1": 100 * float(f1_score(yte, pred, average="macro")),
               "boxes": boxes, "conditions_per_box": cond, **training_errors(m, Xtr, ytr),
               "test_changed": churn["changed"], "test_fixed": churn["fixed"],
               "test_broken": churn["broken"], "nfr_pct": 100 * churn["broken"] / len(yte),
               "seconds": secs, "n_train": len(ytr), "n_test": len(yte),
               "edits_logged": np.nan, "edits_accepted": np.nan}
        if model == "SR-0.3":
            rec.update({"edits_logged": len(edits),
                        "edits_accepted": sum(bool(e["accepted"]) for e in edits)})
        rows.append(rec)
    print(f"{name} seed {seed} (theta-val {tv:g}): " + ", ".join(
        f"{r['model']} F1 {r['macro_f1']:.1f} NFR {r['nfr_pct']:.2f}%" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------ tests
def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def compare(df: pd.DataFrame, key: str, a: str, b: str, metric: str, lower_is_better: bool,
            datasets=None) -> pd.DataFrame:
    """Paired over seeds: a minus b on ``metric``, two-sided Wilcoxon, Holm across datasets.

    The verdict is for ``a``: "better", "worse", "no significant difference", or "no change"
    when a and b are equal on every seed (the test is then not run, p = 1).
    """
    rows = []
    for ds in datasets or list(dict.fromkeys(df.dataset)):
        g = df[df.dataset == ds]
        x = g[g[key] == a].set_index("seed")[metric]
        y = g[g[key] == b].set_index("seed")[metric]
        diff = (x - y.reindex(x.index)).dropna()
        same = bool(len(diff) and np.all(diff == 0))
        p = 1.0 if same or len(diff) < 2 else float(wilcoxon(diff.to_numpy()).pvalue)
        good = (diff < 0) if lower_is_better else (diff > 0)
        bad = (diff > 0) if lower_is_better else (diff < 0)
        rows.append({"dataset": ds, "diff": float(diff.mean()) if len(diff) else np.nan,
                     "seeds_a_better": int(good.sum()), "seeds_a_worse": int(bad.sum()),
                     "n": len(diff), "p": p, "same": same})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy()) if len(out) else []
    verdicts = []
    for _, r in out.iterrows():
        if r.same:
            verdicts.append("no change")
        elif r.p_holm < ALPHA:
            better = r["diff"] < 0 if lower_is_better else r["diff"] > 0
            verdicts.append("better" if better else "worse")
        else:
            verdicts.append("no significant difference")
    out["verdict"] = verdicts
    return out


def _table(L, res, title, unit, note):
    L += ["", f"### {title}", "", note, "",
          f"| Dataset | Mean difference ({unit}) | Seeds better / worse | p | p (Holm) | Verdict |",
          "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r.dataset} | {r['diff']:+.4f} | {r.seeds_a_better} / {r.seeds_a_worse} of {r.n} | "
                 f"{r.p:.3g} | {r.p_holm:.3g} | {r.verdict} |")


def summarise(out: Path) -> str:
    L = ["# Run P10: RQ1 and RQ2 on the held-out datasets", "",
         "Rules: results/heldout/PREREGISTRATION.md. Datasets by the rule in results/heldout/RULE.md. "
         "Paired over seeds; two-sided Wilcoxon signed-rank test; Holm across the 4 datasets, "
         "separately for each primary."]
    p1 = out / "rq1_runs.csv"
    if p1.exists():
        r1 = pd.read_csv(p1)
        cond, v3 = r1[r1.protocol == "cond"], r1[r1.protocol == "v3"]
        chk = r1.drop_duplicates(["dataset", "seed"]).groupby("dataset", sort=False).agg(
            d=("d", "first"), enumerates=("kernel_enumerates", "all"),
            max_diff=("like_for_like_max_abs_diff", "max"))
        L += ["", "## RQ1 (seeds " + ", ".join(str(s) for s in sorted(r1.seed.unique())) + ")", "",
              "| Dataset | Features | KernelSHAP enumerates every coalition | Largest |PACE-SHAP − KernelSHAP|, same rows |",
              "|---|---|---|---|"]
        for ds, r in chk.iterrows():
            L.append(f"| {ds} | {int(r.d)} | {'yes' if r.enumerates else 'no'} | {r.max_diff:.2e} |")
        ok = bool((chk.max_diff < CHECK_TOL).all())
        L += ["", f"**Check: {'PASS' if ok else 'FAIL'}** (every largest difference below {CHECK_TOL:g}, "
              "where KernelSHAP enumerates every coalition)."]
        _table(L, compare(cond, "method", "PACE-SHAP", "KernelSHAP", "margin_auc", True),
               "Primary: deletion AUC, PACE-SHAP minus KernelSHAP, same rows", "AUC",
               "Lower is more faithful; negative favours PACE-SHAP.")
        _table(L, compare(cond, "method", "PACE-SHAP", "KernelSHAP", "insertion_margin_auc", False),
               "Secondary: insertion AUC, same rows", "AUC", "Higher is more faithful; positive favours PACE-SHAP.")
        _table(L, compare(v3, "method", "PACE-SHAP", "KernelSHAP", "margin_auc", True),
               "Secondary: deletion AUC against KernelSHAP at its usual setting (20 rows)", "AUC",
               "The second explanation run's comparison. Negative favours PACE-SHAP.")
        _table(L, compare(v3, "method", "PACE-SHAP", "KernelSHAP", "insertion_margin_auc", False),
               "Secondary: insertion AUC against KernelSHAP at its usual setting", "AUC",
               "Positive favours PACE-SHAP.")
        ms = cond[cond.method.isin(["PACE-SHAP", "KernelSHAP"])].pivot_table(
            index="dataset", columns="method", values="ms_per_explanation", aggfunc="median")
        L += ["", "### Time per explanation, same rows (median over seeds; 6 jobs shared the machine)", "",
              "| Dataset | PACE-SHAP (ms) | KernelSHAP (ms) | Ratio |", "|---|---|---|---|"]
        for ds, r in ms.iterrows():
            L.append(f"| {ds} | {r['PACE-SHAP']:.2f} | {r['KernelSHAP']:.2f} | "
                     f"{r['KernelSHAP'] / r['PACE-SHAP']:.1f}× |")
        _table(L, compare(cond, "method", "PACE-SHAP", "KernelSHAP", "ms_per_explanation", True),
               "Secondary: time per explanation, same rows", "ms",
               "Negative: PACE-SHAP faster. Both explainers are exact here, so time is the comparison.")
    p2 = out / "rq2_runs.csv"
    if p2.exists():
        r2 = pd.read_csv(p2)
        L += ["", "## RQ2 (seeds " + ", ".join(str(s) for s in sorted(r2.seed.unique())) + ")", "",
              "| Dataset | Training errors at θ = 0.3 (%) | Seeds where SR-0.3 changed a test prediction | "
              "θ by validation (seeds) | Boxes: B0-0.3, SR-0.3, B0-val | "
              "Macro-F1: B0-0.3, SR-0.3, B0-val | NFR % vs B0-0.3: SR-0.3, B0-val |",
              "|---|---|---|---|---|---|---|"]
        no_op = []
        for ds, g in r2.groupby("dataset", sort=False):
            m = g.groupby("model").mean(numeric_only=True)
            tv = g[g.model == "B0-val"].theta.value_counts().sort_index()
            sr = g[g.model == "SR-0.3"]
            moved = int((sr.test_changed > 0).sum()) if "test_changed" in sr else 0
            if moved == 0:
                no_op.append(ds)
            L.append(f"| {ds} | {m.loc['B0-0.3', 'train_error_pct']:.2f} | {moved} of {len(sr)} | "
                     + ", ".join(f"{t:g} ({c})" for t, c in tv.items()) + " | "
                     + ", ".join(f"{m.loc[k, 'boxes']:.0f}" for k in ("B0-0.3", "SR-0.3", "B0-val")) + " | "
                     + ", ".join(f"{m.loc[k, 'macro_f1']:.2f}" for k in ("B0-0.3", "SR-0.3", "B0-val")) + " | "
                     + ", ".join(f"{m.loc[k, 'nfr_pct']:.2f}" for k in ("SR-0.3", "B0-val")) + " |")
        a = compare(r2, "model", "SR-0.3", "B0-0.3", "macro_f1", False)
        b = compare(r2, "model", "SR-0.3", "B0-val", "nfr_pct", True)
        b_moved = b[~b.dataset.isin(no_op)]
        _table(L, a, "Primary (a): macro-F1, SR-0.3 minus B0-0.3", "points", "Positive favours self-repair.")
        _table(L, b, "Primary (b): broken test predictions (NFR), SR-0.3 minus B0-val", "% of test rows",
               "Both counted against B0-0.3. Negative: repair broke fewer than re-tuning.")
        _table(L, compare(r2, "model", "SR-0.3", "B0-val", "macro_f1", False),
               "Secondary: macro-F1, SR-0.3 minus B0-val", "points", "Positive favours repair.")
        L += ["", "## Reading (fixed in the pre-registration)", ""]
        L.append(f"- RQ2 (a): self-repair better on {int((a.verdict == 'better').sum())}, worse on "
                 f"{int((a.verdict == 'worse').sum())}, no significant difference on "
                 f"{int((a.verdict == 'no significant difference').sum())}, unchanged on "
                 f"{int((a.verdict == 'no change').sum())} of {len(a)}.")
        L.append(f"- RQ2 (b): repair broke fewer predictions than re-tuning on "
                 f"{int((b.verdict == 'better').sum())}, more on {int((b.verdict == 'worse').sum())}, "
                 f"no significant difference on {int((b.verdict == 'no significant difference').sum())} of {len(b)}.")
        L.append(f"- Where repair changed no test prediction on any seed (it cannot break one): "
                 f"{', '.join(no_op) if no_op else 'none'}. Counting only the other "
                 f"{len(b_moved)}: fewer on {int((b_moved.verdict == 'better').sum())}, more on "
                 f"{int((b_moved.verdict == 'worse').sum())}, no significant difference on "
                 f"{int((b_moved.verdict == 'no significant difference').sum())}.")
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
    ap.add_argument("--smoke", action="store_true", help="seed 194, 10 points: a timing check")
    ap.add_argument("--out")
    ap.add_argument("--parts", nargs="+", choices=["rq1", "rq2"], default=["rq1", "rq2"])
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    if a.smoke and out.resolve() == OUT.resolve():
        raise SystemExit("--smoke writes outside results/heldout: pass --out <dir>")
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        n = 10 if a.smoke else N_EXPLAIN
        if "rq1" in a.parts:
            seeds = [194] if a.smoke else RQ1_SEEDS
            tasks = pending(out / "rq1_runs.csv", [(d, s, n) for d in DATASETS for s in seeds],
                            a.resume, a.fresh)
            run_all(run_rq1, tasks, out / "rq1_runs.csv", a.jobs, COST)
        if "rq2" in a.parts:
            seeds = [194] if a.smoke else RQ2_SEEDS
            tasks = pending(out / "rq2_runs.csv", [(d, s) for d in DATASETS for s in seeds], a.resume, a.fresh)
            run_all(run_rq2, tasks, out / "rq2_runs.csv", a.jobs, COST)
    text = summarise(out)
    print(text)
    if "**Check: FAIL**" in text:
        sys.exit(1)


if __name__ == "__main__":
    main()
