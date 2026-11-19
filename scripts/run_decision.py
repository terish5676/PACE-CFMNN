"""Run P5 (tough review S6 / B3): at ties, does explaining the decision beat explaining the margin?

Rules: results/decision/PREREGISTRATION.md (planned in advance).

PACE-SHAP explains the margin s'_A - max_{c != A} s'_c. At a tie the margin is 0 and the
nearest-centroid rule picks the class, so the margin says little about the decision.
PACE-decision (``pace_cfmnn.decision``) explains the decision itself: v(S) is the share of
background rows z for which (x_S, z_rest) is predicted as A, with CFMNN's full rule, ties included.

Setup: the second explanation run's models and rows (``run_faithfulness.setup``,
``backgrounds_v3``: 200 evaluation rows for the tests, 200 explainer rows), seeds 320-329, 50 test
points per seed. Datasets: the explanation datasets with at least 10 tied explained points in every
seed of the second explanation run (seeds 30-39): MAGIC, Letter and Page Blocks (fixed before this
run; Landsat's fewest was 5 and Shuttle's 3). A point is tied when the runner-up class's rescaled
support is within 2 x tie_tol of the winner's.

Explainers: PACE-decision and PACE-SHAP, both exact for d <= 12, else 32 antithetic permutations
with the seed, on the same 200 explainer rows.

Primary: the decision deletion AUC (the share of fill-ins that keep the class, over k = 0..d
deletions, most supportive feature first; ``run_faithfulness.deletion_auc``'s first output; lower
is more faithful), mean over the seed's tied points, PACE-decision against PACE-SHAP: paired
two-sided Wilcoxon over the 10 seeds, Holm across the 3 datasets.

Check: where exact, PACE-decision equals ``brute_force_decision`` (the model's own prediction on
every mixed input) within 1e-9 on the first 2 points of each seed.

    python -B scripts/run_decision.py --jobs 6
    python -B scripts/run_decision.py --jobs 6 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn.decision import PaceDecision, brute_force_decision  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_error_detection import holm  # noqa: E402

DATASETS = ["magic", "letter", "page_blocks"]
SEEDS = list(range(320, 330))
N_POINTS, N_CHECK, N_PERM = 50, 2, 32
TOL = 1e-9
DEC, SHP = "PACE-decision", "PACE-SHAP"
# (column, lower is better, label); the first is the primary
MEASURES = [("decision_deletion_auc", True, "decision deletion AUC"),
            ("decision_insertion_auc", False, "decision insertion AUC"),
            ("margin_deletion_auc", True, "margin deletion AUC"),
            ("margin_insertion_auc", False, "margin insertion AUC")]
OUT = ROOT / "results" / "decision"


def tied_mask(model, X):
    """Tied points: the runner-up class's rescaled support within 2 x tie_tol of the winner's."""
    det = model.predict_details(X)
    s = np.where(det.best_box >= 0, det.support_rescaled, -np.inf)
    rows = np.arange(len(X))
    own = s[rows, det.cls].copy()
    s[rows, det.cls] = -np.inf
    return own - s.max(axis=1) <= 2 * float(model.tie_tol)


def run_one(name, seed, n=N_POINTS):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    bg, bex = rf.backgrounds_v3(Xtr, seed)
    B = bex[:rf.SHAP_V3_R]
    tied = tied_mask(model, Xte)
    t0 = time.perf_counter()
    pdx = PaceDecision(model, B, n_permutations=N_PERM, seed=seed)
    dec = pdx.explain(Xte)
    t_dec = time.perf_counter() - t0
    t0 = time.perf_counter()
    shp = PaceShap(model, B, n_permutations=N_PERM, seed=seed).explain(Xte)
    t_shp = time.perf_counter() - t0
    check = np.nan
    if pdx.exact:
        check = max(float(np.abs(dec[p] - brute_force_decision(model, Xte[p], B)).max())
                    for p in range(min(N_CHECK, len(Xte))))
    rows = []
    for method, attr, secs in ((DEC, dec, t_dec), (SHP, shp, t_shp)):
        kd, md = rf.deletion_auc(model, Xte, attr, bg, seed=seed, signed=True)
        ki, mi = rf.insertion_auc(model, Xte, attr, bg, seed=seed)
        for p in range(len(Xte)):
            rows.append({"dataset": name, "seed": seed, "point": p, "method": method, "d": Xte.shape[1],
                         "mode": "exact" if pdx.exact else "sampled", "tied": bool(tied[p]),
                         "decision_deletion_auc": kd[p], "margin_deletion_auc": md[p],
                         "decision_insertion_auc": ki[p], "margin_insertion_auc": mi[p],
                         "ms_per_explanation": 1000 * secs / len(Xte), "check_max_diff": check})
    g = pd.DataFrame(rows)
    t = g[g.tied].groupby("method").decision_deletion_auc.mean()
    print(f"{name} seed {seed}: {int(tied.sum())} tied of {len(Xte)}; tied decision deletion AUC "
          + ", ".join(f"{m} {v:.3f}" for m, v in t.items()) + (f"; check {check:.1e}" if pdx.exact else ""),
          flush=True)
    return rows


def compare(df, col, lower, subset="tied"):
    sel = df[df.tied] if subset == "tied" else (df[~df.tied] if subset == "untied" else df)
    per = sel.groupby(["dataset", "seed", "method"])[col].mean().unstack("method")
    out = []
    for ds in [d for d in DATASETS if d in per.index.get_level_values(0)]:
        x = per.loc[ds].dropna()
        dlt = (x[DEC] - x[SHP]).to_numpy()
        p = float(wilcoxon(dlt).pvalue) if np.any(dlt != 0) else 1.0
        better = (dlt < 0) if lower else (dlt > 0)
        out.append({"dataset": ds, "n_seeds": len(dlt), "decision": x[DEC].mean(), "shap": x[SHP].mean(),
                    "diff": dlt.mean(), "wins": int(better.sum()), "losses": int(((dlt != 0) & ~better).sum()),
                    "p": p})
    t = pd.DataFrame(out)
    if len(t):
        t["p_holm"] = holm(t.p)
        good = (t["diff"] < 0) if lower else (t["diff"] > 0)
        t["verdict"] = np.where(t.p_holm >= 0.05, "no significant difference", np.where(good, "better", "worse"))
    return t


def check(df):
    g = df[df.check_max_diff > TOL].drop_duplicates(["dataset", "seed"])
    return g.empty, [f"{r.dataset} seed {r.seed}: PACE-decision differs from the brute force by "
                     f"{r.check_max_diff:.1e}" for _, r in g.iterrows()]


def summarise(df, out: Path) -> str:
    ok, bad = check(df)
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# Run P5: explaining the decision at ties", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and rows, seeds "
         f"{seeds[0]}–{seeds[-1]}, {N_POINTS} points per seed. Written by `scripts/run_decision.py`.", "",
         f"**Check: {'PASS' if ok else 'FAIL'}.** Where exact, PACE-decision equals the brute force "
         f"(the model's prediction on every mixed input) within {TOL:g} on the first {N_CHECK} points of each seed."]
    L += [f"- {b}" for b in bad]
    pts = df[df.method == DEC]
    L += ["", "Tied explained points per seed: " + "; ".join(
        f"{ds} {int(g.groupby('seed').tied.sum().min())}–{int(g.groupby('seed').tied.sum().max())}"
        for ds, g in pts.groupby("dataset", sort=False)) + "."]
    for i, (col, lower, label) in enumerate(MEASURES):
        for subset in (("tied", "untied") if i == 0 else ("tied",)):
            t = compare(df, col, lower, subset)
            if t.empty:
                continue
            head = "Primary" if (i == 0 and subset == "tied") else "Secondary"
            L += ["", f"## {head}: {label}, {subset} points ({'lower' if lower else 'higher'} is better)", "",
                  "| Dataset | Seeds | PACE-decision | PACE-SHAP | Difference | Seeds better / worse | p | p (Holm) | "
                  "Verdict for PACE-decision |", "|---|---|---|---|---|---|---|---|---|"]
            for _, r in t.iterrows():
                L.append(f"| {r.dataset} | {r.n_seeds} | {r.decision:.4f} | {r.shap:.4f} | {r['diff']:+.4f} | "
                         f"{r.wins} / {r.losses} | {r.p:.4f} | {r.p_holm:.4f} | {r.verdict} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P5: decision explanation at ties")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, help="smoke runs only")
    ap.add_argument("--n", type=int, help="points per seed (smoke runs only)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "points.csv"
    if not a.summary_only:
        tasks = [(d, s, a.n or N_POINTS) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=a.jobs, cost={"letter": 3, "magic": 2})
    df = pd.read_csv(csv, float_precision="round_trip")
    print(summarise(df, out))
    sys.exit(0 if check(df)[0] else 1)


if __name__ == "__main__":
    main()
