"""Run P1 (tough review S2 / D2): time to a given accuracy, and one clean re-time of the speed claim.

Rules: results/time_to_accuracy/PREREGISTRATION.md (planned in advance).

The paper compares PACE-SHAP with KernelSHAP at a fixed budget of 200 coalitions, where
KernelSHAP is an estimate and PACE-SHAP (d <= 12) is exact. A fair speed comparison asks how long
KernelSHAP needs to come as close to the exact values as PACE-SHAP is, and the paper's speed
numbers come from several runs. This run does both, one task at a time.

Part (a), seeds 300-302, 10 test points per seed, the 6 explanation datasets with at most 12
features: PACE-SHAP in exact mode against KernelSHAP (``xai_baselines.kernel_shap_game``, paired
coalitions) at 100, 200, 400 and 700 coalitions (those below 2^d - 2) and with every coalition
("all", exact). Per point: the time and the relative L1 error ||phi_K - phi||_1 / ||phi||_1
against PACE-SHAP's exact values. b* is the smallest budget whose median relative error is at most
1%; the primary number is KernelSHAP's time at b* divided by PACE-SHAP's, with a 95% bootstrap
interval over points.

Part (b), seed 303, 10 test points per dataset, the 13 like-for-like datasets: KernelSHAP (200
coalitions) and PACE-SHAP (default mode: exact for d <= 12, else 32 permutations), both on the
same 200 explainer rows, each timed alone. This replaces the paper's "3.5-10x".

Each point is explained on its own call. PACE-SHAP's set-up (built once per task) is shared
equally among the task's points and added to each point's time; KernelSHAP has no set-up.

Both parts use the second explanation run's models (``run_faithfulness.setup``,
``backgrounds_v3``, 200 explainer rows). It measures time, so run it alone:
    python -B scripts/run_time_to_accuracy.py
    python -B scripts/run_time_to_accuracy.py --resume
    python -B scripts/run_time_to_accuracy.py --pilot          # 1 point, 1 seed per part
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

EXACT_SETS = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
D13 = ["S1", "S2", "S3", "S1c", "S2c"] + rf.DATASETS[3:]
SEEDS_A, SEED_B = [300, 301, 302], 303
N_POINTS = 10
BUDGETS = [100, 200, 400, 700]
TARGET = 0.01
N_BOOT = 2000
PACE, KERNEL = "PACE-SHAP", "KernelSHAP"
OUT = ROOT / "results" / "time_to_accuracy"


def _models(name, seed, n):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n)
    B = rf.backgrounds_v3(Xtr, seed)[1][:rf.SHAP_V3_R]
    return model, Xte, B


def rel_l1(phi, ref):
    return float(np.abs(phi - ref).sum() / max(np.abs(ref).sum(), 1e-300))


def run_one(name, seed, n, part):
    model, Xte, B = _models(name, seed, n)
    d = Xte.shape[1]
    clock = time.perf_counter
    PaceShap(model, B, n_permutations=rf.SHAP_V3_P, seed=seed).explain(Xte[:1])     # warm-up
    xb.kernel_shap_game(model, Xte[:1], B, nsamples=20, seed=seed)
    rows = []
    base = {"part": part, "dataset": name, "seed": seed, "d": d, "boxes": int(model.n_boxes_),
            "background_rows": len(B)}
    if part == "a":
        t0 = clock()
        ps = PaceShap(model, B)                                  # exact (d <= 12)
        build = clock() - t0
        if not ps.exact:
            raise ValueError(f"{name}: part (a) needs exact mode")
        for p in range(len(Xte)):
            t0 = clock()
            ref = ps.explain(Xte[p:p + 1])[0]
            rows.append({**base, "point": p, "arm": PACE, "budget": "exact",
                         "ms": 1000 * (clock() - t0 + build / len(Xte)), "rel_l1": 0.0})
            full = (1 << d) - 2
            for b in [b for b in BUDGETS if b < full] + [full]:
                t0 = clock()
                phi = xb.kernel_shap_game(model, Xte[p:p + 1], B, nsamples=b, seed=seed * 1000 + p)[0]
                rows.append({**base, "point": p, "arm": KERNEL, "budget": "all" if b == full else str(b),
                             "ms": 1000 * (clock() - t0), "rel_l1": rel_l1(phi, ref)})
    else:
        t0 = clock()
        ps = PaceShap(model, B, n_permutations=rf.SHAP_V3_P, seed=seed)
        build = clock() - t0
        for p in range(len(Xte)):
            t0 = clock()
            ps.explain(Xte[p:p + 1])
            rows.append({**base, "point": p, "arm": PACE, "budget": "default",
                         "ms": 1000 * (clock() - t0 + build / len(Xte)), "rel_l1": np.nan})
            t0 = clock()
            xb.kernel_shap_game(model, Xte[p:p + 1], B, nsamples=rf.COND_NSAMPLES, seed=seed * 1000 + p)
            rows.append({**base, "point": p, "arm": KERNEL, "budget": str(rf.COND_NSAMPLES),
                         "ms": 1000 * (clock() - t0), "rel_l1": np.nan})
    k = pd.DataFrame(rows).groupby(["arm", "budget"]).ms.mean()
    print(f"{name} seed {seed} part {part}: " + ", ".join(f"{a} {b} {v:.0f} ms" for (a, b), v in k.items()),
          flush=True)
    return rows


def b_star(g):
    """The smallest KernelSHAP budget whose median relative L1 error is at most TARGET."""
    k = g[g.arm == KERNEL]
    med = k.groupby("budget").rel_l1.median()
    order = sorted((b for b in med.index if b != "all"), key=int) + (["all"] if "all" in med.index else [])
    for b in order:
        if med[b] <= TARGET:
            return b, med
    return None, med


def ratio_ci(kms, pms, seed=0):
    """Mean KernelSHAP time / mean PACE-SHAP time, with a 95% bootstrap interval over points."""
    kms, pms = np.asarray(kms, float), np.asarray(pms, float)
    r = kms.mean() / pms.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kms), size=(N_BOOT, len(kms)))
    boot = kms[idx].mean(axis=1) / pms[idx].mean(axis=1)
    return float(r), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarise(df, out: Path) -> str:
    L = ["# Run P1: time to a given accuracy, and one clean re-time", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and 200 explainer rows, one task at "
         "a time. Written by `scripts/run_time_to_accuracy.py`.", ""]
    a = df[df.part == "a"]
    if len(a):
        seeds = sorted(int(s) for s in a.seed.unique())
        L += ["## Part (a): KernelSHAP's time to reach 1% of the exact values", "",
              f"Seeds {', '.join(map(str, seeds))}, {N_POINTS} points per seed. Relative L1 error against "
              "PACE-SHAP's exact values, median over points; b* is the smallest budget with median error "
              f"at most {TARGET:.0%}. Time ratio: KernelSHAP at b* over exact PACE-SHAP (mean over points, "
              "95% bootstrap interval).", "",
              "| Dataset | d | Median error at 100 | 200 | 400 | 700 | all | b* | ms PACE-SHAP (exact) | "
              "ms KernelSHAP at b* | Ratio at b* (95% interval) | Ratio at 200 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for ds in [d for d in EXACT_SETS if d in set(a.dataset)]:
            g = a[a.dataset == ds]
            bs, med = b_star(g)
            cell = lambda b: f"{med[b]:.2%}" if b in med.index else "–"   # noqa: E731
            pm = g[g.arm == PACE].sort_values(["seed", "point"]).ms.to_numpy()
            if bs is None:
                star, kt, rr_ = "none", "–", "–"
            else:
                km = g[(g.arm == KERNEL) & (g.budget == bs)].sort_values(["seed", "point"]).ms.to_numpy()
                r, lo, hi = ratio_ci(km, pm)
                star, kt, rr_ = bs, f"{km.mean():.0f}", f"{r:.1f}× ({lo:.1f}–{hi:.1f})"
            k200 = g[(g.arm == KERNEL) & (g.budget == "200")].sort_values(["seed", "point"]).ms.to_numpy()
            r200 = f"{k200.mean() / pm.mean():.1f}×" if len(k200) else "–"
            L.append(f"| {ds} | {g.d.iloc[0]} | {cell('100')} | {cell('200')} | {cell('400')} | {cell('700')} | "
                     f"{cell('all')} | {star} | {pm.mean():.0f} | {kt} | {rr_} | {r200} |")
        L += [""]
    b = df[df.part == "b"]
    if len(b):
        L += ["## Part (b): one clean re-time of the like-for-like comparison", "",
              f"Seed {SEED_B}, {N_POINTS} points per dataset, each explainer timed alone on the same 200 rows. "
              "KernelSHAP with 200 coalitions; PACE-SHAP in its default mode (exact for d ≤ 12, else 32 "
              "permutations).", "",
              "| Dataset | d | PACE-SHAP mode | ms PACE-SHAP | ms KernelSHAP | KernelSHAP ÷ PACE-SHAP (95% interval) |",
              "|---|---|---|---|---|---|"]
        rs = []
        for ds in [d for d in D13 if d in set(b.dataset)]:
            g = b[b.dataset == ds]
            pm = g[g.arm == PACE].sort_values("point").ms.to_numpy()
            km = g[g.arm == KERNEL].sort_values("point").ms.to_numpy()
            r, lo, hi = ratio_ci(km, pm)
            rs.append(r)
            mode = "exact" if g.d.iloc[0] <= 12 else "32 permutations"
            L.append(f"| {ds} | {g.d.iloc[0]} | {mode} | {pm.mean():.0f} | {km.mean():.0f} | "
                     f"{r:.1f}× ({lo:.1f}–{hi:.1f}) |")
        L += ["", f"Across the {len(rs)} datasets KernelSHAP took {min(rs):.1f}–{max(rs):.1f}× as long as "
              "PACE-SHAP (median {:.1f}×).".format(float(np.median(rs)))]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P1: time to accuracy")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--parts", nargs="+", default=["a", "b"])
    ap.add_argument("--pilot", action="store_true", help="1 point, first seed of each part")
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else (OUT / "pilot" if a.pilot else OUT)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    n = 1 if a.pilot else N_POINTS
    if not a.summary_only:
        tasks = []
        if "a" in a.parts:
            tasks += [(d, s, n, "a") for d in a.datasets or EXACT_SETS
                      for s in (SEEDS_A[:1] if a.pilot else SEEDS_A) if d in EXACT_SETS]
        if "b" in a.parts:
            tasks += [(d, SEED_B, n, "b") for d in a.datasets or D13 if d in D13]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=1)
    df = pd.read_csv(csv, float_precision="round_trip", dtype={"budget": str})
    print(summarise(df, out))


if __name__ == "__main__":
    main()
