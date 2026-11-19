"""Run N: Covertype (54 features) in the sampled-baseline comparison of Run J. Descriptive.

Rules: results/sampled_baselines_covertype/PREREGISTRATION.md. No comparison is tested.

Run J (``scripts/run_sampled_baselines.py``, seeds 160-169) compared PACE-SHAP's sampled mode with
PermutationSHAP and paired-sampling KernelSHAP on the explanation datasets with 13-36 features.
This run adds Covertype, with 54. Everything is Run J's, imported unchanged: the task
(``run_one``: deletion and insertion margin AUC, top-3 stability under a new seed, model rows used,
milliseconds), the timing pass (``time_one``), the methods (``explainers``) and the equal budget
(``budget``: 200 rows x 32 orders x (54 + 1) = 352,000 model rows per explanation).
``run_faithfulness.setup`` accepts the name: the paper's 80/20 split, CFMNN(theta = 0.3, gamma = 4)
on all 464,809 training rows, 50 explained test points, and the explainer rows of
``backgrounds_v3``. No exact reference: 2^54 coalitions.

    python -B scripts/run_sampled_baselines_covertype.py --jobs 5
    python -B scripts/run_sampled_baselines_covertype.py --timing         # alone, after the run
    python -B scripts/run_sampled_baselines_covertype.py --summary-only
    python -B scripts/run_sampled_baselines_covertype.py --seeds 192 --n 10 --out <folder> --fresh   # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sampled_baselines as rj  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

OUT = ROOT / "results" / "sampled_baselines_covertype"
NAME = "covertype"
SEEDS = list(range(230, 235))
TIMING_SEEDS = [230, 231, 232]
N_EXPLAIN = rj.N_EXPLAIN            # 50, as Run J
TIMING_N = rj.TIMING_N              # 10, as Run J
METHODS = rj.METHODS
PACE, PERM, KERNEL, RANDOM = rj.PACE, rj.PERM, rj.KERNEL, rj.RANDOM


def run_one(name: str, seed: int, n_explain: int = N_EXPLAIN):
    return rj.run_one(name, seed, n_explain)


def time_one(name: str, seed: int, n_explain: int = TIMING_N):
    return rj.time_one(name, seed, n_explain)


# ------------------------------------------------------------------ summary
def _ms(v, fmt="{:,.4f}") -> str:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "–"
    s = fmt.format(v.mean())
    return f"{s} ± {fmt.format(v.std(ddof=1))}" if len(v) > 1 else s


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    b = rj.budget(int(df.background_rows.iloc[0]), int(df.d.iloc[0]))
    L = ["# Run N: Covertype in the sampled-baseline comparison", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Descriptive: no comparison is tested. "
         f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}), {int(df.d.iloc[0])} features, "
         f"{int(df.n.iloc[0])} explained test points per seed, {int(df.background_rows.iloc[0])} explainer "
         f"rows, budget {b['rows']:,} model rows per explanation. Mean ± standard deviation over seeds. "
         "Written by `scripts/run_sampled_baselines_covertype.py` with Run J's task, unchanged.", "",
         "| Method | Deletion margin AUC (lower is more faithful) | Insertion margin AUC (higher) | "
         "Top-3 stability | Model rows per explanation | Milliseconds per explanation, during the run |",
         "|---|---|---|---|---|---|"]
    for m in [m for m in METHODS if m in set(df.method)]:
        g = df[df.method == m]
        L.append(f"| {m} | {_ms(g.margin_auc)} | {_ms(g.insertion_margin_auc)} | {_ms(g.stability_top3, '{:.3f}')} | "
                 f"{_ms(g.rows_per_explanation, '{:,.0f}')} | {_ms(g.ms_per_explanation, '{:,.0f}')} |")
    piv = df.pivot_table(index="seed", columns="method", values="margin_auc")
    if PACE in piv:
        L += ["", "Per seed, PACE-SHAP minus each baseline, deletion margin AUC (negative favours PACE-SHAP; "
              "not tested):", "", "| Seed | " + " | ".join(m for m in (PERM, KERNEL) if m in piv) + " |",
              "|---|" + "---|" * sum(m in piv for m in (PERM, KERNEL))]
        for s, r in piv.iterrows():
            L.append(f"| {int(s)} | " + " | ".join(f"{r[PACE] - r[m]:+.4f}" for m in (PERM, KERNEL) if m in piv) + " |")
    over = df[df.rows_per_explanation > df.budget_rows]
    L += ["", f"Every baseline within the budget: {'yes' if over.empty else 'NO'}.", ""]
    tcsv = out / "timing.csv"
    if tcsv.exists():
        t = pd.read_csv(tcsv)
        tm = t.groupby("method").ms_per_explanation.mean()
        L += [f"## Milliseconds per explanation, measured alone (seeds {sorted(int(s) for s in t.seed.unique())}, "
              f"{int(t.n.iloc[0])} points each)", "", "| Method | Milliseconds | Times PACE-SHAP's |", "|---|---|---|"]
        for m in [m for m in (PACE, PERM, KERNEL) if m in tm]:
            L.append(f"| {m} | {tm[m]:,.0f} | {tm[m] / tm[PACE]:.1f}× |")
        L.append("")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run N: Covertype in the sampled-baseline comparison")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--n", type=int, help="explained points per seed (smoke slices only)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--timing", action="store_true", help="the timing pass, alone")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    if (a.seeds or a.n) and out.resolve() == OUT.resolve():
        raise SystemExit("a smoke slice (other seeds or points) needs --out outside results/sampled_baselines_covertype")
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if a.timing:
        tcsv = out / "timing.csv"
        n = a.n or TIMING_N
        todo = pending(tcsv, [(NAME, s, n) for s in a.seeds or TIMING_SEEDS], a.resume, a.fresh)
        run_all(time_one, todo, tcsv, 1)
    elif not a.summary_only:
        n = a.n or N_EXPLAIN
        todo = pending(csv, [(NAME, s, n) for s in a.seeds or SEEDS], a.resume, a.fresh)
        run_all(run_one, todo, csv, a.jobs)             # module-level, so worker processes can run it
    if csv.exists():
        print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
