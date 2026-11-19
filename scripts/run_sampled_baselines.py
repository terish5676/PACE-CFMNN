"""Run J: modern sampled baselines above 12 features, at an equal evaluation budget.

Rules: results/sampled_baselines/PREREGISTRATION.md.

Above 12 features PACE-SHAP samples feature orders (32 per background row). The second
explanation run (scripts/run_faithfulness.py, protocol v3, seeds 30-39) compared it there with
shap's KernelExplainer on 20 rows. This run compares it with two newer sampled estimators that
get the same rows and the same number of model evaluations:

* PACE-SHAP: ``PaceShap(model, B, n_permutations=32)``, sampled mode, as in v3;
* PermutationSHAP: ``shap.PermutationExplainer`` (``sampled_shap.permutation_shap``);
* KernelSHAP-paired: KernelSHAP with paired sampling, Covert and Lee, AISTATS 2021
  (``sampled_shap.paired_kernel_shap``);
* Random: a random ranking, as the floor.

Same model, split, explained points and background rows as v3 (``run_faithfulness.setup`` and
``backgrounds_v3``): B is the first 200 explainer rows (71 on Wine), and the deletion and
insertion tests copy values from the disjoint evaluation rows.

Budget per explanation, in model evaluations (the margin on one hybrid input): R * P * (d + 1),
what PACE-SHAP's 32 orders per row amount to (each order walks through d + 1 coalitions on its
row). PermutationSHAP gets max_evals = P * (d + 1) masks, each on the R rows; KernelSHAP-paired
gets P * (d + 1) - 2 sampled coalitions plus the empty and the full one, each on the R rows.
The rows each baseline really sent to the model are counted and written to runs.csv.

    python -B scripts/run_sampled_baselines.py --jobs 12
    python -B scripts/run_sampled_baselines.py --jobs 12 --resume     # continue after a stop
    python -B scripts/run_sampled_baselines.py --timing               # alone, after the run (10 points, seeds 160-162)
    python -B scripts/run_sampled_baselines.py --summary-only
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from pace_cfmnn.sampled_shap import paired_kernel_shap, permutation_shap  # noqa: E402
from run_error_detection import holm  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUT = ROOT / "results" / "sampled_baselines"
DATASETS = ["wine", "breast_cancer", "thyroid", "letter", "landsat"]   # RQ1's sets with d > 12
SEEDS = list(range(160, 170))
TIMING_SEEDS = [160, 161, 162]
TIMING_N = 10                       # explained points per seed in the --timing pass
N_EXPLAIN = 50                      # as v3 (Wine has 36 test points)
R_ROWS = rf.SHAP_V3_R               # 200 explainer rows, as v3's PACE-SHAP
P_ORDERS = rf.SHAP_V3_P             # 32 orders per row, as v3's PACE-SHAP
EXACT_MAX_D = 16                    # exact values are computed where d <= 16 (Wine, Letter)
N_EXACT = 10                        # ... for the first 10 explained points of each seed
STABILITY_SHIFT = 1000              # the "new seed" of the stability test, as v3
PACE, PERM, KERNEL, RANDOM = "PACE-SHAP", "PermutationSHAP", "KernelSHAP-paired", "Random"
METHODS = [PACE, PERM, KERNEL, RANDOM]
PRIMARY = [(PACE, PERM), (PACE, KERNEL)]
# (column, title, lower is better)
TESTED = [("margin_auc", "deletion margin AUC", True),
          ("insertion_margin_auc", "insertion margin AUC", False)]
# Slowest first: this only orders the tasks.
COST = {"landsat": 5, "letter": 4, "thyroid": 3, "breast_cancer": 2, "wine": 1}


def budget(R: int, d: int, P: int = P_ORDERS) -> dict:
    """The equal budget for one explanation, and what each method is given to stay inside it."""
    return {"rows": R * P * (d + 1),            # model evaluations per explanation
            "perm_max_evals": P * (d + 1),      # shap's masks, each on R rows
            "kernel_coalitions": P * (d + 1) - 2}   # plus the empty and the full coalition


def explainers(model, Xte, B):
    """name -> fn(seed) returning (attributions, model rows per explanation)."""
    d = Xte.shape[1]
    b = budget(len(B), d)

    def pace(r):
        ex = PaceShap(model, B, n_permutations=P_ORDERS, seed=r)
        if ex.exact:
            raise RuntimeError("Run J needs PACE-SHAP's sampled mode (d > 12)")
        return ex.explain(Xte), float(b["rows"])

    def perm(r):
        return permutation_shap(model, Xte, B, b["perm_max_evals"], seed=r, return_rows=True)

    def kernel(r):
        phi, used = paired_kernel_shap(model, Xte, B, b["kernel_coalitions"], seed=r,
                                       return_rows=True)
        return phi, float(used.mean())

    def random(r):
        return np.random.default_rng(r).uniform(size=Xte.shape), 0.0

    return {PACE: pace, PERM: perm, KERNEL: kernel, RANDOM: random}


def top3(attr):
    return np.argsort(-np.abs(attr), axis=1, kind="stable")[:, :3]


def jaccard(a, b) -> float:
    return float(np.mean([len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(a, b)]))


def prepare(name, seed, n_explain=N_EXPLAIN):
    _, Xtr, Xte, model, _ = rf.setup(name, seed, n_explain)
    bg, bex = rf.backgrounds_v3(Xtr, seed)
    return model, Xte, bg, bex[:R_ROWS]


def run_one(name: str, seed: int, n_explain: int = N_EXPLAIN):
    warnings.filterwarnings("ignore")
    model, Xte, bg, B = prepare(name, seed, n_explain)
    n, d = Xte.shape
    if d <= 12:
        raise ValueError(f"{name} has {d} features; Run J is for more than 12")
    b = budget(len(B), d)
    ref = None
    exact_ms = np.nan
    if d <= EXACT_MAX_D:
        t0 = time.perf_counter()
        ref = PaceShap(model, B, exact_max_d=EXACT_MAX_D).explain(Xte[:N_EXACT])
        exact_ms = 1000 * (time.perf_counter() - t0) / len(ref)
    rows = []
    for m, fn in explainers(model, Xte, B).items():
        t0 = time.perf_counter()
        attr, used = fn(seed)
        ms = 1000 * (time.perf_counter() - t0) / n
        stab = jaccard(top3(attr), top3(fn(seed + STABILITY_SHIFT)[0]))
        _, mauc = rf.deletion_auc(model, Xte, attr, bg, seed=seed, signed=True)
        _, imauc = rf.insertion_auc(model, Xte, attr, bg, seed=seed)
        rec = {"dataset": name, "seed": seed, "method": m, "d": d, "n": n, "boxes": model.n_boxes_,
               "background_rows": len(B), "budget_rows": b["rows"], "rows_per_explanation": used,
               "margin_auc": float(mauc.mean()), "insertion_margin_auc": float(imauc.mean()),
               "stability_top3": stab, "ms_per_explanation": ms, "n_exact": 0,
               "rel_l1_error": np.nan, "max_abs_error": np.nan, "top3_overlap_exact": np.nan,
               "exact_ms_per_explanation": exact_ms}
        if ref is not None:
            got = attr[:len(ref)]
            rec.update({"n_exact": len(ref),
                        "rel_l1_error": float(np.mean(np.abs(got - ref).sum(axis=1)
                                                      / np.maximum(np.abs(ref).sum(axis=1), 1e-12))),
                        "max_abs_error": float(np.abs(got - ref).max()),
                        "top3_overlap_exact": jaccard(top3(got), top3(ref))})
        if used > b["rows"]:
            raise RuntimeError(f"{name} seed {seed} {m}: {used:.0f} model rows, over the budget "
                               f"of {b['rows']}")
        rows.append(rec)
        print(f"{name} seed {seed} {m}: deletion {rec['margin_auc']:.3f}, insertion "
              f"{rec['insertion_margin_auc']:.3f}, stability {stab:.2f}, {ms:.0f} ms", flush=True)
    return rows


def time_one(name: str, seed: int, n_explain: int = N_EXPLAIN):
    """Milliseconds per explanation only: one pass per method, nothing else running."""
    warnings.filterwarnings("ignore")
    model, Xte, _, B = prepare(name, seed, n_explain)
    out = []
    for m, fn in explainers(model, Xte, B).items():
        if m == RANDOM:
            continue
        t0 = time.perf_counter()
        fn(seed)
        out.append({"dataset": name, "seed": seed, "method": m, "n": len(Xte),
                    "ms_per_explanation": 1000 * (time.perf_counter() - t0) / len(Xte)})
    print(f"{name} seed {seed} timed: " + ", ".join(
        f"{r['method']} {r['ms_per_explanation']:.0f} ms" for r in out), flush=True)
    return out


# ------------------------------------------------------------------ summary
def compare(df: pd.DataFrame, a: str, b: str, col: str, lower: bool) -> pd.DataFrame:
    """a vs b on ``col``: paired two-sided Wilcoxon over seeds, Holm across the datasets.

    ``diff`` is a - b. The verdict is for a: better, level or worse (``lower`` says which sign
    is better). With every difference zero, p = 1.
    """
    out = []
    for ds in [d for d in DATASETS if d in set(df.dataset)]:
        g = df[df.dataset == ds].pivot_table(index="seed", columns="method", values=col)
        if a not in g or b not in g:
            continue
        x = (g[a] - g[b]).dropna().to_numpy()
        p = wilcoxon(x).pvalue if len(x) > 1 and np.any(x != 0) else 1.0
        good = int(np.sum(x < 0) if lower else np.sum(x > 0))
        out.append({"dataset": ds, "diff": float(x.mean()), "p": float(p), "n": len(x),
                    "seeds_in_favour": good})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = holm(res.p.to_numpy())
    favours = (res["diff"] < 0) if lower else (res["diff"] > 0)
    res["verdict"] = np.where(res.p_holm >= 0.05, "level", np.where(favours, "better", "worse"))
    return res


def tally(res: pd.DataFrame) -> str:
    c = res.verdict.value_counts()
    return f"better {c.get('better', 0)}, level {c.get('level', 0)}, worse {c.get('worse', 0)}"


def _table(L, df, col, fmt, ds, methods):
    piv = df.pivot_table(index="dataset", columns="method", values=col).reindex(index=ds, columns=methods)
    L += ["| Dataset | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    L += [f"| {d} | " + " | ".join("" if np.isnan(v) else fmt.format(v) for v in r) + " |"
          for d, r in piv.iterrows()]
    L.append("")


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    methods = [m for m in METHODS if m in set(df.method)]
    L = ["# Run J: sampled baselines above 12 features, at an equal evaluation budget", "",
         f"Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's model, split, points and rows; seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}). Each comparison: paired two-sided Wilcoxon over seeds, "
         "Holm across the datasets; better / level / worse for PACE-SHAP. Written by "
         "`scripts/run_sampled_baselines.py`.", "",
         "## The budget", "",
         "| Dataset | Features | Background rows | Points per seed | Boxes (mean) | Budget: model rows per "
         "explanation | Rows used: " + " | Rows used: ".join(m for m in methods if m != RANDOM) + " |",
         "|---|---|---|---|---|---|" + "---|" * (len(methods) - (RANDOM in methods))]
    for d in ds:
        g = df[df.dataset == d]
        used = g.groupby("method").rows_per_explanation.mean()
        L.append(f"| {d} | {int(g.d.iloc[0])} | {int(g.background_rows.iloc[0])} | {int(g.n.iloc[0])} | "
                 f"{g.boxes.mean():.0f} | {int(g.budget_rows.iloc[0]):,} | "
                 + " | ".join(f"{used[m]:,.0f}" for m in methods if m != RANDOM) + " |")
    L += ["", "PACE-SHAP's rows are the coalition values its orders walk through (it reads them from "
          "the model's terms and does not call the model); the baselines' rows were counted as they "
          "called the model.", ""]
    for col, title, lower in TESTED:
        head = "Primary" if col == "margin_auc" else "Secondary"
        L += [f"## {head}: {title} ({'lower' if lower else 'higher'} is more faithful)", ""]
        for a, b in PRIMARY:
            r = compare(df, a, b, col, lower)
            if r.empty:
                continue
            L += [f"**{a} vs {b}**: {tally(r)}", "",
                  f"| Dataset | Mean difference ({a} − {b}) | Seeds in favour of {a} | p | Holm p | "
                  "Verdict |", "|---|---|---|---|---|---|"]
            L += [f"| {x.dataset} | {x['diff']:+.4f} | {x.seeds_in_favour} of {x.n} | {x.p:.4f} | "
                  f"{x.p_holm:.4f} | {x.verdict} |" for _, x in r.iterrows()]
            L.append("")
        L += [f"Mean {title} over seeds:", ""]
        _table(L, df, col, "{:.4f}", ds, methods)
    ex = df[df.n_exact > 0]
    if len(ex):
        eds = [d for d in ds if d in set(ex.dataset)]
        em = [m for m in methods if m != RANDOM]
        L += [f"## Error against exact values (d ≤ {EXACT_MAX_D}; the first {N_EXACT} points of each "
              "seed)", "", "Relative L1 error: the sum over features of |estimate − exact|, divided by "
              "the sum of |exact|, averaged over points; mean over seeds.", ""]
        _table(L, ex, "rel_l1_error", "{:.4f}", eds, em)
        L += ["Largest absolute error (mean over seeds):", ""]
        _table(L, ex, "max_abs_error", "{:.5f}", eds, em)
        L += ["Top-3 overlap with the exact values (Jaccard, by absolute value):", ""]
        _table(L, ex, "top3_overlap_exact", "{:.3f}", eds, em)
    L += ["## Top-3 stability under a new seed (Jaccard overlap, mean over seeds)", ""]
    _table(L, df, "stability_top3", "{:.3f}", ds, methods)
    L += ["## Milliseconds per explanation during the run", "",
          "Several tasks shared the CPU, so compare within a dataset only. The times measured alone "
          "are below, if `--timing` has been run.", ""]
    _table(L, df, "ms_per_explanation", "{:,.0f}", ds, [m for m in methods if m != RANDOM])
    tp = out / "timing.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        tm = [m for m in methods if m in set(t.method)]
        L += [f"## Milliseconds per explanation, measured alone (seeds "
              f"{sorted(int(s) for s in t.seed.unique())})", ""]
        _table(L, t, "ms_per_explanation", "{:,.0f}", [d for d in ds if d in set(t.dataset)], tm)
        piv = t.pivot_table(index="dataset", columns="method", values="ms_per_explanation")
        if PACE in piv:
            L += ["Times PACE-SHAP's:", "", "| Dataset | " + " | ".join(m for m in tm if m != PACE) + " |",
                  "|---|" + "---|" * (len(tm) - 1)]
            L += [f"| {d} | " + " | ".join(f"{piv.loc[d, m] / piv.loc[d, PACE]:.1f}×" for m in tm if m != PACE)
                  + " |" for d in ds if d in piv.index]
            L.append("")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run J: sampled baselines at an equal budget")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--n", type=int, help=f"explained points per seed (default {N_EXPLAIN}; "
                    f"{TIMING_N} with --timing)")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--timing", action="store_true", help="time the explainers only (run alone)")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if a.timing:
        if a.jobs != 1:
            raise SystemExit("--timing must run alone (--jobs 1), or the times are distorted")
        rows = [r for d in a.datasets or DATASETS for s in a.seeds or TIMING_SEEDS
                for r in time_one(d, s, a.n or TIMING_N)]
        pd.DataFrame(rows).to_csv(out / "timing.csv", index=False)
    elif not a.summary_only:
        tasks = [(d, s, a.n or N_EXPLAIN) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs, COST)
    if csv.exists():
        print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
