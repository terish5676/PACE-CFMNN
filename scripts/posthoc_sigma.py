"""Post hoc: how B3's conditional game depends on sigma (not pre-registered).

RQ1 extension B3 fixed sigma = 0.1 in advance for the conditional Shapley
game, and its results stand as run. At that sigma, once a coalition holds
half the features, the Gaussian row weights rest on about one explainer row,
and the conditional explainers were less faithful than the interventional
ones, even on the conditional test. This check asks whether the few rows
explain that. It recomputes PACE-SHAP-cond for several sigma, on B3's own
models, test points, explainer rows and tests, and reports for each sigma:

* the effective number of rows (Kish: (sum w)^2 / sum w^2), median over
  points and over coalitions with half the features or more;
* how often the game's inputs are off the data (B3's realism measure);
* faithfulness on B3's conditional and interventional deletion and
  insertion tests;
* ground truth (top 3, on S1, S1c and S2c) and x3's share of |attribution|
  (S1c and S2c);
* how far the attributions are from PACE-SHAP's.

sigma = inf is plain PACE-SHAP. Only datasets where PACE-SHAP is exact
(d <= 12), so every sigma comes from one shared table of margins
(``PaceShap.explain_games``). sigma = 0.1 and inf repeat B3's computations;
the summary checks that they give B3's numbers.

Usage (from the project folder):
    python scripts/posthoc_sigma.py --jobs 5
    python scripts/posthoc_sigma.py --summary-only
Writes results/extensions_v1/conditional/posthoc_sigma.csv and .md.
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

from pace_cfmnn import realism  # noqa: E402
from pace_cfmnn.explain import PaceShap, cond_scale  # noqa: E402
from run_faithfulness import (REALISM_K, SHAP_V3_P, SHAP_V3_R, SYNTHETIC, backgrounds_v3,  # noqa: E402
                              cond_curves, deletion_auc, insertion_auc, setup, top3_random_ties)

B3 = ROOT / "results" / "extensions_v1" / "conditional"
DATASETS = ["S1", "S1c", "S2c", "shuttle", "magic", "page_blocks"]   # exact (d <= 12)
SEEDS = [40, 41, 42, 43, 44]
SIGMAS = (0.1, 0.2, 0.3, 0.5, 1.0, None)       # None: the interventional game (PACE-SHAP)
N_EXPLAIN = 50                                  # B3's test points per dataset and seed
NAMES = {"S1": "S1", "S1c": "S1c", "S2c": "S2c", "shuttle": "Shuttle", "magic": "MAGIC",
         "page_blocks": "Page Blocks"}


def label(sigma) -> str:
    return "inf" if sigma is None else f"{sigma:g}"


def effective_rows(X, B, scale, sigma):
    """Median Kish effective rows of the conditional weights, over points and coalitions.

    Same weights as ``PaceShap(sigma=...)``: exp(-D_S^2 / (2 sigma^2)), D_S^2
    the mean squared standardised difference on the features in S. Returns
    the median over coalitions with half the features or more (but not all),
    and over the smaller non-empty ones."""
    d = X.shape[1]
    masks = np.arange(1 << d)
    ind = ((masks[None, :] >> np.arange(d)[:, None]) & 1).astype(float)   # (d, 2^d)
    size = ind.sum(axis=0)
    large = (size >= d / 2) & (size < d)
    small = (size >= 1) & (size < d / 2)
    kl, ks = [], []
    for x in X:
        logw = -((((x - B) / scale) ** 2) @ ind) / (2.0 * sigma ** 2 * np.maximum(size, 1))
        w = np.exp(logw - logw.max(axis=0))
        w /= w.sum(axis=0)
        k = 1.0 / (w ** 2).sum(axis=0)
        kl.append(k[large])
        ks.append(k[small])
    return float(np.median(np.concatenate(kl))), float(np.median(np.concatenate(ks)))


def game_off_share(man, Xte, B, scale, seed, sigma):
    """Off-data share of a Shapley game's inputs, as B3 measured it.

    Replays ``run_faithfulness.realism_rows``'s random stream, whose first
    scheme is the interventional game and second the conditional one, so
    every sigma gets the same coalitions and draws, and sigma = 0.1 and None
    give B3's own numbers."""
    rng = np.random.default_rng(seed + 900)
    Z = realism.shapley_inputs(Xte, B, REALISM_K, rng)
    if sigma is not None:
        Z = realism.shapley_inputs(Xte, B, REALISM_K, rng, sigma=sigma, scale=scale)
    return man.share(Z)


def run_job(name, seed, n_explain=N_EXPLAIN, sigmas=SIGMAS):
    t0 = time.perf_counter()
    ds, Xtr, Xte, model, _ = setup(name, seed, n_explain)
    bg, bex = backgrounds_v3(Xtr, seed)
    B = bex[:SHAP_V3_R]
    scale = cond_scale(B)
    relevant = set(ds.info.get("relevant", [])) if name in SYNTHETIC else None
    ps = PaceShap(model, B, n_permutations=SHAP_V3_P, seed=seed)
    if not ps.exact:
        raise ValueError(f"{name}: PACE-SHAP is not exact here (d > 12)")
    attrs = ps.explain_games(Xte, sigmas)
    phi_int = attrs[sigmas.index(None)]
    ref = np.abs(phi_int).sum(axis=1)
    man = realism.Manifold(Xtr, seed=seed)
    rows = []
    for s, attr in zip(sigmas, attrs):
        _, del_int = deletion_auc(model, Xte, attr, bg, seed=seed, signed=True)
        _, ins_int = insertion_auc(model, Xte, attr, bg, seed=seed)
        _, del_cond = cond_curves(model, Xte, attr, bg, seed=seed)
        _, ins_cond = cond_curves(model, Xte, attr, bg, seed=seed, insertion=True)
        t3 = top3_random_ties(attr, np.random.default_rng(seed + 7))
        a = np.abs(attr)
        tot = a.sum(axis=1)
        share3 = np.divide(a[:, 3], tot, out=np.zeros(len(a)), where=tot > 0)
        dist = np.divide(np.abs(attr - phi_int).sum(axis=1), ref, out=np.zeros(len(a)), where=ref > 0)
        k_large, k_small = ((float(len(B)),) * 2 if s is None else effective_rows(Xte, B, scale, s))
        rows.append({"dataset": name, "seed": seed, "sigma": label(s),
                     "eff_rows_large": k_large, "eff_rows_small": k_small,
                     "off_share": game_off_share(man, Xte, B, scale, seed, s),
                     "margin_auc": del_int.mean(), "insertion_margin_auc": ins_int.mean(),
                     "cond_margin_auc": del_cond.mean(),
                     "cond_insertion_margin_auc": ins_cond.mean(),
                     "gt_top3_precision": (np.mean([len(set(t) & relevant) / 3 for t in t3])
                                           if relevant else np.nan),
                     "x3_share": float(share3.mean()) if name in ("S1c", "S2c") else np.nan,
                     "dist_to_pace_shap": float(dist.mean()), "n": len(Xte)})
    print(f"{name} seed {seed} done in {time.perf_counter() - t0:.0f} s; conditional deletion "
          + ", ".join(f"{r['sigma']}: {r['cond_margin_auc']:.3f}" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------- summary
TESTS = [("cond_margin_auc", True, "Conditional test, deletion (lower = more faithful)"),
         ("cond_insertion_margin_auc", False, "Conditional test, insertion (higher = more faithful)"),
         ("margin_auc", True, "Interventional test, deletion (lower = more faithful)"),
         ("insertion_margin_auc", False, "Interventional test, insertion (higher = more faithful)")]


def reproduction(df, b3, real) -> list[str]:
    """Largest absolute difference from B3 at sigma = 0.1 and inf (should be about 0)."""
    cols = [m for m, _, _ in TESTS] + ["gt_top3_precision", "x3_share"]
    L = ["## Check: sigma = 0.1 and inf repeat B3", "",
         "Largest absolute difference from B3's own results, over all datasets and seeds here.", "",
         "| sigma | B3 method | Faithfulness, ground truth, x3 share | Inputs off the data |",
         "|---|---|---|---|"]
    for s, method, scheme in (("0.1", "PACE-SHAP-cond", "shap_cond"), ("inf", "PACE-SHAP", "shap_int")):
        mine = df[df.sigma == s].set_index(["dataset", "seed"])
        if mine.empty:
            continue
        ref = b3[b3.method == method].set_index(["dataset", "seed"]).reindex(mine.index)
        rr = real[real.method == scheme].set_index(["dataset", "seed"]).reindex(mine.index)
        da = np.nanmax((mine[cols] - ref[cols]).abs().to_numpy())
        dr = np.nanmax((mine.off_share - rr.off_share).abs().to_numpy())
        L.append(f"| {s} | {method} | {da:.1e} | {dr:.1e} |")
    return L


def summarise(df: pd.DataFrame, out: Path) -> str:
    b3 = pd.read_csv(B3 / "runs.csv")
    real = pd.read_csv(B3 / "realism.csv")
    extra = pd.read_csv(B3 / "extra.csv")
    sig = [label(s) for s in SIGMAS if label(s) in set(df.sigma)]
    conds = [s for s in sig if s != "inf"]
    dsets = [d for d in DATASETS if d in set(df.dataset)] + sorted(set(df.dataset) - set(DATASETS))
    seeds = sorted(int(s) for s in df.seed.unique())
    g = df.groupby(["dataset", "sigma"])
    head = "| Dataset | " + " | ".join(f"σ = {s}" for s in sig) + " |"
    rule = "|---|" + "---|" * len(sig)
    L = ["# Post hoc: B3's conditional game at other values of sigma", "",
         "**Post hoc, not pre-registered.** B3 fixed σ = 0.1 in advance and its results stand as run. "
         "This check asks whether B3's conditional explainers did worse because, at σ = 0.1, the "
         "conditional game rests on about one explainer row for large coalitions.", "",
         f"Datasets {', '.join(NAMES.get(d, d) for d in dsets)} (those where PACE-SHAP is exact, "
         f"d ≤ 12); seeds {seeds}; B3's models, {int(df.n.max())} test points, {SHAP_V3_R} explainer "
         "rows, evaluation rows and tests, recomputed with the same code. Only PACE-SHAP-cond, at "
         f"σ = {', '.join(conds)}; σ = inf is plain PACE-SHAP (all rows weighted equally). Every σ "
         "comes from one shared table of margins, and the realism inputs use the same coalitions "
         "and random draws for every σ.", "",
         f"Descriptive only: with {len(seeds)} seed{'s' * (len(seeds) > 1)} a two-sided Wilcoxon test cannot reach "
         "p < 0.05, so no tests are run. Cells give the mean over seeds and, for σ < inf, in "
         "brackets the number of seeds where PACE-SHAP-cond is more faithful than PACE-SHAP.", ""]
    L += reproduction(df, b3, real)
    sec = iter(range(1, 20))

    L += ["", f"## {next(sec)}. Effective explainer rows", "",
          "Kish effective rows (Σw)² / Σw² of the row weights, median over points and over coalitions "
          "with half the features or more (not all); then the median over seeds. 1 means one row "
          f"carries all the weight; σ = inf weights all {SHAP_V3_R} rows equally.", "", head, rule]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{g.get_group((d, s)).eff_rows_large.median():.1f}" if s != "inf" else f"{SHAP_V3_R}"
            for s in sig) + " |")
    L += ["", "Smaller coalitions (1 feature up to just under half), same summary:", "", head, rule]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{g.get_group((d, s)).eff_rows_small.median():.1f}" if s != "inf" else f"{SHAP_V3_R}"
            for s in sig) + " |")

    pts = real[(real.method == "points") & real.seed.isin(seeds)].groupby("dataset").off_share.mean()
    L += ["", f"## {next(sec)}. Inputs off the data (%)", "",
          "Share of the game's inputs whose nearest training row is farther than the 95th percentile "
          "of the training rows' own nearest-neighbour distances (B3's measure, 200 inputs per point). "
          "Last column: the explained points themselves.", "",
          head[:-1] + "| Points |", rule + "---|"]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{100 * g.get_group((d, s)).off_share.mean():.1f}" for s in sig)
            + f" | {100 * pts.get(d, np.nan):.1f} |")

    for m, lower, title in TESTS:
        L += ["", f"## {next(sec)}. {title}", "", head, rule]
        for d in dsets:
            base = df[(df.dataset == d) & (df.sigma == "inf")].set_index("seed")[m]
            cells = []
            for s in sig:
                v = df[(df.dataset == d) & (df.sigma == s)].set_index("seed")[m]
                if s == "inf":
                    cells.append(f"{v.mean():.3f}")
                    continue
                diff = v - base.reindex(v.index)
                wins = int((diff < 0).sum() if lower else (diff > 0).sum())
                cells.append(f"{v.mean():.3f} ({wins}/{len(v)})")
            L.append(f"| {NAMES.get(d, d)} | " + " | ".join(cells) + " |")

    syn = [d for d in dsets if d in SYNTHETIC]
    if syn:
        L += ["", f"## {next(sec)}. Ground truth: share of the true features x0–x2 in the top 3", "",
              head, rule]
        for d in syn:
            L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
                f"{g.get_group((d, s)).gt_top3_precision.mean():.2f}" for s in sig) + " |")
    xs = [d for d in dsets if d in ("S1c", "S2c")]
    if xs:
        fm = extra[extra.seed.isin(seeds)].groupby("dataset").flip_mass_x3.mean()
        L += ["", f"## {next(sec)}. x3's share of |attribution| (x3 is a noisy copy of x0 the true rule "
              "does not use)",
              "", "Last column: the model's own use of x3 (its share of PACE-flip's flip mass, from B3).",
              "", head[:-1] + "| Model's use |", rule + "---|"]
        for d in xs:
            L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
                f"{g.get_group((d, s)).x3_share.mean():.3f}" for s in sig)
                + f" | {fm.get(d, np.nan):.3f} |")

    L += ["", f"## {next(sec)}. Distance from PACE-SHAP", "",
          "Per point Σ|φ_σ − φ_PACE-SHAP| / Σ|φ_PACE-SHAP|, mean over points and seeds.", "",
          "| Dataset | " + " | ".join(f"σ = {s}" for s in conds) + " |", "|---|" + "---|" * len(conds)]
    for d in dsets:
        L.append(f"| {NAMES.get(d, d)} | " + " | ".join(
            f"{g.get_group((d, s)).dist_to_pace_shap.mean():.2f}" for s in conds) + " |")
    text = "\n".join(L) + "\n"
    (out / "posthoc_sigma.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n", type=int, default=N_EXPLAIN, help="test points (B3: 50; other values "
                    "are for quick trials and do not repeat B3)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(B3))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "posthoc_sigma.csv"
    if not a.summary_only:
        tasks = [(d, s) for d in a.datasets for s in a.seeds]
        done = {}

        def save():
            pd.DataFrame([r for t in tasks if t in done for r in done[t]]).to_csv(csv, index=False)

        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                fut = {ex.submit(run_job, d, s, a.n): (d, s) for d, s in tasks}
                for f in as_completed(fut):
                    done[fut[f]] = f.result()
                    save()
        else:
            for t in tasks:
                done[t] = run_job(*t, a.n)
                save()
    print(summarise(pd.read_csv(csv, dtype={"sigma": str}), out))


if __name__ == "__main__":
    main()
