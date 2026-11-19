"""RQ2: do the trust signals rank CFMNN's errors first? (blueprint 6, 9, 10)

For each dataset and seed:

1. Stratified 50/20/10/10/10 role split (``data.role_split``). The main model
   (fixed centroid + search expansion) is trained on the training rows.
2. The perturbation policy (``Perturber``) is fitted on the training rows.
3. PD, EI and U are computed on the inner-validation rows; their percentile
   ranks there define PRS. Nothing else is fitted.
4. Error detection is evaluated on all remaining rows (pool + acceptance +
   test). Signals never read labels; labels are used only to mark errors.

Scaled rows are clipped to [0, 1], the range the perturbations live in.
A dataset counts for RQ2 only if every seed's evaluation set holds at least
50 errors (blueprint 9.2). ``summary.md`` is rebuilt from ``runs.csv`` alone.

Usage (from the project folder):
    python scripts/run_error_detection.py
    python scripts/run_error_detection.py --datasets S1 S2 --seeds 0 1 --out results/tmp
    python scripts/run_error_detection.py --kappa 0.1 --out results/error_detection/kappa_0.1
    python scripts/run_error_detection.py --summary-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # noqa: E402

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn import trust_baselines as tb  # noqa: E402
from pace_cfmnn.evaluate import error_detection_metrics  # noqa: E402
from pace_cfmnn.neighborhood import Perturber  # noqa: E402
from pace_cfmnn.pace_score import PRS, VARIANT  # noqa: E402
from pace_cfmnn.signals import SignalComputer  # noqa: E402

DATASETS = ["S1", "S2", "S3", "S2-imbalanced", "S2-label-noise", "S2-shifted", "wine",
            "breast_cancer", "digits", "thyroid", "shuttle", "skin"]
# Synthetic variants (blueprint 9.2, S2 row): (base set, keyword arguments).
VARIANTS = {"S2-imbalanced": ("S2", {"minority": 0.1}),
            "S2-label-noise": ("S2", {"label_noise": 0.05})}
SHIFT = 1.0          # S2-shifted: the evaluation rows come from S2 with feature 0 skewed towards 1
MIN_ERRORS = 50
# Blueprint methods first; the starred ones use the corrected U* (a variant added after gate G1).
METHODS = ["PRS", "PD + U", "PD", "EI", "U", "PRS*", "PD + U*", "U*", "Margin", "Entropy", "Random"]

# Confirmatory run (pre-registered in results/error_detection_confirmatory/PREREGISTRATION.md).
CONF_DATASETS = DATASETS + ["magic", "letter", "landsat", "page_blocks"]
CONF_SEEDS = list(range(10, 20))
BASELINES = ["Trust Score", "Order ensemble", "kNN disagreement"]
SMALL = 2_000        # datasets with fewer rows use stratified 5-fold cross-validation
MAX_EVAL = 20_000    # at most this many evaluation rows per seed (fixed random subset)
SYNTHETIC = ["S1", "S2", "S3", "S2-imbalanced", "S2-label-noise", "S2-shifted"]


def load(name: str, seed: int):
    if name in VARIANTS:
        base, kw = VARIANTS[name]
        return data.load(base, seed=seed, **kw)
    if name == "S2-shifted":
        return data.load("S2", seed=seed)
    return data.load(name, seed=seed)


def plausibility(Xtr, X, Z, rng, n=200, per=10) -> float:
    """Share of perturbations at least as close to the training data as 95% of real points.

    Distances are L1 to the nearest training row. A share near 1 means the
    perturbations sit where real data sits. (A ratio of medians fails on data
    with many duplicate rows, where the real points' median distance is 0.)
    """
    pick = rng.choice(len(X), min(n, len(X)), replace=False)
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(Xtr)
    d_x = nn.kneighbors(X[pick])[0].ravel()
    d_z = nn.kneighbors(Z[pick, :per].reshape(-1, X.shape[1]))[0].ravel()
    cut = np.quantile(d_x, 0.95)
    if cut == 0:     # 95% of real points duplicate a training row: the measure is undefined
        return np.nan
    return float((d_z <= cut).mean())


def refresh_plausibility(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute only the plausibility column of existing runs (no model needed)."""
    for (name, seed), idx in df.groupby(["dataset", "seed"]).groups.items():
        ds = load(name, seed)
        roles = data.role_split(ds.y, seed)
        order = np.random.default_rng(seed).permutation(roles["train"])
        scaler = data.fit_scaler(ds.X[order])
        tf = lambda X: np.clip(scaler.transform(X), 0.0, 1.0)
        Xtr = tf(ds.X[order])
        if name == "S2-shifted":
            Xev = tf(data.make_synthetic("S2", seed=seed + 10_000, shift=SHIFT).X)
        else:
            Xev = tf(ds.X[np.concatenate([roles["pool"], roles["accept"], roles["test"]])])
        r = df.loc[idx[0]]
        pert = Perturber(kappa=r.kappa, k=int(r.k)).fit(Xtr)
        Zs = pert.sample(Xev[:2000], np.random.default_rng(seed + 1))
        rng = np.random.default_rng(seed)
        rng.uniform(size=int(r.n_eval))          # same draws as run_one before plausibility
        df.loc[idx, "plausibility"] = plausibility(Xtr, Xev[:2000], Zs, rng)
    return df.drop(columns=["plausibility_ratio"], errors="ignore")


def evaluate_split(Xtr, ytr, Xin, Xev, yev, seed: int, kappa: float, k: int,
                   baselines: bool = False) -> dict:
    """Train on (Xtr, ytr), fit PRS ranks on Xin, score Xev. Returns per-point arrays and scalars."""
    model = CFMNN().fit(Xtr, ytr)
    pert = Perturber(kappa=kappa, k=k).fit(Xtr)
    sc = SignalComputer(model, pert)
    t1 = time.perf_counter()
    inner = sc.compute(Xin, seed=10_000 + seed)
    sig = sc.compute(Xev, seed=seed)
    t_signals = time.perf_counter() - t1

    prs = PRS().fit(inner)
    s_prs, fallback = prs.score(sig)
    s_pdu, _ = prs.score(sig, use=("pd", "u"))
    s_prs2, _ = prs.score(sig, use=VARIANT)
    s_pdu2, _ = prs.score(sig, use=("pd", "u_star"))
    pred = model.predict(Xev)
    err = pred != yev
    rng = np.random.default_rng(seed)
    scores = {"PRS": s_prs, "PD + U": s_pdu, "PD": sig.pd, "U": sig.u,
              "PRS*": s_prs2, "PD + U*": s_pdu2, "U*": sig.u_star,
              "Margin": sig.margin_unc, "Entropy": sig.entropy,
              "Random": rng.uniform(size=len(err))}
    t_base = 0.0
    if baselines:
        t2 = time.perf_counter()
        scores["Trust Score"] = tb.trust_score(Xtr, ytr, Xev, pred)
        scores["Order ensemble"] = tb.order_ensemble_disagreement(model, Xtr, ytr, Xev, pred,
                                                                  n_models=5, seed=seed + 3)
        scores["kNN disagreement"] = tb.knn_disagreement(Xtr, ytr, Xev, pred)
        t_base = time.perf_counter() - t2
    Zs = pert.sample(Xev[:2000], np.random.default_rng(seed + 1))
    return {"scores": scores, "err": err, "ei": sig.ei, "tie": sig.tie, "pd": sig.pd,
            "u": sig.u, "entropy": sig.entropy, "mode_switch": sig.mode_switch,
            "box_switch": sig.box_switch, "fallback": fallback,
            "n_train": len(ytr), "n_inner": len(Xin), "n_boxes": model.n_boxes_,
            "clip_share": pert.clip_share(Xev[:2000], Zs),
            "plausibility": plausibility(Xtr, Xev[:2000], Zs, rng),
            "signals_s": t_signals, "baselines_s": t_base}


def pool(parts: list[dict]) -> dict:
    """Join per-fold results: per-point arrays are concatenated, scalars averaged."""
    out = {"scores": {m: np.concatenate([p["scores"][m] for p in parts]) for m in parts[0]["scores"]}}
    for key, v in parts[0].items():
        if key == "scores":
            continue
        if isinstance(v, np.ndarray):
            out[key] = np.concatenate([p[key] for p in parts])
        else:
            out[key] = float(np.nanmean([p[key] for p in parts]))
    return out


def run_one(name: str, seed: int, kappa: float, k: int, protocol: str = "exploratory") -> list[dict]:
    """One dataset and seed.

    Exploratory (seeds 0-9): role split for every dataset, all evaluation rows.
    Confirmatory (pre-registered, seeds 10-19): datasets under 2,000 rows use
    stratified 5-fold cross-validation with pooled out-of-fold scores (blueprint
    9.1); larger ones use the role split with at most 20,000 evaluation rows;
    the three standard baselines are added.
    """
    t0 = time.perf_counter()
    conf = protocol == "confirmatory"
    ds = load(name, seed)
    if conf and len(ds.y) < SMALL:
        parts = []
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for f, (tr_all, ev) in enumerate(outer.split(ds.X, ds.y)):
            inner_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + f)
            tr_i, in_i = next(inner_split.split(tr_all, ds.y[tr_all]))
            tr = np.random.default_rng(seed + f).permutation(tr_all[tr_i])
            scaler = data.fit_scaler(ds.X[tr])
            tf = lambda X: np.clip(scaler.transform(X), 0.0, 1.0)
            parts.append(evaluate_split(tf(ds.X[tr]), ds.y[tr], tf(ds.X[tr_all[in_i]]),
                                        tf(ds.X[ev]), ds.y[ev], seed * 10 + f, kappa, k,
                                        baselines=True))
        r = pool(parts)
        split_kind = "5-fold CV"
    else:
        roles = data.role_split(ds.y, seed)
        order = np.random.default_rng(seed).permutation(roles["train"])
        scaler = data.fit_scaler(ds.X[order])
        tf = lambda X: np.clip(scaler.transform(X), 0.0, 1.0)
        Xtr, ytr = tf(ds.X[order]), ds.y[order]
        Xin = tf(ds.X[roles["inner"]])
        if name == "S2-shifted":
            shifted = data.make_synthetic("S2", seed=seed + 10_000, shift=SHIFT)
            Xev, yev = tf(shifted.X), shifted.y
        else:
            ev = np.concatenate([roles["pool"], roles["accept"], roles["test"]])
            if conf and len(ev) > MAX_EVAL:
                ev = np.sort(np.random.default_rng(seed + 2).choice(ev, MAX_EVAL, replace=False))
            Xev, yev = tf(ds.X[ev]), ds.y[ev]
        r = evaluate_split(Xtr, ytr, Xin, Xev, yev, seed, kappa, k, baselines=conf)
        split_kind = "role split"

    err, has_ei, ei = r["err"], np.isfinite(r["ei"]), r["ei"]
    rho = lambda a, b, m=slice(None): float(stats.spearmanr(a[m], b[m])[0])
    base = {
        "dataset": name, "seed": seed, "kappa": kappa, "k": k,
        "n_train": r["n_train"], "n_inner": r["n_inner"], "n_eval": len(err),
        "n_features": ds.X.shape[1], "n_boxes": r["n_boxes"],
        "accuracy": 100 * float(1 - err.mean()), "n_errors": int(err.sum()),
        "ei_coverage": float(has_ei.mean()), "fallback_share": float(r["fallback"].mean()),
        "tie_share": float(r["tie"].mean()),
        "tie_error_rate": float(err[r["tie"]].mean()) if r["tie"].any() else np.nan,
        "mode_switch": float(r["mode_switch"].mean()), "box_switch": float(r["box_switch"].mean()),
        "clip_share": r["clip_share"], "plausibility": r["plausibility"],
        "u_std": float(r["u"].std()), "entropy_std": float(r["entropy"].std()),
        "rho_pd_u": rho(r["pd"], r["u"]), "rho_pd_ei": rho(r["pd"], ei, has_ei),
        "rho_ei_u": rho(ei, r["u"], has_ei),
        "signals_s": r["signals_s"], "total_s": time.perf_counter() - t0,
    }
    if conf:
        base.update(protocol=protocol, split=split_kind, baselines_s=r["baselines_s"])
    rows = []
    for m in (METHODS + BASELINES if conf else METHODS):
        if m == "EI":   # EI alone is scored only where it exists; missing EI is never 0
            met = error_detection_metrics(err[has_ei], ei[has_ei])
        else:
            met = error_detection_metrics(err, r["scores"][m])
        rows.append({**base, "method": m, **met})
    return rows


# --------------------------------------------------------------------- summary
def ci95(v) -> float:
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def fmt(m, c, digits=3) -> str:
    return f"{m:.{digits}f} ± {c:.{digits}f}"


def paired(df, a: str, b: str, metric="auprc"):
    """Per-dataset paired comparison of method a vs b over seeds."""
    rows = []
    for ds, g in df.groupby("dataset", sort=False):
        x = g[g.method == a].set_index("seed")[metric]
        y = g[g.method == b].set_index("seed")[metric]
        diff = (x - y.reindex(x.index)).dropna()
        p = stats.wilcoxon(diff).pvalue if len(diff) > 1 and (diff != 0).any() else 1.0
        dz = diff.mean() / diff.std(ddof=1) if len(diff) > 1 and diff.std(ddof=1) > 0 else np.nan
        rows.append({"dataset": ds, "diff": diff.mean(), "ci": ci95(diff), "wins": int((diff > 0).sum()),
                     "n": len(diff), "p": p, "dz": dz})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
    return out


def summarise(df: pd.DataFrame, out: Path) -> str:
    meta = df.drop_duplicates(["dataset", "seed"])
    elig = meta.groupby("dataset", sort=False)["n_errors"].min() >= MIN_ERRORS
    ok = [d for d in elig.index if elig[d]]
    seeds = sorted(int(s) for s in df.seed.unique())
    kappa, k = df.kappa.iloc[0], df.k.iloc[0]
    L = ["# RQ2: error detection by the trust signals", "",
         f"Main model (fixed centroid + search). Role split 50/20/10/10/10; PRS ranks fitted on the "
         f"inner-validation rows; errors scored on all other rows. Perturbations: kappa = {kappa}, "
         f"K = {k}. Seeds {seeds}. Mean ± 95% CI over seeds. A dataset is eligible for RQ2 if every "
         f"seed has at least {MIN_ERRORS} errors in its evaluation rows.", "",
         "Methods: PRS = mean percentile rank of PD, EI and U (PD + U where EI is missing); "
         "PD + U = the same without EI; Margin = 1 - (s'1 - s'2); Entropy = normalised entropy of s'; "
         "EI alone is scored only on points where it exists.", "",
         "Starred methods use U*, a corrected U added after gate G1 (a variant, not the planned "
         "analysis): the blueprint's U ranks points where the top two classes tie below most other "
         "points, although ties are where CFMNN errs most. U* ranks every tie first and grades ties by "
         "the centroid-distance ratio.", "",
         "## Datasets", "",
         "| Dataset | Eligible | Accuracy (%) | Errors (min over seeds) | Evaluation rows | Boxes | "
         "Tied points | Error rate in ties | EI coverage | Mode switches | Box switches |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, g in meta.groupby("dataset", sort=False):
        L.append(f"| {ds} | {'yes' if elig[ds] else 'no'} | {fmt(g.accuracy.mean(), ci95(g.accuracy), 1)} | "
                 f"{g.n_errors.min()} | {int(g.n_eval.mean())} | {g.n_boxes.mean():.0f} | "
                 f"{g.tie_share.mean():.3f} | {g.tie_error_rate.mean():.3f} | "
                 f"{g.ei_coverage.mean():.2f} | {g.mode_switch.mean():.3f} | {g.box_switch.mean():.3f} |")

    for metric, title in [("auprc", "AUPRC (higher is better; compare with the base error rate)"),
                          ("auroc", "AUROC (0.5 = random)"),
                          ("precision_at", "Precision when the top 10% are flagged")]:
        L += ["", f"## {title}", "",
              "| Dataset | Base error rate | " + " | ".join(METHODS) + " |",
              "|---|---|" + "---|" * len(METHODS)]
        for ds in ok:
            g = df[df.dataset == ds]
            cells = []
            best = g.groupby("method")[metric].mean().drop("Random").max()
            for m in METHODS:
                v = g[g.method == m][metric]
                c = fmt(v.mean(), ci95(v))
                cells.append(f"**{c}**" if np.isclose(v.mean(), best) else c)
            L.append(f"| {ds} | {g.base_rate.mean():.3f} | " + " | ".join(cells) + " |")

    sub = df[df.dataset.isin(ok)]
    for a, b, title in [("PRS", "PD + U", "Primary endpoint: does EI add value? AUPRC of PRS minus PD + U"),
                        ("PRS*", "PD + U*", "Variant: does EI add value with U*? AUPRC of PRS* minus PD + U*"),
                        ("PRS*", "Margin", "PRS* minus the plain margin (AUPRC)"),
                        ("PD + U*", "Margin", "PD + U* minus the plain margin (AUPRC)")]:
        t = paired(sub, a, b)
        L += ["", f"## {title}", "",
              "Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; "
              "d_z = mean difference / its standard deviation.", "",
              "| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |",
              "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.dataset} | {fmt(r['diff'], r.ci)} | {r.wins} of {r.n} | {r.p:.3g} | "
                     f"{r.p_holm:.3g} | {r.dz:.2f} |")

    L += ["", "## Signal checks (gate G1)", "",
          "Plausibility = share of perturbations whose L1 distance to the nearest training row is at "
          "most the 95th percentile of the original points' distances (near 0.95 or above = "
          "perturbations sit where real data sits). Spread = standard deviation over evaluation points.", "",
          "| Dataset | Clipped coordinates | Plausibility | Spread of U | Spread of entropy | "
          "rho(PD, U) | rho(PD, EI) | rho(EI, U) | Signal time (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds, g in meta.groupby("dataset", sort=False):
        L.append(f"| {ds} | {g.clip_share.mean():.3f} | {g.plausibility.mean():.2f} | "
                 f"{g.u_std.mean():.3f} | {g.entropy_std.mean():.3f} | {g.rho_pd_u.mean():.2f} | "
                 f"{g.rho_pd_ei.mean():.2f} | {g.rho_ei_u.mean():.2f} | {g.signals_s.mean():.1f} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def summarise_confirmatory(df: pd.DataFrame, out: Path) -> str:
    """Summary of the pre-registered run: endpoints tested on the synthetic family + real datasets."""
    meta = df.drop_duplicates(["dataset", "seed"])
    elig = meta.groupby("dataset", sort=False)["n_errors"].min() >= MIN_ERRORS
    ok = [d for d in elig.index if elig[d]]
    syn_ok = [d for d in SYNTHETIC if d in ok]
    real_ok = [d for d in ok if d not in SYNTHETIC]
    methods = METHODS + BASELINES
    cols = ["auprc", "auroc", "precision_at", "base_rate"]
    fam = (df[df.dataset.isin(syn_ok)].groupby(["seed", "method"], as_index=False)[cols].mean()
           .assign(dataset="Synthetic family"))
    units = pd.concat([fam, df[df.dataset.isin(real_ok)]], ignore_index=True)
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# RQ2 confirmatory run (pre-registered)", "",
         f"Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Seeds {seeds}. Main model (fixed "
         f"centroid + search). Datasets under {SMALL:,} rows: stratified 5-fold cross-validation with "
         f"pooled out-of-fold scores; others: role split 50/20/10/10/10 with at most {MAX_EVAL:,} "
         f"evaluation rows. PRS ranks fitted on inner-validation rows only. Eligible: at least "
         f"{MIN_ERRORS} errors in every seed. Mean ± 95% CI over seeds.", "",
         f"Tested units: the synthetic family (per seed, the mean over {len(syn_ok)} synthetic sets: "
         f"{', '.join(syn_ok)}) and each eligible real dataset ({', '.join(real_ok) or 'none'}). "
         "Per-set synthetic rows below are descriptive only.", "",
         "Baselines (all read the training labels; the PACE signals do not): Trust Score (Jiang et "
         "al. 2018), disagreement of 5 CFMNNs trained on the same rows in other orders, and the share "
         "of the 10 nearest training rows with another label.", "",
         "## Datasets", "",
         "| Dataset | Split | Eligible | Accuracy (%) | Errors (min over seeds) | Evaluation rows | "
         "Boxes | Tied points | Error rate in ties | EI coverage |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, g in meta.groupby("dataset", sort=False):
        L.append(f"| {ds} | {g.split.iloc[0]} | {'yes' if elig[ds] else 'no'} | "
                 f"{fmt(g.accuracy.mean(), ci95(g.accuracy), 1)} | {g.n_errors.min()} | "
                 f"{int(g.n_eval.mean())} | {g.n_boxes.mean():.0f} | {g.tie_share.mean():.3f} | "
                 f"{g.tie_error_rate.mean():.3f} | {g.ei_coverage.mean():.2f} |")

    for metric, title in [("auprc", "AUPRC (higher is better; compare with the base error rate)"),
                          ("auroc", "AUROC (0.5 = random)"),
                          ("precision_at", "Precision when the top 10% are flagged")]:
        L += ["", f"## {title}", "",
              "| Dataset | Base error rate | " + " | ".join(methods) + " |",
              "|---|---|" + "---|" * len(methods)]
        frame = pd.concat([df[df.dataset.isin(ok)], fam], ignore_index=True)
        for ds in syn_ok + ["Synthetic family"] + real_ok:
            g = frame[frame.dataset == ds]
            best = g.groupby("method")[metric].mean().drop("Random").max()
            cells = []
            for m in methods:
                v = g[g.method == m][metric]
                c = fmt(v.mean(), ci95(v))
                cells.append(f"**{c}**" if np.isclose(v.mean(), best) else c)
            L.append(f"| {ds} | {g.base_rate.mean():.3f} | " + " | ".join(cells) + " |")

    tie_units = [d for d in real_ok if meta[meta.dataset == d].tie_share.mean() >= 0.01]
    endpoints = [
        ("PD + U*", "Margin", units, "P1 (primary): PD + U* minus the plain margin"),
        ("PRS*", "PD + U*", units, "P2 (primary): does EI add value? PRS* minus PD + U*"),
        ("PD + U*", "Trust Score", units, "Secondary: PD + U* minus Trust Score"),
        ("PD + U*", "Order ensemble", units, "Secondary: PD + U* minus order-ensemble disagreement"),
        ("PD + U*", "kNN disagreement", units, "Secondary: PD + U* minus kNN disagreement"),
        ("U*", "U", units[units.dataset.isin(tie_units)],
         "Secondary: U* minus U, on real datasets with at least 1% tied points"),
    ]
    for a, b, frame, title in endpoints:
        t = paired(frame, a, b)
        L += ["", f"## {title} (AUPRC)", "",
              "Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across the units "
              "in this table; d_z = mean difference / its standard deviation.", "",
              "| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |",
              "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.dataset} | {fmt(r['diff'], r.ci)} | {r.wins} of {r.n} | {r.p:.3g} | "
                     f"{r.p_holm:.3g} | {r.dz:.2f} |")

    L += ["", "## Signal checks and cost", "",
          "| Dataset | Clipped coordinates | Plausibility | rho(PD, U) | rho(PD, EI) | Signal time (s) | "
          "Baseline time (s) |",
          "|---|---|---|---|---|---|---|"]
    for ds, g in meta.groupby("dataset", sort=False):
        L.append(f"| {ds} | {g.clip_share.mean():.3f} | {g.plausibility.mean():.2f} | "
                 f"{g.rho_pd_u.mean():.2f} | {g.rho_pd_ei.mean():.2f} | {g.signals_s.mean():.1f} | "
                 f"{g.baselines_s.mean():.1f} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["exploratory", "confirmatory"], default="exploratory",
                    help="exploratory = seeds 0-9 as first run; confirmatory = the pre-registered run")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--refresh-plausibility", action="store_true",
                    help="recompute only the plausibility column of an existing runs.csv")
    a = ap.parse_args()
    conf = a.protocol == "confirmatory"
    datasets = a.datasets or (CONF_DATASETS if conf else DATASETS)
    seeds = a.seeds or (CONF_SEEDS if conf else list(range(10)))
    out = Path(a.out or ROOT / "results" / ("error_detection_confirmatory" if conf else "error_detection"))
    out.mkdir(parents=True, exist_ok=True)
    if a.refresh_plausibility:
        refresh_plausibility(pd.read_csv(out / "runs.csv")).to_csv(out / "runs.csv", index=False)
    elif not a.summary_only:
        rows = []
        for name in datasets:
            for seed in seeds:
                try:
                    r = run_one(name, seed, a.kappa, a.k, a.protocol)
                except FileNotFoundError as e:
                    print(f"skip {name}: {e}", flush=True)
                    break
                rows += r
                m = {x["method"]: x for x in r}
                print(f"{name} seed {seed}: acc {r[0]['accuracy']:.1f}% errors {r[0]['n_errors']} "
                      f"AUPRC PRS {m['PRS']['auprc']:.3f} PD+U* {m['PD + U*']['auprc']:.3f} "
                      f"margin {m['Margin']['auprc']:.3f} EI cov {r[0]['ei_coverage']:.2f} "
                      f"({r[0]['total_s']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)   # saved after each dataset
    df = pd.read_csv(out / "runs.csv")
    print(summarise_confirmatory(df, out) if "protocol" in df.columns else summarise(df, out))


if __name__ == "__main__":
    main()
