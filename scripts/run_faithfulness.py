"""RQ1: faithfulness, stability and time per explanation.

For each explained test point, features are replaced one at a time in the
order a method ranks them, by copying that feature's value from one random
real background row (10 repetitions, each with its own row). We track whether
the model still predicts the original class A, and A's normalised margin.

Two protocols:

* ``v2`` (default, the fair protocol):
  - **ranking:** features in order of signed attribution, most supportive of
    A first. That is the usual deletion order. Methods whose scores are never
    negative (PACE-flip, occlusion) rank exactly as before;
  - **what is explained:** KernelSHAP and LIME explain A's margin, the
    quantity the test measures. KernelSHAP uses 20 real background rows;
  - **deletion** (lower AUC = more faithful) and **insertion**: start from the
    background row and put the top features back (higher AUC = more
    faithful);
  - the old |attribution| ranking is kept as a secondary column;
  - top-3 ground truth breaks ties at random.
* ``v3`` (pre-registered, seeds 30–39, results/faithfulness_v3/): the v2
  tests, plus PACE-SHAP (exact Shapley values of the margin from CFMNN's
  terms), PACE-SHAP-20 (KernelSHAP's own 20 rows) and KernelSHAP-strong
  (2,048 samples). The tests copy values from one set of training rows and
  every explainer uses a second, disjoint set (``backgrounds_v3``).
* ``v1``: the original protocol (|attribution| ranking, pseudo-probability
  targets, 10 k-means background centres), kept so the earlier results
  reproduce exactly.
* ``cond`` (RQ1 extension B3, pre-registered in
  results/extensions_v1/PREREGISTRATION.md; seeds 40–49; output in
  results/extensions_v1/conditional/): interventional vs conditional
  replacement. Eight explainers on the same explainer rows: PACE-SHAP and
  PACE-SHAP-cond, PACE-flip and PACE-flip-cond, KernelSHAP and
  KernelSHAP-cond (one implementation, ``xai_baselines.kernel_shap_games``,
  differing only in how rows are weighted), LIME and Random. The two
  KernelSHAPs, and the two PACE-SHAPs where exact, share their model calls
  (same results as running them apart, about half the time); a paired
  method's time is the shared part plus its own part. Each is scored
  by the v3 tests (interventional) and by conditional versions in which the
  features not taken from x come from the evaluation rows nearest to x on
  the features taken from x. Also: the off-manifold share of every
  replacement scheme (``pace_cfmnn.realism``), and S1 and S2 with a
  correlated distractor (S1c, S2c: feature 3 is a noisy copy of feature 0).
  The two estimands are reported in separate tables.

``--timing`` measures milliseconds per explanation only (no curves). Run it
alone, so the times are not distorted by other work; ``--jobs`` runs dataset x
seed pairs in parallel for the quality metrics, whose results do not depend on
it.

Usage:
    python scripts/run_faithfulness.py --jobs 3
    python scripts/run_faithfulness.py --timing --seeds 0 1 2
    python scripts/run_faithfulness.py --protocol v1 --out results/faithfulness_v1
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

from pace_cfmnn import CFMNN, data  # noqa: E402
from pace_cfmnn.explain import (PaceFlip, PaceShap, cond_neighbour_values, cond_scale,  # noqa: E402
                                contrastive, contrastive_multi, decomposition, pace_flip,
                                pace_flip_cond)
from pace_cfmnn import realism  # noqa: E402
from pace_cfmnn import xai_baselines as xb  # noqa: E402

DATASETS = ["S1", "S2", "S3", "wine", "breast_cancer", "thyroid", "shuttle",
            "magic", "letter", "landsat", "page_blocks"]
SYNTHETIC = ("S1", "S2", "S3", "S1c", "S2c")
# cond protocol (RQ1 extension B3)
COND_DATASETS = ["S1", "S2", "S3", "S1c", "S2c"] + DATASETS[3:]
COND_SIGMA = 0.1            # Aas, Jullum and Løland (2021)
COND_K = 32                 # neighbour rows per feature for PACE-flip-cond
COND_NSAMPLES = 200         # KernelSHAP coalitions per point (as the v2/v3 KernelSHAP)
REALISM_K = 200             # sampled inputs per point and scheme
SCHEMES = {"shap_int": "Interventional Shapley game (PACE-SHAP, KernelSHAP)",
           "shap_cond": "Conditional Shapley game (PACE-SHAP-cond, KernelSHAP-cond)",
           "flip_int": "Quantile values (PACE-flip)",
           "flip_cond": "Neighbour values (PACE-flip-cond)",
           "lime": "LIME's samples (kernel-weighted)",
           "test_int": "Interventional deletion/insertion test",
           "test_cond": "Conditional deletion/insertion test",
           "points": "The explained points themselves"}
SHAP_BG_ROWS = 20
# v3 (PACE-SHAP). R and P come from scripts/pace_shap_settings.py (approximation
# quality only, seed 99); see results/faithfulness_v3/settings.md.
SHAP_V3_R = 200
SHAP_V3_P = 32
STRONG_NSAMPLES = 2048      # >= 2^d - 2 for d <= 11, so shap enumerates every coalition there
NO_STABILITY_RERUN = {"KernelSHAP-strong"}   # its re-run would double the slowest method


def _margin(det, cls):
    """Rescaled support of the original class minus the best other class."""
    s = det.support_rescaled.copy()
    rows = np.arange(len(cls))
    a = s[rows, cls].copy()
    s[rows, cls] = -np.inf
    return a - s.max(axis=1)


def _order(attr, signed: bool):
    return np.argsort(-attr if signed else -np.abs(attr), axis=1, kind="stable")


def deletion_auc(model, X, attr, background, reps=10, seed=0, signed=False, with_raw=False):
    """Per-point deletion AUCs (class kept, and margin) over k = 0..d deletions.

    Each repetition copies the deleted features from one random background row,
    so binary features stay binary. ``signed=False`` is the original
    |attribution| order; ``signed=True`` deletes the most supportive features first.

    The margin curve is divided by the point's own margin (at least 1e-12) and
    clipped to [-1, 1]. For a tied point (margin 0) that turns the curve into
    signs. ``with_raw=True`` also returns the AUC of the raw, undivided margin.
    """
    n, d = X.shape
    det0 = model.predict_details(X)
    orig = det0.cls
    mx = _margin(det0, orig)
    m0 = np.maximum(mx, 1e-12)
    order = _order(attr, signed)
    rng = np.random.default_rng(seed)
    kept_all, marg_all, raw_all = [], [], []
    for _ in range(reps):
        B = background[rng.integers(len(background), size=n)]
        Z = X.copy()
        kept, marg, raw = [np.ones(n)], [np.ones(n)], [mx]
        for k in range(d):
            idx = order[:, k]
            Z[np.arange(n), idx] = B[np.arange(n), idx]
            det = model.predict_details(Z)
            mg = _margin(det, orig)
            kept.append((det.cls == orig).astype(float))
            marg.append(np.clip(mg / m0, -1, 1))
            raw.append(mg)
        kept_all.append(np.trapezoid(np.array(kept), dx=1.0 / d, axis=0))
        marg_all.append(np.trapezoid(np.array(marg), dx=1.0 / d, axis=0))
        raw_all.append(np.trapezoid(np.array(raw), dx=1.0 / d, axis=0))
    out = np.mean(kept_all, axis=0), np.mean(marg_all, axis=0)
    return out + (np.mean(raw_all, axis=0),) if with_raw else out


def insertion_auc(model, X, attr, background, reps=10, seed=0, with_raw=False):
    """Per-point insertion AUCs (class back, and margin): HIGHER = more faithful.

    Each repetition starts from one random background row and puts x's features
    back in signed-attribution order. At k = d the point is x again (normalised
    margin 1, or 0 for a tied point). ``with_raw=True`` also returns the AUC of
    the raw margin, as in ``deletion_auc``.
    """
    n, d = X.shape
    det0 = model.predict_details(X)
    orig = det0.cls
    m0 = np.maximum(_margin(det0, orig), 1e-12)
    order = _order(attr, signed=True)
    rng = np.random.default_rng(seed + 1)
    kept_all, marg_all, raw_all = [], [], []
    for _ in range(reps):
        Z = background[rng.integers(len(background), size=n)].copy()
        det = model.predict_details(Z)
        mg = _margin(det, orig)
        kept, marg, raw = [(det.cls == orig).astype(float)], [np.clip(mg / m0, -1, 1)], [mg]
        for k in range(d):
            idx = order[:, k]
            Z[np.arange(n), idx] = X[np.arange(n), idx]
            det = model.predict_details(Z)
            mg = _margin(det, orig)
            kept.append((det.cls == orig).astype(float))
            marg.append(np.clip(mg / m0, -1, 1))
            raw.append(mg)
        kept_all.append(np.trapezoid(np.array(kept), dx=1.0 / d, axis=0))
        marg_all.append(np.trapezoid(np.array(marg), dx=1.0 / d, axis=0))
        raw_all.append(np.trapezoid(np.array(raw), dx=1.0 / d, axis=0))
    out = np.mean(kept_all, axis=0), np.mean(marg_all, axis=0)
    return out + (np.mean(raw_all, axis=0),) if with_raw else out


def cond_curves(model, X, attr, eval_bg, reps=10, seed=0, insertion=False):
    """Conditional deletion (or insertion) AUCs: per-point (class kept, margin).

    As ``deletion_auc(signed=True)`` / ``insertion_auc``, but the features not
    taken from x come from the q-th nearest evaluation row to x on the
    features taken from x (repetition q = 0..reps-1; standardised Euclidean
    distance), so replaced values fit the rest of the point. When nothing is
    taken from x (all deleted, or nothing inserted yet) a random row is used.
    """
    n, d = X.shape
    det0 = model.predict_details(X)
    orig = det0.cls
    m0 = np.maximum(_margin(det0, orig), 1e-12)
    order = _order(attr, signed=True)
    Q = ((X[:, None, :] - eval_bg[None, :, :]) / cond_scale(eval_bg)) ** 2   # (n, R, d)
    nr = min(reps, len(eval_bg))
    rand = np.random.default_rng(seed + (1 if insertion else 0)).integers(len(eval_bg), size=(reps, n))
    pts = np.arange(n)[:, None]
    kept = np.ones((reps, d + 1, n))
    marg = np.ones((reps, d + 1, n))
    for k in range(d + 1):
        take = np.zeros((n, d), dtype=bool) if insertion else np.ones((n, d), dtype=bool)
        take[pts, order[:, :k]] = insertion
        if not insertion and k == 0:
            continue                                      # x itself: kept 1, margin 1
        near = (np.argsort((Q * take[:, None, :]).sum(axis=2), axis=1, kind="stable")[:, :nr]
                if take.any() else None)
        for q in range(reps):
            rows = near[:, q % nr] if near is not None else rand[q]
            det = model.predict_details(np.where(take, X, eval_bg[rows]))
            kept[q, k] = det.cls == orig
            marg[q, k] = np.clip(_margin(det, orig) / m0, -1, 1)
    auc = lambda c: np.trapezoid(c, dx=1.0 / d, axis=1).mean(axis=0)
    return auc(kept), auc(marg)


def top3_random_ties(attr, rng):
    """Top-3 features by |attribution|, ties broken at random (not by feature index)."""
    n, d = attr.shape
    perm = rng.permuted(np.tile(np.arange(d), (n, 1)), axis=1)
    a = np.take_along_axis(-np.abs(attr), perm, axis=1)
    return np.take_along_axis(perm, np.argsort(a, axis=1, kind="stable"), axis=1)[:, :3]


def setup(name, seed, n_explain):
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    te = np.random.default_rng(seed).permutation(sp["test"])[:n_explain]
    Xte = sc.transform(ds.X[te])
    model = CFMNN().fit(Xtr, ytr)
    bg = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(200, len(Xtr)), replace=False)]
    return ds, Xtr, Xte, model, bg


def backgrounds_v3(Xtr, seed, size=200):
    """Two disjoint sets of training rows: (evaluation background, explainer background).

    The deletion and insertion tests copy values from the first; every
    explainer uses only the second, so no method is graded on rows it saw.
    Each holds ``size`` rows, or half the training set if that is smaller.
    """
    perm = np.random.default_rng(seed + 500).permutation(len(Xtr))
    k = min(size, len(Xtr) // 2)
    return Xtr[perm[:k]], Xtr[perm[k:2 * k]]


def _same_rows(bg, rows, seed):
    """The rows KernelSHAP picks for this seed (see xai_baselines._kernel_shap_margin)."""
    if len(bg) <= rows:
        return bg
    return bg[np.random.default_rng(seed).choice(len(bg), rows, replace=False)]


def methods_v3(model, Xte, Xtr, bex):
    """v3: explainers see only the explainer background ``bex``."""
    return {
        "PACE-SHAP": lambda r: PaceShap(model, bex[:SHAP_V3_R], n_permutations=SHAP_V3_P,
                                        seed=r).explain(Xte),
        "PACE-SHAP-20": lambda r: PaceShap(model, _same_rows(bex, SHAP_BG_ROWS, r),
                                           n_permutations=SHAP_V3_P, seed=r).explain(Xte),
        "PACE-flip": lambda r: pace_flip(model, Xte, bex),
        "PACE-multi": lambda r: contrastive_multi(model, Xte),
        "PACE-contrastive": lambda r: contrastive(model, Xte).e,
        "PACE-decomposition": lambda r: decomposition(model, Xte, bex).phi,
        "KernelSHAP": lambda r: xb.kernel_shap(model, Xte, bex, seed=r, target="margin",
                                               bg_rows=SHAP_BG_ROWS),
        "KernelSHAP-strong": lambda r: xb.kernel_shap(model, Xte, bex, seed=r, target="margin",
                                                      bg_rows=SHAP_BG_ROWS,
                                                      nsamples=STRONG_NSAMPLES, l1_reg=False),
        "LIME": lambda r: xb.lime_tabular(model, Xte, Xtr, seed=r, target="margin"),
        "Random": lambda r: np.random.default_rng(r).uniform(size=Xte.shape),
    }


def methods(model, Xte, Xtr, bg, protocol):
    out = {
        "PACE-contrastive": lambda r: contrastive(model, Xte).e,
        "PACE-multi": lambda r: contrastive_multi(model, Xte),
        "PACE-flip": lambda r: pace_flip(model, Xte, bg),
        "Occlusion (brute force)": lambda r: xb.occlusion(model, Xte, bg),
        "PACE-decomposition": lambda r: decomposition(model, Xte, bg).phi,
    }
    if protocol == "v1":
        out["KernelSHAP"] = lambda r: xb.kernel_shap(model, Xte, bg, seed=r)
        out["LIME"] = lambda r: xb.lime_tabular(model, Xte, Xtr, seed=r)
    else:
        out["KernelSHAP"] = lambda r: xb.kernel_shap(model, Xte, bg, seed=r, target="margin",
                                                     bg_rows=SHAP_BG_ROWS)
        out["LIME"] = lambda r: xb.lime_tabular(model, Xte, Xtr, seed=r, target="margin")
    out["Random"] = lambda r: np.random.default_rng(r).uniform(size=Xte.shape)
    return out


def run(name, seed, n_explain, protocol="v2"):
    ds, Xtr, Xte, model, bg = setup(name, seed, n_explain)
    bex = bg
    if protocol == "v3":
        bg, bex = backgrounds_v3(Xtr, seed)
    relevant = set(ds.info.get("relevant", [])) if name in ("S1", "S2", "S3") else None
    rows, extra = [], []
    if relevant is not None:
        # How much of the model's own single-feature decision change comes from the relevant
        # features: the share of PACE-flip's flip mass on them (computed from the model, no explainer).
        _, flip, _ = PaceFlip(model, background=bex).explain(Xte, return_parts=True)
        rel = sorted(relevant)
        extra.append({"dataset": name, "seed": seed,
                      "flip_mass_relevant": float(flip[:, rel].sum() / max(flip.sum(), 1e-12)),
                      "share_points_flip_irrelevant": float((flip[:, [i for i in range(flip.shape[1])
                                                                       if i not in relevant]].sum(axis=1) > 0).mean())})
    gt_rng = np.random.default_rng(seed + 7)
    ms = methods_v3(model, Xte, Xtr, bex) if protocol == "v3" else methods(model, Xte, Xtr, bg, protocol)
    for m, fn in ms.items():
        t0 = time.perf_counter()
        attr = fn(seed)
        dt = (time.perf_counter() - t0) / len(Xte)
        top3 = np.argsort(-np.abs(attr), axis=1, kind="stable")[:, :3]
        if protocol == "v3" and m in NO_STABILITY_RERUN:
            stab = np.nan
        else:
            top3b = np.argsort(-np.abs(fn(seed + 1000)), axis=1, kind="stable")[:, :3]
            stab = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(top3, top3b)])
        if protocol == "v1":
            auc, mauc = deletion_auc(model, Xte, attr, bg, seed=seed)
            gt = (np.mean([len(set(t) & relevant) / 3 for t in top3]) if relevant else np.nan)
            row = {"deletion_auc": auc.mean(), "margin_auc": mauc.mean(), "gt_top3_precision": gt}
        else:
            auc, mauc = deletion_auc(model, Xte, attr, bg, seed=seed, signed=True)
            _, mauc_abs = deletion_auc(model, Xte, attr, bg, seed=seed, signed=False)
            iauc, imauc = insertion_auc(model, Xte, attr, bg, seed=seed)
            t3 = top3_random_ties(attr, gt_rng)
            gt = (np.mean([len(set(t) & relevant) / 3 for t in t3]) if relevant else np.nan)
            row = {"deletion_auc": auc.mean(), "margin_auc": mauc.mean(),
                   "margin_auc_abs": mauc_abs.mean(), "insertion_auc": iauc.mean(),
                   "insertion_margin_auc": imauc.mean(), "gt_top3_precision": gt,
                   "share_negative": float((attr < 0).mean())}
        rows.append({"dataset": name, "seed": seed, "method": m, "protocol": protocol, **row,
                     "stability_top3": stab, "ms_per_explanation": 1000 * dt, "n": len(Xte)})
        print(f"{name} seed {seed} {m}: margin {row['margin_auc']:.3f} "
              f"{'insertion %.3f ' % row['insertion_margin_auc'] if protocol == 'v2' else ''}"
              f"stab {stab:.2f}, {1000 * dt:.2f} ms", flush=True)
    return rows, extra


def timing(name, seed, n_explain, protocol="v2"):
    """Milliseconds per explanation only (one pass per method, no curves)."""
    _, Xtr, Xte, model, bg = setup(name, seed, n_explain)
    ms = (methods_v3(model, Xte, Xtr, backgrounds_v3(Xtr, seed)[1]) if protocol == "v3"
          else methods(model, Xte, Xtr, bg, protocol))
    out = []
    for m, fn in ms.items():
        t0 = time.perf_counter()
        fn(seed)
        out.append({"dataset": name, "seed": seed, "method": m,
                    "ms_per_explanation": 1000 * (time.perf_counter() - t0) / len(Xte)})
    print(f"{name} seed {seed} timed", flush=True)
    return out


# ------------------------------------------------- cond protocol (extension B3)
def methods_cond(model, Xte, B):
    """Eight explainers, all on the same explainer rows ``B``, in groups run together.

    Each group maps a seed r to {method: (attributions, seconds per explanation)},
    in the order the methods are reported. KernelSHAP and KernelSHAP-cond, and
    PACE-SHAP and PACE-SHAP-cond where exact (d <= 12), differ only in how the
    rows are weighted, so each pair makes its model calls once
    (``xai_baselines.kernel_shap_games``, ``PaceShap.explain_games``); the
    attributions are identical to computing them apart. A paired method's time
    is the shared part plus its own part, an estimate of its time alone.
    """
    n = len(Xte)
    clock = time.perf_counter

    def single(name, fn):
        def run(r):
            t0 = clock()
            attr = fn(r)
            return {name: (attr, (clock() - t0) / n)}
        return run

    def pair(names, fn):
        def run(r):
            times = {}
            t0 = clock()
            attrs = fn(r, times)
            shared = clock() - t0 - sum(times["own"])
            return {m: (a, (shared + own) / n) for m, a, own in zip(names, attrs, times["own"])}
        return run

    def pace_shap_pair(r, times):
        t0 = clock()
        ps = PaceShap(model, B, n_permutations=SHAP_V3_P, seed=r)
        if ps.exact:
            return ps.explain_games(Xte, (None, COND_SIGMA), times=times)
        # Sampled mode: the two games sample differently, so nothing is shared.
        a = ps.explain(Xte)
        t1 = clock()
        b = PaceShap(model, B, n_permutations=SHAP_V3_P, seed=r, sigma=COND_SIGMA).explain(Xte)
        times["own"] = [t1 - t0, clock() - t1]
        return a, b

    def kernel_shap_pair(r, times):
        return xb.kernel_shap_games(model, Xte, B, nsamples=COND_NSAMPLES, seed=r,
                                    sigmas=(None, COND_SIGMA), times=times)

    return [
        pair(("PACE-SHAP", "PACE-SHAP-cond"), pace_shap_pair),
        single("PACE-flip", lambda r: pace_flip(model, Xte, B)),
        single("PACE-flip-cond", lambda r: pace_flip_cond(model, Xte, B, k=COND_K)),
        pair(("KernelSHAP", "KernelSHAP-cond"), kernel_shap_pair),
        single("LIME", lambda r: xb.lime_tabular(model, Xte, B, seed=r, target="margin")),
        single("Random", lambda r: np.random.default_rng(r).uniform(size=Xte.shape)),
    ]


def realism_rows(name, seed, Xtr, Xte, bg, B):
    """Off-manifold share of every replacement scheme (one row per scheme)."""
    man = realism.Manifold(Xtr, seed=seed)
    rng = np.random.default_rng(seed + 900)
    scale = cond_scale(B)
    k = REALISM_K
    quant = np.quantile(B, (np.arange(32) + 0.5) / 32, axis=0)
    neigh = np.stack([cond_neighbour_values(x, B, COND_K, scale) for x in Xte])
    inputs = {
        "shap_int": (realism.shapley_inputs(Xte, B, k, rng), None),
        "shap_cond": (realism.shapley_inputs(Xte, B, k, rng, sigma=COND_SIGMA, scale=scale), None),
        "flip_int": (realism.occlusion_inputs(Xte, quant, k, rng), None),
        "flip_cond": (realism.occlusion_inputs(Xte, neigh, k, rng), None),
        "lime": realism.lime_inputs(Xte, B, k, rng),
        "test_int": (realism.curve_inputs(Xte, bg, k, rng, conditional=False), None),
        "test_cond": (realism.curve_inputs(Xte, bg, k, rng, conditional=True), None),
        "points": (Xte[:, None, :], None),
    }
    return [{"dataset": name, "seed": seed, "method": s, "off_share": man.share(Z, w),
             "threshold": man.threshold_} for s, (Z, w) in inputs.items()]


def run_cond(name, seed, n_explain):
    ds, Xtr, Xte, model, _ = setup(name, seed, n_explain)
    bg, bex = backgrounds_v3(Xtr, seed)
    B = bex[:SHAP_V3_R]
    relevant = set(ds.info.get("relevant", [])) if name in SYNTHETIC else None
    gt_rng = np.random.default_rng(seed + 7)
    extra = []
    if relevant is not None:
        _, flip, _ = PaceFlip(model, background=B).explain(Xte, return_parts=True)
        tot = max(flip.sum(), 1e-12)
        extra.append({"dataset": name, "seed": seed,
                      "flip_mass_relevant": float(flip[:, sorted(relevant)].sum() / tot),
                      "flip_mass_x3": float(flip[:, 3].sum() / tot)})
    rows = []
    for group in methods_cond(model, Xte, B):
        for m, (attr, dt) in group(seed).items():
            _, del_int = deletion_auc(model, Xte, attr, bg, seed=seed, signed=True)
            _, ins_int = insertion_auc(model, Xte, attr, bg, seed=seed)
            _, del_cond = cond_curves(model, Xte, attr, bg, seed=seed)
            _, ins_cond = cond_curves(model, Xte, attr, bg, seed=seed, insertion=True)
            t3 = top3_random_ties(attr, gt_rng)
            a = np.abs(attr)
            share3 = np.divide(a[:, 3], a.sum(axis=1), out=np.zeros(len(a)), where=a.sum(axis=1) > 0)
            rows.append({"dataset": name, "seed": seed, "method": m, "protocol": "cond",
                         "margin_auc": del_int.mean(), "insertion_margin_auc": ins_int.mean(),
                         "cond_margin_auc": del_cond.mean(),
                         "cond_insertion_margin_auc": ins_cond.mean(),
                         "gt_top3_precision": (np.mean([len(set(t) & relevant) / 3 for t in t3])
                                               if relevant else np.nan),
                         "x3_share": float(share3.mean()) if name in ("S1c", "S2c") else np.nan,
                         "ms_per_explanation": 1000 * dt, "n": len(Xte)})
            print(f"{name} seed {seed} {m}: deletion {del_int.mean():.3f} / cond "
                  f"{del_cond.mean():.3f}, {1000 * dt:.1f} ms", flush=True)
    return rows, extra, realism_rows(name, seed, Xtr, Xte, bg, B)


COND_COMPARISONS = [   # (focus, other, metric, lower is better); the first two are primary
    ("PACE-SHAP-cond", "KernelSHAP-cond", "cond_margin_auc", True),
    ("PACE-SHAP-cond", "KernelSHAP-cond", "cond_insertion_margin_auc", False),
    ("PACE-SHAP", "KernelSHAP", "margin_auc", True),
    ("PACE-SHAP", "KernelSHAP", "insertion_margin_auc", False),
    ("PACE-SHAP-cond", "PACE-SHAP", "cond_margin_auc", True),
    ("PACE-SHAP-cond", "PACE-SHAP", "cond_insertion_margin_auc", False),
    ("PACE-SHAP-cond", "PACE-SHAP", "margin_auc", True),
    ("PACE-SHAP-cond", "PACE-SHAP", "insertion_margin_auc", False),
    ("PACE-flip-cond", "PACE-flip", "cond_margin_auc", True),
    ("PACE-flip-cond", "PACE-flip", "margin_auc", True),
]
REALISM_COMPARISONS = [("shap_cond", "shap_int"), ("flip_cond", "flip_int"), ("test_cond", "test_int")]
METRIC_NAMES = {"margin_auc": "interventional deletion margin AUC",
                "insertion_margin_auc": "interventional insertion margin AUC",
                "cond_margin_auc": "conditional deletion margin AUC",
                "cond_insertion_margin_auc": "conditional insertion margin AUC"}


def _paired_block(L, df, focus, other, metric, lower, title, tag=""):
    from run_error_detection import paired
    res = paired(df, focus, other, metric=metric)
    L += ["", f"## {title}{tag}", "",
          "Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. "
          f"Better for {focus}: {'negative' if lower else 'positive'}.", "",
          f"| Dataset | Difference | Seeds where {focus} is higher | p | p (Holm) |",
          "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r.dataset} | {r['diff']:+.3f} ± {r.ci:.3f} | {r.wins} of {r.n} | "
                 f"{r.p:.3g} | {r.p_holm:.3g} |")


def summarise_cond(df, extra, real, out: Path) -> str:
    from run_error_detection import ci95
    seeds = sorted(int(s) for s in df.seed.unique())
    f = lambda v: f"{v.mean():.3f} ± {ci95(v):.3f}" if v.notna().any() else "–"
    L = ["# RQ1 extension B3: interventional vs conditional replacement", "",
         f"{int(df.n.max())} test points per dataset and seed; seeds {seeds}. Every explainer uses the "
         f"same {SHAP_V3_R} explainer rows (half the training set if smaller); the tests copy values "
         "from a disjoint set of evaluation rows. Interventional: replaced features come from a random "
         "row. Conditional: they come from rows close to the point on the features it keeps "
         f"(PACE-SHAP-cond and KernelSHAP-cond: Gaussian weights, sigma = {COND_SIGMA}, on standardised "
         f"distances; PACE-flip-cond: the {COND_K} nearest rows on the other features; conditional "
         "tests: the q-th nearest evaluation row in repetition q). KernelSHAP and KernelSHAP-cond share "
         f"one implementation ({COND_NSAMPLES} coalitions per point) and differ only in the row "
         "weights. Each such pair (the two KernelSHAPs; the two PACE-SHAPs where exact, d ≤ 12) "
         "makes its model calls once; a paired method's time is the shared part plus its own part. "
         "The two kinds of test answer different questions, so they are reported separately and "
         "never combined. Mean ± 95% CI over seeds.", ""]
    for metric_pair, title in [(("margin_auc", "insertion_margin_auc"), "Interventional test"),
                               (("cond_margin_auc", "cond_insertion_margin_auc"), "Conditional test")]:
        L += ["", f"## {title}", "", "Deletion margin AUC: lower = more faithful. Insertion margin AUC: "
              "higher = more faithful.", "",
              "| Dataset | Method | Deletion margin AUC | Insertion margin AUC |", "|---|---|---|---|"]
        for (ds, m), g in df.groupby(["dataset", "method"], sort=False):
            L.append(f"| {ds} | {m} | {f(g[metric_pair[0]])} | {f(g[metric_pair[1]])} |")
    for k, (a, b, metric, lower) in enumerate(COND_COMPARISONS):
        _paired_block(L, df, a, b, metric, lower, f"{a} minus {b}: {METRIC_NAMES[metric]}",
                      " (primary)" if k < 2 else " (secondary)")
    L += ["", "## Realism: off-manifold share of the inputs each scheme evaluates", "",
          "An input is off-manifold if its nearest training row is farther than the 95th percentile "
          "of the training rows' own nearest-neighbour distances (standardised features), so about "
          f"5% of real points are. {REALISM_K} sampled inputs per point and scheme; LIME's are weighted "
          "by its kernel. Lower = more realistic.", "",
          "| Dataset | " + " | ".join(SCHEMES[s] for s in SCHEMES) + " |", "|---|" + "---|" * len(SCHEMES)]
    for ds, g in real.groupby("dataset", sort=False):
        L.append(f"| {ds} | " + " | ".join(f(g[g.method == s].off_share) for s in SCHEMES) + " |")
    for k, (a, b) in enumerate(REALISM_COMPARISONS):
        _paired_block(L, real, a, b, "off_share", True,
                      f"{SCHEMES[a]} minus {SCHEMES[b]}: off-manifold share",
                      " (primary)" if k == 0 else " (secondary)")
    syn = df[df.dataset.isin(["S1c", "S2c"])]
    if len(syn):
        L += ["", "## Correlated distractor (S1c, S2c)", "",
              "Feature 3 is a noisy copy of feature 0 that the true rule does not use. x3 share = mean "
              "over points of |attribution of x3| / sum of |attributions|. The model's own reliance: "
              "share of PACE-flip's single-feature decision flips (interventional) that come from x3.", "",
              "| Dataset | Method | x3 share | Ground truth top-3 |", "|---|---|---|---|"]
        for (ds, m), g in syn.groupby(["dataset", "method"], sort=False):
            L.append(f"| {ds} | {m} | {f(g.x3_share)} | {f(g.gt_top3_precision)} |")
        if len(extra):
            L += ["", "| Dataset | Flip mass on x3 | Flip mass on x0–x2 |", "|---|---|---|"]
            for ds, g in extra[extra.dataset.isin(["S1c", "S2c"])].groupby("dataset", sort=False):
                L.append(f"| {ds} | {f(g.flip_mass_x3)} | {f(g.flip_mass_relevant)} |")
    gt = df[df.dataset.isin(list(SYNTHETIC))]
    if len(gt):
        L += ["", "## Ground truth top-3 (synthetic sets)", "",
              "| Method | " + " | ".join(gt.dataset.unique()) + " |", "|---|" + "---|" * gt.dataset.nunique()]
        for m, g in gt.groupby("method", sort=False):
            L.append(f"| {m} | " + " | ".join(f"{g[g.dataset == ds].gt_top3_precision.mean():.2f}"
                                             for ds in gt.dataset.unique()) + " |")
    L += ["", "## Milliseconds per explanation", "",
          "Measured with several runs sharing the CPU: compare within a dataset only.", "",
          "| Dataset | " + " | ".join(df.method.unique()) + " |", "|---|" + "---|" * df.method.nunique()]
    for ds, g in df.groupby("dataset", sort=False):
        L.append(f"| {ds} | " + " | ".join(f"{g[g.method == m].ms_per_explanation.mean():.1f}"
                                           for m in df.method.unique()) + " |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------- summary
V3_COMPARISONS =[("PACE-SHAP", "KernelSHAP"), ("PACE-SHAP-20", "KernelSHAP"),
                  ("PACE-SHAP", "KernelSHAP-strong"), ("PACE-SHAP", "LIME"),
                  ("PACE-SHAP", "PACE-flip")]


def summarise_v2(df: pd.DataFrame, extra: pd.DataFrame, out: Path, protocol: str = "v2") -> str:
    from run_error_detection import ci95, paired
    seeds = sorted(int(s) for s in df.seed.unique())
    f = lambda v: f"{v.mean():.3f} ± {ci95(v):.3f}"
    if protocol == "v3":
        head = ["# RQ1 faithfulness, v3 (PACE-SHAP; separate explainer background)", "",
                f"{int(df.n.max())} test points per dataset and seed (fewer if the test split is "
                f"smaller); seeds {seeds}. Same tests as v2 (signed ranking, margin targets), but the "
                "deletion and insertion tests copy values from one set of training rows and every "
                "explainer uses a second, disjoint set (200 rows each, or half the training set). "
                f"PACE-SHAP uses the first {SHAP_V3_R} explainer rows (exact for d <= 12, otherwise "
                f"{SHAP_V3_P} orders per row); PACE-SHAP-20 uses the same {SHAP_BG_ROWS} rows as "
                f"KernelSHAP; KernelSHAP-strong uses {STRONG_NSAMPLES} samples and no l1 feature "
                "selection (every coalition when d <= 11; no stability re-run). Each deletion or insertion copies values from one random "
                "evaluation row (10 repetitions). Mean ± 95% CI over seeds.", ""]
        comparisons = V3_COMPARISONS
    else:
        head = ["# RQ1 faithfulness, fair protocol (v2)", "",
                f"{int(df.n.max())} test points per dataset and seed (fewer if the test split is "
                f"smaller); seeds {seeds}. Features are ranked by signed attribution, most supportive "
                "of the predicted class first; KernelSHAP and LIME explain the predicted class's "
                f"margin (KernelSHAP with {SHAP_BG_ROWS} real background rows). Each deletion or "
                "insertion copies values from one random real background row (10 repetitions). "
                "Mean ± 95% CI over seeds.", ""]
        comparisons = [("PACE-flip", o) for o in ("KernelSHAP", "LIME", "PACE-multi")]
    L = head + [
         "Deletion margin AUC: lower = more faithful. Insertion margin AUC: higher = more faithful. "
         "'|attr| deletion' is the original ranking (secondary). Stability = top-3 overlap when the "
         "explainer is re-run with another seed. Ground truth = share of the top 3 (ties broken at "
         "random) that are truly relevant (synthetic sets only).", "",
         "| Dataset | Method | Deletion margin AUC | Insertion margin AUC | |attr| deletion (old) | "
         "Class-kept deletion AUC | Stability | Ground truth top-3 | Negative attributions |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (ds, m), g in df.groupby(["dataset", "method"], sort=False):
        gt = "–" if g.gt_top3_precision.isna().all() else f"{g.gt_top3_precision.mean():.2f}"
        stab = "–" if g.stability_top3.isna().all() else f"{g.stability_top3.mean():.2f}"
        L.append(f"| {ds} | {m} | {f(g.margin_auc)} | {f(g.insertion_margin_auc)} | "
                 f"{f(g.margin_auc_abs)} | {f(g.deletion_auc)} | {stab} | {gt} | "
                 f"{g.share_negative.mean():.2f} |")
    for metric, title, sign in [("margin_auc", "deletion margin AUC (lower is better)", -1),
                                ("insertion_margin_auc", "insertion margin AUC (higher is better)", 1)]:
        for focus, other in comparisons:
            res = paired(df, focus, other, metric=metric)
            better = f"{focus} better" if sign > 0 else f"{focus} better (more negative)"
            L += ["", f"## {focus} minus {other}: {title}", "",
                  f"Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. "
                  f"Sign: {'positive' if sign > 0 else 'negative'} = {better}.", "",
                  f"| Dataset | Difference | Seeds where {focus} is higher | p | p (Holm) |",
                  "|---|---|---|---|---|"]
            for _, r in res.iterrows():
                L.append(f"| {r.dataset} | {r['diff']:+.3f} ± {r.ci:.3f} | {r.wins} of {r.n} | "
                         f"{r.p:.3g} | {r.p_holm:.3g} |")
    if len(extra):
        L += ["", "## The model's own decision sensitivity (synthetic sets)", "",
              "Share of all single-feature decision flips (PACE-flip's flip mass, computed from the "
              "model) that come from the truly relevant features 0–2, and share of points where "
              "replacing some irrelevant feature can flip the decision.", "",
              "| Dataset | Flip mass on relevant features | Points with an irrelevant feature that flips |",
              "|---|---|---|"]
        for ds, g in extra.groupby("dataset", sort=False):
            L.append(f"| {ds} | {f(g.flip_mass_relevant)} | {f(g.share_points_flip_irrelevant)} |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def summarise_v1(df: pd.DataFrame, out: Path) -> str:
    g = df.groupby(["dataset", "method"], sort=False)[["deletion_auc", "margin_auc", "stability_top3",
                                                        "gt_top3_precision", "ms_per_explanation"]]
    mean, sd = g.mean(), g.std()
    seeds = sorted(int(s) for s in df.seed.unique())
    lines = ["# Deletion faithfulness, original protocol (v1; lower AUC = more faithful)", "",
             f"Seeds {seeds}; features deleted in |attribution| order by copying values from real "
             "background rows (10 draws); main model (fixed centroid + search).", "",
             "| Dataset | Method | Class-kept AUC | Margin AUC | Stability | Ground truth top-3 | ms / explanation |",
             "|---|---|---|---|---|---|---|"]
    for (dsn, m), r in mean.iterrows():
        s = sd.loc[(dsn, m)]
        lines.append(f"| {dsn} | {m} | {r.deletion_auc:.3f} ± {s.deletion_auc:.3f} | "
                     f"{r.margin_auc:.3f} ± {s.margin_auc:.3f} | {r.stability_top3:.2f} | "
                     f"{'–' if pd.isna(r.gt_top3_precision) else f'{r.gt_top3_precision:.2f}'} | "
                     f"{r.ms_per_explanation:.2f} |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--protocol", choices=["v1", "v2", "v3", "cond"], default="v2")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timing", action="store_true", help="time the explainers only (run alone)")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    cond = a.protocol == "cond"
    if a.timing and cond:
        raise SystemExit("--timing is not available for the cond protocol")
    datasets = a.datasets or (COND_DATASETS if cond else DATASETS)
    if a.timing:
        seeds = a.seeds or ([30, 31, 32] if a.protocol == "v3" else [0, 1, 2])
    else:
        seeds = a.seeds or {"v1": [0, 1, 2], "v2": list(range(10)), "v3": list(range(30, 40)),
                            "cond": list(range(40, 50))}[a.protocol]
    out = Path(a.out or ROOT / "results" / {"v1": "faithfulness_v1", "v2": "faithfulness",
                                            "v3": "faithfulness_v3",
                                            "cond": "extensions_v1/conditional"}[a.protocol])
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(n, s) for n in datasets for s in seeds]
    if cond:
        if not a.summary_only:
            done = {}
            if a.jobs > 1:
                for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    os.environ[var] = "1"
                with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                    fut = {ex.submit(run_cond, n, s, a.n): (n, s) for n, s in tasks}
                    for f in as_completed(fut):
                        done[fut[f]] = f.result()
            else:
                for t in tasks:
                    done[t] = run_cond(*t, a.n)
            for i, fname in enumerate(("runs.csv", "extra.csv", "realism.csv")):
                pd.DataFrame([r for t in tasks for r in done[t][i]]).to_csv(out / fname, index=False)
        ex_path = out / "extra.csv"
        extra = pd.read_csv(ex_path) if ex_path.exists() and ex_path.stat().st_size > 2 else pd.DataFrame()
        print(summarise_cond(pd.read_csv(out / "runs.csv"), extra, pd.read_csv(out / "realism.csv"), out))
        return
    if a.timing:
        if a.jobs != 1:
            raise SystemExit("--timing must run alone (--jobs 1), or the times are distorted")
        rows = [r for t in tasks for r in timing(*t, a.n, a.protocol)]
        t = pd.DataFrame(rows)
        t.to_csv(out / "timing.csv", index=False)
        m = t.groupby(["dataset", "method"], sort=False).ms_per_explanation.mean().unstack()
        m = m[list(dict.fromkeys(t.method))]
        rows_md = ["| Dataset | " + " | ".join(m.columns) + " |", "|---|" + "---|" * len(m.columns)]
        rows_md += [f"| {ds} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for ds, r in m.iterrows()]
        text = ("# Milliseconds per explanation (batch of points, measured alone)\n\n"
                f"Seeds {seeds}. Mean over seeds.\n\n" + "\n".join(rows_md) + "\n")
        (out / "timing.md").write_text(text, encoding="utf-8")
        print(text)
        return
    if not a.summary_only:
        done = {}
        if a.jobs > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = "1"
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                fut = {ex.submit(run, n, s, a.n, a.protocol): (n, s) for n, s in tasks}
                for f in as_completed(fut):
                    done[fut[f]] = f.result()
        else:
            for t in tasks:
                done[t] = run(*t, a.n, a.protocol)
        pd.DataFrame([r for t in tasks for r in done[t][0]]).to_csv(out / "runs.csv", index=False)
        pd.DataFrame([r for t in tasks for r in done[t][1]]).to_csv(out / "flip_mass.csv", index=False)
    df = pd.read_csv(out / "runs.csv")
    if a.protocol == "v1":
        print(summarise_v1(df, out))
    else:
        fm = out / "flip_mass.csv"
        extra = pd.read_csv(fm) if fm.exists() and fm.stat().st_size > 2 else pd.DataFrame()
        print(summarise_v2(df, extra, out, a.protocol))


if __name__ == "__main__":
    main()
