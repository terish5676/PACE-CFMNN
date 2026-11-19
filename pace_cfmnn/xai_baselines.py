"""XAI baselines on the same model, inputs and splits as PACE (blueprint 9.3).

CFMNN has no probabilities. Two targets are available for KernelSHAP and LIME:

* ``target="proba"`` (the original setting): a pseudo-probability, the
  rescaled class supports s'_c normalised to sum to 1.
* ``target="margin"``: the predicted class's margin s'_A - max_{c != A} s'_c,
  the quantity the deletion and insertion tests measure. The class A is fixed
  per explained point.

Defaults keep the original settings, so earlier results reproduce exactly.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pandas as pd

from .cfmnn_core import CFMNN


def class_scores(model: CFMNN, X) -> np.ndarray:
    """(n, C) rescaled supports normalised per row (uniform if all zero)."""
    s = model.predict_details(np.atleast_2d(X)).support_rescaled
    tot = s.sum(axis=1, keepdims=True)
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1.0 / s.shape[1])


def margin_score(model: CFMNN, X, cls_idx) -> np.ndarray:
    """Margin of class index A: s'_A - max over other classes of s'_c, shape (n,)."""
    s = model.predict_details(np.atleast_2d(X)).support_rescaled.copy()
    a = s[:, cls_idx].copy()
    s[:, cls_idx] = -np.inf
    return a - s.max(axis=1)


def kernel_shap(model: CFMNN, X, background, nsamples=200, seed=0, target="proba",
                bg_rows=None, l1_reg=None) -> np.ndarray:
    """KernelSHAP attributions for the predicted class, shape (n, d).

    ``target="margin"`` explains the predicted class's margin; ``bg_rows`` then
    uses that many real background rows (seeded) instead of 10 k-means centres.
    ``l1_reg`` is passed to shap when given (margin target only); ``None`` keeps
    shap's default. With ``l1_reg=False`` and enough samples to enumerate every
    coalition, KernelSHAP is exact; shap's default feature selection can move
    small values by about 1e-4 even then.

    shap draws its coalition samples from numpy's global random state, so the
    ``seed`` given to KernelExplainer alone does not make results repeatable.
    The global state is seeded for this call and restored afterwards.
    """
    import shap
    if target == "margin":
        return _kernel_shap_margin(model, X, background, nsamples, seed, bg_rows, l1_reg)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        bg = shap.kmeans(background, min(10, len(background))) if len(background) > 10 else background
        ex = shap.KernelExplainer(lambda Z: class_scores(model, Z), bg, seed=seed)
        cls = model.predict_details(X).cls
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = ex.shap_values(np.asarray(X), nsamples=nsamples, silent=True)
    finally:
        np.random.set_state(state)
    vals = np.asarray(vals)
    if vals.ndim == 3 and vals.shape[0] == len(X):      # (n, d, C)
        return vals[np.arange(len(X)), :, cls]
    return np.stack([vals[c][i] for i, c in enumerate(cls)])  # list of (n, d) per class


def _kernel_shap_margin(model, X, background, nsamples, seed, bg_rows, l1_reg=None) -> np.ndarray:
    import shap
    X = np.asarray(X)
    background = np.asarray(background)
    if bg_rows is not None and len(background) > bg_rows:
        background = background[np.random.default_rng(seed).choice(len(background), bg_rows,
                                                                    replace=False)]
    cls = model.predict_details(X).cls
    out = np.zeros(X.shape)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        for a in np.unique(cls):
            rows = cls == a
            ex = shap.KernelExplainer(lambda Z, a=a: margin_score(model, Z, a), background, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kw = {} if l1_reg is None else {"l1_reg": l1_reg}
                out[rows] = np.asarray(ex.shap_values(X[rows], nsamples=nsamples, silent=True, **kw))
    finally:
        np.random.set_state(state)
    return out


def exact_shap(model: CFMNN, X, background) -> np.ndarray:
    """Exact interventional Shapley values of the predicted class's margin, shape (n, d).

    A general-purpose exact computation with shap's ExactExplainer: every
    coalition is evaluated on every background row by calling the model, with
    no use of CFMNN's structure. It is the same game, on the same rows, as
    ``PaceShap(model, background)`` in exact mode, so the two agree up to
    rounding. Points are grouped by predicted class, as for KernelSHAP's
    margin target. The cost grows as 2^d, so use it for small d only.
    """
    import shap
    X = np.asarray(X, float)
    B = np.asarray(background, float)
    cls = model.predict_details(X).cls
    out = np.zeros(X.shape)
    masker = shap.maskers.Independent(B, max_samples=len(B))   # all rows (the default keeps 100)
    for a in np.unique(cls):
        rows = cls == a
        ex = shap.explainers.Exact(lambda Z, a=a: margin_score(model, Z, a), masker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[rows] = ex(X[rows], max_evals=max(100_000, 2 ** X.shape[1] + 2), silent=True).values
    return out


def lime_tabular(model: CFMNN, X, X_train, num_samples=1000, seed=0, target="proba",
                 discretize: bool = False) -> np.ndarray:
    """LIME attributions (local linear weights) for the predicted class, shape (n, d).

    ``target="margin"`` fits LIME in regression mode to the predicted class's margin
    and returns each feature's contribution at x, w_i * (x_i - mean_i) / std_i: LIME
    fits its surrogate on features standardised around the training mean, so this is
    the surrogate's own split of the prediction. (The raw weights are slopes, which
    do not say whether a feature raises or lowers the margin at this point.)

    ``discretize=True`` (margin target) uses LIME's default quartile bins instead.
    The surrogate's features are then "in the same bin as x", which is 1 at x, so
    each weight is that feature's contribution at x.
    """
    from lime.lime_tabular import LimeTabularExplainer
    d = X_train.shape[1]
    if target == "margin":
        ex = LimeTabularExplainer(np.asarray(X_train), discretize_continuous=discretize,
                                  random_state=seed, mode="regression")
        mu, sd = ex.scaler.mean_, ex.scaler.scale_
        cls = model.predict_details(X).cls
        out = np.zeros((len(X), d))
        for r, x in enumerate(np.asarray(X)):
            a = int(cls[r])
            e = ex.explain_instance(x, lambda Z: margin_score(model, Z, a), num_features=d,
                                    num_samples=num_samples)
            for i, w in e.as_map()[1]:
                out[r, i] = w if discretize else w * (x[i] - mu[i]) / sd[i]
        return out
    ex = LimeTabularExplainer(np.asarray(X_train), discretize_continuous=False,
                              random_state=seed, mode="classification")
    cls = model.predict_details(X).cls
    out = np.zeros((len(X), d))
    for r, x in enumerate(np.asarray(X)):
        e = ex.explain_instance(x, lambda Z: class_scores(model, Z), labels=(int(cls[r]),),
                                num_features=d, num_samples=num_samples)
        for i, w in e.as_map()[int(cls[r])]:
            out[r, i] = w
    return out


def nearest_unlike_neighbour(model: CFMNN, x, X_train, target) -> np.ndarray | None:
    """Nearest training point (L1) that the model predicts as the target class."""
    X_train = np.asarray(X_train)
    pred = model.predict(X_train)
    pts = X_train[pred == target]
    return None if len(pts) == 0 else pts[np.abs(pts - x).sum(axis=1).argmin()]


def dice_counterfactual(model: CFMNN, x, X_train, y_train, target, seed=0, locked=(),
                        n_cfs=1, sparse=False, pick="first", method="random"):
    """A DiCE counterfactual for x, or None. ``method`` is DiCE's "random" (the default, and the
    only one used before run B), "genetic" or "kdtree".

    Defaults are the original setting (one counterfactual, DiCE's pre-sparsity
    output). The fairer setting asks for ``n_cfs`` candidates, uses DiCE's own
    post-hoc sparsity output (``sparse=True``) and keeps the nearest one (L1)
    that CFMNN predicts as the target (``pick="nearest_valid"``), else the
    nearest overall.
    """
    import dice_ml
    if method not in ("random", "genetic", "kdtree"):
        raise ValueError("method must be 'random', 'genetic' or 'kdtree'")
    X_train = np.asarray(X_train)
    names = [f"f{i}" for i in range(X_train.shape[1])]
    df = pd.DataFrame(X_train, columns=names)
    df["y"] = model.predict(X_train)
    classes = list(model.classes_)

    class Wrap:
        def predict_proba(self, Z):
            return class_scores(model, np.asarray(Z, float))

        def predict(self, Z):
            return model.predict(np.asarray(Z, float))

    d = dice_ml.Data(dataframe=df, continuous_features=names, outcome_name="y")
    m = dice_ml.Model(model=Wrap(), backend="sklearn")
    exp = dice_ml.Dice(d, m, method=method)
    vary = [n for i, n in enumerate(names) if i not in set(locked)]
    extra = {"random_seed": seed} if method == "random" else {}
    if method != "random":                       # the other methods draw from numpy's global state
        np.random.seed(seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = exp.generate_counterfactuals(pd.DataFrame([x], columns=names), total_CFs=n_cfs,
                                             desired_class=classes.index(target),
                                             features_to_vary=vary, **extra)
        ex0 = r.cf_examples_list[0]
        cf = ex0.final_cfs_df_sparse if (sparse and ex0.final_cfs_df_sparse is not None)             else ex0.final_cfs_df
        if cf is None or len(cf) == 0:
            return None
        Z = cf[names].to_numpy(float)
        if pick == "first":
            return Z[0]
        l1 = np.abs(Z - np.asarray(x, float)).sum(axis=1)
        ok = model.predict(Z) == target
        pool = np.flatnonzero(ok) if ok.any() else np.arange(len(Z))
        return Z[pool[np.argmin(l1[pool])]]
    except Exception:
        return None


def occlusion(model: CFMNN, X, background, n_quantiles: int | None = 32,
              values=None) -> np.ndarray:
    """Brute-force single-feature occlusion with the same replacement values and
    scoring as PACE-flip, but computed by re-running the full model on every
    modified input. It must equal ``pace_flip`` exactly; it exists to measure
    PACE-flip's speed-up, not as a different explanation.

    ``values`` (n, r, d), if given, sets each point's own replacement values
    (column i for feature i); ``background`` is then ignored. With the values
    of ``explain.cond_neighbour_values`` it must equal ``pace_flip_cond``.
    """
    X = model._check_X(X)
    n, d = X.shape
    if values is None:
        B = model._check_X(background)
        if n_quantiles is not None:
            B = np.quantile(B, (np.arange(n_quantiles) + 0.5) / n_quantiles, axis=0)
        per_point = np.broadcast_to(B, (n,) + B.shape)
    else:
        per_point = np.asarray(values, float)
        if per_point.ndim != 3 or per_point.shape[0] != n or per_point.shape[2] != d:
            raise ValueError("values must have shape (n, r, d)")
    r = per_point.shape[1]
    det0 = model.predict_details(X)

    def margin(det, a):
        s = det.support_rescaled.copy()
        rows = np.arange(len(s))
        ma = s[rows, a].copy()
        s[rows, a] = -np.inf
        other = s.max(axis=1)
        return ma - np.where(np.isfinite(other), other, 0.0)

    m0 = margin(det0, det0.cls)
    flip = np.zeros((n, d))
    drop = np.zeros((n, d))
    for p in range(n):
        a = np.full(r, det0.cls[p])
        B = per_point[p]
        for i in range(d):
            Y = np.repeat(X[p][None], r, axis=0)
            Y[:, i] = B[:, i]
            det = model.predict_details(Y)
            flip[p, i] = np.mean(det.cls != det0.cls[p])
            drop[p, i] = m0[p] - margin(det, a).mean()
    lo, hi = drop.min(axis=1, keepdims=True), drop.max(axis=1, keepdims=True)
    norm = np.divide(drop - lo, hi - lo, out=np.zeros_like(drop), where=(hi - lo) > 0)
    return flip + norm * (0.5 / r)


def occlusion_batched(model: CFMNN, X, background, n_quantiles: int | None = 32,
                      values=None, max_rows: int = 100_000) -> np.ndarray:
    """``occlusion`` with the modified inputs of many points sent to the model in
    one call, instead of one call per point and feature.

    Same replacement values, scoring and result as ``occlusion`` (up to
    floating-point rounding). It exists so that PACE-flip's speed-up can also be
    measured against a brute-force baseline without per-call overhead.
    ``max_rows`` caps the modified inputs per model call.
    """
    X = model._check_X(X)
    n, d = X.shape
    if values is None:
        B = model._check_X(background)
        if n_quantiles is not None:
            B = np.quantile(B, (np.arange(n_quantiles) + 0.5) / n_quantiles, axis=0)
        per_point = np.broadcast_to(B, (n,) + B.shape)
    else:
        per_point = np.asarray(values, float)
        if per_point.ndim != 3 or per_point.shape[0] != n or per_point.shape[2] != d:
            raise ValueError("values must have shape (n, r, d)")
    r = per_point.shape[1]
    det0 = model.predict_details(X)

    def margin(det, a):
        s = det.support_rescaled.copy()
        rows = np.arange(len(s))
        ma = s[rows, a].copy()
        s[rows, a] = -np.inf
        other = s.max(axis=1)
        return ma - np.where(np.isfinite(other), other, 0.0)

    m0 = margin(det0, det0.cls)
    flip = np.zeros((n, d))
    drop = np.zeros((n, d))
    ii = np.arange(d)
    step = max(1, max_rows // (d * r))
    for s in range(0, n, step):
        e = min(n, s + step)
        Y = np.repeat(X[s:e, None, None, :], d, axis=1).repeat(r, axis=2)  # (k, d, r, d)
        Y[:, ii, :, ii] = per_point[s:e].transpose(2, 0, 1)  # block i: feature i takes its r values
        det = model.predict_details(Y.reshape(-1, d))
        a = np.repeat(det0.cls[s:e], d * r)
        flip[s:e] = (det.cls.reshape(e - s, d, r) != det0.cls[s:e, None, None]).mean(axis=2)
        drop[s:e] = m0[s:e, None] - margin(det, a).reshape(e - s, d, r).mean(axis=2)
    lo, hi = drop.min(axis=1, keepdims=True), drop.max(axis=1, keepdims=True)
    norm = np.divide(drop - lo, hi - lo, out=np.zeros_like(drop), where=(hi - lo) > 0)
    return flip + norm * (0.5 / r)


# ------------------------------------------------- KernelSHAP on a chosen game
def shapley_kernel_coalitions(d: int, nsamples: int, rng):
    """Coalitions (k, d) bool and their regression weights for KernelSHAP.

    * ``nsamples >= 2^d - 2``: every non-empty proper coalition, weighted by
      the Shapley kernel (d - 1) / (C(d, s) s (d - s)); KernelSHAP is then exact.
    * otherwise ``nsamples`` coalitions in complementary pairs: the size s is
      drawn with probability proportional to (d - 1) / (s (d - s)) and the
      members uniformly, so coalitions are drawn in proportion to the kernel
      and all get weight 1 (paired sampling, Covert and Lee 2021).
    """
    if d < 2:
        raise ValueError("KernelSHAP needs at least 2 features")
    if nsamples >= (1 << d) - 2:
        masks = np.arange(1, (1 << d) - 1)
        Z = ((masks[:, None] >> np.arange(d)) & 1).astype(bool)
        s = Z.sum(axis=1)
        comb = np.array([math.comb(d, int(k)) for k in s], float)
        return Z, (d - 1) / (comb * s * (d - s))
    sizes = np.arange(1, d)
    p = (d - 1) / (sizes * (d - sizes))
    half = max(1, nsamples // 2)
    draw = rng.choice(sizes, size=half, p=p / p.sum())
    Z = np.zeros((2 * half, d), dtype=bool)
    for k, s in enumerate(draw):
        Z[2 * k, rng.choice(d, int(s), replace=False)] = True
    Z[1::2] = ~Z[0::2]
    return Z, np.ones(2 * half)


def kernel_solve(Z, w, y, total) -> np.ndarray:
    """phi minimising sum_k w_k (y_k - Z_k . phi)^2 subject to sum(phi) = total.

    The last feature is eliminated with the constraint, as shap does, and the
    rest is solved by least squares.
    """
    Z = np.asarray(Z, float)
    last = Z[:, -1]
    A = Z[:, :-1] - last[:, None]
    b = np.asarray(y, float) - last * total
    sw = np.sqrt(np.asarray(w, float))
    sol = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
    return np.r_[sol, total - sol.sum()]


def kernel_shap_game(model: CFMNN, X, background, nsamples: int = 200, seed: int = 0,
                     sigma: float | None = None, scale=None) -> np.ndarray:
    """KernelSHAP of the predicted class's margin on a chosen game, shape (n, d).

    A black-box estimator: every coalition's value comes from model calls on
    hybrid inputs (x on the coalition, one background row elsewhere), one per
    background row. ``sigma=None`` is the interventional game (rows weighted
    equally, as shap's KernelExplainer does); a number gives the conditional
    game of ``PaceShap(sigma=...)``, with each row weighted by
    exp(-D_S^2 / (2 sigma^2)) on the coalition's features. Coalitions come from
    ``shapley_kernel_coalitions`` (new ones for each point, from ``seed``);
    the efficiency constraint is enforced exactly. With every coalition it is
    exact and equals PACE-SHAP (or PACE-SHAP-cond) on the same rows.
    """
    return kernel_shap_games(model, X, background, nsamples, seed, (sigma,), scale)[0]


def kernel_shap_games(model: CFMNN, X, background, nsamples: int = 200, seed: int = 0,
                      sigmas=(None,), scale=None, times: dict | None = None) -> list[np.ndarray]:
    """``kernel_shap_game`` for several games at once: one (n, d) array per sigma.

    The games differ only in how the background rows are weighted, so they
    share the coalitions and the model calls, which are most of the cost. Each
    result is identical to ``kernel_shap_game`` with that sigma and seed.
    ``times``, if a dict, gets ``"own"``: each game's own seconds (its weights,
    values and regression); the rest of the time is shared.
    """
    from .explain import cond_scale
    X = model._check_X(X)
    B = model._check_X(background)
    n, d = X.shape
    R = len(B)
    if any(s is not None for s in sigmas):
        scale = cond_scale(B) if scale is None else np.asarray(scale, float)
    cls = model.predict_details(X).cls
    rng = np.random.default_rng(seed)
    out = [np.zeros((n, d)) for _ in sigmas]
    spent = np.zeros(len(sigmas))
    for p in range(n):
        x, a = X[p], int(cls[p])
        Z, w = shapley_kernel_coalitions(d, nsamples, rng)
        k = len(Z)
        hybrid = np.where(Z[:, None, :], x, B[None, :, :]).reshape(k * R, d)
        f = margin_score(model, hybrid, a).reshape(k, R)
        f0 = margin_score(model, B, a).mean()
        fx = margin_score(model, x[None], a)[0]
        for g, sigma in enumerate(sigmas):
            t0 = time.perf_counter()
            if sigma is None:
                v = f.mean(axis=1)
            else:
                q = ((x - B) / scale) ** 2                               # (R, d)
                logw = -(Z.astype(float) @ q.T) / (2.0 * sigma ** 2 * Z.sum(axis=1)[:, None])
                wt = np.exp(logw - logw.max(axis=1, keepdims=True))
                v = (wt * f).sum(axis=1) / wt.sum(axis=1)
            out[g][p] = kernel_solve(Z, w, v - f0, fx - f0)
            spent[g] += time.perf_counter() - t0
    if times is not None:
        times["own"] = spent.tolist()
    return out
