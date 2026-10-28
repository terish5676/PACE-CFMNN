"""XAI baselines on the same model, inputs and splits as PACE (blueprint 9.3).

CFMNN has no probabilities, so baselines explain a pseudo-probability: the
rescaled class supports s'_c normalised to sum to 1. This is a convenience
score for the baselines, not a calibrated probability.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .cfmnn_core import CFMNN


def class_scores(model: CFMNN, X) -> np.ndarray:
    """(n, C) rescaled supports normalised per row (uniform if all zero)."""
    s = model.predict_details(np.atleast_2d(X)).support_rescaled
    tot = s.sum(axis=1, keepdims=True)
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1.0 / s.shape[1])


def kernel_shap(model: CFMNN, X, background, nsamples=200, seed=0) -> np.ndarray:
    """KernelSHAP attributions for the predicted class, shape (n, d).

    shap draws its coalition samples from numpy's global random state, so the
    ``seed`` given to KernelExplainer alone does not make results repeatable.
    The global state is seeded for this call and restored afterwards.
    """
    import shap
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


def lime_tabular(model: CFMNN, X, X_train, num_samples=1000, seed=0) -> np.ndarray:
    """LIME attributions (local linear weights) for the predicted class, shape (n, d)."""
    from lime.lime_tabular import LimeTabularExplainer
    d = X_train.shape[1]
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


def dice_counterfactual(model: CFMNN, x, X_train, y_train, target, seed=0, locked=()):
    """One DiCE (random method) counterfactual for x, or None."""
    import dice_ml
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
    exp = dice_ml.Dice(d, m, method="random")
    vary = [n for i, n in enumerate(names) if i not in set(locked)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = exp.generate_counterfactuals(pd.DataFrame([x], columns=names), total_CFs=1,
                                             desired_class=classes.index(target),
                                             features_to_vary=vary, random_seed=seed)
        cf = r.cf_examples_list[0].final_cfs_df
        return None if cf is None or len(cf) == 0 else cf[names].to_numpy(float)[0]
    except Exception:
        return None


def occlusion(model: CFMNN, X, background, n_quantiles: int | None = 32) -> np.ndarray:
    """Brute-force single-feature occlusion with the same replacement values and
    scoring as PACE-flip, but computed by re-running the full model on every
    modified input. It must equal ``pace_flip`` exactly; it exists to measure
    PACE-flip's speed-up, not as a different explanation.
    """
    X = model._check_X(X)
    B = model._check_X(background)
    if n_quantiles is not None:
        B = np.quantile(B, (np.arange(n_quantiles) + 0.5) / n_quantiles, axis=0)
    n, d = X.shape
    r = len(B)
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
        for i in range(d):
            Y = np.repeat(X[p][None], r, axis=0)
            Y[:, i] = B[:, i]
            det = model.predict_details(Y)
            flip[p, i] = np.mean(det.cls != det0.cls[p])
            drop[p, i] = m0[p] - margin(det, a).mean()
    lo, hi = drop.min(axis=1, keepdims=True), drop.max(axis=1, keepdims=True)
    norm = np.divide(drop - lo, hi - lo, out=np.zeros_like(drop), where=(hi - lo) > 0)
    return flip + norm * (0.5 / r)
