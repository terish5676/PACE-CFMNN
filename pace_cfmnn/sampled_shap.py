"""Two modern sampled Shapley estimators, as black-box baselines for PACE-SHAP (Run J).

Both estimate the game PACE-SHAP computes: the interventional Shapley values of the predicted
class's margin, with the background rows as reference. Both see the model only through calls
on hybrid inputs (x on a coalition, one background row elsewhere), and every row they send to
the model is counted, so the methods can be given the same budget.

* ``permutation_shap``: ``shap.PermutationExplainer`` (antithetic feature orders, each run
  forward and backward; the same orders for every background row).
* ``paired_kernel_shap``: KernelSHAP with paired sampling (Covert and Lee, "Improving
  KernelSHAP: Practical Shapley Value Estimation Using Linear Regression", AISTATS 2021,
  PMLR 130:3457-3465; the arXiv version, 2012.01536, says "via Linear Regression").
  Coalitions are drawn from the Shapley kernel together with their complements, and the
  values come from least squares under the efficiency constraint. The sampler and the solver
  are ``xai_baselines.shapley_kernel_coalitions`` and ``kernel_solve``, which the conditional
  run already uses; this function adds the row count, a lean model call and chunking.

``MarginModel`` is the model call both use. It gives the same number as
``xai_baselines.margin_score`` when every class has a box, without the centroid distances and
per-class bookkeeping a prediction needs, so the baselines are not slowed by work the margin
does not use.
"""

from __future__ import annotations

import warnings

import numpy as np

from .cfmnn_core import CFMNN
from .xai_baselines import kernel_solve, shapley_kernel_coalitions


class MarginModel:
    """f(Z) = margin of class index ``a`` at each row of Z; ``rows`` counts the rows evaluated.

    The margin is s'_A - max over other classes of s'_c, with s' = 2 * (best membership) - 1,
    as in ``PaceShap._margin``. With a single class it is s'_A.
    """

    def __init__(self, model: CFMNN, a: int):
        is_a = model.box_class_ == a
        self._va, self._wa = model.V_[is_a], model.W_[is_a]
        self._vo, self._wo = model.V_[~is_a], model.W_[~is_a]
        self._gamma = float(model.gamma)
        self._step = model._chunk_size()
        self._d = model.n_features_in_
        self.rows = 0

    def _best(self, Zc, V, W) -> np.ndarray:
        """Highest membership of each row of Zc (chunk, 1, d) in the boxes (V, W).

        A point is outside a box on at most one side of a feature, so the feature term
        [f(x - w) + f(v - x)] / 2 equals 1 - min(1, gamma * min(1, e)) / 2 with
        e = max(x - w, v - x, 0). Same value as ``feature_terms``, with fewer passes.
        """
        e = np.maximum(Zc - W, V - Zc)
        np.clip(e, 0.0, 1.0, out=e)
        e *= self._gamma
        np.minimum(e, 1.0, out=e)
        return 1.0 - 0.5 * e.mean(axis=2).min(axis=1)

    def __call__(self, Z) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        if Z.shape[1] != self._d:
            raise ValueError(f"expected {self._d} features, got {Z.shape[1]}")
        self.rows += len(Z)
        out = np.empty(len(Z))
        for s in range(0, len(Z), self._step):
            Zc = Z[s:s + self._step, None, :]
            ga = self._best(Zc, self._va, self._wa)
            if len(self._vo):
                out[s:s + self._step] = 2.0 * ga - 2.0 * self._best(Zc, self._vo, self._wo)
            else:
                out[s:s + self._step] = 2.0 * ga - 1.0
        return out


def paired_kernel_shap(model: CFMNN, X, background, n_coalitions: int, seed: int = 0,
                       chunk_elems: int = 4_000_000, return_rows: bool = False):
    """KernelSHAP with paired sampling for the predicted class's margin, shape (n, d).

    For each point, ``n_coalitions`` coalitions are drawn in complementary pairs in proportion
    to the Shapley kernel (an odd number is rounded down to even; with ``n_coalitions >=
    2^d - 2`` every coalition is used and the result is exact). Each coalition's value is the
    mean of the margin over the background rows; the empty and the full coalition are evaluated
    too, and the values solve the least-squares problem under sum(phi) = f(x) - mean f(z).

    Model rows per explanation: ``k * R + R + 1`` for k coalitions and R background rows.
    ``return_rows=True`` also returns the rows actually sent to the model, per point.
    """
    X = model._check_X(X)
    B = model._check_X(background)
    n, d = X.shape
    R = len(B)
    cls = model.predict_details(X).cls
    rng = np.random.default_rng(seed)
    step = max(1, int(chunk_elems // (R * d)))
    phi = np.zeros((n, d))
    used = np.zeros(n, dtype=np.int64)
    for p in range(n):
        x = X[p]
        f = MarginModel(model, int(cls[p]))
        Z, w = shapley_kernel_coalitions(d, n_coalitions, rng)
        v = np.empty(len(Z))
        for s in range(0, len(Z), step):
            Zc = Z[s:s + step]
            hybrid = np.where(Zc[:, None, :], x, B[None, :, :]).reshape(len(Zc) * R, d)
            v[s:s + step] = f(hybrid).reshape(len(Zc), R).mean(axis=1)
        f0 = f(B).mean()
        fx = f(x[None])[0]
        phi[p] = kernel_solve(Z, w, v - f0, fx - f0)
        used[p] = f.rows
    return (phi, used) if return_rows else phi


def permutation_shap(model: CFMNN, X, background, max_evals: int, seed: int = 0,
                     return_rows: bool = False):
    """``shap.PermutationExplainer`` for the predicted class's margin, shape (n, d).

    ``max_evals`` is shap's budget of masks per explanation; each mask is evaluated on the
    background rows (shap skips a row whose masked input did not change, so the rows sent to
    the model can be fewer than ``max_evals * R``). shap runs ``max_evals // (2 m + 1)`` random
    orders, each forward and backward, where m is the number of features on which x differs
    from the background. Every background row is used (shap's default keeps 100).

    Points are explained one per call. Given several points at once, shap 0.54 also computes
    "main effects" (d + 1 more masks) for every point after the first, although they were not
    asked for and are not used; one point per call avoids that, so the rows sent to the model
    are at most ``max_evals * R``.

    shap shuffles with numpy's global random state, so that state is seeded for this call and
    restored afterwards, as ``xai_baselines.kernel_shap`` does. Points are grouped by predicted
    class. ``return_rows=True`` also returns the mean rows sent to the model per point.
    """
    import shap
    X = model._check_X(X)
    B = model._check_X(background)
    cls = model.predict_details(X).cls
    out = np.zeros(X.shape)
    masker = shap.maskers.Independent(B, max_samples=len(B))
    rows_used = 0
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        for a in np.unique(cls):
            rows = cls == a
            f = MarginModel(model, int(a))
            ex = shap.explainers.Permutation(f, masker, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for p in np.flatnonzero(rows):      # one point per call: see the docstring
                    out[p] = ex(X[p:p + 1], max_evals=int(max_evals), silent=True).values[0]
            rows_used += f.rows
    finally:
        np.random.set_state(state)
    return (out, rows_used / len(X)) if return_rows else out
