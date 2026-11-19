"""Proposition 5: exact PACE-SHAP values are #P-hard, by a reduction from voting games.

A weighted voting game has players 1..n with positive integer weights w_i and a
quota q; a coalition S wins when w(S) = sum_{i in S} w_i >= q. A player's
Shapley value in this game is its Shapley-Shubik index SS_i. Computing SS_i is
#P-complete when the weights are written in binary (Deng and Papadimitriou
1994: #P-metric-complete; Faliszewski and Hemaspaandra 2008: many-one complete
for #P).

``voting_game_model`` builds, in time polynomial in the input, a CFMNN with
two classes and four boxes, one explained point x and one background row z,
such that for every coalition S of features

    g_A(x_S, z_rest) = 2 eps * [w(S) >= q],

where g_A is the margin PACE-SHAP explains and eps = 1 / (4 n max_i w_i).
Hence PACE-SHAP's value of weight feature i is 2 eps * SS_i, and every padding
feature gets 0. A polynomial-time algorithm for exact PACE-SHAP values would
therefore compute Shapley-Shubik indices in polynomial time.

Construction (d = 2n features: n weight features, then n padding features;
boxes a1, a2 of class A = 0 and b1, b2 of class B = 1; r = 1/gamma):

* weight feature i: a1 and b1 span [0.5, 0.5 + r]; x_i = 0.5 is inside, and
  z_i = 0.5 - t_i lies below them with t_i = 2 d eps w_i / gamma <= r, so the
  term drops by gamma t_i / 2 and Delta_{a1,i} = Delta_{b1,i} = eps w_i
  (Proposition 2, with its 1/d). a2 and b2 span [0, 1], so their Delta is 0.
* padding feature k: x_k = z_k = 0.5, so no box changes with it. Boxes start
  at 0.5 + s with s = 2 (1 - p) / gamma, which sets their term at 0.5 to p.
  With p = 1/2 for a2 and b2, 1/2 + d eps (W - q + 1) / n for a1 and
  1/2 + d eps (W - q) / n for b1 (W = sum of weights), the memberships at z
  are b_{a2} = b_{b2} = K, b_{a1} = K - eps (q - 1) and b_{b1} = K - eps q.

Then s_A(S) = max(K + eps (w(S) - q + 1), K) and s_B(S) = max(K + eps (w(S) - q), K),
which differ by eps exactly when w(S) >= q (w(S) is an integer) and are equal
otherwise. The margin is 2 (s_A - s_B).

The hardness needs weights in binary. With weights bounded by a polynomial in
n, ``shapley_shubik_dp`` counts winning coalitions by size and weight in
pseudo-polynomial time, so the statement is "#P-hard in general".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN


@dataclass
class VotingGameModel:
    model: CFMNN          # four boxes: a1, a2 (class 0) and b1, b2 (class 1)
    x: np.ndarray         # explained point, shape (1, 2n)
    z: np.ndarray         # the one background row, shape (1, 2n)
    eps: float            # phi_i = 2 eps SS_i on weight features
    weights: np.ndarray
    quota: int


def _check_game(weights, quota) -> tuple[np.ndarray, int]:
    w = np.asarray(weights)
    if w.ndim != 1 or len(w) == 0 or not np.issubdtype(w.dtype, np.integer) or (w <= 0).any():
        raise ValueError("weights must be a non-empty 1-D array of positive integers")
    q = int(quota)
    if not 1 <= q <= int(w.sum()):
        raise ValueError("quota must lie in [1, sum of weights]")
    return w.astype(np.int64), q


def voting_game_model(weights, quota, gamma: float = 4.0) -> VotingGameModel:
    """The CFMNN of Proposition 5 for the voting game (weights; quota)."""
    w, q = _check_game(weights, quota)
    if gamma < 2.0:
        raise ValueError("gamma >= 2 keeps every coordinate in [0, 1]")
    n = len(w)
    d = 2 * n
    W_tot = int(w.sum())
    eps = 1.0 / (4.0 * n * int(w.max()))
    r = 1.0 / gamma

    t = 2.0 * d * eps * w / gamma                       # <= r by the choice of eps
    x = np.full(d, 0.5)
    z = x.copy()
    z[:n] = 0.5 - t

    def pad_start(p: float) -> float:                   # box [0.5 + s, 1] scores p at 0.5
        return 0.5 + 2.0 * (1.0 - p) / gamma

    p_a1 = 0.5 + d * eps * (W_tot - q + 1) / n
    p_b1 = 0.5 + d * eps * (W_tot - q) / n
    V = np.zeros((4, d))
    Wc = np.ones((4, d))
    for k, (lo_w, hi_w, p) in enumerate([(0.5, 0.5 + r, p_a1),   # a1
                                         (0.0, 1.0, 0.5),        # a2
                                         (0.5, 0.5 + r, p_b1),   # b1
                                         (0.0, 1.0, 0.5)]):      # b2
        V[k, :n], Wc[k, :n] = lo_w, hi_w
        V[k, n:] = pad_start(p)
    model = CFMNN.from_boxes(V, Wc, box_class=[0, 0, 1, 1], gamma=gamma)
    return VotingGameModel(model, x[None, :], z[None, :], eps, w, q)


def shapley_shubik(weights, quota) -> np.ndarray:
    """Shapley-Shubik index of every player by enumerating coalitions (small n only)."""
    w, q = _check_game(weights, quota)
    n = len(w)
    if n > 20:
        raise ValueError("enumeration is for n <= 20; use shapley_shubik_dp")
    fact = [math.factorial(k) for k in range(n + 1)]
    out = np.zeros(n)
    masks = np.arange(1 << n)
    bits = (masks[:, None] >> np.arange(n)[None, :]) & 1
    ws = bits @ w
    size = bits.sum(axis=1)
    for i in range(n):
        lo = bits[:, i] == 0
        pivotal = lo & (ws < q) & (ws + w[i] >= q)
        out[i] = sum(fact[s] * fact[n - s - 1] for s in size[pivotal]) / fact[n]
    return out


def shapley_shubik_dp(weights, quota) -> np.ndarray:
    """The same indices by dynamic programming: O(n^2 * sum of weights) per player.

    Pseudo-polynomial: fast when the weights are small, which is why the
    hardness of Proposition 5 needs weights written in binary.
    """
    w, q = _check_game(weights, quota)
    n = len(w)
    fact = [math.factorial(k) for k in range(n + 1)]
    out = np.zeros(n)
    for i in range(n):
        others = np.delete(w, i)
        # count[s, t] = number of coalitions of the others with s members and weight t
        count = np.zeros((n, q), dtype=object)
        count[0, 0] = 1
        for wj in others:
            for s in range(n - 1, 0, -1):
                if wj < q:
                    count[s, wj:] = count[s, wj:] + count[s - 1, :q - wj]
        total = 0
        for s in range(n):
            lo = max(0, q - int(w[i]))
            total += fact[s] * fact[n - s - 1] * int(sum(count[s, lo:q]))
        out[i] = total / fact[n]
    return out
