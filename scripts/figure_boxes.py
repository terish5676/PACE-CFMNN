"""Figure: CFMNN boxes in 2-D, a tied training error, the shrink that fixes it and its certificate.

Fits CFMNN (theta = 0.3, the paper's setting) to a small 2-D problem with two overlapping
classes, finds a training row that the model gets wrong while it lies inside a wrong-class
box (Proposition 1 says every training error is such a tie), shrinks that box with the
self-repair rule (the feature that cuts the smallest share of the box's width), and shades
the certificate: the inputs whose prediction the edit can change (``audit.impact_set``).

    python -B scripts/figure_boxes.py [--out results/peer_review/figure_boxes.png]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.audit import impact_set  # noqa: E402
from pace_cfmnn.refine import feature_volume, shrink_box, shrink_target  # noqa: E402

COLOURS = ("#2a6fb0", "#d9822b")      # class 0 blue, class 1 orange
GRID = 300


def problem(seed: int, n: int = 70):
    """Two overlapping classes in the unit square."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    centre = np.where(y[:, None] == 0, [0.38, 0.42], [0.62, 0.58])
    X = np.clip(centre + rng.normal(0, 0.15, (n, 2)), 0, 1)
    return X, y


def find_case(max_seed: int = 200):
    """First seed with a training error inside a wrong-class box that one shrink fixes."""
    for seed in range(max_seed):
        X, y = problem(seed)
        m = CFMNN(theta=0.3).fit(X, y)
        if not 4 <= m.n_boxes_ <= 14:
            continue
        det = m.predict_details(X)
        y_idx = np.searchsorted(m.classes_, y)
        for r in np.flatnonzero(det.y_pred != y):
            k = shrink_target(m, X[r], int(y_idx[r]))
            if k < 0:
                continue
            i = feature_volume(m, X[r], k, int(det.cls[r]), int(y_idx[r]))
            after = copy.deepcopy(m)
            info = shrink_box(after, k, X[r], i, X)
            if info["removed"] or after.predict(X[r][None])[0] != y[r]:
                continue
            return seed, X, y, m, after, r, k, i, info
    raise RuntimeError("no suitable case found")


def draw_boxes(ax, model, highlight=None):
    for k in range(model.n_boxes_):
        c = COLOURS[int(model.box_class_[k])]
        v, w = model.V_[k], model.W_[k]
        lw = 2.8 if k == highlight else 0.9
        face = matplotlib.colors.to_rgba(c, 0.08)
        edge = matplotlib.colors.to_rgba(c, 1.0 if k == highlight else 0.55)
        ax.add_patch(Rectangle(v, *(w - v), facecolor=face, edgecolor=edge, lw=lw,
                               zorder=2 if k == highlight else 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join("results", "peer_review", "figure_boxes.png"))
    args = ap.parse_args()

    seed, X, y, before, after, r, k, i, info = find_case()
    g = np.linspace(0, 1, GRID)
    G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    cert = impact_set(before, after, G).reshape(GRID, GRID)
    changed = (before.predict(G) != after.predict(G)).reshape(GRID, GRID)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, model, title in ((axes[0], before, "(a) Trained CFMNN: a tied training error"),
                             (axes[1], after, "(b) After one shrink, with its certificate")):
        if model is after:
            ax.contourf(g, g, cert.astype(float), levels=[0.5, 1.5], colors=["#9a9a9a"], alpha=0.30)
            ax.contour(g, g, changed.astype(float), levels=[0.5], colors="black", linewidths=0.8,
                       linestyles="dotted")
        draw_boxes(ax, model, highlight=k)
        for c in (0, 1):
            ax.scatter(*X[y == c].T, s=14, color=COLOURS[c], edgecolor="white", lw=0.4, zorder=3)
        ax.scatter(*X[r], s=140, facecolor="none", edgecolor="black", lw=1.6, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("feature 1")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("feature 2")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"seed {seed}: {before.n_boxes_} boxes; row {r} (class {y[r]}) inside wrong box {k}; "
          f"cut feature {i + 1}, {info['n_cut']} rows cut off; certificate covers "
          f"{cert.mean():.1%} of the square, prediction changed on {changed.mean():.1%}; "
          f"changed outside certificate: {int((changed & ~cert).sum())}; wrote {args.out}")


if __name__ == "__main__":
    main()
