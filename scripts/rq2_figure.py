"""The key figure for RQ2 (tough review B5): accuracy, size and broken predictions across theta.

One panel per dataset, from committed results only (seeds 70-89):

* gray line: CFMNN's mean test accuracy against its mean number of boxes for theta from 0.5
  (left, fewest boxes) to 0.05, from ``results/theta_churn/runs.csv``;
* black ring: CFMNN at theta = 0.3, the model the paper explains and repairs;
* hollow square: CFMNN at theta chosen by validation (per seed; the mean position);
* blue: self-repair at theta = 0.3; orange: self-repair at the chosen theta
  (``results/repair_theta/runs.csv``); a thin arrow joins each repair to its starting model.

A marker's area is its negative flip rate: the share of test rows that CFMNN at theta = 0.3 got
right and the model gets wrong. For self-repair at the chosen theta it is against the model it
repairs (``test_broken`` in the repair run).

    python -B scripts/rq2_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
CURVE = "#8a8984"
SR03, SRVAL = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated, light)
NAMES = {"S1": "S1", "S2": "S2", "S3": "S3", "wine": "Wine", "breast_cancer": "Breast Cancer",
         "digits": "Digits", "thyroid": "Thyroid", "shuttle": "Shuttle", "skin": "Skin",
         "magic": "MAGIC", "letter": "Letter", "landsat": "Landsat", "page_blocks": "Page Blocks"}


def area(nfr):
    """Marker area (points^2) from the negative flip rate in percent; 0 still shows."""
    return 10 + 9 * np.asarray(nfr, float)


def frames(th: pd.DataFrame, rt: pd.DataFrame):
    curve = th.groupby(["dataset", "theta"])[["accuracy", "boxes", "nfr"]].mean().reset_index()
    val = th[th.is_val].groupby("dataset")[["accuracy", "boxes", "nfr"]].mean()
    r = rt.assign(nfr=100 * rt.test_broken / rt.n_test)
    pts = r.groupby(["dataset", "model"])[["accuracy", "boxes", "nfr"]].mean()
    return curve, val, pts


def draw(th: pd.DataFrame, rt: pd.DataFrame, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from accuracy_table import DATASETS

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "svg.fonttype": "none"})
    curve, val, pts = frames(th, rt)
    ds = [d for d in DATASETS if d in set(curve.dataset)]
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.6), facecolor=SURFACE)
    for ax, d in zip(axes.flat, ds):
        c = curve[curve.dataset == d].sort_values("theta", ascending=False)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.plot(c.boxes, c.accuracy, color=CURVE, linewidth=1.4, zorder=2, solid_capstyle="round")
        ax.scatter(c.boxes, c.accuracy, s=area(c.nfr), color=CURVE, edgecolor=SURFACE, linewidth=0.8,
                   zorder=3)
        b0 = c[np.isclose(c.theta, 0.3)].iloc[0]
        ax.scatter([b0.boxes], [b0.accuracy], s=area(0) + 18, facecolor="none", edgecolor=INK,
                   linewidth=1.0, zorder=5)
        v = val.loc[d]
        ax.scatter([v.boxes], [v.accuracy], s=area(v.nfr), marker="s", facecolor=SURFACE,
                   edgecolor=INK2, linewidth=1.0, zorder=4)
        for model, start, color in (("SR-0.3", (b0.boxes, b0.accuracy), SR03),
                                    ("SR-val", (v.boxes, v.accuracy), SRVAL)):
            p = pts.loc[(d, model)]
            if abs(p.accuracy - start[1]) > 0.05:
                ax.annotate("", xy=(p.boxes, p.accuracy), xytext=start, zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=3, shrinkB=3,
                                            mutation_scale=6))
            ax.scatter([p.boxes], [p.accuracy], s=area(p.nfr), color=color, edgecolor=SURFACE,
                       linewidth=0.8, zorder=6)
        ax.set_xscale("log")
        xs = np.r_[c.boxes, v.boxes, pts.loc[d].boxes]
        lx0, lx1 = np.log10(xs.min()), np.log10(xs.max())
        span = max(lx1 - lx0, 0.3)
        ax.set_xlim(10 ** (lx0 - 0.12 * span), 10 ** (lx1 + 0.12 * span))
        ax.set_title(NAMES.get(d, d), color=INK, fontsize=8, loc="left", pad=3)
        lo = min(c.accuracy.min(), pts.loc[d].accuracy.min())
        hi = max(c.accuracy.max(), pts.loc[d].accuracy.max())
        pad = max(0.4, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(length=2, pad=1.5, labelsize=6.5)
        ax.minorticks_off()
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.flat[len(ds):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test accuracy (%)", fontsize=7)
    for i, ax in enumerate(axes.flat[:len(ds)]):
        if i >= len(ds) - 4:
            ax.set_xlabel("Boxes (log scale)", fontsize=7)
    lg = axes.flat[len(ds)]
    handles = [Line2D([], [], color=CURVE, marker="o", markersize=4, linewidth=1.4,
                      label="CFMNN over θ (0.5 → 0.05)"),
               Line2D([], [], color=INK, marker="o", markerfacecolor="none", markersize=6, linewidth=0,
                      label="CFMNN, θ = 0.3"),
               Line2D([], [], color=INK2, marker="s", markerfacecolor=SURFACE, markersize=5, linewidth=0,
                      label="CFMNN, θ by validation"),
               Line2D([], [], color=SR03, marker="o", markersize=5, linewidth=0,
                      label="Self-repair, θ = 0.3"),
               Line2D([], [], color=SRVAL, marker="o", markersize=5, linewidth=0,
                      label="Self-repair, θ by validation")]
    lg.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.8, handletextpad=0.6,
              borderaxespad=0, labelcolor=INK)
    sz = axes.flat[len(ds) + 1]
    for k, n in enumerate((0, 1, 5, 10)):
        sz.scatter([0.15 + 0.22 * k], [0.55], s=area(n), color=CURVE, edgecolor=SURFACE)
        sz.text(0.15 + 0.22 * k, 0.32, f"{n}%", ha="center", va="top", color=INK2, fontsize=6.5)
    sz.text(0.0, 0.95, "Marker area: test predictions\nCFMNN at θ = 0.3 got right\nthat the model gets wrong",
            va="top", color=INK2, fontsize=6.5, transform=sz.transAxes)
    sz.set_xlim(0, 1)
    sz.set_ylim(0, 1)
    sz.axis("off")
    fig.text(0.01, 0.005, "Means over seeds 70–89. Arrows join each repair to the model it starts from "
             "(omitted where repair changed accuracy by under 0.05 points).", color=MUTED, fontsize=6.2)
    fig.tight_layout(rect=(0, 0.015, 1, 1), h_pad=0.9, w_pad=0.6)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "rq2_theta.png"
    fig.savefig(png, dpi=220, facecolor=SURFACE)
    fig.savefig(out / "rq2_theta.svg", facecolor=SURFACE)
    plt.close(fig)
    return png


if __name__ == "__main__":
    res = ROOT / "results"
    print(draw(pd.read_csv(res / "theta_churn" / "runs.csv"), pd.read_csv(res / "repair_theta" / "runs.csv"),
               res / "tough_review"))
