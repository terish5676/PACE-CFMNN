"""Figures and numbers for docs/RQ3_self_repair.md.

Every toy figure is drawn from our own code (CFMNN, refine, active, audit), not
by hand, and every fact the document states about a toy is checked by an
assert here. The result figures and numbers are recomputed from the committed
RQ3 run (results/refinement_v2_confirmatory/), which is only read.

    python -B docs/rq3_figures.py

Writes docs/figures/rq3_*.png and docs/figures/rq3_numbers.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.audit import certify, top_groups  # noqa: E402
from pace_cfmnn.cfmnn_core import feature_terms  # noqa: E402
from pace_cfmnn.refine import MARGIN, face_cut, shrink_box, shrink_target  # noqa: E402

FIG = ROOT / "docs" / "figures"
RUN = ROOT / "results" / "refinement_v2_confirmatory"
GAMMA = 4.0
BLUE, ORANGE, RED, PURPLE, GREY = "#2b6cb0", "#dd6b20", "#c53030", "#6b46c1", "#718096"
NAMES = {"thyroid": "Thyroid", "landsat": "Landsat", "page_blocks": "Page Blocks",
         "magic": "MAGIC", "letter": "Letter", "skin": "Skin"}
ORDER = list(NAMES)
LINES: list[str] = []


def note(text: str) -> None:
    LINES.append(text)
    print(text)


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def membership_1d(x, v, w):
    return feature_terms(np.asarray(x, float)[:, None], np.array([v]), np.array([w]), GAMMA)[:, 0]


def draw_box(ax, model, j, color, ls="-", lw=2.0, alpha=0.12, label=None):
    v, w = model.V_[j], model.W_[j]
    ax.add_patch(Rectangle(v, *(w - v), fill=True, facecolor=color, alpha=alpha, edgecolor="none"))
    ax.add_patch(Rectangle(v, *(w - v), fill=False, edgecolor=color, lw=lw, ls=ls, label=label))


# ---------------------------------------------------------------- figure 1
def fig_membership():
    v, w = 0.40, 0.60
    xs = np.linspace(0, 1, 1001)
    m = membership_1d(xs, v, w)
    assert np.allclose(m[(xs >= v) & (xs <= w)], 1.0)
    assert np.isclose(membership_1d([w + 0.05], v, w)[0], 1 - GAMMA * 0.05 / 2)
    assert np.isclose(membership_1d([w + 0.30], v, w)[0], 0.5)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axvspan(v, w, color=BLUE, alpha=0.12)
    ax.plot(xs, m, color=BLUE, lw=2.5)
    ax.annotate("inside the box:\nmembership 1", (0.5, 1.0), (0.5, 0.80), ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.annotate(f"falls by γ/2 = {GAMMA / 2:g}\nper unit of distance", (0.66, membership_1d([0.66], v, w)[0]),
                (0.76, 0.86), arrowprops=dict(arrowstyle="->", color=GREY))
    ax.annotate("floor 0.5 once the distance\nis 1/γ = 0.25 or more", (0.08, 0.5), (0.02, 0.90),
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set(xlim=(0, 1), ylim=(0.45, 1.05), xlabel="feature value x", ylabel="membership of the box",
           title="One box [0.40, 0.60] on one feature (γ = 4)")
    ax.text(v, 0.47, " v", color=BLUE)
    ax.text(w, 0.47, " w", color=BLUE)
    save(fig, "rq3_1_membership.png")


# ---------------------------------------------------------------- figure 2
def fig_order():
    a_rows = [0.10, 0.35, 0.60]
    b_row, test = 0.45, 0.42
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.6), sharex=True)
    preds = []
    for ax, order in zip(axes, ([0, 1, 2], [1, 2, 0])):
        X = np.array([[a_rows[i]] for i in order] + [[b_row]])
        y = np.array([0, 0, 0, 1])
        m = CFMNN(theta=0.3).fit(X, y)
        p = int(m.predict(np.array([[test]]))[0])
        preds.append(p)
        boxes = sorted((m.V_[j, 0], m.W_[j, 0], int(m.box_class_[j])) for j in range(m.n_boxes_))
        for lo, hi, c in boxes:
            col = BLUE if c == 0 else ORANGE
            ax.add_patch(Rectangle((lo, 0.35), max(hi - lo, 0.006), 0.3, facecolor=col, alpha=0.3,
                                   edgecolor=col, lw=2))
        ax.scatter(a_rows, [0.5] * 3, color=BLUE, zorder=3, s=40)
        ax.scatter([b_row], [0.5], color=ORANGE, zorder=3, s=40, marker="s")
        ax.axvline(test, color=PURPLE, ls="--", lw=1.5)
        seq = ", ".join(f"{a_rows[i]:.2f}" for i in order)
        ax.set_title(f"Class A rows in the order {seq}: test point {test} → class {'AB'[p]}",
                     fontsize=10, loc="left")
        ax.set_yticks([])
        ax.set_ylim(0.2, 0.8)
    axes[-1].set_xlabel("feature value (θ = 0.3, so a box may be at most 0.3 wide)")
    axes[-1].set_xlim(0, 0.75)
    assert preds == [1, 0], preds
    save(fig, "rq3_2_order.png")
    note(f"order: same rows, two orders -> the test point at {test} is class B, then class A")


# ---------------------------------------------------------------- toy 2-D model
A_ROWS = [(0.10, 0.20), (0.38, 0.30), (0.15, 0.30), (0.22, 0.25), (0.28, 0.38), (0.12, 0.48),
          (0.37, 0.46), (0.30, 0.22)]
B_ROWS = [(0.30, 0.65), (0.58, 0.68), (0.50, 0.40), (0.40, 0.52), (0.45, 0.55), (0.55, 0.47),
          (0.42, 0.60), (0.39, 0.50)]
ERR = 6  # index in A_ROWS of the row CFMNN gets wrong


def toy():
    X = np.array(A_ROWS + B_ROWS)
    y = np.array([0] * len(A_ROWS) + [1] * len(B_ROWS))
    m = CFMNN(theta=0.3, gamma=GAMMA).fit(X, y)
    return X, y, m


def fig_tie(X, y, m):
    assert m.n_boxes_ == 2, m.n_boxes_
    x = X[ERR]
    det = m.predict_details(x[None])
    M = m.memberships(x[None])[0]
    assert np.allclose(M, 1.0), M                      # inside both boxes
    assert int(det.cls[0]) == 1 and y[ERR] == 0        # predicted B, true A
    errors = np.flatnonzero(m.predict(X) != y)
    assert errors.tolist() == [ERR], errors            # the only training error
    d2 = ((x - m.centroid_) ** 2).sum(axis=1)
    a, b = (int(np.flatnonzero(m.box_class_ == c)[0]) for c in (0, 1))
    assert d2[b] < d2[a]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    draw_box(ax, m, a, BLUE, label="box of class A")
    draw_box(ax, m, b, ORANGE, label="box of class B")
    ax.scatter(*X[y == 0].T, color=BLUE, s=35, zorder=3)
    ax.scatter(*X[y == 1].T, color=ORANGE, s=35, marker="s", zorder=3)
    ax.scatter(*x, s=220, facecolor="none", edgecolor=RED, lw=2.2, zorder=4, label="training row x (class A)")
    for j, col in ((a, BLUE), (b, ORANGE)):
        ax.scatter(*m.centroid_[j], marker="X", s=140, color=col, edgecolor="black", zorder=5)
        ax.plot(*zip(x, m.centroid_[j]), color=col, ls=":", lw=1.8)
        mid = (x + m.centroid_[j]) / 2
        ax.text(*mid, f" {np.sqrt(d2[j]):.3f}", color=col, fontsize=10)
    ax.text(0.02, 0.97, f"membership of x: box A = {M[a]:.3f}, box B = {M[b]:.3f} (a tie)\n"
            f"nearest centroid is B's, so x is predicted B: wrong",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor=GREY))
    ax.set(xlim=(0.0, 0.75), ylim=(0.1, 0.8), xlabel="feature 1", ylabel="feature 2",
           title="Illustration: a training error is a tie between two boxes")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    save(fig, "rq3_3_tie_error.png")
    note(f"toy: x = {tuple(x)}, distance to A's centroid {np.sqrt(d2[a]):.3f}, to B's {np.sqrt(d2[b]):.3f}")
    return a, b


def fig_shrink(X, y, m, a, b):
    x = X[ERR]
    k = shrink_target(m, x, 0)
    assert k == b
    cuts = [face_cut(m, k, x, i) for i in range(2)]
    shares = [c[2] for c in cuts]
    i_best = int(np.argmin(shares))
    # The real self-repair run on the toy: gate on the training rows, no new labels.
    sr, _, edits = run_self_repair(m, X, y, X, y, chooser="volume", budget=300, n_rounds=1, seed=0,
                                   gate_labels_are_new=False)
    acc = [e for e in edits if e["edit"] == "shrink"]
    assert len(acc) == 1 and acc[0]["accepted"] and acc[0]["shrink_feature"] == i_best
    side, new, share = cuts[i_best]
    after = sr.memberships(x[None])[0]
    d = X.shape[1]
    assert np.isclose(after[b], 1 - GAMMA * MARGIN / (2 * d))       # 1 - γε/(2d)
    assert np.isclose(after[a], 1.0)
    assert sr.predict(x[None])[0] == 0                               # x is now right
    assert (sr.predict(X) == y).all()                                # and no row is wrong
    members_before = set(m.members_[b].tolist())
    members_after = set(sr.members_[b].tolist())
    dropped = sorted(members_before - members_after)
    assert len(dropped) == acc[0]["shrink_n_cut"] >= 1
    assert all(r not in np.concatenate(sr.members_) for r in dropped)   # in no box now

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    titles = ["(a) Candidate cuts: one per feature", "(b) The thinnest cut is made",
              "(c) After: x is predicted A"]
    for ax, t in zip(axes, titles):
        ax.set(xlim=(0.0, 0.75), ylim=(0.1, 0.8), xlabel="feature 1", ylabel="feature 2")
        ax.set_title(t, loc="left", fontsize=11)
        ax.set_aspect("equal")
    # (a)
    ax = axes[0]
    draw_box(ax, m, a, BLUE)
    draw_box(ax, m, b, ORANGE)
    ax.scatter(*X[y == 0].T, color=BLUE, s=25, zorder=3)
    ax.scatter(*X[y == 1].T, color=ORANGE, s=25, marker="s", zorder=3)
    ax.scatter(*x, s=200, facecolor="none", edgecolor=RED, lw=2, zorder=4)
    v, w = m.V_[b], m.W_[b]
    for i, (sd, nw, sh) in enumerate(cuts):
        lo = (v[i], nw) if sd == "low" else (nw, w[i])
        if i == 0:
            rect = Rectangle((lo[0], v[1]), lo[1] - lo[0], w[1] - v[1])
            tx, ty, va = (lo[0] + lo[1]) / 2, w[1] + 0.02, "bottom"
        else:
            rect = Rectangle((v[0], lo[0]), w[0] - v[0], lo[1] - lo[0])
            tx, ty, va = (v[0] + w[0]) / 2 + 0.08, v[1] - 0.03, "top"
        rect.set(facecolor="none", edgecolor=RED if i == i_best else GREY, hatch="///", lw=1.2,
                 ls="-" if i == i_best else "--")
        ax.add_patch(rect)
        ax.text(tx, ty, f"feature {i + 1}: cuts {100 * sh:.1f}%\nof the width",
                color=RED if i == i_best else GREY, fontsize=9, ha="center", va=va)
    # (b)
    ax = axes[1]
    draw_box(ax, m, b, ORANGE, ls="--", lw=1.2, alpha=0.04)
    draw_box(ax, sr, int(np.flatnonzero(sr.box_class_ == 1)[0]), ORANGE)
    draw_box(ax, m, a, BLUE)
    ax.scatter(*X[y == 0].T, color=BLUE, s=25, zorder=3)
    keep_b = [r for r in np.flatnonzero(y == 1) if r not in dropped]
    ax.scatter(*X[keep_b].T, color=ORANGE, s=25, marker="s", zorder=3)
    ax.scatter(*X[dropped].T, color=ORANGE, s=90, marker="s", facecolor="none", lw=2, zorder=3,
               label="member dropped from box B")
    ax.scatter(*x, s=200, facecolor="none", edgecolor=RED, lw=2, zorder=4)
    face = sr.V_[1, i_best] if side == "low" else sr.W_[1, i_best]
    if i_best == 1:
        ax.axhline(face, color=RED, lw=1, ls=":")
    else:
        ax.axvline(face, color=RED, lw=1, ls=":")
    ax.text(0.02, 0.97, f"face of box B on feature {i_best + 1} moved\nfrom {v[i_best]:.3f} to "
            f"x + ε = {new:.3f} (ε = {MARGIN})", transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor=GREY))
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=9)
    ins = ax.inset_axes([0.66, 0.05, 0.31, 0.24])
    zx, zy = x[0], x[1]
    ins.set(xlim=(zx - 0.004, zx + 0.004), ylim=(zy - 0.004, zy + 0.004), xticks=[], yticks=[])
    ins.scatter([zx], [zy], color=BLUE, s=30)
    if i_best == 1:
        ins.axhline(face, color=ORANGE, lw=2)
        ins.axhline(v[1], color=ORANGE, lw=1, ls="--")
    else:
        ins.axvline(face, color=ORANGE, lw=2)
    ins.set_title("zoom: x is just outside B", fontsize=8)
    # (c)
    ax = axes[2]
    bb = int(np.flatnonzero(sr.box_class_ == 1)[0])
    draw_box(ax, sr, a, BLUE)
    draw_box(ax, sr, bb, ORANGE)
    ax.scatter(*X[y == 0].T, color=BLUE, s=25, zorder=3)
    ax.scatter(*X[keep_b].T, color=ORANGE, s=25, marker="s", zorder=3)
    ax.scatter(*X[dropped].T, color=ORANGE, s=90, marker="s", facecolor="none", lw=2, zorder=3)
    ax.scatter(*x, s=200, facecolor="none", edgecolor=BLUE, lw=2.2, zorder=4)
    ax.scatter(*m.centroid_[b], marker="X", s=120, color="white", edgecolor=ORANGE, lw=1.5, zorder=5)
    ax.scatter(*sr.centroid_[bb], marker="X", s=140, color=ORANGE, edgecolor="black", zorder=5)
    ax.annotate("", sr.centroid_[bb], m.centroid_[b], arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.text(0.02, 0.97, f"membership of x: box A = {after[a]:.3f}, box B = {after[b]:.3f}\n"
            f"no tie, so x is predicted A (right)\nB's centroid moves: its members changed",
            transform=ax.transAxes, va="top", fontsize=9.5, bbox=dict(facecolor="white", edgecolor=GREY))
    fig.suptitle("Illustration: one self-repair edit, made by run_self_repair on the toy model",
                 fontsize=12)
    save(fig, "rq3_4_shrink_steps.png")
    note(f"toy cut shares: feature 1 {100 * shares[0]:.1f}%, feature 2 {100 * shares[1]:.1f}%; "
         f"chosen feature {i_best + 1}, face {v[i_best]:.3f} -> {new:.3f}; dropped members {len(dropped)}; "
         f"membership of x in B after = {after[b]:.4f} = 1 - γε/(2d) with d = {d}")
    return sr


def fig_map(m, sr, a, b):
    g = np.linspace(0, 0.8, 321)
    G = np.array([(p, q) for q in g for p in g])
    pb, pa = m.predict(G), sr.predict(G)
    tb, _ = top_groups(m, G)
    ta, _ = top_groups(sr, G)
    loose, tight = certify(m, sr, tb, ta, removed=None)
    changed = pb != pa
    assert changed.any()
    assert not (changed & ~tight).any() and not (changed & ~loose).any()
    inside_old_b = ((G >= m.V_[b]) & (G <= m.W_[b])).all(axis=1)
    out_changed = changed & ~inside_old_b
    shape = (len(g), len(g))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#bee3f8", "#feebc8"])
    for ax, pred, model, t in ((axes[0], pb, m, "Before the edit"), (axes[1], pa, sr, "After the edit")):
        ax.imshow(pred.reshape(shape), origin="lower", extent=(0, 0.8, 0, 0.8), cmap=cmap,
                  vmin=0, vmax=1, interpolation="nearest")
        for j in range(model.n_boxes_):
            draw_box(ax, model, j, BLUE if model.box_class_[j] == 0 else ORANGE, alpha=0)
        ax.set(xlabel="feature 1", ylabel="feature 2", title=t)
    ax = axes[1]
    draw_box(ax, m, b, ORANGE, ls="--", lw=1.0, alpha=0)
    over = np.zeros(shape + (4,))
    over[tight.reshape(shape)] = matplotlib.colors.to_rgba(PURPLE, 0.35)
    over[changed.reshape(shape)] = matplotlib.colors.to_rgba(RED, 0.9)
    ax.imshow(over, origin="lower", extent=(0, 0.8, 0, 0.8), interpolation="nearest")
    ax.scatter([], [], marker="s", color=RED, label="prediction changed")
    ax.scatter([], [], marker="s", color=PURPLE, alpha=0.35, label="in the tight certificate, unchanged")
    ax.plot([], [], color=ORANGE, ls="--", label="box B before the edit")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Illustration: predicted class over the plane (blue = A, orange = B)", fontsize=12)
    save(fig, "rq3_5_decision_map.png")
    note(f"toy map: {100 * changed.mean():.2f}% of grid points changed, all inside the tight certificate, "
         f"which holds {100 * tight.mean():.1f}% of them; {100 * out_changed.sum() / changed.sum():.0f}% "
         f"of the changed points lie outside the old box B")
    return out_changed.any()


# ---------------------------------------------------------------- figure 6
def fig_outside():
    X = np.array([[0.0], [0.195], [0.40], [0.50], [0.60]])
    y = np.array([0, 0, 1, 1, 1])
    m = CFMNN(theta=0.3, gamma=GAMMA).fit(X, y)
    k = int(np.flatnonzero(m.box_class_ == 1)[0])
    z = int(np.flatnonzero(m.box_class_ == 0)[0])
    assert np.allclose([m.V_[k, 0], m.W_[k, 0], m.V_[z, 0], m.W_[z, 0]], [0.40, 0.60, 0.0, 0.195])
    after = CFMNN(theta=0.3, gamma=GAMMA).fit(X, y)
    shrink_box(after, k, np.array([0.410]), 0, X)
    assert np.isclose(after.V_[k, 0], 0.411)
    p = np.array([[0.30]])
    mb, ma = m.memberships(p)[0], after.memberships(p)[0]
    assert np.allclose([mb[k], mb[z], ma[k]], [0.800, 0.790, 0.778])
    assert m.predict(p)[0] == 1 and after.predict(p)[0] == 0
    xs = np.linspace(0, 0.8, 1601)[:, None]
    flip = m.predict(xs) != after.predict(xs)
    lo, hi = xs[flip, 0].min(), xs[flip, 0].max()
    assert hi < 0.40   # every flipped point lies outside the old box
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, (x0, x1, y0, y1) in zip(axes, ((0, 0.8, 0.48, 1.03), (0.285, 0.315, 0.76, 0.82))):
        ax.axvspan(lo, hi, color=RED, alpha=0.15, label="prediction changed")
        ax.plot(xs, m.memberships(xs)[:, z], color=BLUE, lw=2, label="box 0 (class 0), unchanged")
        ax.plot(xs, m.memberships(xs)[:, k], color=ORANGE, lw=2, ls="--", label="box k (class 1), before")
        ax.plot(xs, after.memberships(xs)[:, k], color=ORANGE, lw=2,
                label="box k after its lower face moves 0.40 → 0.411")
        ax.axvline(0.30, color=GREY, ls=":")
        ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="feature value", ylabel="membership")
    axes[0].axvspan(0.40, 0.60, color=ORANGE, alpha=0.06)
    axes[0].text(0.50, 0.52, "old box k", ha="center", color=ORANGE)
    axes[0].set_title("Illustration: a shrink changes a prediction outside the old box", loc="left", fontsize=11)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.8, -0.16), ncol=2, fontsize=8.5, frameon=False)
    axes[1].set_title("Zoom near x = 0.30", loc="left", fontsize=11)
    axes[1].text(0.3005, 0.812, "at x = 0.30:\nbox k 0.800 → 0.778\nbox 0 stays 0.790\nclass 1 → class 0",
                 fontsize=9, va="top")
    save(fig, "rq3_6_outside_change.png")
    note(f"outside: predictions changed for x in [{lo:.4f}, {hi:.4f}], all below the old face 0.40")


# ---------------------------------------------------------------- results
def results():
    r = pd.read_csv(RUN / "runs.csv")
    e = pd.read_csv(RUN / "edits.csv.gz")
    mean = r.groupby(["dataset", "arm"]).mean(numeric_only=True)

    def val(arm, col):
        return [mean.loc[(d, arm), col] for d in ORDER]

    arms = ["B0", "SR", "AL", "SR+AL", "Retrain, shuffled", "Retrain, appended", "RF, every label"]
    note("\nTest macro-F1 (%), mean over seeds 50-59")
    for arm in arms:
        note(f"  {arm:18s} " + "  ".join(f"{NAMES[d]} {v:.2f}" for d, v in zip(ORDER, val(arm, 'test_macro_f1'))))
    note("Lowest class recall (%)")
    for arm in ("B0", "SR", "SR+AL"):
        note(f"  {arm:18s} " + "  ".join(f"{NAMES[d]} {v:.2f}" for d, v in zip(ORDER, val(arm, 'test_min_class_recall'))))
    for col, lab in (("churn_fixed_share", "Fixed (% of test rows)"), ("churn_broken_share", "Broken (% of test rows)"),
                     ("churn_fix_share", "Fix share (%)"), ("churn_changed_share", "Changed (% of test rows)")):
        note(lab)
        for arm in ("SR", "SR+AL", "Retrain, shuffled", "Retrain, appended"):
            note(f"  {arm:18s} " + "  ".join(f"{NAMES[d]} {v:.2f}" for d, v in zip(ORDER, val(arm, col))))
    note("Sizes and time (SR arm unless named)")
    for d in ORDER:
        sr = mean.loc[(d, "SR")]
        note(f"  {NAMES[d]}: n_train {sr.n_train:.0f}, boxes {sr.boxes_start:.1f} -> {sr.boxes_end:.1f}, "
             f"train error {sr.train_error_start:.2f}% -> {sr.train_error_end:.2f}%, SR time {sr.time_s:.1f}s, "
             f"retrain shuffled {mean.loc[(d, 'Retrain, shuffled'), 'time_s']:.2f}s, "
             f"tight cert precision {sr.cert_tight_precision:.1f}%, tight cert size {sr.cert_tight_mean_size_share:.1f}%")
    # Edit log, SR arm.
    s = e[e["arm"] == "SR"]
    per = s.groupby(["dataset", "seed"])
    note("Edit log, SR arm, mean per seed")
    for d in ORDER:
        g = s[s["dataset"] == d]
        n_seeds = g["seed"].nunique()
        sh = g[g["edit"] == "shrink"]
        acc = sh[sh["accepted"] == True]  # noqa: E712
        reasons = g["reason"].value_counts()
        note(f"  {NAMES[d]}: errors taken {len(g) / n_seeds:.1f}, already right {reasons.get('already right', 0) / n_seeds:.1f}, "
             f"not inside {reasons.get('not inside the winning box', 0) / n_seeds:.1f}, shrinks tried {len(sh) / n_seeds:.1f}, "
             f"accepted {len(acc) / n_seeds:.1f}, rejected for macro-F1 {reasons.get('macro-F1', 0) / n_seeds:.1f}, "
             f"for class recall {reasons.get('class recall', 0) / n_seeds:.1f}; accepted edits that fixed their row "
             f"{100 * acc['x_correct_after'].astype(bool).mean():.1f}%; members dropped per accepted edit: mean "
             f"{acc['shrink_n_cut'].mean():.1f}, median {acc['shrink_n_cut'].median():.0f}, max {acc['shrink_n_cut'].max():.0f}; "
             f"share of width cut: mean {100 * acc['shrink_share_cut'].mean():.1f}%; boxes removed per seed "
             f"{acc['shrink_removed'].astype(bool).sum() / n_seeds:.2f}")
    tot = s[s["edit"] == "shrink"]
    note(f"  all SR shrinks: {len(tot)} tried, {int(tot['accepted'].sum())} accepted; "
         f"cert misses (loose, tight) over all edit arms: "
         f"{int(r['cert_loose_misses'].sum())}, {int(r['cert_tight_misses'].sum())} in {int(r['cert_edits'].sum())} edits")
    assert per.size().max() <= 300

    # Figure 7: macro-F1.
    show = [("B0", "Starting model", GREY), ("SR", "Self-repair (no new labels)", ORANGE),
            ("AL", "Active learning", "#38a169"), ("SR+AL", "SR then AL", "#2c7a7b"),
            ("Retrain, shuffled", "Retrained, shuffled", PURPLE), ("RF, every label", "Random Forest, same labels", "#a0aec0")]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    wdt = 0.13
    xs = np.arange(len(ORDER))
    for i, (arm, lab, col) in enumerate(show):
        ax.bar(xs + (i - 2.5) * wdt, val(arm, "test_macro_f1"), wdt, label=lab, color=col)
    ax.set_xticks(xs, [NAMES[d] for d in ORDER])
    ax.set(ylim=(55, 101), ylabel="test macro-F1 (%)", title="RQ3 second run: test macro-F1, mean over seeds 50–59")
    ax.legend(ncol=6, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    save(fig, "rq3_7_macro_f1.png")

    # Figure 8: fixed and broken.
    show = [("SR", "Self-repair", ORANGE), ("SR+AL", "SR then AL", "#2c7a7b"),
            ("Retrain, shuffled", "Retrained, shuffled", PURPLE), ("Retrain, appended", "Retrained, appended", "#a0aec0")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    wdt = 0.2
    for ax, col, t in ((axes[0], "churn_fixed_share", "Fixed: wrong → right (% of test rows)"),
                       (axes[1], "churn_broken_share", "Broken: right → wrong (% of test rows)")):
        for i, (arm, lab, c) in enumerate(show):
            ax.bar(xs + (i - 1.5) * wdt, val(arm, col), wdt, label=lab, color=c)
        ax.set_xticks(xs, [NAMES[d] for d in ORDER], fontsize=9)
        ax.set_title(t, fontsize=11)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Test predictions changed against the starting model, mean over seeds 50–59", fontsize=12)
    save(fig, "rq3_8_fixed_broken.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    fig_membership()
    fig_order()
    X, y, m = toy()
    a, b = fig_tie(X, y, m)
    sr = fig_shrink(X, y, m, a, b)
    fig_map(m, sr, a, b)
    fig_outside()
    results()
    (FIG / "rq3_numbers.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
