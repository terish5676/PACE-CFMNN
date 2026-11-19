"""Run P4 (tough review B2): exact PACE-SHAP for min-membership (GFMM-family) networks.

Rules: results/gfmm_exact/PREREGISTRATION.md (planned in advance).

GFMM models score a point by the minimum, not the mean, of one-feature terms. ``gfmm_explain``
builds every coalition's minimum from a table of subset minima, so PACE-SHAP-min should give the
exact Shapley values of the margin, as PACE-SHAP does for CFMNN. This run checks that on models
fitted by hyperbox-brain 0.1.6 and measures the time.

Models: Online GFMM (theta = theta_min = 0.3, gamma = 4) and IOL-GFMM (theta = 0.3, gamma = 4),
fitted on the paper's 80/20 split of each seed (features scaled on the training rows; explained
points clipped to [0, 1], the range hyperbox-brain expects). Explainer rows: ``backgrounds_v3``'s
200 explainer rows.

* ``check`` (seeds 310-319, 10 test points each, any number of jobs): the adapter predicts like
  hyperbox-brain on the explained points; PACE-SHAP-min (exact, with pruning) explains all 10, and
  equals the brute force (the margin on every mixed input, ``brute_force_min``) within 1e-12 on
  the first 3 (brute force takes up to 3 minutes per point on MAGIC with IOL-GFMM).
* ``timing`` (seeds 310-312, 1 point each, alone): PACE-SHAP-min, brute force and shap's
  ExactExplainer on the same margin, each timed alone with its set-up; ExactExplainer must agree
  within 1e-9.

    python -B scripts/run_gfmm_exact.py --part check --jobs 10
    python -B scripts/run_gfmm_exact.py --part timing
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
from pace_cfmnn import data  # noqa: E402
from pace_cfmnn.gfmm_explain import GFMMBoxes, PaceShapMin, brute_force_min  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

DATASETS = ["S1", "S2", "S3", "shuttle", "magic", "page_blocks"]
MODELS = ["Online GFMM", "IOL-GFMM"]
SEEDS = {"check": list(range(310, 320)), "timing": [310, 311, 312]}
CHUNK_ELEMS = 4_000_000   # as brute_force_min's default
N_POINTS = {"check": 10, "timing": 1}
N_BRUTE = 3                 # check part: points per seed compared with the brute force
TOL_BRUTE, TOL_EXACT = 1e-12, 1e-9
PACE, BRUTE, EXACT = "PACE-SHAP-min", "Brute force", "shap ExactExplainer"
OUT = ROOT / "results" / "gfmm_exact"


def make(kind):
    if kind == "Online GFMM":
        from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
        return OnlineGFMM(theta=0.3, theta_min=0.3, gamma=4)
    from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
    return ImprovedOnlineGFMM(theta=0.3, gamma=4)


def prepare(name, seed, n):
    """The paper's split and scaling as in ``run_faithfulness.setup``, with the labels."""
    ds = data.load(name, seed=seed)
    sp = data.paper_split(ds.y, seed)
    order = np.random.default_rng(seed).permutation(sp["train"])
    sc = data.fit_scaler(ds.X[order])
    Xtr, ytr = sc.transform(ds.X[order]), ds.y[order]
    te = np.random.default_rng(seed).permutation(sp["test"])[:n]
    Xte = np.clip(sc.transform(ds.X[te]), 0.0, 1.0)
    B = np.clip(rf.backgrounds_v3(Xtr, seed)[1][:rf.SHAP_V3_R], 0.0, 1.0)
    return Xtr, ytr, Xte, B


def blocked_margin(boxes: GFMMBoxes, Z, a, chunk_elems: int | None = None):
    """The margin of class index ``a`` on the rows of Z, in blocks of at most ``chunk_elems``
    (row, box, feature) numbers, as ``brute_force_min`` does (P4 Amendment 2). The values are the
    same as one call; shap's ExactExplainer passes 2^d x 200 rows at once, which needs about
    35 GiB for one call on S1-S3 and MAGIC."""
    chunk_elems = CHUNK_ELEMS if chunk_elems is None else chunk_elems
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    step = max(1, chunk_elems // max(1, len(boxes.V) * Z.shape[1]))
    return np.concatenate([boxes.margin(Z[s:s + step], np.full(len(Z[s:s + step]), a))
                           for s in range(0, len(Z), step)]) if len(Z) else np.zeros(0)


def exact_shap_min(boxes: GFMMBoxes, X, B):
    """shap's ExactExplainer on the GFMM margin of the predicted class (as ``xai_baselines.exact_shap``)."""
    import shap
    cls = boxes.predict_index(X)
    out = np.zeros(X.shape)
    masker = shap.maskers.Independent(B, max_samples=len(B))
    for a in np.unique(cls):
        rows = cls == a
        ex = shap.explainers.Exact(lambda Z, a=a: blocked_margin(boxes, Z, a), masker)
        out[rows] = ex(X[rows], max_evals=max(100_000, 2 ** X.shape[1] + 2), silent=True).values
    return out


def run_one(name, seed, part):
    warnings.filterwarnings("ignore")
    n = N_POINTS[part]
    Xtr, ytr, Xte, B = prepare(name, seed, n)
    rows = []
    for kind in MODELS:
        t0 = time.perf_counter()
        est = make(kind).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        bx = GFMMBoxes.from_hbbrain(est)
        same_pred = bool(np.array_equal(bx.predict(Xte), np.asarray(est.predict(Xte))))
        base = {"part": part, "dataset": name, "seed": seed, "model": kind, "d": Xte.shape[1],
                "boxes": int(len(bx.V)), "fit_s": fit_s, "adapter_predicts_alike": same_pred}
        if part == "timing":
            PaceShapMin(bx, B).explain(Xte[:1])                       # warm-up, not timed
        t0 = time.perf_counter()
        ps = PaceShapMin(bx, B)
        phi = ps.explain(Xte)
        t_pace = time.perf_counter() - t0
        nb = N_BRUTE if part == "check" else len(Xte)
        t0 = time.perf_counter()
        ref = np.array([brute_force_min(bx, x, B) for x in Xte[:nb]])
        t_brute = time.perf_counter() - t0
        res = {PACE: (phi, t_pace, len(Xte)), BRUTE: (ref, t_brute, nb)}
        if part == "timing":
            t0 = time.perf_counter()
            res[EXACT] = (exact_shap_min(bx, Xte, B), time.perf_counter() - t0, len(Xte))
        for arm, (v, secs, k) in res.items():
            rows.append({**base, "arm": arm, "n": k, "ms_per_explanation": 1000 * secs / k,
                         "max_abs_diff_brute": float(np.abs(v[:nb] - ref).max()),
                         "max_abs_diff_pace": float(np.abs(v - phi[:k]).max()),
                         "max_abs_value": float(np.abs(ref).max()),
                         "share_pairs_skipped": ps.stats_.get("share_of_boxes_skipped", np.nan)
                         if arm == PACE else np.nan})
        print(f"{name} seed {seed} {kind} ({len(bx.V)} boxes): PACE-SHAP-min vs brute force "
              f"{np.abs(phi[:nb] - ref).max():.1e}" + (f", vs ExactExplainer {np.abs(res[EXACT][0] - phi).max():.1e}"
                                                  if EXACT in res else ""), flush=True)
    return rows


def check(df: pd.DataFrame) -> tuple[bool, list[str]]:
    bad = [f"{r.dataset} seed {r.seed} {r.model}: the adapter predicts unlike hyperbox-brain"
           for _, r in df[~df.adapter_predicts_alike.astype(bool)].drop_duplicates(["dataset", "seed", "model"]).iterrows()]
    g = df[(df.arm == PACE) & (df.max_abs_diff_brute > TOL_BRUTE)]
    bad += [f"{r.dataset} seed {r.seed} {r.model}: PACE-SHAP-min differs from brute force by "
            f"{r.max_abs_diff_brute:.1e}" for _, r in g.iterrows()]
    g = df[(df.arm == EXACT) & (df.max_abs_diff_pace > TOL_EXACT)]
    bad += [f"{r.dataset} seed {r.seed} {r.model}: ExactExplainer differs by {r.max_abs_diff_pace:.1e}"
            for _, r in g.iterrows()]
    return not bad, bad


def summarise(df, out: Path) -> str:
    ok, bad = check(df)
    L = ["# Run P4: exact PACE-SHAP for GFMM-family networks", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Written by `scripts/run_gfmm_exact.py`.", "",
         f"**Check: {'PASS' if ok else 'FAIL'}.** The adapter predicts like hyperbox-brain on every explained "
         f"point; PACE-SHAP-min equals the brute force within {TOL_BRUTE:g} on every point; where timed, "
         f"shap's ExactExplainer agrees within {TOL_EXACT:g}."]
    L += [f"- {b}" for b in bad]
    c = df[(df.part == "check") & (df.arm == PACE)]
    if len(c):
        L += ["", "## Check (seeds {}–{})".format(c.seed.min(), c.seed.max()), "",
              "| Dataset | Model | Points explained | Points compared | Boxes (mean) | Largest difference from brute force | (Row, box) pairs skipped |",
              "|---|---|---|---|---|---|---|"]
        cb = df[(df.part == "check") & (df.arm == BRUTE)]
        for (ds, m), g in c.groupby(["dataset", "model"], sort=False):
            nb = int(cb[(cb.dataset == ds) & (cb.model == m)].n.sum())
            L.append(f"| {ds} | {m} | {int(g.n.sum())} | {nb} | {g.boxes.mean():.0f} | {g.max_abs_diff_brute.max():.1e} | "
                     f"{g.share_pairs_skipped.mean():.3f} |")
    t = df[df.part == "timing"]
    if len(t):
        L += ["", "## Timing (each alone, with its set-up; mean over seeds)", "",
              "| Dataset | Model | ms PACE-SHAP-min | ms brute force | ms ExactExplainer | Brute force ÷ PACE | "
              "ExactExplainer ÷ PACE |", "|---|---|---|---|---|---|---|"]
        for (ds, m), g in t.groupby(["dataset", "model"], sort=False):
            ms = g.groupby("arm").ms_per_explanation.mean()
            L.append(f"| {ds} | {m} | {ms[PACE]:.1f} | {ms[BRUTE]:.0f} | {ms[EXACT]:.0f} | "
                     f"{ms[BRUTE] / ms[PACE]:.1f}× | {ms[EXACT] / ms[PACE]:.1f}× |")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P4: GFMM exactness")
    ap.add_argument("--part", choices=["check", "timing"], required=True)
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, help="smoke runs only")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{a.part}.csv"
    if a.part == "timing" and a.jobs != 1:
        raise SystemExit("the timing part runs alone: use --jobs 1")
    if not a.summary_only:
        tasks = [(d, s, a.part) for d in a.datasets or DATASETS for s in a.seeds or SEEDS[a.part]]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=a.jobs)
    frames = [pd.read_csv(out / f"{p}.csv", float_precision="round_trip") for p in ("check", "timing")
              if (out / f"{p}.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    print(summarise(df, out))
    sys.exit(0 if check(df)[0] else 1)


if __name__ == "__main__":
    main()
