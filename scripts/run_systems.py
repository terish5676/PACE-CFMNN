"""Run G: CFMNN + PACE-SHAP against other models that explain themselves exactly
(planned in advance).

Rules: results/systems/PREREGISTRATION.md. A "system" is a model together with its own
exact explanation:

* **CFMNN + PACE-SHAP**: CFMNN at theta = 0.3, explained by PACE-SHAP;
* **EBM + terms**: an explainable boosting machine with main effects only, explained by
  its own per-feature terms;
* **Tree + TreeSHAP**: a decision tree with its number of leaves tuned, explained by
  ``shap.TreeExplainer`` (interventional).

Each system is explained and tested on its OWN predicted-class margin (the score of the
class it predicts minus the best other class), with the deletion and insertion tests of
scripts/run_faithfulness.py. Margins of different models are on different scales, so
each system is compared with a random ranking of the features on the same model and the
same points: the normalised lead is (random - method) / |random| for deletion and
(method - random) / |random| for insertion, so a larger lead is always better.

    python -B scripts/run_systems.py --jobs 12
    python -B scripts/run_systems.py --jobs 12 --resume      # continue after a stop
    python -B scripts/run_systems.py --summary-only
    python -B scripts/run_systems.py --timing                # alone, after the run (amendment 1)

Amendment 1 (PREREGISTRATION.md, before the run): the same AUCs, leads and gains on the
explained points whose margin is not zero ("untied"), a comparison on them, and a timing pass
run alone.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_faithfulness as rf  # noqa: E402
import run_readable_baselines as rb  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402

DATASETS = list(rf.DATASETS)                  # the 11 explanation datasets
SYNTHETIC = ["S1", "S2", "S3"]
FAMILY = "Synthetic family"
SEEDS = list(range(120, 130))
N_EXPLAIN = 50                                # 36 on Wine, whose test set has 36 rows
BG_ROWS = 200                                 # explainer rows (and, separately, test rows)
PERMUTATIONS = rf.SHAP_V3_P                   # PACE-SHAP's sampled mode, as in RQ1 (d > 12)
RANDOM_RANKINGS = 5                           # random rankings averaged per model
NEAR_ZERO = 0.05                              # |random AUC| below this: the lead is flagged
# Amendment 1, written before the run (PREREGISTRATION.md, "Amendment 1").
TIE_TOL = 1e-12                               # a margin at or below this is zero: a tied point
MIN_UNTIED = 10                               # untied points every seed needs (as run_review_rq1)
TIE_HEAVY = 0.10                              # CFMNN's mean share of tied points from which the
                                              # primary verdict is read beside the untied one
TIMING_SEEDS = [120]                          # the --timing pass: one seed, one process, alone
CFMNN_SYS, EBM_SYS, TREE_SYS = "CFMNN + PACE-SHAP", "EBM + terms", "Tree + TreeSHAP"
SYSTEMS = [CFMNN_SYS, EBM_SYS, TREE_SYS]
OTHERS = [EBM_SYS, TREE_SYS]
OUT = ROOT / "results" / "systems"


# ------------------------------------------------------------------ the three systems
class _Scored:
    """What the deletion and insertion tests need from a model: ``predict_details(X)``
    with the predicted class (a column index) and one score per class."""

    def scores(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_details(self, X):
        s = self.scores(X)
        return SimpleNamespace(cls=s.argmax(axis=1), support_rescaled=s)


class EBMScored(_Scored):
    """EBM with main effects only. Score of a class = its logit. With two classes the
    logits are (0, z), z being the model's log-odds, so the margin is +z or -z."""

    def __init__(self, ebm):
        self.ebm = ebm
        feats = [tuple(int(i) for i in f) for f in ebm.term_features_]
        if feats != [(i,) for i in range(len(feats))]:
            raise ValueError("expected one main-effect term per feature, in feature order")

    @staticmethod
    def _per_class(a):
        """(n, ...) for two classes -> (n, ..., 2) with zeros for the first class."""
        return np.stack([np.zeros_like(a), a], axis=-1)

    def scores(self, X):
        z = np.asarray(self.ebm.decision_function(X), float)
        return self._per_class(z) if z.ndim == 1 else z

    def terms(self, X):
        """Each feature's term for each class: (n, d, C). Their sum over the features,
        plus the intercept, is ``scores(X)``."""
        t = np.asarray(self.ebm.eval_terms(X), float)
        return self._per_class(t) if t.ndim == 2 else t


class TreeScored(_Scored):
    """Decision tree. Score of a class = its share of the training rows in the leaf."""

    def __init__(self, tree):
        self.tree = tree

    def scores(self, X):
        return np.asarray(self.tree.predict_proba(X), float)


def class_weights(scores, cls, tol: float = 1e-12):
    """Weights (n, C) that turn per-class values into the predicted class's margin:
    +1 on the predicted class and -1 on the best other class. If several other classes
    share the best score (a pure tree leaf: every other class has 0), the -1 is split
    equally among them, so no class is picked by its index."""
    scores = np.asarray(scores, float)
    rows = np.arange(len(scores))
    other = scores.copy()
    other[rows, cls] = -np.inf
    tied = other >= other.max(axis=1, keepdims=True) - tol
    w = -tied.astype(float) / tied.sum(axis=1, keepdims=True)
    w[rows, cls] = 1.0
    return w


def margin_attribution(phi, scores, cls):
    """Per-class attributions (n, d, C) -> attributions of the predicted class's margin
    (n, d). The class weights are fixed at the explained point, so the result adds up to
    the point's margin minus the background mean of the same weighted score."""
    return np.einsum("ndc,nc->nd", phi, class_weights(scores, cls))


def explain_ebm(sm: EBMScored, X, background):
    """EBM's own terms, centred on the background rows. For an additive model these are
    also its exact interventional Shapley values."""
    phi = sm.terms(X) - sm.terms(background).mean(axis=0, keepdims=True)
    s = sm.scores(X)
    return margin_attribution(phi, s, s.argmax(axis=1))


def _f32(X):
    """scikit-learn trees compare float32 inputs with their thresholds. Rounding the rows
    the same way makes TreeExplainer follow exactly the paths the tree follows."""
    return np.asarray(X, np.float32).astype(np.float64)


def float32_floor(thresholds):
    """The largest float32 value at or below each threshold (Amendment 2).

    scikit-learn compares a float32 input with the float64 threshold; shap's TreeExplainer
    rounds the threshold to the nearest float32 and compares with that. When the rounding
    goes up to exactly the input's value, shap sends the point left where the tree sends it
    right. For a float32 x, x <= t holds exactly when x <= float32_floor(t), so with these
    thresholds shap follows the tree's own paths."""
    t = np.asarray(thresholds, np.float64)
    t32 = t.astype(np.float32)
    up = t32.astype(np.float64) > t
    t32[up] = np.nextafter(t32[up], np.float32(-np.inf))
    return t32.astype(np.float64)


def tree_shap_values(tree, X, background):
    """Interventional TreeSHAP values of every class's score: (n, d, C)."""
    import shap
    bg = _f32(background)
    # shap keeps only 100 background rows unless told otherwise; we want all of them.
    ex = shap.TreeExplainer(tree, data=shap.maskers.Independent(bg, max_samples=len(bg)),
                            feature_perturbation="interventional")
    assert ex.model.input_dtype == np.float32
    ex.model.thresholds[:] = float32_floor(ex.model.thresholds)     # Amendment 2
    sv = ex.shap_values(_f32(X))
    return np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv, float)


def explain_tree(sm: TreeScored, X, background):
    s = sm.scores(X)
    return margin_attribution(tree_shap_values(sm.tree, X, background), s, s.argmax(axis=1))


# ------------------------------------------------------------------ the tests
def lead(method, random, higher_better: bool = False) -> float:
    """Normalised lead over a random ranking on the same model; larger is better.
    Deletion (lower AUC is better): (random - method) / |random|.
    Insertion (higher is better):   (method - random) / |random|."""
    if not np.isfinite(random) or abs(random) < 1e-12:
        return float("nan")
    diff = method - random if higher_better else random - method
    return float(diff / abs(random))


def point_curves(model, X, attr, bev, seed: int) -> dict:
    """Deletion and insertion margin AUCs of one ranking, one value per explained point."""
    _, dele, dele_raw = rf.deletion_auc(model, X, attr, bev, seed=seed, signed=True, with_raw=True)
    _, ins, ins_raw = rf.insertion_auc(model, X, attr, bev, seed=seed, with_raw=True)
    return {"deletion_auc": dele, "insertion_auc": ins,
            "deletion_auc_raw": dele_raw, "insertion_auc_raw": ins_raw}


def curves(model, X, attr, bev, seed: int) -> dict:
    """Deletion and insertion margin AUCs (mean over the points) of one ranking."""
    return {k: float(v.mean()) for k, v in point_curves(model, X, attr, bev, seed).items()}


def _leads(out: dict, suffix: str = "") -> None:
    """Add the lead and the gain over the random ranking to ``out`` (keys end in ``suffix``)."""
    for kind, higher in (("deletion", False), ("insertion", True)):
        m, r = out[f"{kind}_auc{suffix}"], out[f"random_{kind}_auc{suffix}"]
        out[f"{kind}_lead{suffix}"] = lead(m, r, higher)
        out[f"{kind}_gain{suffix}"] = (m - r) * (1 if higher else -1)


def faithfulness(model, X, attr, bev, seed: int) -> dict:
    """The method's AUCs, the same for random rankings on the same model and points (the
    mean of ``RANDOM_RANKINGS`` rankings), and the leads.

    Amendment 1: the deletion and insertion AUCs, leads and gains are also given on the
    untied points only (``*_untied``; margin above ``TIE_TOL``), with ``n_untied``. They are
    NaN when no point is untied. The all-point values are unchanged."""
    mine = point_curves(model, X, attr, bev, seed)
    rnd = [point_curves(model, X, np.random.default_rng(seed + 1000 * (k + 1)).uniform(size=X.shape),
                        bev, seed) for k in range(RANDOM_RANKINGS)]
    out = {key: float(v.mean()) for key, v in mine.items()}
    for key in list(out):
        out["random_" + key] = float(np.mean([float(r[key].mean()) for r in rnd]))
    _leads(out)
    det = model.predict_details(X)
    untied = rf._margin(det, det.cls) > TIE_TOL
    out["n_untied"] = int(untied.sum())
    for key in ("deletion_auc", "insertion_auc"):
        if untied.any():
            out[key + "_untied"] = float(mine[key][untied].mean())
            out["random_" + key + "_untied"] = float(np.mean([float(r[key][untied].mean()) for r in rnd]))
        else:
            out[key + "_untied"] = out["random_" + key + "_untied"] = float("nan")
    _leads(out, "_untied")
    return out


# ------------------------------------------------------------------ one dataset and seed
def build(system: str, seed: int, Xtr, ytr, Xp, cfmnn, bex):
    """One system on one dataset and seed: (key, fitted model, scored model, the tree's
    setting, seconds to fit, a function that explains the points)."""
    t0 = time.perf_counter()
    setting = None
    if system == CFMNN_SYS:
        key, raw = "CFMNN-0.3", cfmnn
        model = cfmnn
        explain = lambda: PaceShap(cfmnn, bex, n_permutations=PERMUTATIONS, seed=seed).explain(Xp)
    elif system == EBM_SYS:
        key = "EBM"
        raw = rb.make(key, None, seed, 1).fit(Xtr, ytr)
        model = EBMScored(raw)
        explain = lambda: explain_ebm(model, Xp, bex)
    elif system == TREE_SYS:
        key = "Tree"
        setting, _ = rb.tune(key, rb.GRIDS[key], Xtr, ytr, seed, 1)
        raw = rb.make(key, setting, seed, 1).fit(Xtr, ytr)
        model = TreeScored(raw)
        explain = lambda: explain_tree(model, Xp, bex)
    else:
        raise ValueError(system)
    fit_s = time.perf_counter() - t0              # CFMNN: about 0, it was fitted in setup
    return key, raw, model, setting, fit_s, explain


def _inputs(name: str, seed: int, n_explain: int):
    Xtr, ytr, Xte, yte = rb.prepare(name, seed)
    # The explained points and the CFMNN come from RQ1's own setup (the same split, row
    # order and scaling as ``prepare``); the two sets of rows are RQ1 v3's: the tests copy
    # values from ``bev`` and every explainer sees only ``bex``.
    _, Xtr_rf, Xp, cfmnn, _ = rf.setup(name, seed, n_explain)
    assert np.array_equal(Xtr, Xtr_rf) and cfmnn.theta == rb.FIXED["CFMNN-0.3"] and cfmnn.gamma == rb.GAMMA
    bev, bex = rf.backgrounds_v3(Xtr, seed, BG_ROWS)
    return Xtr, ytr, Xte, yte, Xp, cfmnn, bev, bex


def run_one(name: str, seed: int, n_explain: int = N_EXPLAIN, systems=None):
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte, Xp, cfmnn, bev, bex = _inputs(name, seed, n_explain)
    rows = []
    for system in systems or SYSTEMS:
        key, raw, model, setting, fit_s, explain = build(system, seed, Xtr, ytr, Xp, cfmnn, bex)
        pred = np.asarray(raw.predict(Xte))
        rules, cond = rb.model_size(key, raw)
        t0 = time.perf_counter()
        attr = explain()
        ms = 1000 * (time.perf_counter() - t0) / len(Xp)
        det = model.predict_details(Xp)
        rows.append({"dataset": name, "seed": seed, "model": system, "setting": setting,
                     "accuracy": 100 * float(np.mean(pred == yte)),
                     "macro_f1": 100 * float(f1_score(yte, pred, average="macro")),
                     "rules": rules, "conditions_per_rule": cond,
                     "ms_per_explanation": ms, "fit_s": fit_s,
                     **faithfulness(model, Xp, attr, bev, seed),
                     "share_zero_margin": float(np.mean(rf._margin(det, det.cls) <= 1e-12)),
                     "n_explained": len(Xp), "n_features": Xp.shape[1],
                     "n_train": len(ytr), "n_test": len(yte)})
    print(f"{name} seed {seed}: " + "; ".join(
        f"{r['model']} acc {r['accuracy']:.1f}, lead {r['deletion_lead']:+.2f}, "
        f"{r['ms_per_explanation']:.2f} ms" for r in rows), flush=True)
    return rows


def time_one(name: str, seed: int, n_explain: int = N_EXPLAIN, systems=None):
    """Amendment 1's timing pass: the same models and explanations as ``run_one``, with only
    the times recorded. Run alone, one task at a time.

    shap is imported before anything is timed. In the main run a worker process imports it
    inside its first TreeSHAP call, which adds seconds to that one task's time."""
    warnings.filterwarnings("ignore")
    import shap  # noqa: F401  (one-off import cost, kept out of the times)
    Xtr, ytr, _, _, Xp, cfmnn, _, bex = _inputs(name, seed, n_explain)
    rows = []
    for system in systems or SYSTEMS:
        _, _, _, _, fit_s, explain = build(system, seed, Xtr, ytr, Xp, cfmnn, bex)
        t0 = time.perf_counter()
        explain()
        rows.append({"dataset": name, "seed": seed, "model": system,
                     "ms_per_explanation": 1000 * (time.perf_counter() - t0) / len(Xp),
                     "fit_s": fit_s, "n_explained": len(Xp)})
    print(f"{name} seed {seed} timed: " + "; ".join(
        f"{r['model']} {r['ms_per_explanation']:.2f} ms" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------ summary
def units(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """S1-S3 averaged into one unit per seed and system, beside the real datasets."""
    syn = df[df.dataset.isin(SYNTHETIC)].groupby(["seed", "model"], as_index=False)[col].mean()
    return pd.concat([syn.assign(dataset=FAMILY),
                      df[~df.dataset.isin(SYNTHETIC)][["dataset", "seed", "model", col]]])


def compare(df, a, b, col, order, lower_better: bool = False) -> pd.DataFrame:
    """Paired two-sided Wilcoxon over seeds, Holm across the units in ``order``: a vs b.
    ``diff`` is a minus b; the verdict is for a."""
    out = []
    for u in [u for u in order if u in set(df.dataset)]:
        g = df[df.dataset == u].pivot_table(index="seed", columns="model", values=col)
        if a not in g or b not in g:
            continue
        d = (g[a] - g[b]).dropna().to_numpy()
        if len(d) == 0:
            continue
        p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else 1.0
        out.append({"dataset": u, "a": float(g[a].mean()), "b": float(g[b].mean()),
                    "diff": float(d.mean()), "p": float(p), "n": len(d)})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = rb.holm(res.p.to_numpy())
    good = (res["diff"] < 0) if lower_better else (res["diff"] > 0)
    res["verdict"] = np.where(res.p_holm >= 0.05, "level", np.where(good, "better", "worse"))
    return res


def flagged(df: pd.DataFrame, a: str, b: str) -> set:
    """Datasets where a seed of system a or b has a random deletion AUC near zero."""
    f = df[df.model.isin([a, b]) & (df.random_deletion_auc.abs() < NEAR_ZERO)]
    return set(f.dataset)


def near_zero_rule(df: pd.DataFrame, other: str, present) -> dict:
    """The primary verdicts, with flagged datasets read from the deletion gain instead of
    the lead. Per dataset, and with S1-S3 as one unit (flagged if any of them is)."""
    bad = flagged(df, CFMNN_SYS, other)
    views = {"per dataset": (df, present, bad)}
    if set(SYNTHETIC) <= set(present):
        fam_bad = {d for d in bad if d not in SYNTHETIC} | ({FAMILY} if bad & set(SYNTHETIC) else set())
        views["synthetic family as one unit"] = (
            None, [FAMILY] + [d for d in present if d not in SYNTHETIC], fam_bad)
    out = {}
    for label, (data, order, bad_units) in views.items():
        get = (lambda col: df) if data is not None else (lambda col: units(df, col))
        lead_res = compare(get("deletion_lead"), CFMNN_SYS, other, "deletion_lead", order)
        gain_res = compare(get("deletion_gain"), CFMNN_SYS, other, "deletion_gain", order)
        if lead_res.empty:
            continue
        res = lead_res[["dataset"]].copy()
        res["from_gain"] = res.dataset.isin(bad_units).to_numpy()
        res["verdict"] = np.where(res.from_gain, gain_res.set_index("dataset").verdict.reindex(res.dataset),
                                  lead_res.verdict)
        out[label] = res
    return out


# ---- amendment 1 (PREREGISTRATION.md, "Amendment 1"): untied points and the timing pass
def untied_datasets(df: pd.DataFrame, present) -> list:
    """Datasets where every seed of every system has at least ``MIN_UNTIED`` untied points."""
    if "n_untied" not in df:
        return []
    low = df.groupby("dataset").n_untied.min()
    return [d for d in present if low.get(d, 0) >= MIN_UNTIED]


def tie_heavy(df: pd.DataFrame, present) -> list:
    """Datasets where CFMNN's mean share of tied explained points is at least ``TIE_HEAVY``."""
    share = df[df.model == CFMNN_SYS].groupby("dataset").share_zero_margin.mean()
    return [d for d in present if share.get(d, 0.0) >= TIE_HEAVY]


def reading(primary: str, untied) -> str:
    """How amendment 1 reads a primary verdict on a tie-heavy dataset."""
    if untied is None:
        return "not checked: too few untied points"
    return "holds on untied points" if primary == untied else "depends on the tied points"


def untied_reading(df: pd.DataFrame, other: str, present) -> pd.DataFrame:
    """Per dataset: the primary verdict (lead, all points), the same on untied points, and
    the reading. Datasets that are not tie-heavy keep their primary verdict as it is."""
    prim = compare(df, CFMNN_SYS, other, "deletion_lead", present)
    if prim.empty:
        return prim
    ok = untied_datasets(df, present)
    unt = compare(df[df.dataset.isin(ok)], CFMNN_SYS, other, "deletion_lead_untied", ok)
    unt = dict(zip(unt.dataset, unt.verdict)) if not unt.empty else {}
    heavy = set(tie_heavy(df, present))
    share = df[df.model == CFMNN_SYS].groupby("dataset").share_zero_margin.mean()
    rows = []
    for _, r in prim.iterrows():
        u = unt.get(r.dataset)
        rows.append({"dataset": r.dataset, "tied_share": float(share.get(r.dataset, np.nan)),
                     "primary": r.verdict, "untied": u if u is not None else "not tested",
                     "reading": reading(r.verdict, u) if r.dataset in heavy else "not tie-heavy"})
    return pd.DataFrame(rows)


def amendment_1(df, L, present, systems) -> None:
    if "deletion_lead_untied" not in df:
        return
    ok = untied_datasets(df, present)
    L.extend(["## Amendment 1: the primary on untied points", "",
              "CFMNN can tie (a zero margin); for a tied point the deletion curve is divided by "
              "almost zero and becomes signs, which the other two models' points do not do. Below, "
              f"every system is scored on its untied explained points only. Tested on the {len(ok)} "
              f"dataset(s) where every seed of every system has at least {MIN_UNTIED} untied points: "
              + (", ".join(ok) or "none") + ".", ""])
    for other in [s for s in OTHERS if s in systems]:
        res = untied_reading(df, other, present)
        if res.empty:
            continue
        L.extend([f"**{CFMNN_SYS} vs {other}**: how the primary verdict reads (tie-heavy = CFMNN's mean "
                  f"share of tied points is at least {TIE_HEAVY:.0%})", "",
                  "| Dataset | CFMNN's tied share | Primary (all points) | Untied points | Reading |",
                  "|---|---|---|---|---|"])
        L.extend(f"| {r.dataset} | {r.tied_share:.1%} | {r.primary} | {r.untied} | {r.reading} |"
                 for _, r in res.iterrows())
        L.append("")
    sub_df = df[df.dataset.isin(ok)]
    if len(sub_df):
        _blocks_on(sub_df, L, ok, systems)
    near0 = df[df.random_deletion_auc_untied.abs() < NEAR_ZERO].groupby("model").size()
    L += [f"Runs whose random-ranking deletion AUC on untied points is within {NEAR_ZERO} of zero: "
          + ", ".join(f"{s} {int(near0.get(s, 0))} of {int((df.model == s).sum())}" for s in systems),
          "", "Untied explained points per run (mean over seeds):", "",
          "| Dataset | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    t = df.pivot_table(index="dataset", columns="model", values="n_untied").reindex(present)
    L.extend(f"| {ds} | " + " | ".join(f"{r[s]:.1f}" for s in systems) + " |" for ds, r in t.iterrows())
    L.append("")


def _blocks_on(df, L, order, systems) -> None:
    """The untied lead and gain comparisons on the eligible datasets (Holm across them)."""
    for title, col in (("Amendment 1: normalised deletion lead, untied points", "deletion_lead_untied"),
                       ("Amendment 1: deletion gain, untied points (not divided)", "deletion_gain_untied")):
        L.extend([f"## {title}", ""])
        for other in [s for s in OTHERS if s in systems]:
            res = compare(df, CFMNN_SYS, other, col, order)
            if res.empty:
                continue
            c = res.verdict.value_counts()
            L.append(f"**{CFMNN_SYS} vs {other}** ({col}): better {c.get('better', 0)}, "
                     f"level {c.get('level', 0)}, worse {c.get('worse', 0)}")
            L.extend(["", f"| Dataset | {CFMNN_SYS} | {other} | Mean difference | p | Holm p | Verdict |",
                      "|---|---|---|---|---|---|---|"])
            L.extend(f"| {r.dataset} | {r.a:.3f} | {r.b:.3f} | {r['diff']:+.3f} | {r.p:.4f} | "
                     f"{r.p_holm:.4f} | {r.verdict} |" for _, r in res.iterrows())
            L.append("")


def timing_table(out: Path, L, present, systems) -> None:
    """Amendment 1: the times measured alone, if the --timing pass has been run."""
    tp = out / "timing.csv"
    if not tp.exists():
        return
    t = pd.read_csv(tp)
    seeds = sorted(int(s) for s in t.seed.unique())
    sy = [s for s in systems if s in set(t.model)]
    piv = t.pivot_table(index="dataset", columns="model", values="ms_per_explanation")
    piv = piv.reindex([d for d in present if d in piv.index])
    L += [f"## Amendment 1: milliseconds per explanation, measured alone (seeds {seeds})", "",
          "One task at a time with nothing else running. These are the times to quote; the times in "
          "the run above were taken while 12 tasks shared the CPU.", "",
          "| Dataset | " + " | ".join(sy) + " | " + " | ".join(
              f"{CFMNN_SYS} ÷ {s}" for s in sy if s != CFMNN_SYS) + " |",
          "|---|" + "---|" * (2 * len(sy) - 1)]
    for ds, r in piv.iterrows():
        L.append(f"| {ds} | " + " | ".join(f"{r[s]:,.3f}" for s in sy) + " | " + " | ".join(
            f"{r[CFMNN_SYS] / r[s]:,.1f}×" for s in sy if s != CFMNN_SYS) + " |")
    L.append("")


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    present = [d for d in DATASETS if d in set(df.dataset)]
    systems = [s for s in SYSTEMS if s in set(df.model)]
    L = ["# Run G: CFMNN + PACE-SHAP against other models that explain themselves", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). "
         f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}). Each system is tested on its own "
         "predicted-class margin. Lead = how far the system's ranking beats a random ranking "
         "on the same model, as a share of the random ranking's AUC (larger is better). "
         "Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; "
         f"better / level / worse for {CFMNN_SYS}.", ""]

    def table(title, col, fmt):
        t = df.pivot_table(index="dataset", columns="model", values=col).reindex(present)
        L.extend([f"## {title}", "", "| Dataset | " + " | ".join(systems) + " |",
                  "|---|" + "---|" * len(systems)])
        L.extend(f"| {ds} | " + " | ".join(format(r[s], fmt) for s in systems) + " |"
                 for ds, r in t.iterrows())
        L.append("")

    def block(title, col, lower=False, fmt="+.3f"):
        L.extend([f"## {title}", ""])
        views = [("per dataset", df, present)]
        if set(SYNTHETIC) <= set(present):
            views.append(("synthetic family as one unit", units(df, col),
                          [FAMILY] + [d for d in present if d not in SYNTHETIC]))
        for other in [s for s in OTHERS if s in systems]:
            for label, data, order in views:
                res = compare(data, CFMNN_SYS, other, col, order, lower)
                if res.empty:
                    continue
                c = res.verdict.value_counts()
                L.append(f"**{CFMNN_SYS} vs {other}**, {label} ({col}): better {c.get('better', 0)}, "
                         f"level {c.get('level', 0)}, worse {c.get('worse', 0)}")
                L.extend(["", f"| Dataset | {CFMNN_SYS} | {other} | Mean difference | p | Holm p | Verdict |",
                          "|---|---|---|---|---|---|---|"])
                L.extend(f"| {r.dataset} | {format(r.a, fmt[1:])} | {format(r.b, fmt[1:])} | "
                         f"{format(r['diff'], fmt)} | {r.p:.4f} | {r.p_holm:.4f} | {r.verdict} |"
                         for _, r in res.iterrows())
                L.append("")

    table("Normalised deletion lead (mean over seeds)", "deletion_lead", ".3f")
    block("Primary: normalised deletion lead", "deletion_lead")
    L.extend(["## Primary with the near-zero rule", "",
              f"Where any seed of a compared pair has a random deletion AUC within {NEAR_ZERO} "
              "of zero, the verdict is read from the deletion gain (pre-registered).", ""])
    for other in [s for s in OTHERS if s in systems]:
        for label, res in near_zero_rule(df, other, present).items():
            c = res.verdict.value_counts()
            L.append(f"**{CFMNN_SYS} vs {other}**, {label}: better {c.get('better', 0)}, "
                     f"level {c.get('level', 0)}, worse {c.get('worse', 0)}; read from the gain: "
                     + (", ".join(res.dataset[res.from_gain]) or "none"))
            L.append("")
    amendment_1(df, L, present, systems)
    block("Secondary: normalised insertion lead", "insertion_lead")
    block("Secondary: test accuracy (%)", "accuracy", fmt="+.2f")
    block("Secondary: macro-F1 (%)", "macro_f1", fmt="+.2f")
    block("Secondary: milliseconds per explanation (lower is better)", "ms_per_explanation",
          lower=True, fmt="+.3f")
    block("Sensitivity: deletion gain (random minus method, not divided)", "deletion_gain")
    block("Sensitivity: insertion gain (method minus random, not divided)", "insertion_gain")

    L += ["## Descriptive", ""]
    table("Size: rules (boxes, terms, leaves)", "rules", ".1f")
    table("Conditions per rule", "conditions_per_rule", ".2f")
    table("Deletion margin AUC of the system (lower is better)", "deletion_auc", ".3f")
    table("Deletion margin AUC of a random ranking on the same model", "random_deletion_auc", ".3f")
    table("Insertion margin AUC of the system (higher is better)", "insertion_auc", ".3f")
    table("Insertion margin AUC of a random ranking on the same model", "random_insertion_auc", ".3f")
    table("Share of explained points whose margin is zero", "share_zero_margin", ".3f")
    # The lead divides by the random ranking's AUC, so it is unstable when that is near zero.
    near0 = df[df.random_deletion_auc.abs() < NEAR_ZERO].groupby("model").size()
    L += [f"Runs whose random-ranking deletion AUC is within {NEAR_ZERO} of zero (lead unstable): "
          + ", ".join(f"{s} {int(near0.get(s, 0))} of {int((df.model == s).sum())}" for s in systems),
          ""]
    tree = df[df.model == TREE_SYS]
    if len(tree):
        L += ["Leaves the tree was allowed (count over seeds):", ""]
        L += [f"- {ds}: " + ", ".join(f"{int(k)} ×{v}" for k, v in g.setting.value_counts().items())
              for ds, g in tree.groupby("dataset", sort=False)]
        L.append("")
    timing_table(out, L, present, systems)
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run G: explainable systems compared")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS)
    ap.add_argument("--n-explain", type=int, default=N_EXPLAIN)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--timing", action="store_true",
                    help="amendment 1: time the systems only, alone (seed 120 unless --seeds)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if a.timing:
        if a.jobs != 1:
            raise SystemExit("--timing must run alone (--jobs 1), or the times are distorted")
        tcsv = out / "timing.csv"
        tasks = [(d, s, a.n_explain, a.systems) for d in a.datasets or DATASETS
                 for s in a.seeds or TIMING_SEEDS]
        run_all(time_one, pending(tcsv, tasks, a.resume, a.fresh), tcsv, 1)
    elif not a.summary_only:
        tasks = [(d, s, a.n_explain, a.systems) for d in a.datasets or DATASETS
                 for s in a.seeds or SEEDS]
        # Slowest first (EBM's fit is most of the time). This only orders the tasks.
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs,
                {"letter": 6, "shuttle": 5, "S1": 2, "S2": 2, "S3": 2, "magic": 1, "landsat": 1})
    if csv.exists():
        print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
