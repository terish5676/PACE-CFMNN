"""Re-analysis of committed results for the tough review (reviews/tough_review_2026-11-16.md).

Covers D1, D2, D3, S7, B5 and B6 (parts 1 and 3). Writes ``results/tough_review/``:

* ``reanalysis.md``, the report;
* ``d1_primaries.csv``, one row per primary comparison and dataset;
* ``rq2_theta.png`` and ``rq2_theta.svg``, the accuracy-size-churn figure (B5);
* ``repair_gain.csv`` and ``churn_bound.csv``, the tables of B6.

Nothing is refitted: every number comes from a committed ``runs.csv``. Post hoc: none of this
was pre-registered. Each section names the review item it answers.

Rules fixed here before any number below was computed (2026-11-16):

* **D1, the corrected test.** The Nadeau-Bengio corrected resampled t-test (Nadeau and Bengio,
  2003) on the per-seed differences d_1..d_J: t = mean(d) / sqrt((1/J + n2/n1) var(d)), J - 1
  degrees of freedom, two-sided. n1 is the number of training rows and n2 the number of held-out
  rows of the split the measure comes from:
    - runs on the paper's 80/20 split: n2/n1 = n_test/n_train where a run saved them, else 0.25;
    - repair runs on the role split (50/20/10/10/10): n_test/n_train as saved (about 0.2);
    - trust-signal runs: n_eval/n_train on the role split (the pool, acceptance and test rows
      are scored, about 0.8), and (n_eval/5)/n_train under 5-fold cross-validation (one fold's
      held-out rows over one fold's training rows); the synthetic family takes the mean over its
      sets; runs 2 and 3 of the trust signals use run 1's ratios (same datasets and splits).
  If all differences are equal, p = 1 when they are 0 and 0 otherwise.
* **D1, the families.** Holm across the datasets of one comparison, as in the paper; and Holm
  across every comparison and dataset of the run's primaries ("run-level"). The verdict is
  better / not significant / worse at 0.05, in the direction of the comparison's measure.
* **D1, which comparisons.** Every primary comparison in Appendix E (rev 575) that is a paired
  test over seeds. Left out: the interaction test (a share of seeds), and the checks (pass or
  fail). The readable-models run also named CFMNN-0.3 vs GFMM-0.3 as primary (its P4), which
  Appendix E leaves out; it is included and flagged.
* **D2.** One value per unit (the mean difference over seeds), a two-sided signed-rank test
  across units. The synthetic sets count as one unit where noted (per-seed mean, as RQ2 does).
* **D3.** Equivalence (TOST, both one-sided Wilcoxon tests, Holm across datasets) at margins of
  5%, 10% and 20% of KernelSHAP's lead over the random ranking.
* **B6, part 1.** The bound is the one-sided 95% Clopper-Pearson upper limit on the share of the
  audit rows (min(2,000, test rows)) that fall in the union of the tight certificates of an
  arm's accepted edits; it is set beside the share of all test rows whose prediction changed.
* **B6, part 3.** Spearman correlations across the 13 datasets, between repair's mean gain in
  accuracy and (a) the starting model's training error, (b) the share of tied test predictions.

Changed after the first computation, and only these: wording and formatting of the report, a
paragraph explaining the Wilcoxon floor under run-level Holm, and the B6 sentence on datasets
without training errors (threshold "under 0.05%" instead of "exactly 0", since S2's mean is 0.01%).

    python -B scripts/tough_review_reanalysis.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import peer_review_reanalysis as pr  # noqa: E402
import review_reanalysis as rr  # noqa: E402
from accuracy_table import DATASETS as D_ACC  # noqa: E402
from pace_cfmnn.audit import churn_bound  # noqa: E402
from run_error_detection import MIN_ERRORS, SYNTHETIC, holm  # noqa: E402

RES = ROOT / "results"
OUT = RES / "tough_review"
RQ1, D13 = rr.RQ1, rr.D13
RQ3 = pr.RQ3
ALPHA = 0.05


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(RES / rel)


def order(datasets) -> list[str]:
    """Datasets in the paper's order (D13, then the accuracy table's), others after."""
    seen = list(dict.fromkeys(datasets))
    ref = [d for d in ["Synthetic family"] + D13 + D_ACC if d in seen]
    return list(dict.fromkeys(ref + seen))


# ------------------------------------------------------------------ the corrected test
def nb_p(d: np.ndarray, ratio: float) -> float:
    """Nadeau-Bengio corrected resampled t-test, two-sided."""
    d = np.asarray(d, float)
    J = len(d)
    if J < 2:
        return 1.0
    m, v = d.mean(), d.var(ddof=1)
    if v == 0:
        return 1.0 if m == 0 else 0.0
    t = m / np.sqrt((1.0 / J + ratio) * v)
    return float(2 * stats.t.sf(abs(t), J - 1))


def w_p(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    return float(stats.wilcoxon(d).pvalue) if len(d) > 1 and (d != 0).any() else 1.0


def verdict(p, diff, lower):
    good = np.where(lower, np.asarray(diff) < 0, np.asarray(diff) > 0)
    return np.where(np.asarray(p) < ALPHA, np.where(good, "better", "worse"), "not significant")


def tally(v) -> str:
    v = pd.Series(v).value_counts()
    return f"{v.get('better', 0)} / {v.get('not significant', 0)} / {v.get('worse', 0)}"


# ------------------------------------------------------------------ difference frames
def paired(df, a, b, col, method="method", ratio=None, units=None) -> pd.DataFrame:
    """Per dataset and seed: a minus b in ``col``, and the Nadeau-Bengio ratio.

    ``ratio``: a number, a dict by dataset, or None to use n_test / n_train from the frame."""
    rows = []
    for ds in units or order(df.dataset):
        g = df[df.dataset == ds]
        x = g[g[method] == a].groupby("seed")[col].mean()
        y = g[g[method] == b].groupby("seed")[col].mean()
        d = (x - y.reindex(x.index)).dropna()
        if ratio is None:
            r = float((g.n_test / g.n_train).mean())
        elif isinstance(ratio, dict):
            r = float(ratio[ds])
        else:
            r = float(ratio)
        rows += [{"dataset": ds, "seed": int(s), "d": float(v), "ratio": r} for s, v in d.items()]
    return pd.DataFrame(rows)


def trust_ratio() -> dict:
    m = read("error_detection_confirmatory/runs.csv").drop_duplicates(["dataset", "seed"])
    g = m.groupby("dataset").agg(n_train=("n_train", "mean"), n_eval=("n_eval", "mean"),
                                 split=("split", "first"))
    r = np.where(g.split.str.contains("CV"), g.n_eval / 5 / g.n_train, g.n_eval / g.n_train)
    out = dict(zip(g.index, r))
    elig = m.groupby("dataset").n_errors.min() >= MIN_ERRORS
    syn = [d for d in SYNTHETIC if elig.get(d, False)]
    out["Synthetic family"] = float(np.mean([out[d] for d in syn]))
    return out


def trust_units(rel: str, metric="auprc") -> pd.DataFrame:
    """The units of the trust-signal runs: the synthetic family (per-seed mean over the eligible
    sets) and each eligible real dataset; eligible = at least MIN_ERRORS errors in every seed."""
    df = read(rel)
    elig = df.drop_duplicates(["dataset", "seed"]).groupby("dataset").n_errors.min() >= MIN_ERRORS
    ok = [d for d in elig.index if elig[d]]
    syn = [d for d in SYNTHETIC if d in ok]
    fam = (df[df.dataset.isin(syn)].groupby(["seed", "method"], as_index=False)[metric].mean()
           .assign(dataset="Synthetic family"))
    real = df[df.dataset.isin([d for d in ok if d not in SYNTHETIC])][["dataset", "seed", "method", metric]]
    return pd.concat([fam, real], ignore_index=True)


def cf_valid_pairs(df, a, b, col, ratio) -> pd.DataFrame:
    """Per seed: mean of a minus b over points where both counterfactuals are valid (the
    extensions run's rule, ``run_counterfactuals.paired_seeds``)."""
    valid = df.valid.astype(str) == "True"
    rows = []
    for ds in order(df.dataset):
        g = df[valid & (df.dataset == ds)]
        A = g[g.method == a].set_index(["seed", "point"])[col].astype(float)
        B = g[g.method == b].set_index(["seed", "point"])[col].astype(float)
        per = (A - B.reindex(A.index)).dropna().groupby(level="seed").mean()
        rows += [{"dataset": ds, "seed": int(s), "d": float(v), "ratio": ratio} for s, v in per.items()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the primaries (Appendix E)
@dataclass
class Primary:
    run: str
    seeds: str
    name: str
    lower: bool
    frame: Callable[[], pd.DataFrame]
    note: str = ""


def primaries() -> list[Primary]:
    P = []
    tr = trust_ratio()

    # trust signals, confirmatory (10-19)
    u = trust_units("error_detection_confirmatory/runs.csv")
    P += [Primary("Trust signals, confirmatory", "10–19", "AUPRC, PD + U* vs the plain margin", False,
                  lambda u=u: paired(u, "PD + U*", "Margin", "auprc", ratio=tr)),
          Primary("Trust signals, confirmatory", "10–19", "AUPRC, PRS* vs PD + U*", False,
                  lambda u=u: paired(u, "PRS*", "PD + U*", "auprc", ratio=tr))]
    # first repair run (10-19)
    rc = read("refinement_confirmatory/runs.csv")
    P.append(Primary("First repair run", "10–19", "macro-F1, cut by the contrastive explanation vs by width",
                     False, lambda: paired(rc, "order_ensemble | A+B+C", "order_ensemble | A+B+C'-volume",
                                           "test_macro_f1", method="arm")))
    # trust signals, combination (20-29)
    u2 = trust_units("error_detection_v2/runs.csv")
    P.append(Primary("Trust signals, combination", "20–29",
                     "AUPRC, LR on kNN + TS + order ensemble vs on kNN + TS", False,
                     lambda: paired(u2, "LR: kNN + TS + order ens.", "LR: kNN + TS", "auprc", ratio=tr)))
    # RQ1, second run (30-39)
    v3 = read("faithfulness_v3/runs.csv")
    P.append(Primary("RQ1, second run", "30–39", "deletion margin AUC, PACE-SHAP vs KernelSHAP", True,
                     lambda: paired(v3, "PACE-SHAP", "KernelSHAP", "margin_auc", ratio=0.25, units=RQ1)))
    # extensions (40-49)
    cf = read("extensions_v1/counterfactuals/runs.csv")
    gi = read("extensions_v1/global_importance/runs.csv")
    cond = read("extensions_v1/conditional/runs.csv")
    real = read("extensions_v1/conditional/realism.csv")
    P += [Primary("Extensions", "40–49", "features changed, PACE-sparse vs DiCE", True,
                  lambda: cf_valid_pairs(cf, "PACE-sparse", "DiCE", "n_changed", 0.25)),
          Primary("Extensions", "40–49", "agreement with permutation importance, mean |PACE-SHAP| vs "
                  "mean |KernelSHAP|", False,
                  lambda: paired(gi, "Mean abs PACE-SHAP", "Mean abs KernelSHAP", "rho_Permutation importance",
                                 ratio=0.25)),
          Primary("Extensions", "40–49", "conditional deletion margin AUC, PACE-SHAP-cond vs "
                  "KernelSHAP-cond", True,
                  lambda: paired(cond, "PACE-SHAP-cond", "KernelSHAP-cond", "cond_margin_auc", ratio=0.25,
                                 units=D13)),
          Primary("Extensions", "40–49", "conditional insertion margin AUC, PACE-SHAP-cond vs "
                  "KernelSHAP-cond", False,
                  lambda: paired(cond, "PACE-SHAP-cond", "KernelSHAP-cond", "cond_insertion_margin_auc",
                                 ratio=0.25, units=D13)),
          Primary("Extensions", "40–49", "share of inputs off the data, conditional vs usual game", True,
                  lambda: paired(real, "shap_cond", "shap_int", "off_share", ratio=0.25, units=D13))]
    # stability and ties (50-59)
    rv = read("review_v1/faithfulness/runs.csv")
    for col, label in (("stability_fixed", "rows kept"), ("stability_new", "new rows")):
        P.append(Primary("Stability and ties", "50–59", f"stability with {label}, PACE-SHAP vs KernelSHAP",
                         False, lambda col=col: paired(rv, "PACE-SHAP", "KernelSHAP", col, ratio=0.25,
                                                       units=RQ1)))
    # repair, second run (50-59)
    v2 = read("refinement_v2_confirmatory/runs.csv")
    P += [Primary("Repair, second run (RQ2)", "50–59", "macro-F1, SR + AL vs SR", False,
                  lambda: paired(v2, "SR+AL", "SR", "test_macro_f1", method="arm", units=RQ3)),
          Primary("Repair, second run (RQ2)", "50–59", "fix share, SR + AL vs shuffled retraining", False,
                  lambda: paired(v2, "SR+AL", "Retrain, shuffled", "churn_fix_share", method="arm",
                                 units=RQ3))]
    # ensembles (60-69)
    u3 = trust_units("review_v1/ensembles/runs.csv")
    P.append(Primary("Ensembles", "60–69", "AUPRC, order ensemble vs bagged ensemble, each added to kNN + TS",
                     False, lambda: paired(u3, "LR: kNN + TS + order ens.", "LR: kNN + TS + bagged ens.",
                                           "auprc", ratio=tr)))
    # readable models (70-89)
    rb = read("readable_baselines/runs.csv")
    for b in ("Tree", "LogReg", "EBM"):
        P.append(Primary("Readable models", "70–89", f"accuracy, CFMNN-val vs {b}", False,
                         lambda b=b: paired(rb, "CFMNN-val", b, "accuracy", method="model")))
    P.append(Primary("Readable models", "70–89", "accuracy, CFMNN-0.3 vs GFMM-0.3", False,
                     lambda: paired(rb, "CFMNN-0.3", "GFMM-0.3", "accuracy", method="model"),
                     "primary in the plan (P4), missing from Appendix E"))
    # repair at the chosen theta (70-89)
    rt = read("repair_theta/runs.csv")
    P += [Primary("Repair at the chosen θ", "70–89", "accuracy, SR-0.3 vs B0-val", False,
                  lambda: paired(rt, "SR-0.3", "B0-val", "accuracy", method="model")),
          Primary("Repair at the chosen θ", "70–89", "accuracy, SR-val vs B0-val", False,
                  lambda: paired(rt, "SR-val", "B0-val", "accuracy", method="model"))]
    # follow-up: more points (110-119)
    mp = read("rq1_more_points/runs.csv")
    for b in ("KernelSHAP", "KernelSHAP-same"):
        for col, lower, label in (("margin_auc", True, "deletion"), ("insertion_margin_auc", False, "insertion")):
            P.append(Primary("Follow-up: more points", "110–119", f"{label} margin AUC, PACE-SHAP vs {b}", lower,
                             lambda b=b, col=col: paired(mp, "PACE-SHAP", b, col, ratio=0.25, units=RQ1)))
    # follow-up: conditional settings (110-119)
    cs = read("conditional_settings/runs.csv")
    for st in dict.fromkeys(cs.setting):
        for col, lower, label in (("cond_margin_auc", True, "deletion"),
                                  ("cond_insertion_margin_auc", False, "insertion")):
            P.append(Primary("Follow-up: conditional settings", "110–119",
                             f"conditional {label}, PACE-SHAP-cond vs KernelSHAP-cond, {st}", lower,
                             lambda st=st, col=col: paired(cs[cs.setting == st], "PACE-SHAP-cond",
                                                           "KernelSHAP-cond", col, ratio=0.25)))
    # follow-up: counterfactuals (110-119)
    cr = read("cf_robustness/runs.csv")
    for a in ("PACE", "PACE-sparse"):
        for col, lower, label in (("r1", True, "distance moved after a nudge (r1)"),
                                  ("r3", False, "validity after a refit (r3)")):
            P.append(Primary("Follow-up: counterfactuals", "110–119", f"{label}, {a} vs DiCE random", lower,
                             lambda a=a, col=col: paired(cr, a, "DiCE random", col, ratio=0.25)))
    # follow-up: budget and gate (110-119)
    rs = read("repair_sensitivity/runs.csv")
    for a in ("SR-b100", "SR-b1000", "SR-ball", "SR-gate-none", "SR-gate-strict", "SR-gate-loose",
              "SR-gate-cumulative"):
        P.append(Primary("Follow-up: budget and gate", "110–119", f"macro-F1, {a} vs SR-default", False,
                         lambda a=a: paired(rs, a, "SR-default", "test_macro_f1", method="arm")))
    # self-repair vs contraction (120-129)
    ct = read("contraction/runs.csv")
    P.append(Primary("Self-repair vs contraction", "120–129", "macro-F1, SR vs contraction on every overlap",
                     False, lambda: paired(ct, "SR", "SC-all", "test_macro_f1", method="arm")))
    # whole systems (120-129)
    sy = read("systems/runs.csv")
    for b in ("EBM + terms", "Tree + TreeSHAP"):
        for col, label in (("deletion_lead", "normalised deletion lead"),
                           ("deletion_lead_untied", "normalised deletion lead, untied points (added)")):
            P.append(Primary("Whole systems", "120–129", f"{label}, CFMNN + PACE-SHAP vs {b}", False,
                             lambda b=b, col=col: paired(sy, "CFMNN + PACE-SHAP", b, col, method="model")))
    # repair order (130-139)
    ro = read("repair_order/runs.csv")
    P += [Primary("Repair order", "130–139", "macro-F1, greedy-300 vs random-300", False,
                  lambda: paired(ro, "greedy-300", "random-300", "test_macro_f1", method="arm")),
          Primary("Repair order", "130–139", "macro-F1, greedy-300 vs random-300 counting tries that reach "
                  "the gate (added)", False,
                  lambda: paired(ro, "greedy-300", "random-300-real", "test_macro_f1", method="arm"))]
    # gamma (150-159)
    ge = read("generality/runs.csv")
    for a in ("gamma=2", "gamma=8"):
        P.append(Primary("γ and Simpson's contraction", "150–159", f"accuracy, {a.replace('gamma', 'γ ')} vs γ = 4",
                         False, lambda a=a: paired(ge, a, "gamma=4", "test_accuracy", method="model")))
    # modern samplers (160-169)
    sb = read("sampled_baselines/runs.csv")
    for b in ("PermutationSHAP", "KernelSHAP-paired"):
        P.append(Primary("Modern samplers", "160–169", f"deletion margin AUC, PACE-SHAP vs {b}", True,
                         lambda b=b: paired(sb, "PACE-SHAP", b, "margin_auc", ratio=0.25)))
    # re-tune theta or repair (70-89, reused)
    import run_theta_churn as tc
    th, rt2 = read("theta_churn/runs.csv"), read("repair_theta/runs.csv")
    ratio = (th.n_test / th.n_train).groupby(th.dataset).mean().to_dict()
    P.append(Primary("Re-tune θ or repair", "70–89, reused", "negative flip rate, SR-0.3 vs B0-val", True,
                     lambda: paired(tc.p1_frame(th, rt2), "SR-0.3", "B0-val", "nfr", method="model",
                                    ratio=ratio)))
    return P


def run_d1() -> pd.DataFrame:
    rows = []
    for k, p in enumerate(primaries()):
        f = p.frame()
        for ds in order(f.dataset):
            d = f[f.dataset == ds].d.to_numpy()
            rows.append({"k": k, "run": p.run, "seeds": p.seeds, "comparison": p.name, "note": p.note,
                         "lower": p.lower, "dataset": ds, "n": len(d), "diff": d.mean(),
                         "ratio": f[f.dataset == ds].ratio.iloc[0], "p_w": w_p(d),
                         "p_nb": nb_p(d, f[f.dataset == ds].ratio.iloc[0])})
    t = pd.DataFrame(rows)
    for col in ("w", "nb"):
        t[f"p_{col}_holm"] = t.groupby("k")[f"p_{col}"].transform(lambda s: holm(s.to_numpy()))
        t[f"p_{col}_run"] = t.groupby("run")[f"p_{col}"].transform(lambda s: holm(s.to_numpy()))
    for col in ("w_holm", "w_run", "nb_holm", "nb_run"):
        t[f"v_{col}"] = verdict(t[f"p_{col}"], t["diff"], t.lower)
    return t


def s_d1(L, t):
    by = t.groupby("k", sort=True)
    changed = int((t.v_w_holm != t.v_nb_run).sum())
    sig = int((t.v_w_holm != "not significant").sum())
    lost = int(((t.v_w_holm != "not significant") & (t.v_nb_run == "not significant")).sum())
    gained = int(((t.v_w_holm == "not significant") & (t.v_nb_run != "not significant")).sum())
    flipped = int(((t.v_w_holm != "not significant") & (t.v_nb_run != "not significant")
                   & (t.v_w_holm != t.v_nb_run)).sum())
    L += ["", "## 1. Every primary comparison, corrected (D1)", "",
          f"{len(by)} primary comparisons from {t.run.nunique()} runs, {len(t)} dataset-level verdicts. "
          f"The paper's verdicts (Wilcoxon, Holm across datasets) call {sig} of them significant. With "
          f"the Nadeau–Bengio correction and Holm across each run's primaries, **{changed} verdicts change: "
          f"{lost} significant results become not significant, {gained} go the other way, and {flipped} "
          "change direction.**", "",
          "Columns give better / not significant / worse for the first method:", "",
          "- **Paper:** two-sided Wilcoxon over seeds, Holm across datasets;",
          "- **+ run Holm:** the same test, Holm across every primary comparison and dataset of the run;",
          "- **NB:** the Nadeau–Bengio corrected t-test, Holm across datasets;",
          "- **NB + run Holm:** both corrections. \"Survives\" counts the paper's significant verdicts "
          "that keep their direction here.",
          "- **Across units:** a signed-rank test of the per-dataset mean differences (D2).", "",
          "Two things drive the changes. First, the Wilcoxon test over 10 seeds cannot give p below "
          "0.00195, so Holm across k tests can reject only when 0.00195 × k < 0.05, that is k ≤ 25; in a "
          "run with more dataset-level tests, \"+ run Holm\" loses every verdict whatever the data (with 20 "
          "seeds the floor is 1.9e-6 and this does not bite). The t-test has no such floor. Second, the "
          "Nadeau–Bengio factor (1/J + n2/n1) against 1/J multiplies the variance by 3.5 for 10 seeds on "
          "the 80/20 split, by 6 for 20 seeds, and by 7–9 for the trust-signal runs, which score 40% of the "
          "rows. Several plans said their primaries answer separate questions and are not corrected for "
          "each other; the run-level columns are the review's stricter reading, not the plans'.", "",
          "| Run (seeds) | Comparison | Units | Paper | + run Holm | NB | NB + run Holm | Survives | "
          "Across units p |", "|---|---|---|---|---|---|---|---|---|"]
    for k, g in by:
        r0 = g.iloc[0]
        s = int(((g.v_w_holm != "not significant") & (g.v_w_holm == g.v_nb_run)).sum())
        n_sig = int((g.v_w_holm != "not significant").sum())
        d = g["diff"].to_numpy()
        p_across = w_p(d) if len(d) > 1 else np.nan
        name = r0.comparison + (f" ({r0.note})" if r0.note else "")
        L.append(f"| {r0.run} ({r0.seeds}) | {name} | {len(g)} | {tally(g.v_w_holm)} | {tally(g.v_w_run)} | "
                 f"{tally(g.v_nb_holm)} | {tally(g.v_nb_run)} | {s} of {n_sig} | {p_across:.3f} |")
    ch = t[t.v_w_holm != t.v_nb_run]
    L += ["", "**Verdicts that change** (paper → NB + run Holm):", "",
          "| Run | Comparison | Dataset | Mean difference | Paper Holm p | NB + run Holm p | Paper | Corrected |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in ch.iterrows():
        L.append(f"| {r.run} | {r.comparison} | {r.dataset} | {r['diff']:+.4g} | {r.p_w_holm:.3f} | "
                 f"{r.p_nb_run:.3f} | {r.v_w_holm} | {r.v_nb_run} |")
    L += ["", "Ratios n2/n1 used (mean over the run's datasets): " + "; ".join(
        f"{run} {g.ratio.mean():.2f}" for run, g in t.groupby("run", sort=False)) + "."]


# ------------------------------------------------------------------ D2
def across_line(t, label, sign=1.0, min_p=True):
    d = sign * t["diff"].to_numpy()
    p = w_p(d)
    n = len(d)
    floor = 2.0 / 2 ** n
    return (f"| {label} | {n} | {int((d > 0).sum())} of {n} | {np.median(d):+.3f} | {p:.4f} | "
            f"{floor:.4f} |")


def s_d2(L):
    L += ["", "## 2. Headlines across datasets (D2)", "",
          "One value per unit (the mean difference over seeds), a two-sided signed-rank test across units. "
          "Positive favours the first method (for deletion AUC and the negative flip rate, where lower is "
          "better, the sign is flipped). The last column is the smallest p the test can give with that many "
          "units, when every unit points the same way.", "",
          "| Headline | Units | Units in favour | Median difference | p | Smallest possible p |",
          "|---|---|---|---|---|---|"]
    v3 = read("faithfulness_v3/runs.csv")
    cond = read("extensions_v1/conditional/runs.csv")
    for name, df, members, units0 in (("usual setting (30–39)", v3, pr.SYN_V3, RQ1),
                                      ("like for like (40–49)", cond, pr.SYN_COND, D13)):
        for metric, lower, m in (("margin_auc", True, "deletion"), ("insertion_margin_auc", False, "insertion")):
            sign = -1.0 if lower else 1.0
            t = rr.compare(df, "PACE-SHAP", "KernelSHAP", metric, lower, units0)
            L.append(across_line(t, f"RQ1 {name}, {m}, PACE-SHAP vs KernelSHAP, every dataset", sign))
            fam = pr.family(df, members)
            units = ["synthetic"] + [d for d in units0 if d not in members]
            t = rr.compare(fam, "PACE-SHAP", "KernelSHAP", metric, lower, units)
            L.append(across_line(t, f"RQ1 {name}, {m}, synthetic sets as one unit", sign))
    v2 = read("refinement_v2_confirmatory/runs.csv").rename(columns={"arm": "method"})
    for a, b in (("SR", "B0"), ("SR+AL", "SR"), ("SR+AL", "Retrain, appended"), ("SR+AL", "Retrain, shuffled")):
        t = rr.compare(v2, a, b, "test_macro_f1", False, RQ3)
        L.append(across_line(t, f"RQ2 macro-F1, {a} vs {b} (50–59)"))
    import run_theta_churn as tc
    p1 = tc.p1_frame(read("theta_churn/runs.csv"), read("repair_theta/runs.csv")).rename(
        columns={"model": "method"})
    sr = p1[p1.method == "SR-0.3"].groupby("dataset").changed_pct.mean()
    informative = [d for d in D_ACC if sr.get(d, 0) > 0]
    t = rr.compare(p1, "SR-0.3", "B0-val", "nfr", True, informative)
    L.append(across_line(t, f"RQ2 negative flip rate, repair vs re-tuning θ, the {len(informative)} datasets "
                            "where repair changed a prediction (70–89)", -1.0))
    t = rr.compare(p1, "SR-0.3", "B0-val", "nfr", True, [d for d in D_ACC if d in sr.index])
    L.append(across_line(t, "RQ2 negative flip rate, repair vs re-tuning θ, all 13 datasets", -1.0))
    rt = read("repair_theta/runs.csv").rename(columns={"model": "method"})
    for a, b in (("SR-0.3", "B0-val"), ("SR-val", "B0-val"), ("SR-0.3", "B0-0.3")):
        t = rr.compare(rt, a, b, "accuracy", False, [d for d in D_ACC if d in set(rt.dataset)])
        L.append(across_line(t, f"RQ2 accuracy, {a} vs {b} (70–89)"))
    L += ["", f"Datasets where repair changed a prediction: {', '.join(informative)}."]


# ------------------------------------------------------------------ D3
def s_d3(L):
    L += ["", "## 3. Equivalence at other margins (D3)", "",
          "TOST against KernelSHAP: both one-sided Wilcoxon tests against the margin, Holm across datasets. "
          "The margin is a share of KernelSHAP's lead over the random ranking on that dataset. The paper's "
          "count uses 10%, a margin fixed in the peer-review reanalysis after the runs but before its numbers "
          "were computed, so all of these are post hoc.", "",
          "| Comparison | Measure | Datasets | 5% | 10% | 20% |", "|---|---|---|---|---|---|"]
    v3 = read("faithfulness_v3/runs.csv")
    cond = read("extensions_v1/conditional/runs.csv")
    for name, df, ds in (("Usual setting (30–39)", v3, RQ1), ("Like for like (40–49)", cond, D13)):
        for metric, lower, m in (("margin_auc", True, "deletion"), ("insertion_margin_auc", False, "insertion")):
            c = []
            for share in (0.05, 0.10, 0.20):
                mg = pr.lead_margins(df, metric, lower, ds, share)
                c.append(int((pr.tost(df, "PACE-SHAP", "KernelSHAP", metric, ds, mg) < 0.05).sum()))
            L.append(f"| {name} | {m} | {len(ds)} | " + " | ".join(f"{x} of {len(ds)}" for x in c) + " |")


# ------------------------------------------------------------------ S7
def s_s7(L):
    L += ["", "## 4. Raw and untied scores lead (S7)", "",
          "The deletion and insertion scores divide each point's margin by its starting margin, floored at "
          "1e-12, and clip the ratio to [−1, 1] (`scripts/run_faithfulness.py:134, :148`). At a tie the "
          "starting margin is 0, so the curve becomes a sequence of signs. The tie test (seeds 50–59) and "
          "the more-points run (110–119) also saved the raw margin AUC (no division) and the scores on "
          "untied points. A dataset enters the untied rows if every seed has at least 10 untied points.", "",
          "PACE-SHAP vs KernelSHAP: better / not significant / worse (Wilcoxon over seeds, Holm across "
          "datasets).", "",
          "| Run | Measure | Deletion | Insertion | Datasets |", "|---|---|---|---|---|"]
    for run, rel, a, b in (("Tie test (50–59)", "review_v1/faithfulness/runs.csv", "PACE-SHAP", "KernelSHAP"),
                           ("More points (110–119)", "rq1_more_points/runs.csv", "PACE-SHAP", "KernelSHAP"),
                           ("More points, same rows (110–119)", "rq1_more_points/runs.csv", "PACE-SHAP",
                            "KernelSHAP-same")):
        df = read(rel)
        g = df[df.method == a]
        untied = (g.n - g.n_tied).groupby(g.dataset).min()
        ok = [d for d in RQ1 if untied.get(d, 0) >= 10]
        for label, dcol, icol, ds in (
                ("normalised and clipped (the pre-registered score), all points", "margin_auc",
                 "insertion_margin_auc", RQ1),
                ("raw margin, all points", "margin_auc_raw", "insertion_margin_auc_raw", RQ1),
                ("normalised, untied points", "margin_auc_untied", "insertion_margin_auc_untied", ok),
                ("raw margin, untied points", "margin_auc_raw_untied", "insertion_margin_auc_raw_untied", ok)):
            vd = rr.compare(df, a, b, dcol, True, ds).verdict.replace("level", "not significant")
            vi = rr.compare(df, a, b, icol, False, ds).verdict.replace("level", "not significant")
            L.append(f"| {run.replace('PACE', a)} | {label} | {tally(vd)} | {tally(vi)} | {len(ds)} |")
    rv = read("review_v1/faithfulness/runs.csv")
    g = rv[rv.method == "PACE-SHAP"]
    share = (100 * g.n_tied / g.n).groupby(g.dataset).mean().reindex(RQ1)
    L += ["", "Share of explained points that are tied (tie test, mean over seeds): " + ", ".join(
        f"{d} {v:.0f}%" for d, v in share.items()) + "."]


# ------------------------------------------------------------------ B6
def s_b6(L):
    v2 = read("refinement_v2_confirmatory/runs.csv")
    rows = []
    for _, r in v2[v2.arm.isin(["SR", "AL", "SR+AL"])].iterrows():
        m = int(min(2000, r.n_test))
        for kind in ("tight", "loose"):
            k = int(round(r[f"cert_{kind}_union_share"] / 100 * m))
            b = churn_bound(np.r_[np.ones(k, bool), np.zeros(m - k, bool)])
            rows.append({"dataset": r.dataset, "seed": r.seed, "arm": r.arm, "kind": kind, "m": m,
                         "cert_share": 100 * k / m, "bound": 100 * b["upper"],
                         "changed": r.churn_changed_share, "misses": r[f"cert_{kind}_misses"]})
    cb = pd.DataFrame(rows)
    cb.to_csv(OUT / "churn_bound.csv", index=False)
    t = cb[cb.kind == "tight"].groupby(["dataset", "arm"], sort=False).agg(
        changed=("changed", "mean"), cert=("cert_share", "mean"), bound=("bound", "mean"),
        holds=("bound", "size"), misses=("misses", "sum"))
    hold = cb[cb.kind == "tight"].assign(ok=lambda d: d.bound >= d.changed).groupby(
        ["dataset", "arm"], sort=False).ok.sum()
    L += ["", "## 5. When repair helps, and how much it can change (B6)", "",
          "**Part 1, the churn bound.** Before an edit is used, the share of unlabelled rows inside its "
          "certificate bounds how many predictions it can change (Proposition 4). For a sequence of edits, the "
          "union of their certificates does. Below: the repair run (seeds 50–59), tight certificates on the "
          "audit rows (up to 2,000 test rows), the one-sided 95% Clopper–Pearson upper limit on that share, "
          "and the share of all test rows whose prediction actually changed. Means over the 10 seeds.", "",
          "| Dataset | Arm | Predictions changed (%) | In the certificates (%) | 95% bound (%) | "
          "Seeds where the bound holds | Missed changes |", "|---|---|---|---|---|---|---|"]
    for (ds, arm), r in t.iterrows():
        L.append(f"| {ds} | {arm} | {r.changed:.2f} | {r.cert:.1f} | {r.bound:.1f} | {hold[(ds, arm)]} of "
                 f"{int(r.holds)} | {int(r.misses)} |")
    ratio = (t.bound / t.changed.where(t.changed > 0)).replace(np.inf, np.nan)
    L += ["", f"The bound held in {int(hold.sum())} of {int(t.holds.sum())} arm-seeds. It is loose: "
          f"{ratio.min():.0f}–{ratio.max():.0f} times the actual change (median {ratio.median():.0f}), "
          "because a certificate flags every row whose top group the edited box can touch, not the rows "
          "whose class it changes. It is a guarantee for an operator, not an estimate."]

    rt = read("repair_theta/runs.csv")
    ties = read("peer_review/ties_all.csv").groupby("dataset").sum(numeric_only=True)
    tie = 100 * ties.n_tied / ties.n
    m = rt.pivot_table(index="dataset", columns="model", values="accuracy")
    b0 = rt[rt.model == "B0-0.3"].groupby("dataset")[["train_error_pct", "boxes"]].mean()
    acc = rt[rt.model == "SR-0.3"].groupby("dataset").edits_accepted.mean()
    g = pd.DataFrame({"train_error_pct": b0.train_error_pct, "tied_test_pct": tie,
                      "gain_0.3": m["SR-0.3"] - m["B0-0.3"], "gain_val": m["SR-val"] - m["B0-val"],
                      "edits_accepted": acc}).reindex([d for d in D_ACC if d in m.index])
    g.to_csv(OUT / "repair_gain.csv")
    rho = {(x, y): stats.spearmanr(g[x], g[y]) for x in ("train_error_pct", "tied_test_pct")
           for y in ("gain_0.3", "gain_val")}
    zero = g[g.train_error_pct < 0.05]
    L += ["", "**Part 3, which datasets gain.** Repair's gain in test accuracy (points, mean over seeds 70–89) "
          "beside the starting model's training error and its share of tied test predictions (θ = 0.3; ties "
          "from the base model on seeds 0–4).", "",
          "| Dataset | Training error (%) | Tied test predictions (%) | Gain at θ = 0.3 | Gain at θ by "
          "validation | Edits accepted at θ = 0.3 |", "|---|---|---|---|---|---|"]
    for ds, r in g.iterrows():
        L.append(f"| {ds} | {r.train_error_pct:.2f} | {r.tied_test_pct:.1f} | {r['gain_0.3']:+.2f} | "
                 f"{r.gain_val:+.2f} | {r.edits_accepted:.0f} |")
    L += ["", "Spearman correlations across the 13 datasets:", ""]
    for (x, y), res in rho.items():
        L.append(f"- {y.replace('gain_', 'gain at θ = ').replace('= val', 'by validation')} vs "
                 f"{x.replace('_pct', ' (%)').replace('_', ' ')}: ρ = {res.statistic:+.2f}, p = {res.pvalue:.3f}")
    L += ["", f"With training error under 0.05% ({', '.join(zero.index)}) repair accepted "
          f"{zero.edits_accepted.max():.0f} edits and gained {zero['gain_0.3'].abs().max():.2f} points at most: "
          "self-repair fixes training errors, so with none there is nothing to fix."]
    return g


# ------------------------------------------------------------------ B5
def b5_figure(L):
    import rq2_figure
    path = rq2_figure.draw(read("theta_churn/runs.csv"), read("repair_theta/runs.csv"), OUT)
    L += ["", "## 6. Accuracy, size and broken predictions across θ (B5)", "",
          f"![Accuracy against boxes for every θ, with repair](rq2_theta.png)", "",
          f"Drawn by `scripts/rq2_figure.py` ({path.name}, and an SVG beside it). One panel per dataset: "
          "CFMNN's mean test accuracy against its mean number of boxes (log scale) for θ from 0.5 to 0.05, "
          "and self-repair at θ = 0.3 and at θ chosen by validation. A marker's area is the share of test "
          "predictions the model at θ = 0.3 got right that the update gets wrong (the negative flip rate); "
          "for repair at the chosen θ it is against the model it repairs. Means over seeds 70–89."]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = run_d1()
    t.to_csv(OUT / "d1_primaries.csv", index=False)
    L = ["# Tough review: re-analysis of committed results", "",
         "Written by `scripts/tough_review_reanalysis.py` from committed `runs.csv` files; nothing was refitted. "
         "Post hoc: none of this was pre-registered. The rules (the test, the ratios, the families, the "
         "margins) are in the script's docstring, fixed before any number was computed."]
    s_d1(L, t)
    s_d2(L)
    s_d3(L)
    s_s7(L)
    s_b6(L)
    b5_figure(L)
    (OUT / "reanalysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'reanalysis.md'}")


if __name__ == "__main__":
    main()
