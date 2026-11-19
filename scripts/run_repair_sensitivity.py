"""Run A: repair budgets, gates and certificates.

Rules: results/repair_sensitivity/PREREGISTRATION.md. Plan: research_audit/LIMITATIONS_PLAN.md, run A.

RQ3's split and SR arm (`scripts/run_refinement_v2.py`) on the same 6 datasets, theta = 0.3
(`CFMNN()` defaults), the volume cut, gated on the training rows. Every arm starts from the same
fitted model B0 and uses the same seeded order of training errors:

* B0: no repair;
* SR-default: 300 edits (`min(300, len(pool) // 4)`), gate 0.5 / 1.0;
* budgets at that gate: SR-b100 (100 edits), SR-b1000 (1,000), SR-ball (every training error);
* gates at the default budget: SR-gate-none (both limits infinite), SR-gate-strict (0 / 0),
  SR-gate-loose (1.0 / 2.0), SR-gate-cumulative (0.5 / 1.0 against the model before any edit).

All arms keep RQ3's box cap (1.1 times the start). The cached gate (`fast_gate=True`) is used only
if run E's summary says **Verdict: PASS** (`results/fast_gate_check/summary.md`).

Certificates, on SR-default only, on 100 test rows: for every accepted edit, each certificate is
scored against a brute-force recomputation before and after the edit (misses, precision, size):
loose and tight (`audit.certify`), centroid (`audit.certify_centroid`), and PACE-flip and
PACE-decomposition (`audit.explanation_certificate`). PACE-SHAP has none.

    python -B scripts/run_repair_sensitivity.py --jobs 12
    python -B scripts/run_repair_sensitivity.py --jobs 12 --resume
    python -B scripts/run_repair_sensitivity.py --summary-only
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

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.audit import (certify, certify_centroid, explanation_certificate,  # noqa: E402
                              prediction_churn, top_groups)
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.explain import PaceFlip, decomposition  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_readable_baselines import compare, tally  # noqa: E402
from run_refinement import MAX_BUDGET, N_ROUNDS, split  # noqa: E402

OUT = ROOT / "results" / "repair_sensitivity"
E_SUMMARY = ROOT / "results" / "fast_gate_check" / "summary.md"
DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = list(range(110, 120))
N_CERT, N_FLIP_BG, N_QUANTILES = 100, 200, 32     # certificate rows; RQ3's PACE-flip background
INF = float("inf")
# (arm, group, budget, max_f1_drop, max_recall_drop, gate_reference); budget "default" is RQ3's.
ARMS = [("SR-default", "reference", "default", 0.5, 1.0, "last"),
        ("SR-b100", "budget", 100, 0.5, 1.0, "last"),
        ("SR-b1000", "budget", 1000, 0.5, 1.0, "last"),
        ("SR-ball", "budget", "all", 0.5, 1.0, "last"),
        ("SR-gate-none", "gate", "default", INF, INF, "last"),
        ("SR-gate-strict", "gate", "default", 0.0, 0.0, "last"),
        ("SR-gate-loose", "gate", "default", 1.0, 2.0, "last"),
        ("SR-gate-cumulative", "gate", "default", 0.5, 1.0, "start")]
MODELS = ["B0"] + [a[0] for a in ARMS]
CERTS = ["loose", "tight", "centroid", "flip", "decomposition"]
CERT_CHANGE = {"loose": "pred", "tight": "pred", "centroid": "pred", "flip": "flip",
               "decomposition": "dec"}
REASONS = {"macro-F1": "rej_macro_f1", "class recall": "rej_class_recall", "box cap": "rej_box_cap",
           "already right": "skip_already_right", "not inside the winning box": "skip_not_inside"}
EXPL_TOL = 1e-12
# Slowest first: this only orders the tasks.
COST = {"skin": 6, "letter": 5, "magic": 4, "landsat": 3, "page_blocks": 2, "thyroid": 1}


def use_fast_gate(path: Path = E_SUMMARY) -> bool:
    """True only if run E's summary says Verdict: PASS."""
    try:
        return "**Verdict: PASS**" in Path(path).read_text(encoding="utf-8")
    except OSError:
        return False


class ExplainCertificates:
    """``on_accept`` hook: every certificate of every accepted edit, scored against brute force.

    The brute-force truth before an edit is the state after the previous accepted one (a
    rejected edit is rolled back), so each model's predictions and explanations are computed once.
    """

    def __init__(self, X, background):
        self.X, self.bg = X, background
        self.values = np.quantile(background, (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES, axis=0)
        self.n_edits = 0
        self.misses = dict.fromkeys(CERTS, 0)
        self.flagged = dict.fromkeys(CERTS, 0)
        self.changed = dict.fromkeys(("pred", "flip", "dec"), 0)
        self.caught = dict.fromkeys(CERTS, 0)
        self._last = None

    def _state(self, model):
        key = (model.V_, model.W_, model.centroid_, model.box_class_)
        if self._last is not None and all(a.shape == b.shape and np.array_equal(a, b)
                                          for a, b in zip(key, self._last[0])):
            return self._last[1]
        tied, cls = top_groups(model, self.X)
        st = {"tied": tied, "cls": cls, "flip": PaceFlip(model, self.bg).explain(self.X),
              "phi": decomposition(model, self.X, self.bg).phi}
        return st

    def __call__(self, before, after, removed):
        sb = self._state(before)
        sa = self._state(after)
        self._last = ((after.V_.copy(), after.W_.copy(), after.centroid_.copy(),
                       after.box_class_.copy()), sa)
        truth = {"pred": sb["cls"] != sa["cls"],
                 "flip": np.abs(sb["flip"] - sa["flip"]).max(axis=1) > EXPL_TOL,
                 "dec": np.abs(sb["phi"] - sa["phi"]).max(axis=1) > EXPL_TOL}
        loose, tight = certify(before, after, sb["tied"], sa["tied"], removed)
        flip, dec = explanation_certificate(before, after, self.X, self.values, removed)
        flags = {"loose": loose, "tight": tight,
                 "centroid": certify_centroid(before, after, self.X, removed),
                 "flip": flip, "decomposition": dec}
        self.n_edits += 1
        for kind, v in truth.items():
            self.changed[kind] += int(v.sum())
        for name in CERTS:
            ch = truth[CERT_CHANGE[name]]
            self.misses[name] += int((ch & ~flags[name]).sum())
            self.flagged[name] += int(flags[name].sum())
            self.caught[name] += int((ch & flags[name]).sum())

    def summary(self) -> dict:
        n = len(self.X) * max(1, self.n_edits)
        out = {"cert_edits": self.n_edits,
               "cert_pred_changes": self.changed["pred"], "cert_flip_changes": self.changed["flip"],
               "cert_dec_changes": self.changed["dec"]}
        for name in CERTS:
            f = self.flagged[name]
            out[f"cert_{name}_misses"] = self.misses[name]
            out[f"cert_{name}_size_share"] = 100 * f / n
            out[f"cert_{name}_precision"] = 100 * self.caught[name] / f if f else np.nan
        return out


COLUMNS = (["dataset", "seed", "arm", "group", "gate_impl", "budget", "max_f1_drop",
            "max_recall_drop", "gate_reference", "n_train", "n_pool", "n_test", "boxes_start",
            "boxes_end", "train_error_start", "train_error_end", "test_accuracy", "test_macro_f1",
            "test_min_class_recall", "test_balanced_accuracy", "edits_logged", "edits_accepted"]
           + list(REASONS.values()) + ["churn_changed", "churn_fixed", "churn_broken", "seconds"]
           + ["cert_edits", "cert_pred_changes", "cert_flip_changes", "cert_dec_changes"]
           + [f"cert_{c}_{m}" for c in CERTS for m in ("misses", "size_share", "precision")])


def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    P = split(name, seed)
    (Xtr, ytr), (Xp, _), (Xt, yt) = P["train"], P["pool"], P["test"]
    model = CFMNN().fit(Xtr, ytr)
    fast = use_fast_gate()
    rng = np.random.default_rng(seed + 3)
    Xcert = Xt[rng.permutation(len(Xt))[:N_CERT]]
    bg = Xtr[:N_FLIP_BG]
    p0 = np.asarray(model.predict(Xt))
    err0 = 100 * float(np.mean(model.predict(Xtr) != ytr))
    base = {c: np.nan for c in COLUMNS}
    base.update({"dataset": name, "seed": seed, "gate_impl": "cached" if fast else "standard",
                 "n_train": len(ytr), "n_pool": len(Xp), "n_test": len(yt),
                 "boxes_start": model.n_boxes_, "train_error_start": err0})
    METR = {"accuracy": "test_accuracy", "macro_f1": "test_macro_f1",
            "min_class_recall": "test_min_class_recall", "balanced_accuracy": "test_balanced_accuracy"}

    def record(arm, group, budget, f1d, recd, ref, m, secs, edits=None, cert=None):
        pred = np.asarray(m.predict(Xt))
        met = classification_metrics(yt, pred)
        churn = prediction_churn(yt, p0, pred)
        rec = {**base, "arm": arm, "group": group, "budget": budget, "max_f1_drop": f1d,
               "max_recall_drop": recd, "gate_reference": ref, "boxes_end": m.n_boxes_,
               "train_error_end": 100 * float(np.mean(m.predict(Xtr) != ytr)),
               **{col: met[k] for k, col in METR.items()},
               "churn_changed": churn["changed"], "churn_fixed": churn["fixed"],
               "churn_broken": churn["broken"], "seconds": secs}
        if edits is not None:
            rec.update({"edits_logged": len(edits), "edits_accepted": sum(bool(e["accepted"]) for e in edits),
                        **{col: sum(e["reason"] == r for e in edits) for r, col in REASONS.items()}})
        if cert is not None:
            rec.update(cert.summary())
        print(f"  {name} seed {seed} {arm}: macro-F1 {rec['test_macro_f1']:.2f}, boxes "
              f"{rec['boxes_end']}, accepted {rec['edits_accepted']}, {secs:.0f}s", flush=True)
        return rec

    rows = [record("B0", "reference", 0, np.nan, np.nan, "", model, 0.0)]
    default_budget = min(MAX_BUDGET, len(Xp) // 4)
    for arm, group, budget, f1d, recd, ref in ARMS:
        n_edits = default_budget if budget == "default" else len(ytr) if budget == "all" else budget
        cert = ExplainCertificates(Xcert, bg) if arm == "SR-default" else None
        t0 = time.perf_counter()
        m, _, edits = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=n_edits,
                                      n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False,
                                      max_f1_drop=f1d, max_recall_drop=recd, on_accept=cert,
                                      fast_gate=fast, gate_reference=ref)
        secs = time.perf_counter() - t0
        rows.append(record(arm, group, n_edits, f1d, recd, ref, m, secs, edits, cert))
    return rows


# ------------------------------------------------------------------ summary
def primaries(df: pd.DataFrame, col: str = "test_macro_f1"):
    """Each arm vs SR-default: paired two-sided Wilcoxon over seeds, Holm across datasets."""
    d = df.rename(columns={"arm": "model"})
    return {a[0]: compare(d, a[0], "SR-default", col) for a in ARMS if a[0] != "SR-default"}


def summarise(df: pd.DataFrame, out: Path) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    ds = [d for d in DATASETS if d in set(df.dataset)]
    models = [m for m in MODELS if m in set(df.arm)]
    L = ["# Run A: repair budgets, gates and certificates", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). RQ3's split and SR arm, seeds "
         f"{seeds[0]}–{seeds[-1]} ({len(seeds)}), gate: {', '.join(sorted(set(df.gate_impl)))}. "
         "Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; better / "
         "level / worse for the first arm. Written by `scripts/run_repair_sensitivity.py`.", ""]
    L += ["## Primary: each arm vs SR-default, test macro-F1", ""]
    for arm, res in primaries(df).items():
        if res.empty:
            continue
        L += [f"**{arm} vs SR-default**: {tally(res)}", "",
              "| Dataset | Mean difference (points) | p | Holm p | Verdict |", "|---|---|---|---|---|"]
        L += [f"| {r.dataset} | {r['diff']:+.2f} | {r.p:.4f} | {r.p_holm:.4f} | {r.verdict} |"
              for _, r in res.iterrows()]
        L.append("")
    for title, col, fmt in (("Test macro-F1 (%)", "test_macro_f1", "{:.2f}"),
                            ("Test accuracy (%)", "test_accuracy", "{:.2f}"),
                            ("Lowest class recall (%)", "test_min_class_recall", "{:.2f}"),
                            ("Boxes at the end", "boxes_end", "{:.0f}"),
                            ("Training error at the end (%)", "train_error_end", "{:.2f}"),
                            ("Edits accepted", "edits_accepted", "{:.0f}"),
                            ("Test predictions fixed against B0", "churn_fixed", "{:.1f}"),
                            ("Test predictions broken against B0", "churn_broken", "{:.1f}"),
                            ("Seconds", "seconds", "{:.0f}")):
        piv = df.pivot_table(index="dataset", columns="arm", values=col).reindex(index=ds, columns=models)
        L += [f"## {title}, mean over seeds", "", "| Dataset | " + " | ".join(models) + " |",
              "|---|" + "---|" * len(models)]
        L += [f"| {d} | " + " | ".join("" if np.isnan(v) else fmt.format(v) for v in r) + " |"
              for d, r in piv.iterrows()]
        L.append("")
    rep = df[df.arm != "B0"]
    L += ["## Edits rejected or skipped, by reason (mean over seeds)", ""]
    cols = list(REASONS.values())
    means = rep.groupby(["dataset", "arm"], sort=False)[cols].mean().round(1)
    L += ["```", means.to_string(), "```", ""]
    cert = df[df.arm == "SR-default"]
    if cert.cert_edits.notna().any():
        L += ["## Certificates on SR-default (100 test rows, every accepted edit)", "",
              "Misses: rows whose prediction or explanation changed but the certificate did not "
              "flag them (sum over seeds; any miss is a failure). Size: share of rows flagged per "
              "edit (smaller is tighter). Precision: share of flagged rows that really changed "
              "(mean over seeds).", ""]
        head = ["Certificate", "Misses", "Size (% of rows per edit)", "Precision (%)"]
        L += ["| " + " | ".join(head) + " |", "|---|---|---|---|"]
        for c in CERTS:
            L.append(f"| {c} | {int(cert[f'cert_{c}_misses'].sum())} | "
                     f"{cert[f'cert_{c}_size_share'].mean():.2f} | "
                     f"{cert[f'cert_{c}_precision'].mean():.2f} |")
        L += ["", f"Edits scored: {int(cert.cert_edits.sum())}. Rows changed (prediction / PACE-flip / "
              f"PACE-decomposition, summed over edits): {int(cert.cert_pred_changes.sum())} / "
              f"{int(cert.cert_flip_changes.sum())} / {int(cert.cert_dec_changes.sum())}.", ""]
        by = cert.groupby("dataset", sort=False)
        L += ["Per dataset (mean size %, then misses):", "", "```"]
        tab = pd.concat({"size%": by[[f"cert_{c}_size_share" for c in CERTS]].mean().rename(
            columns=lambda x: x[5:-11]),
            "misses": by[[f"cert_{c}_misses" for c in CERTS]].sum().rename(
                columns=lambda x: x[5:-7])}, axis=1).reindex(ds)
        L += [tab.round(2).to_string(), "```", ""]
        if int(cert["cert_centroid_misses"].sum()) > 0:
            L += ["**The centroid certificate missed, so it is dropped.**", ""]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run A: repair budgets, gates and certificates")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue a stopped run")
    ap.add_argument("--fresh", action="store_true", help="delete earlier results and start over")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "runs.csv"
    if not a.summary_only:
        tasks = [(d, s) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, a.jobs, COST)
    print(summarise(pd.read_csv(csv), out))


if __name__ == "__main__":
    main()
