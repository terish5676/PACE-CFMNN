"""Self-repair as the method: the RQ3 time and gain tables, and RQ4 for self-repair alone.

Writes results/sr_method/tables.md from saved results only:

* results/refinement_v2_confirmatory/runs.csv (RQ3, seeds 50-59): metrics, labels, churn and
  certificates;
* results/repair_timing/runs.csv (Run T, seed 50, one machine): times. Until Run T has run, the
  time section says so.

What is pre-registered and what is not:

* the gains over B0 are means of paired differences; markers come only from RQ3's
  pre-registered endpoints, recomputed with the same function (`run_refinement_v2.test_endpoint`)
  and checked against RQ3's summary.md: `*` for S3a (SR over B0) and P3 (SR+AL over SR), two-sided;
  `†` for S3d and S3c (lowest class recall no worse than B0 within 2 points), one-sided.
  Every other cell is descriptive;
* RQ4 for self-repair alone (SR against shuffled retraining) is post hoc: RQ3's P4, S4a and S4b
  tests applied to SR instead of SR+AL. The retrain used every label SR+AL used; SR used none.

    python -B scripts/sr_method_tables.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_refinement_v2 import ENDPOINTS, F1_MARGIN, RECALL_MARGIN, test_endpoint  # noqa: E402

RQ3 = ROOT / "results" / "refinement_v2_confirmatory"
RUN_T = ROOT / "results" / "repair_timing" / "runs.csv"
OUT = ROOT / "results" / "sr_method"
DS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
NAMES = {"thyroid": "Thyroid", "landsat": "Landsat", "page_blocks": "Page Blocks",
         "magic": "MAGIC", "letter": "Letter", "skin": "Skin"}
SETUPS = ["Train CFMNN", "SR, old gate", "SR, cached gate", "AL", "SR+AL"]
ALPHA = 0.05


def header(first: str) -> list[str]:
    return [f"| {first} | " + " | ".join(NAMES[d] for d in DS) + " |", "|---|" + "---|" * len(DS)]


def endpoint(runs, name: str) -> pd.DataFrame:
    """One RQ3 endpoint, recomputed exactly as RQ3's summary computed it."""
    _, _, value, a, b, kind, margin = next(e for e in ENDPOINTS if e[0] == name)
    return test_endpoint(runs, value, a, b, kind, margin).set_index("dataset")


def check_against_summary(runs) -> None:
    """The recomputed endpoint means must equal RQ3's summary.md (to its 2 decimals)."""
    text = (RQ3 / "summary.md").read_text(encoding="utf-8")
    for name in ("P3", "S3a", "S3b", "S3c", "S3d", "P4"):
        line = next(x for x in text.splitlines() if x.startswith(f"| {name} |"))
        shown = [float(v.rstrip("*")) for v in re.findall(r"[+-]\d+\.\d+\*?", line)]
        mine = endpoint(runs, name).reindex(DS)["diff"].to_numpy()
        if not np.allclose(np.round(mine, 2), shown, atol=0.006):
            raise SystemExit(f"{name}: recomputed {np.round(mine, 2)} but summary.md shows {shown}")


def gains(runs) -> list[str]:
    piv = runs.pivot_table(index=["dataset", "seed"], columns="arm")
    marks = {("test_macro_f1", "SR"): ("S3a", "*"), ("test_macro_f1", "SR+AL over SR"): ("P3", "*"),
             ("test_min_class_recall", "SR"): ("S3d", "†"),
             ("test_min_class_recall", "SR+AL"): ("S3c", "†")}
    L = ["## Gains over B0 (points, mean of paired differences over seeds 50–59)", "",
         "B0 is the absolute start. `*`: pre-registered two-sided test, Holm p < 0.05 (S3a for SR "
         "over B0, P3 for SR+AL over SR). `†`: pre-registered non-inferiority test passed, lowest "
         f"class recall no worse than B0 within {RECALL_MARGIN:g} points (S3d for SR, S3c for "
         "SR+AL). Other cells are descriptive.", ""]
    for title, col in (("Macro-F1", "test_macro_f1"), ("Accuracy", "test_accuracy"),
                       ("Lowest class recall", "test_min_class_recall")):
        v = piv[col]
        L += header(title)
        L.append("| B0 (absolute) | " + " | ".join(f"{v.loc[d, 'B0'].mean():.2f}" for d in DS) + " |")
        rows = [("SR only", v["SR"] - v["B0"], "SR"), ("AL only", v["AL"] - v["B0"], "AL"),
                ("SR+AL", v["SR+AL"] - v["B0"], "SR+AL")]
        if col == "test_macro_f1":
            rows.append(("SR+AL over SR", v["SR+AL"] - v["SR"], "SR+AL over SR"))
        for label, diff, key in rows:
            mark = marks.get((col, key))
            sig = endpoint(runs, mark[0]).p_holm < ALPHA if mark else None
            cells = []
            for d in DS:
                x = diff.loc[d].mean()
                cells.append(f"{x:+.2f}".replace("-", "−") + (mark[1] if mark and sig[d] else ""))
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    s3b = endpoint(runs, "S3b")
    ok = [NAMES[d] for d in DS if s3b.p_holm[d] < ALPHA]
    L += [f"S3b (SR no worse than AL alone, within {F1_MARGIN:g} point of macro-F1, one-sided): "
          f"passed on {len(ok)} of 6 ({', '.join(ok) or 'none'}).", ""]
    lab = runs.pivot_table(index="arm", columns="dataset", values="total_labels").reindex(columns=DS)
    L += header("Labels beyond the training rows")
    for a in ("SR", "AL", "SR+AL"):
        L.append(f"| {a} | " + " | ".join(f"{lab.loc[a, d]:,.0f}" for d in DS) + " |")
    return L + [""]


def times() -> list[str]:
    L = ["## Time on one machine (Run T, seed 50, seconds)", ""]
    if not RUN_T.exists():
        return L + ["Run T has not run yet (`scripts/time_repair_arms.py`, on the home PC).", ""]
    t = pd.read_csv(RUN_T)
    ok = bool(t.matches_committed.astype(bool).all()) and len(t) == len(DS) * len(SETUPS)
    m = t.iloc[0]
    L += [f"{m.machine}, {int(m.cpu_count)} logical CPUs, Python {m.python}; one task at a time, "
          "no certificate audit. Every setup's test macro-F1 equals RQ3's: "
          f"{'yes' if ok else 'NO, see results/repair_timing/summary.md'}. SR+AL is SR with the "
          "cached gate, then AL (the AL stage alone in brackets). AL has no cached gate.", ""]
    piv = t.pivot_table(index="setup", columns="dataset", values="seconds").reindex(index=SETUPS, columns=DS)
    stage = t[t.setup == "SR+AL"].set_index("dataset").stage_seconds
    L += header("Setup")
    for s in SETUPS:
        cells = [f"{piv.loc[s, d]:.2f}" + (f" ({stage[d]:.2f})" if s == "SR+AL" else "") for d in DS]
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    base = piv.loc["SR, cached gate"]
    L += ["", "As a multiple of SR with the cached gate:", ""] + header("Setup")
    for s in ("SR, old gate", "AL", "SR+AL"):
        L.append(f"| {s} | " + " | ".join(f"{piv.loc[s, d] / base[d]:.1f}×" for d in DS) + " |")
    return L + [""]


def rq4_sr_alone(runs) -> list[str]:
    """Post hoc: RQ3's P4, S4a and S4b tests with SR in place of SR+AL."""
    L = ["## RQ4 for self-repair alone (post hoc)", "",
         "RQ3's P4, S4a and S4b tests (paired two-sided Wilcoxon over seeds, Holm across "
         "datasets), with SR in place of SR+AL. Not pre-registered. The shuffled retrain used every "
         "label SR+AL used (training rows, bought labels, acceptance rows); SR used none. "
         "`*`: Holm p < 0.05.", ""]
    # churn_* columns are already percentages; expl_topk_change_share is a share in [0, 1].
    tests = [("Fix share: SR − retrain (points)", "churn_fix_share", "SR", "Retrain, shuffled", (), 1),
             ("Broken test predictions: retrain − SR (points of the test rows)", "churn_broken_share",
              "Retrain, shuffled", "SR", (), 1),
             ("Explanations changed (PACE-flip top 3): retrain − SR (points of the explained rows)",
              "expl_topk_change_share", "Retrain, shuffled", "SR", ("skin",), 100)]
    L += header("Comparison")
    for label, col, a, b, excl, scale in tests:
        r = test_endpoint(runs, col, a, b, "two-sided", 0.0, excl).set_index("dataset")
        cells = ["–" if d in excl else f"{scale * r['diff'][d]:+.2f}".replace("-", "−")
                 + ("*" if r.p_holm[d] < ALPHA else "") for d in DS]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "Skin has 3 features, so its top-3 set never changes (excluded, as in S4b).", ""]
    mean = runs.pivot_table(index="arm", columns="dataset", values="churn_fix_share").reindex(columns=DS)
    L += header("Fix share (%), mean")
    for a in ("SR", "SR+AL", "Retrain, shuffled"):
        L.append(f"| {a} | " + " | ".join(f"{mean.loc[a, d]:.1f}" for d in DS) + " |")
    sr = runs[runs.arm == "SR"]
    L += ["", f"Certificates on SR's accepted edits ({int(sr.cert_edits.sum()):,} edits, 2,000 test "
          f"rows each): loose missed {int(sr.cert_loose_misses.sum())}, tight missed "
          f"{int(sr.cert_tight_misses.sum())} (a miss is a changed prediction the certificate did "
          "not flag).", ""]
    size = sr.groupby("dataset")[["cert_loose_mean_size_share", "cert_tight_mean_size_share"]].mean()
    L += header("Certificate size (% of test rows per edit)")
    for kind in ("loose", "tight"):
        L.append(f"| {kind} | " + " | ".join(f"{size.loc[d, f'cert_{kind}_mean_size_share']:.2f}"
                                            for d in DS) + " |")
    return L + [""]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = pd.read_csv(RQ3 / "runs.csv")
    check_against_summary(runs)
    L = ["# Self-repair as the method: time, gains and RQ4 for self-repair alone", "",
         "Written by `scripts/sr_method_tables.py` from saved results (RQ3: "
         "`results/refinement_v2_confirmatory/runs.csv`; Run T: `results/repair_timing/runs.csv`). "
         "The RQ3 endpoints it marks were recomputed and match RQ3's summary.md.", ""]
    L += times() + gains(runs) + rq4_sr_alone(runs)
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L)
    (OUT / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
