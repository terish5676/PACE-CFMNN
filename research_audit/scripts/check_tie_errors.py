"""Check the paper's claim (Section 6): every CFMNN training error is a cross-class tie.

Claim: on the 6 repair datasets, at seeds 0 and 10, each training row CFMNN gets
wrong lies inside (membership 1) a box of its own class and inside a box of the
wrong class, and the nearest-centroid rule picks the wrong one. The claim was
stated in the v2 pre-registration but no committed script or output backed it.

Uses the repair runs' own split (``scripts/run_refinement.split``) and the main
model (``CFMNN()``). Seeds 0 and 10 (the claim) and 50 (the first confirmatory
seed). Writes results/final_review/tie_errors/{runs.csv,summary.md}.

    python -B research_audit/scripts/check_tie_errors.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from run_refinement import split  # noqa: E402

DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = [0, 10, 50]
OUT = ROOT / "results" / "final_review" / "tie_errors"


def check(name: str, seed: int) -> dict:
    t0 = time.perf_counter()
    P = split(name, seed)
    Xtr, ytr = P["train"]
    m = CFMNN().fit(Xtr, ytr)
    det = m.predict_details(Xtr)
    t_idx = np.searchsorted(m.classes_, ytr)
    err = np.flatnonzero(det.cls != t_idx)
    own_support = det.support[err, t_idx[err]]          # best own-class membership
    win_support = det.support[err, det.cls[err]]        # winning (wrong) class membership
    inside_own = own_support >= 1.0
    inside_wrong = win_support >= 1.0
    tied = det.n_tied[err] > 1
    both = inside_own & inside_wrong
    return {"dataset": name, "seed": seed, "n_train": len(ytr), "n_errors": len(err),
            "share_inside_own_box": float(inside_own.mean()) if len(err) else np.nan,
            "share_inside_wrong_box": float(inside_wrong.mean()) if len(err) else np.nan,
            "share_tied_at_top": float(tied.mean()) if len(err) else np.nan,
            "share_cross_class_tie_inside_both": float(both.mean()) if len(err) else np.nan,
            "n_not_cross_class_tie": int((~both).sum()), "seconds": time.perf_counter() - t0}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for name in DATASETS:
        for seed in SEEDS:
            r = check(name, seed)
            recs.append(r)
            print(r, flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "runs.csv", index=False)
    lines = ["# Are all CFMNN training errors cross-class ties?", "",
             "Written by `research_audit/scripts/check_tie_errors.py` (final review, post hoc check of a claim "
             "in the paper's Section 6). Split: `scripts/run_refinement.split` (the repair runs' role split); "
             "model: `CFMNN()` (θ = 0.3, γ = 4, fixed centroid, search expansion).", "",
             "A cross-class tie: the row has membership 1 in a box of its own class and in a box of the wrong "
             "class, and the nearest-centroid rule picks the wrong one.", "",
             "| Dataset | Seed | Training rows | Errors | Inside own-class box | Inside wrong-class box | "
             "Cross-class tie | Errors that are not |", "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['n_train']:,} | {r['n_errors']:,} | "
                     f"{100 * r['share_inside_own_box']:.1f}% | {100 * r['share_inside_wrong_box']:.1f}% | "
                     f"{100 * r['share_cross_class_tie_inside_both']:.1f}% | {r['n_not_cross_class_tie']} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
