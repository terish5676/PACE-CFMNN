"""Smoke-run every experiment script once: one dataset, one seed, few points, outputs outside the repo.

A smoke run checks that a script runs end to end and writes its outputs. Its numbers mean nothing.
Usage (from the repository root):
    python -B research_audit/scripts/smoke_all.py OUTBASE [--jobs 3] [--only NAME ...]
Writes OUTBASE/<name>/ per run and OUTBASE/smoke_log.csv; prints one line per run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PY = [sys.executable, "-B"]
RED = "research_audit/scripts/smoke_redirect.py"


def runs(base: Path) -> dict[str, list[str]]:
    def o(name):
        return ["--out", str(base / name)]

    s = "scripts/"
    return {
        "g0": [s + "run_g0.py", "--datasets", "thyroid", "--seeds", "0"] + o("g0"),
        "faith_v1": [s + "run_faithfulness.py", "--protocol", "v1", "--datasets", "wine", "--seeds", "0", "--n", "5"] + o("faith_v1"),
        "faith_v2": [s + "run_faithfulness.py", "--datasets", "wine", "--seeds", "0", "--n", "5"] + o("faith_v2"),
        "faith_v2_timing": [s + "run_faithfulness.py", "--timing", "--datasets", "wine", "--seeds", "0", "--n", "5"] + o("faith_v2_timing"),
        "faith_v3": [s + "run_faithfulness.py", "--protocol", "v3", "--datasets", "wine", "--seeds", "30", "--n", "5"] + o("faith_v3"),
        "faith_v3_timing": [s + "run_faithfulness.py", "--protocol", "v3", "--timing", "--datasets", "wine", "--seeds", "30", "--n", "5"] + o("faith_v3_timing"),
        "faith_cond": [s + "run_faithfulness.py", "--protocol", "cond", "--datasets", "wine", "--seeds", "40", "--n", "5"] + o("faith_cond"),
        "ed_explore": [s + "run_error_detection.py", "--datasets", "S1", "--seeds", "0"] + o("ed_explore"),
        "ed_conf": [s + "run_error_detection.py", "--protocol", "confirmatory", "--datasets", "S1", "--seeds", "10"] + o("ed_conf"),
        "ed2_explore": [s + "run_error_detection_v2.py", "--protocol", "exploratory", "--datasets", "S1", "--seeds", "0"] + o("ed2_explore"),
        "ed2_conf": [s + "run_error_detection_v2.py", "--protocol", "confirmatory", "--datasets", "S1", "--seeds", "20"] + o("ed2_conf"),
        "ed2_ens": [s + "run_error_detection_v2.py", "--protocol", "ensembles", "--datasets", "S1", "--seeds", "60"] + o("ed2_ens"),
        "ref_pilot": [s + "run_refinement.py", "--datasets", "page_blocks", "--seeds", "0"] + o("ref_pilot"),
        "ref_elig": [s + "run_refinement.py", "--protocol", "eligibility", "--datasets", "page_blocks", "--seeds", "0"] + o("ref_elig"),
        "ref_conf": [s + "run_refinement.py", "--protocol", "confirmatory", "--datasets", "page_blocks", "--seeds", "10"] + o("ref_conf"),
        "ref_explore": [s + "run_refinement_explore.py", "--datasets", "page_blocks", "--seeds", "0"] + o("ref_explore"),
        "ref_v2": [s + "run_refinement_v2.py", "--datasets", "page_blocks", "--seeds", "50"] + o("ref_v2"),
        "cf": [s + "run_counterfactuals.py", "--datasets", "wine", "--seeds", "0", "--n", "5"] + o("cf"),
        "cf_ext": [s + "run_counterfactuals.py", "--extended", "--datasets", "wine", "--seeds", "40", "--n", "5"] + o("cf_ext"),
        "gi": [s + "run_global_importance.py", "--datasets", "wine", "--seeds", "0"] + o("gi"),
        "gi_ext": [s + "run_global_importance.py", "--extended", "--datasets", "wine", "--seeds", "40"] + o("gi_ext"),
        "interactions": [s + "run_interactions.py", "--datasets", "S3", "--seeds", "40", "--n", "5"] + o("interactions"),
        "speed": [s + "time_pace_flip.py", "--datasets", "wine", "--seeds", "0", "--n", "5", "--reps", "1"] + o("speed"),
        "occl_batched": [s + "time_occlusion_batched.py", "--datasets", "wine", "--seeds", "0", "--n", "5", "--reps", "1"] + o("occl_batched"),
        "accuracy": [s + "accuracy_table.py", "--datasets", "wine", "--seeds", "0"] + o("accuracy"),
        "exact": [s + "exact_baseline.py", "--datasets", "wine", "--seeds", "30", "--n", "3"] + o("exact"),
        "review_rq1": [s + "run_review_rq1.py", "--datasets", "wine", "--seeds", "50", "--n", "5"] + o("review_rq1"),
        "posthoc_sigma": [s + "posthoc_sigma.py", "--datasets", "S1", "--seeds", "40", "--n", "5"] + o("posthoc_sigma"),
        "posthoc_rows": [s + "posthoc_rows.py", "--datasets", "S1", "--seeds", "40", "--n", "5"] + o("posthoc_rows"),
        # No --out option: run through the redirecting wrapper (full run, outputs sent to OUTBASE).
        "pace_shap_settings": [RED, str(base / "pace_shap_settings"), "--only-datasets", "wine", s + "pace_shap_settings.py"],
        "summarise_v1": [RED, str(base / "summarise_v1"), s + "summarise_refinement_v1.py"],
        "posthoc_shuttle": [RED, str(base / "posthoc_shuttle"), s + "posthoc_shuttle_sparse.py"],
        "posthoc_review": [RED, str(base / "posthoc_review"), s + "posthoc_review.py", "--points", "page_blocks"],
        "reanalysis": [RED, str(base / "reanalysis"), s + "review_reanalysis.py", "--jobs", "3"],
        "knn_reference": [RED, str(base / "knn_reference"), s + "knn_reference.py"],
        "cf_table": [RED, str(base / "cf_table"), s + "counterfactual_table_seeds.py"],
    }


def one(name: str, cmd: list[str], base: Path) -> dict:
    (base / name).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(PY + cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=40 * 60)
        code, err, out = r.returncode, r.stderr, r.stdout
    except subprocess.TimeoutExpired:
        code, err, out = "timeout", "", ""
    (base / name / "stdout.txt").write_text(out, encoding="utf-8")
    (base / name / "stderr.txt").write_text(err, encoding="utf-8")
    files = sorted(str(p.relative_to(base / name)) for p in (base / name).rglob("*")
                   if p.is_file() and p.name not in ("stdout.txt", "stderr.txt"))
    last = (err.strip().splitlines() or [""])[-1][:200]
    row = dict(name=name, exit=code, minutes=round((time.time() - t0) / 60, 2), n_files=len(files),
               files=" ".join(files)[:300], last_stderr=last if code != 0 else "")
    print(f"{name:20s} exit={code} {row['minutes']:6.2f} min files={len(files)} {row['last_stderr']}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    base = Path(a.base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in runs(base).items() if not a.only or k in a.only}
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda kv: one(kv[0], kv[1], base), todo.items()))
    log = base / "smoke_log.csv"
    df = pd.DataFrame(rows)
    if log.exists() and a.only:
        old = pd.read_csv(log)
        df = pd.concat([old[~old.name.isin(df.name)], df], ignore_index=True)
    df.to_csv(log, index=False)
    print(f"\n{(df.exit.astype(str) == '0').sum()} of {len(df)} runs exited 0")


if __name__ == "__main__":
    main()
