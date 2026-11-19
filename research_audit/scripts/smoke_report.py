"""Write research_audit/SMOKE.md from a smoke_all.py run.

For the scripts run in full through smoke_redirect.py, also compare each output with the committed file
of the same path (byte for byte, ignoring line endings).
Usage (from the repository root):
    python -B research_audit/scripts/smoke_report.py OUTBASE
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL = ["summarise_v1", "posthoc_shuttle", "posthoc_review", "reanalysis", "knn_reference", "cf_table"]


def main() -> None:
    base = Path(sys.argv[1]).resolve()
    log = pd.read_csv(base / "smoke_log.csv").sort_values("name")
    n_ok = int((log.exit.astype(str) == "0").sum())
    L = ["# Smoke runs", "",
         "Every experiment script run once by `research_audit/scripts/smoke_all.py`: one dataset, one seed and a",
         "few points per run, with every output written to a scratch folder outside the repository. These are",
         "small checks that the scripts run end to end; their numbers mean nothing.", "",
         f"**Result: {n_ok} of {len(log)} runs exited 0.**", "",
         "| Run | Exit | Minutes | Output files |", "|---|---|---|---|"]
    for r in log.itertuples():
        files = str(r.files) if isinstance(r.files, str) else ""
        L.append(f"| {r.name} | {r.exit} | {r.minutes:.2f} | {files.replace('|', '/')} |")
    L += ["", "## Scripts run in full: outputs against the committed files", "",
          "These scripts have no `--out` option, so `smoke_redirect.py` ran them in full and sent their writes to",
          "the scratch folder. Their outputs can be compared with the committed ones.", "",
          "| Run | Output | Same as committed |", "|---|---|---|"]
    for name in FULL:
        for p in sorted((base / name).rglob("*")):
            if not p.is_file() or p.name in ("stdout.txt", "stderr.txt"):
                continue
            rel = p.relative_to(base / name)
            committed = ROOT / "results" / rel
            if committed.exists():
                same = (p.read_bytes().replace(b"\r\n", b"\n") == committed.read_bytes().replace(b"\r\n", b"\n"))
                verdict = "yes" if same else "no"
            else:
                verdict = "no committed file"
            L.append(f"| {name} | `results/{rel.as_posix()}` | {verdict} |")
    L += ["", "Notes:",
          "- `pace_shap_settings` ran on Wine only (`--only-datasets wine`), so it is a smoke run, not a "
          "reproduction.",
          "- `reanalysis` resumes from the committed `results/review_v1/ties_v3.csv`, so its tie counts were "
          "read, not recomputed; the rest of the analysis ran in full.", ""]
    (ROOT / "research_audit" / "SMOKE.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
