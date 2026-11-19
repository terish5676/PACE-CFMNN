"""Run a script that has no --out option, sending every file it writes under results/ elsewhere.

Usage (from the repository root):
    python -B research_audit/scripts/smoke_redirect.py OUTDIR [--only-datasets a,b] SCRIPT [ARGS...]

Writes through Path.write_text / write_bytes / mkdir, DataFrame.to_csv and open(..., "w"/"a"/"x")
whose target lies under <repo>/results are redirected to OUTDIR/<same relative path>. Reads are not
touched, so the script still reads the committed results. --only-datasets replaces
run_faithfulness.DATASETS (used by pace_shap_settings.py) for a one-dataset smoke run.
"""

from __future__ import annotations

import builtins
import os
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = (ROOT / "results").resolve()


def main() -> None:
    args = sys.argv[1:]
    outdir = Path(args.pop(0)).resolve()
    only = None
    if args[0] == "--only-datasets":
        args.pop(0)
        only = args.pop(0).split(",")
    script = ROOT / args[0]

    def redirect(p):
        try:
            q = Path(os.fspath(p)).resolve()
        except TypeError:
            return p
        if q == RESULTS or RESULTS in q.parents:
            new = outdir / q.relative_to(RESULTS)
            new.parent.mkdir(parents=True, exist_ok=True)
            return new
        return p

    wt, wb, mk, csv, op = Path.write_text, Path.write_bytes, Path.mkdir, pd.DataFrame.to_csv, builtins.open
    Path.write_text = lambda self, *a, **k: wt(redirect(self), *a, **k)
    Path.write_bytes = lambda self, *a, **k: wb(redirect(self), *a, **k)
    Path.mkdir = lambda self, *a, **k: mk(redirect(self), *a, **k)

    def to_csv(self, path_or_buf=None, *a, **k):
        if isinstance(path_or_buf, (str, os.PathLike)):
            path_or_buf = redirect(path_or_buf)
        return csv(self, path_or_buf, *a, **k)

    def open_(file, mode="r", *a, **k):
        if isinstance(file, (str, os.PathLike)) and any(c in mode for c in "wax"):
            file = redirect(file)
        return op(file, mode, *a, **k)

    pd.DataFrame.to_csv = to_csv
    builtins.open = open_

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    if only is not None:
        import run_faithfulness

        run_faithfulness.DATASETS = only
    sys.argv = [str(script)] + args[1:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
