"""Save-as-you-go running for long experiments.

Each finished task's rows are appended to the results CSV at once, so a stopped
run loses only the tasks in progress, and ``--resume`` skips the
(dataset, seed) pairs already in the CSV.
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def finished(csv_path) -> set:
    """(dataset, seed) pairs already in the CSV."""
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    df = pd.read_csv(p, usecols=["dataset", "seed"])
    return set(zip(df.dataset.astype(str), df.seed.astype(int)))


def append_rows(csv_path, rows) -> None:
    """Append rows (a list of dicts) to the CSV, in the column order of its header."""
    if not rows:
        return
    p = Path(csv_path)
    df = pd.DataFrame(rows)
    if p.exists() and p.stat().st_size > 0:
        cols = list(pd.read_csv(p, nrows=0).columns)
        extra = [c for c in df.columns if c not in cols]
        if extra:
            raise ValueError(f"{p} has no columns {extra}; start again with --fresh")
        df.reindex(columns=cols).to_csv(p, mode="a", header=False, index=False)
    else:
        df.to_csv(p, index=False)


def pending(csv_path, tasks, resume: bool = False, fresh: bool = False) -> list:
    """The tasks still to run. Each task is a tuple that starts with (dataset, seed).

    Existing results are never overwritten silently: ``resume`` continues them,
    ``fresh`` deletes them first.
    """
    p = Path(csv_path)
    has_rows = p.exists() and p.stat().st_size > 0
    if has_rows and fresh:
        p.unlink()
        has_rows = False
    if has_rows and not resume:
        raise SystemExit(f"{p} already has results. Run the same command with --resume to "
                         "continue, or with --fresh to delete them and start over.")
    done = finished(p) if has_rows else set()
    todo = [t for t in tasks if (str(t[0]), int(t[1])) not in done]
    print(f"{len(tasks) - len(todo)} of {len(tasks)} dataset-seed pairs already done; "
          f"{len(todo)} to run.", flush=True)
    return todo


def run_all(fn, tasks, csv_path, jobs: int = 1, cost: dict | None = None) -> list:
    """Run ``fn(*task)`` for every task, appending its rows as soon as it finishes.

    A failed task is reported and skipped, and the others go on; rerun with
    ``--resume`` to retry it. Returns the failed tasks.
    """
    cost = cost or {}
    failed = []

    def report(t, err):
        failed.append(t)
        print(f"FAILED {t}: {err!r}", flush=True)

    if jobs > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            todo = sorted(tasks, key=lambda t: -cost.get(t[0], 0))
            fut = {ex.submit(fn, *t): t for t in todo}
            for f in as_completed(fut):
                try:
                    append_rows(csv_path, f.result())
                except Exception as err:  # noqa: BLE001 - keep the other tasks going
                    report(fut[f], err)
    else:
        for t in tasks:
            try:
                append_rows(csv_path, fn(*t))
            except Exception as err:  # noqa: BLE001
                traceback.print_exc()
                report(t, err)
    if failed:
        print(f"{len(failed)} task(s) failed: {failed}. Run the same command with --resume "
              "to retry them.", flush=True)
    return failed
