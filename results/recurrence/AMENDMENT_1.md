# Run P8, Amendment 1: the ExactExplainer comparison (2026-11-17)

Written after the first run, which is kept unedited in [`first_run/`](first_run/). The
pre-registration ([`PREREGISTRATION.md`](PREREGISTRATION.md)) is not changed.

## What happened

The first run on the home PC (2026-11-16/17, 14,164 s) exited 1 with **Check: FAIL**: on Letter
seed 350, both PACE-SHAP modes "differ from ExactExplainer by 3.9e-02".

The cause was the comparison in `scripts/run_recurrence.py`, not the values. On Letter,
ExactExplainer explains one point (the design in the plan), while matmul and the recurrence
explain 10. The difference was computed as `phi - exact[:k]` with k = 10, and `exact[:10]` holds
only that one point, so numpy broadcast it against all 10 PACE-SHAP points: points 2–10 were
compared with point 1's values. On the one point all arms explained, the first run's `runs.csv`
shows ExactExplainer within 1.9e-16 of matmul, and brute force within 1.9e-16 of ExactExplainer.
Wine was not affected (every arm explains 10 points there).

## The change

`run_one` now compares each arm with ExactExplainer only on the points both explained (the first
`min(k, points ExactExplainer explained)`). Nothing else changes: no method, setting, dataset,
tolerance, point, seed or timing. A test (`tests/test_recurrence_run.py`,
`test_exact_comparison_uses_only_the_points_both_explained`) fails on the old code and passes on
the new.

## The rerun

With the user's approval (2026-11-17), the whole run is repeated with the fixed script, alone, with
the plan's command plus `--fresh`; its `runs.csv`, `summary.md` and `run_log.txt` in this folder
are the results. The first run's numbers are not used.
