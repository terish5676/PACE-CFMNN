# Run E, the cached gate: pre-registration

Written on 2026-11-11 and pushed, with the code it describes, before the run.
The code is `scripts/check_fast_gate.py` (tests in `tests/test_fast_gate.py`).
It adds `CachedGate` to `pace_cfmnn/refine.py` and the keyword arguments
`fast_gate=False` and `gate_reference="last"` to `run_self_repair` in
`pace_cfmnn/active.py`. Plan: `research_audit/LIMITATIONS_PLAN.md`, run E.

## What ran before this was written

- **A laptop prototype, reverted.** A prototype of the cached gate was built and
  tried on the laptop on 2026-11-11 (commit ab5a166). It was reverted (fa2f890),
  so everything here was rebuilt on the home PC. Its results are not used.
- **Setup on the home PC** (Intel64 Family 6 Model 191, Python 3.14.10): the
  package install and the test suite. The suite ran after `CachedGate` and
  `tests/test_fast_gate.py` were written, so it ran 214 tests, the 198 existing
  ones and 16 new ones, and all 214 passed. The 198 were not run on their own
  before the edit.
- **A smoke slice of the script,** as the runbook allows for E: Thyroid and Page
  Blocks, seeds 50 and 51 (RQ3's seeds, so these runs were compared with the
  committed RQ3 log), 4 processes, written outside the repository.
  - All 4 replays matched the committed edit log and macro-F1 exactly (macro-F1
    difference 0).
  - On seed 50 the standard gate gave the same edit log, model and test
    predictions as the cached gate on both datasets.
  - Times, one process each: Thyroid cached 0.3 s, standard 0.8 s;
    Page Blocks cached 0.9 s, standard 2.5 s.
  - The summary printed FAIL because only 4 of 60 runs were present.

Nothing changed after the smoke slice.

## The question

Does the cached gate make exactly the standard gate's decisions? The repair gate
reads every gate row after every edit (n rows × H boxes × d features), which is
92–99% of self-repair's time (`results/rq3_explainer_timing/`). A self-repair
edit changes one box: it moves one face, or removes the box. `CachedGate` keeps
the gate rows' membership and distance tables and re-scores only the rows that
edit can change. If it gives the same decisions, run A can use it.

## Design

**The cached gate** (`refine.CachedGate`): see its docstring. It keeps, for the
gate rows, the membership table M, the squared centroid distances D, each row's
top membership and predicted class, and copies of V, W, the centroids and the
box classes. M and D use the expressions of `CFMNN.predict_details`. On each
measure it finds what changed:

- nothing: the cache stands;
- one box moved: recompute that box's column, and re-score the rows where it was
  tied at the top before or reaches the top now;
- one box removed: delete its column, and re-score the rows where it was tied
  at the top;
- anything else, including a change of box class: full recompute.

A rejected edit restores the cache exactly.

**Replay.** RQ3's SR arm exactly as `scripts/run_refinement_v2.py` ran it:

- the split from `run_refinement.split`;
- `CFMNN().fit` on the training rows;
- `budget = min(MAX_BUDGET, len(pool) // 4)`;
- `n_rounds = N_ROUNDS` (10);
- `chooser="volume"`;
- gated on the training rows, with `gate_labels_are_new=False`;
- `fast_gate=True` and no certificate hook (the hook does not change the model).

**Scope.** 6 datasets (Thyroid, Landsat, Page Blocks, MAGIC, Letter, Skin),
seeds 50–59: 60 runs. On seed 50 only, the standard gate is also run on the same
machine.

**Compared:**

- each edit's `train_index`, `accepted`, `reason` and `x_correct_after`, against
  `results/refinement_v2_confirmatory/edits.csv.gz` (arm SR);
- the final test macro-F1, against that folder's `runs.csv` (arm SR), to 1e-12;
- on seed 50: the standard-gate and cached-gate edit logs, repaired models and
  test predictions, which must be identical.

## Primary and pass rule

**PASS** only if both hold:

1. all 60 replayed runs equal the committed edit log and macro-F1;
2. on seed 50, the standard-gate run equals the cached-gate run on all 6
   datasets.

Otherwise **FAIL**, and the cached gate is not used anywhere. A missing run or
a failed task is not a pass.

**Speed.** The ratio of standard to cached time on seed 50 is reported. No
speed is required.

## Defaults

The defaults stay whatever the results. `fast_gate=False` and
`gate_reference="last"` reproduce today's output exactly (tested). RQ3 and the
other pushed results are untouched. `run_refinement` (active learning) stays on
the standard gate. If E passes, run A uses `fast_gate=True`; if it fails, run A
uses the standard gate, and A's pre-registration says so.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.

## How to run

```
python -B scripts/check_fast_gate.py --jobs 12
python -B scripts/check_fast_gate.py --jobs 12 --resume     # continue after a stop
python -B scripts/check_fast_gate.py --summary-only
```

Outputs: `results/fast_gate_check/runs.csv` (one row per dataset and seed, the
same columns on every row, NaN where a column does not apply), `summary.md` and
`run_log.txt`.
