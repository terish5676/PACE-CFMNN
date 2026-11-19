# Run P7, exact flip sets against dense occlusion: pre-registration (check, with a descriptive secondary)

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review S4 / B8).
The code is `scripts/run_flip_intervals.py` (tests in `tests/test_flip_intervals_run.py`).
- It uses `flip_exact.flip_sets` (commit 7843fee), changed in 4043ddf in two ways found while
  building this run:
  - box-edge breakpoints outside the feature's domain are no longer candidates (they could make
    flip intervals below 0 or above 1);
  - each piece between breakpoints compares only the boxes that can reach the tied top group
    there. This is exact (a box whose value on the piece stays below the best floor minus the tie
    tolerance is never in the group), and it makes large models feasible: under 1 s per point on
    Landsat, where the old loop did not finish one pilot in 25 minutes. On 6 toy models the new
    and old sets differ only at crossing points computed about 1e-16 apart.
- It uses `PaceFlip(model, background).values_` for PACE-flip's 32 values, and the second
  explanation run's set-up (`run_faithfulness.setup`, `backgrounds_v3`).

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 11.

## What ran before this was written

- **Smoke slices on the laptop, seed 194** (not a study seed, never analysed), 2 points per dataset,
  written outside the repository. The check printed PASS on all 9 datasets tried (S1, Wine, Breast
  Cancer, Thyroid, Shuttle, MAGIC, Letter, Landsat, Page Blocks). PACE-flip's 32 values missed 1 of
  7 flipping (point, feature) pairs on S1, 4 of 10 on Thyroid, 8 of 15 on Shuttle, 2 of 10 on MAGIC,
  2 of 21 on Letter, 3 of 42 on Landsat and 10 of 20 on Page Blocks; Wine and Breast Cancer had no
  flipping pair. So the likely direction of the secondary is known: the 32 values miss a real share
  of flips on some datasets.
- After those slices one rule changed: 32-value grid points exactly on a breakpoint are now
  compared (they had been set aside with the rounding-level near misses).
- **Tests while building:** `tests/test_flip_intervals_run.py` (7 tests) and
  `tests/test_tough_explain.py` pass on the laptop.

## The question

PACE-flip tries 32 replacement values per feature. Proposition B8 says the class, as one feature
varies, changes only at breakpoints of the piecewise-linear memberships, so the set of values that
change the class can be computed exactly as a union of intervals. **Are the exact flip sets right
when checked against the model itself on a dense grid, and how much does the 32-value grid miss?**

## Setup

- **Datasets:** the 11 explanation datasets (`run_faithfulness.DATASETS`).
- **Seeds:** 340–349 (new). **Points:** 50 test points per seed; the dense check uses the first 10.
- **Domain:** each feature on [min(0, x_i), max(1, x_i)], as `flip_sets` widens it.
- **Dense grid:** 10,001 evenly spaced values of the feature on the domain, the class from
  `model.predict_details` with every box.

## Check (pass/fail; the script exits 1 if it fails; stop and ask)

On every densely checked (point, feature) pair:
1. every grid value that changes the class lies in an exact interval, within 1e-9;
2. every exact interval wider than the grid step contains a grid value that changes the class.

On every pair: the class at PACE-flip's 32 values agrees with the exact set, leaving out values
within 1e-9 of a breakpoint but not on it (ties up to rounding; their number is reported).

## Secondary (descriptive)

- The share of flipping (point, feature) pairs where none of the 32 values flips, per dataset and
  overall.
- Intervals per flipping pair.
- The smallest single-feature change per point (the distance from x_i to the nearest flipping
  value, minimised over features): exact against the 32-value grid's, as medians, and the number of
  points the 32 values call unflippable although the exact set says otherwise.

**No primary test.**

## How the result will be read (written now, before the run)

- **Check passes:** the paper states Proposition B8 with "checked against a 10,001-value grid on
  N pairs from 11 datasets" and adds exact flip sets as an option of PACE-flip.
- **Check fails:** stop; the failure is a bug in `flip_sets` or in the proposition. The paper
  keeps the review's flip-set wording (S4) without the exact option until it is found.
- **Secondary:** the paper reports the miss share per dataset, whatever it is. If it is above 10%
  on any dataset, the PACE-flip section says the 32-value grid can miss flips there and points to
  the exact sets; the RQ1 results, which use the 32 values, are not changed.

## Time

Laptop smoke slices took 7–99 s per dataset for 2 points (dense on both). The full run, with 10
dense points and 40 more points per seed, is about 8 minutes per seed on S1–S3, Letter and Landsat
and less elsewhere, about 8 CPU-hours in all: **about 45–90 minutes on the home PC with 12 jobs.**
Stop and ask if it is past 3 hours.

## Tests

`tests/test_flip_intervals_run.py`:
- the exact sets pass the dense check on a toy model and stay inside the domain;
- a wrong (empty) flip set is caught;
- the exact class lookup matches the model at random values and at every breakpoint;
- the check fails on a wrong count and on a disagreement at the 32 values;
- the summary's miss count and share;
- one end-to-end task on S1, smoke seed 194.

## How to run (home PC, first in batch 1, may run with P3)

```
.\.venv\Scripts\python -B scripts/run_flip_intervals.py --datasets S1 shuttle --seeds 194 --n 2 --out $env:TEMP\flip_smoke
.\.venv\Scripts\python -B scripts/run_flip_intervals.py --jobs 12 *>&1 | Tee-Object results/flip_intervals/run_log.txt
.\.venv\Scripts\python -B scripts/run_flip_intervals.py --jobs 12 --resume *>&1 | Tee-Object -Append results/flip_intervals/run_log.txt   # after a stop
```

Commit `results/flip_intervals/` unedited (`runs.csv`, `summary.md`, `run_log.txt`) as "Run P7:
results of the pre-registered run on the home PC".
