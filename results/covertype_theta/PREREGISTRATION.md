# Run L, Covertype with θ chosen by validation, then self-repair: pre-registration

Written on 2026-11-15 on the home PC (i5-14400F) and pushed, with the code it describes, before
the run. Plan: `research_audit/PART_B_PLAN.md`, Run L.

- **The code:** `scripts/run_covertype_theta.py` (tests in `tests/test_covertype_theta.py`).
- **Used as it is:** `prepare`, `peak_memory_mb` and `timed` from `scripts/run_covertype.py`
  (Run K); `tune`, `cv_folds` and `GRIDS["CFMNN-val"]` from `scripts/run_readable_baselines.py`;
  `run_self_repair` from `pace_cfmnn/active.py` with `fast_gate=True`; `training_error_ties` and
  `centroid_decided` from `pace_cfmnn/ties.py`; `PaceFlip`, `PaceShap` and
  `run_faithfulness.backgrounds_v3` as Run K uses them. No existing file was changed.

## Why

On Covertype (Run K, seeds 170–174) CFMNN at θ = 0.3 reached 51.4% test accuracy, against 92.7%
for 10-nearest neighbours and 95.6% for a Random Forest. (The plan's "51.5%" is Run K's summary
value rounded twice; the raw mean is 51.447%.) With θ = 0.3 it grew only about 119 boxes. Does a
smaller θ, chosen by validation as in the readable-baselines run, or self-repair, help?

## What ran before this was written

All on smoke seeds; output outside the repository (`$HOME\scratch\run_l`), never analysed.

- **Unit tests** on made-up data shaped like Covertype (`tests/test_covertype.py`'s
  `fake_covertype`).
- **Run K check, seed 190** (the seed the plan names: Run K's own smoke seed, whose full-data
  B0-0.3 accuracy is in Run K's Amendment 1): `--check-k 190` fitted B0-0.3 on Run K's split of
  the full data and got 55.9667%, Run K's 55.97%: **PASS**, so the split and the model are
  Run K's (18 s).
- **Smoke slice, seed 192, a stratified 10% of the rows** (46,480 training rows, 11,621 test
  rows), the whole task, alone: 40.7 minutes, peak memory 3.9 GB. It printed, for seed 192:
  - tuning on all 46,480 training rows: 267 s; CV accuracy 49.8% at θ = 0.5 rising to 72.9% at
    θ = 0.05, so θ = 0.05 (the smallest in the grid) was chosen;
  - B0-0.3: 83 boxes, trained in 1 s; SR-0.3: 34 of 300 edits accepted, 28 s;
  - B0-val (θ = 0.05): 3,069 boxes, trained in 9 s; SR-val: 290 of 300 edits accepted, 461 s;
  - test accuracy B0-0.3 56.68%, SR-0.3 56.96%, B0-val 72.42%, SR-val 72.83%; training errors
    that are ties 100% before repair and about 95% after; test predictions decided by the
    nearest centroid 97% at θ = 0.3 and 39% at θ = 0.05;
  - PACE-flip 0.7–9.1 ms and PACE-SHAP 261–9,919 ms per explanation (9.9 s at 3,069 boxes).
- **A full-data probe, seed 192:** B0 at θ = 0.05 on all 464,809 training rows: 7,788 boxes,
  877 s to train and 1,306 s to predict the 116,203 test rows (test accuracy 74.05%, printed,
  not analysed), peak memory 0.9 GB.

**What changed because of these.** The plan asked for the full data, with a tuning subsample if
tuning projected over 2 hours and a gate subsample or "repair infeasible" if repair was too
slow. On all rows the cached gate keeps two tables of (gate rows × boxes): about 58 GB at 7,788
boxes, against the PC's 15.8 GB; tuning projected to about 3 hours per seed; and counting the
training-row ties and scoring the test rows took about 2 hours per model. The whole run
projected to about 50 hours. The user was told the times and chose to run **the whole of Run L
on a stratified 10% of Covertype's rows** (`SUBSAMPLE = 0.1`), with the method unchanged and
every row of that share used for tuning, the repair's errors and the gate (`TUNE_ROWS` and
`GATE_ROWS` are `None`). No setting of CFMNN, the grid, the budget or the gate changed.

## Setup

- **Data:** Covertype (581,012 rows, 54 features, 7 classes). For each seed, a stratified 10%
  of the rows (`run_covertype.prepare(seed, 0.1)`: `train_test_split(..., train_size=0.1,
  stratify=y, random_state=seed)`, 58,101 rows), then the paper's 80/20 split of those rows,
  the training rows in the seed's order, min-max scaling fitted on them: 46,480 training rows
  and 11,621 test rows. **This is not the full data**, so the numbers are not directly
  comparable with Run K's (which used all rows); the summary says so.
- **Seeds:** 210–214 (5). No earlier run or smoke slice used them. Each seed draws its own 10%.
- **One task at a time,** with nothing else running: the times are measures.

## Models (the four of the repair-at-θ run)

| Name | What it is |
|---|---|
| **B0-0.3** | `CFMNN(theta=0.3, gamma=4)` (Run K's model) |
| **SR-0.3** | B0-0.3, then RQ3's self-repair: `run_self_repair(B0, X_train, y_train, X_train, y_train, chooser="volume", budget=300, n_rounds=10, seed=seed, gate_labels_are_new=False, fast_gate=True)` |
| **B0-val** | CFMNN at the θ with the best mean accuracy in 3-fold stratified CV on the training rows (`tune("CFMNN-val", [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05], ...)`, the readable-baselines run's rule: on an exact tie the larger θ) |
| **SR-val** | B0-val, then the same self-repair |

The gate (macro-F1 may fall by at most 0.5 points, no class's recall by more than 1 point, boxes
at most 1.1 times the start) reads the training rows; the cached gate makes the same decisions
as the standard one (Run E).

## Recorded

Per seed and model (`runs.csv`): θ and where it came from; the CV accuracy of every θ; boxes;
training error, its ties (both counts of `pace_cfmnn/ties.py`; Proposition 1's form is the share
inside a box of both classes); the share of test predictions decided by the nearest centroid
and the share of test errors among them; test accuracy and macro-F1; for SR, edits tried, already
right and accepted; seconds to train, to repair and to tune; PACE-flip and PACE-SHAP
milliseconds per explanation on the 50 explained test points, with the explainer rows of the
second explanation run (as Run K); peak memory and minutes per seed. 10-NN and the Random Forest
are quoted from Run K, not rerun.

## Design: descriptive

**No comparison is tested and there is no primary.** One dataset cannot support a paired test
across datasets. Everything is reported as mean ± standard deviation over the 5 seeds, or pooled
over them, whichever way it goes, including a θ at the end of the grid and a repair that helps
little.

## What the result changes

- Results are reported whichever way they go, including "it stays poor".
- Nothing in the paper's method or settings changes because of this run; θ = 0.3 stays the
  explained model's setting.
- The paper may say only what a 10% subsample of one dataset supports.
- Any change to this plan after the push is a dated amendment (before the run) or a reported
  deviation (after it). Any analysis not listed here is labelled post hoc.

## Time

From the smoke slice: about 41 minutes per seed, so about **3.5 hours** for the 5 seeds, under
the plan's 6-hour limit; peak memory about 4 GB.

## Tests

`tests/test_covertype_theta.py` checks that:

- the seeds, models, γ, θ grid, budget, rounds, chooser and the 10% share are as above, and the
  split is Run K's `prepare`;
- the stratified subsample helper keeps every row when asked for more than there are, and is
  sorted, stratified and seeded;
- one task on made-up data writes four rows with the same columns; B0-0.3 and B0-val equal a
  CFMNN fitted directly at their θ; the chosen θ is in the grid; repair stays within the budget;
  every training error before repair is a tie; PACE-SHAP runs in sampled mode;
- the summary says the run is descriptive and prints every model.

## How to run

```
python -B scripts/run_covertype_theta.py *>&1 | Tee-Object results/covertype_theta/run_log.txt
python -B scripts/run_covertype_theta.py --resume            # continue after a stop
python -B scripts/run_covertype_theta.py --summary-only
```

Outputs in `results/covertype_theta/`: `runs.csv` (one row per seed and model; 5 × 4 = **20
rows**), `summary.md` and `run_log.txt`.
