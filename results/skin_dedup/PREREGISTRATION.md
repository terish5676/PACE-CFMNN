# Run P3, Skin without its duplicate rows: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review D5).
The code is `scripts/run_skin_dedup.py` (tests in `tests/test_skin_dedup_run.py`).
- It uses `data.load("skin_dedup")` (commit d003ed1: the first copy of each row kept, 51,444 of
  245,057 rows; the paper's stratified split proportions unchanged).
- It calls the existing runs unchanged: `run_refinement_v2.run_one`, `accuracy_table.run_one`,
  `run_theta_churn.run_one` and `run_repair_theta.run_one` (with the new `tags=("0.3",)`
  argument, which only skips the θ-by-validation models).

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 11.

## What ran before this was written

- **The refinement run** (`results/refinement_v2_confirmatory/`, seeds 50–59): on Skin,
  self-repair (SR) − the starting model (B0) = **+5.66 macro-F1 points** (better on 10 of 10),
  the largest gain of the six datasets. Skin has 245,057 rows but only 51,444 distinct ones, so a
  test row is often an exact copy of a training row (review D5).
- **Smoke slice on the laptop, seed 194** (not a study seed, never analysed), every part except
  the reference, written outside the repository, about 6 minutes in all:
  - refinement: B0 87.84, SR 98.19 (+10.35), AL 98.73, SR+AL 99.00 macro-F1;
  - accuracy: CFMNN 92.9%, 10-NN 99.7%, Random Forest 99.7%;
  - θ by validation 0.05 (accuracy 99.4% with 1,991 boxes; 92.9% with 90 boxes at θ = 0.3);
  - repair on the paper's split: B0-0.3 92.9%, SR-0.3 98.3%.

  So on one seed self-repair still helps a lot without the duplicates. The design did not change
  after the slice.
- **Tests while building:** `tests/test_skin_dedup_run.py` (5 tests) passes on the laptop. The
  reference arm reproduces the refinement run's stored B0 and SR on Thyroid seed 50 to 9 decimals.

## The question

**Does self-repair still help on Skin once the duplicate rows are removed, and by how much?** The
secondary parts ask whether the other Skin numbers in the paper (accuracy, the θ sweep, broken
predictions) change without the duplicates.

## Setup

- **Dataset:** `skin_dedup`; the reference arm uses the original `skin`.
- **Seeds:** 260–269 (new), for every part.
- **Parts** (each its own CSV in `results/skin_dedup/`):
  - `refinement`: `run_refinement_v2.run_one` (B0, B9, AL, SR, SR+AL, both retrains, kNN, RF), the
    refinement run's split, budget, rounds, gate and box cap;
  - `reference`: original Skin, B0 and SR only, with the same calls as `run_refinement_v2.run_one`
    (AL on the original Skin takes about 11 minutes per seed and is not needed for the primary);
  - `accuracy`: `accuracy_table.run_one` (CFMNN, 10-NN, Random Forest, the paper's split);
  - `theta`: `run_theta_churn.run_one` (CFMNN at 7 values of θ, θ by validation tuned here, flip
    rates against θ = 0.3);
  - `repair`: `run_repair_theta.run_one(tags=("0.3",))` (B0-0.3 and SR-0.3 on the paper's split).
    SR at θ by validation is not run: it took 6,401 s per seed on Skin.

## Primary test

SR − B0, test macro-F1 on `skin_dedup`, paired two-sided Wilcoxon signed-rank test over the 10
seeds (`scipy.stats.wilcoxon`; a seed with no change is dropped). One test, so no Holm correction.
The threshold is half the paper's gain: **5.66 / 2 = 2.83 points.**

## Secondary (reported, no correction, not used for the reading)

- The reference: SR − B0 on the original Skin, same seeds, same test.
- SR+AL − SR on `skin_dedup` (do bought labels add anything?), and every arm's mean macro-F1,
  accuracy, lowest class recall, labels and broken-prediction share.
- The accuracy table on `skin_dedup`.
- The θ sweep (accuracy, boxes, NFR per θ) and θ by validation.
- Broken correct predictions (NFR), repair (SR-0.3) against re-tuning (B0 at θ by validation),
  paired Wilcoxon over seeds.
- Repair on the paper's split, SR-0.3 − B0-0.3 accuracy.

## How the result will be read (written now, before the run)

- **p < 0.05 and gain ≥ 2.83:** the Skin result stands. The paper keeps Skin among the datasets
  where self-repair helps, adds the deduplicated gain next to 5.66, and says the duplicates did
  not drive it.
- **p < 0.05 and 0 < gain < 2.83:** the paper reports the deduplicated gain as the Skin number,
  says the duplicates inflated the original, and keeps Skin as a dataset where self-repair helps.
- **p ≥ 0.05:** the paper attributes Skin's gain to the duplicate rows and no longer counts Skin
  as a dataset where self-repair helps (SR beat B0 on all 6 refinement datasets; this becomes
  "on 5 of 6, and on Skin only with its duplicates", and the across-dataset test is reported
  both ways).
- **p < 0.05 and gain < 0:** as above, and the paper says self-repair hurts on Skin without them.
- **Either way:** the paper's data section reports Skin's duplicate share, and every Skin number
  that changes by more than its seed-to-seed spread is given both ways.

## Time

The laptop smoke slice took about 6 minutes for one seed (all parts but the reference); the
reference is about 2 minutes per seed (B0 and SR on the original Skin). **About 30–60 minutes on
the home PC with 4 jobs.** Random Forest in the accuracy part uses every core (`n_jobs=-1`), so
this run uses 4 jobs, not 12. Stop and ask if it is past 2 hours.

## Tests

`tests/test_skin_dedup_run.py`:
- the reference arm reproduces the refinement run's stored B0 and SR (Thyroid, seed 50);
- each part uses the right dataset;
- the four readings, and the threshold 2.83;
- the paired test (10 of 10 gives p = 0.00195; no change gives 1);
- the summary's primary line.

## How to run (home PC, first in batch 1, may run with P7)

```
.\.venv\Scripts\python -B scripts/run_skin_dedup.py --parts reference accuracy --seeds 194 --out $env:TEMP\skin_dedup_smoke
.\.venv\Scripts\python -B scripts/run_skin_dedup.py --jobs 4 *>&1 | Tee-Object results/skin_dedup/run_log.txt
.\.venv\Scripts\python -B scripts/run_skin_dedup.py --jobs 4 --resume *>&1 | Tee-Object -Append results/skin_dedup/run_log.txt   # after a stop
```

Commit `results/skin_dedup/` unedited (the five CSVs, `summary.md`, `run_log.txt`) as "Run P3:
results of the pre-registered run on the home PC".
