# Run P2, CFMNN with self-repair against tuned fuzzy min-max networks: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review D4).
The code is `scripts/run_fmm_baselines.py` (tests in `tests/test_fmm_baselines_run.py`). It uses
`pace_cfmnn/fmm_baselines.py` (commit d003ed1: `make_fmm`, `tune_theta`, the grid and the tie rule)
and self-repair exactly as `scripts/run_repair_theta.py` runs it.

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 13.

## Background (in plain words)

The paper compares self-repair with Simpson's 1992 contraction, and with a GFMM left at the θ chosen
for CFMNN. A reviewer at IEEE TFS would ask: how does CFMNN with self-repair compare with the modern
fuzzy min-max networks that were designed to handle overlapping boxes? All of these are in
hyperbox-brain 0.1.6:

- **EFMNN** (enhanced FMNN): a new overlap test and new contraction cases.
- **KNEFMNN**: EFMNN, choosing among the k = 5 nearest boxes when it expands one (the library's
  default k).
- **RFMNN** (refined FMNN): new expansion, overlap, contraction and prediction rules.
- **IOL-GFMM** (improved online GFMM): GFMM's minimum membership, expansion without contraction,
  and ties broken by distance.

Every model, CFMNN included, gets θ the same way, so no model wins by having been tuned more.

## What ran before this was written

- **The repair-at-θ-val run** (`results/repair_theta/`, seeds 70–89, the same split as here):
  - on RQ2's six datasets, cross-validation chose θ = 0.05 for MAGIC, Letter and Skin (20 of 20
    seeds), Landsat (13 of 20) and Page Blocks (19 of 20). On Thyroid it chose 0.5 (11 seeds),
    0.4 (7) or 0.3 (2);
  - self-repair at that θ took a median of 6,516 s per seed on Skin on the home PC (1,148 s on
    MAGIC).
- **Tests while building:** `tests/test_fmm_baselines.py`. Every baseline fits and predicts on a
  toy problem, and tuning picks from the grid.
- **Laptop timing check, seed 194** (one fit on all training rows at one θ, not a study seed;
  Intel i7-8650U, other runs sharing the machine, so the times are rough). Each cell is fit
  seconds / boxes:

  | Dataset (training rows) | θ | EFMNN | KNEFMNN | RFMNN | IOL-GFMM |
  |---|---|---|---|---|---|
  | Thyroid (5,760) | 0.3 | 13 / 1,211 | 7 / 408 | 1 / 1,244 | 1 / 387 |
  | Thyroid | 0.1 | 21 / 2,792 | 18 / 1,355 | 3 / 2,794 | 2 / 1,170 |
  | Page Blocks (4,378) | 0.3 | 5 / 514 | 2 / 96 | 1 / 739 | 0 / 181 |
  | Page Blocks | 0.1 | 13 / 1,415 | 8 / 472 | 1 / 1,491 | 1 / 422 |
  | MAGIC (15,216) | 0.3 | 568 / 6,958 | 209 / 1,510 | 15 / 9,241 | 106 / 2,221 |
  | MAGIC | 0.1 | 317 / 12,249 | 867 / 8,281 | 44 / 12,341 | 43 / 4,037 |
  | Skin, all rows (196,045) | 0.3 | 187 / 878 | – | – | 97 / 1,012 |
  | Skin, all rows | 0.05 | 1,237 / 11,212 | 320 / 2,333 | 273 / 11,288 | 101 / 2,075 |
  | Skin, 20% (39,208) | 0.3 | 24 / 448 | 8 / 74 | 9 / 1,321 | 10 / 281 |
  | Skin, 20% | 0.05 | 200 / 3,890 | 76 / 1,211 | 21 / 3,921 | 14 / 1,122 |

  The slowest fits are EFMNN and KNEFMNN on MAGIC and EFMNN on the full Skin. Tuning costs about
  14 such fits per model (7 values of θ × 3 folds, each on two thirds of the rows), plus the refit.
  The home PC has been about 3 times as fast as the laptop in earlier runs.

- **Smoke slice:** the end-to-end test runs CFMNN (with repair) and IOL-GFMM on Wine, seed 194.

The design did not change after these checks.

## The question

**Is CFMNN with self-repair as accurate as the fuzzy min-max networks that handle overlaps by
design, when every model is tuned the same way?**

## Setup

- **Datasets:** RQ2's six: Thyroid, Landsat, Page Blocks, MAGIC, Letter and Skin. Skin is the
  original, with its duplicate rows, as in RQ2; Run P3 asks what the duplicates do.
- **Seeds:** 250–259 (new). The pilot uses seed 194.
- **Split:** the paper's 80/20 split (`run_readable_baselines.prepare`): the seed's row order, and
  min-max scaling fitted on the training rows. Test rows are not clipped, as in the
  readable-baselines run. This is the split of the readable-baselines and repair-at-θ-val runs,
  the two runs this one extends.
- **Tuning, for every model:** θ is chosen from {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5} by mean
  accuracy over the same 3 stratified folds of the training rows (`cv_folds(y, seed)`). An exact tie
  goes to the largest θ, which gives the fewest boxes. Then the model is refitted on all training
  rows. γ = 4 throughout; the libraries' defaults otherwise.
- **Models, per dataset and seed:**
  - **CFMNN-SR-val** (the method): CFMNN at the θ chosen by validation, then self-repair. This is
    RQ2's SR arm as in `run_repair_theta.py`: up to 300 of the model's own training errors, the
    smallest cut, 10 rounds, gated on the training rows, no label beyond them.
  - **EFMNN-val, KNEFMNN-val, RFMNN-val, IOL-GFMM-val:** each at the θ its own cross-validation
    chose.
  - The same five at a fixed **θ = 0.3** (secondary).
  - **CFMNN-val** and **CFMNN-0.3** without repair (descriptive).
- **A task** is one (dataset, seed, model family), so a stopped run keeps the families it finished.
  If validation picks θ = 0.3, the "0.3" rows reuse the same fitted models.
- **Output:** `results/fmm_baselines/runs.csv`, one row per model, dataset and seed.

## Primary test

Test macro-F1 of CFMNN-SR-val against each of the four tuned baselines, on each dataset:

- a paired two-sided Wilcoxon signed-rank test over the 10 seeds (`scipy.stats.wilcoxon`);
- Holm correction across all **6 datasets × 4 baselines = 24 tests** together;
- the verdict for CFMNN-SR-val is better, worse or no significant difference (Holm p ≥ 0.05).

**Power, stated in advance.** With 10 seeds the smallest possible p is 0.00195, and 24 × 0.00195 =
0.047. So the first rejections need all 10 seeds to agree. This test finds only consistent
differences. "No significant difference" is not evidence that two models are equally accurate, so
the write-up also gives each mean difference and the count of seeds better and worse.

## Secondary (reported, not used for the reading)

- Test accuracy, the same 24 comparisons.
- Test macro-F1 at a fixed θ = 0.3: CFMNN-SR-0.3 against each baseline at 0.3 (24 tests, Holm).
- CFMNN-val (no repair) against each tuned baseline (24 tests, Holm): how much of any gap the
  repair closes.
- Boxes, and seconds for tuning, fitting and repair. Times are measured with 6 jobs sharing the
  machine, so they are rough.
- The θ each model's validation chose, per dataset.

## How the result will be read (written now, before the run)

Counted over the 24 primary comparisons:

- **More than 12 are "worse" for CFMNN-SR-val:** most tuned baselines beat CFMNN with repair. RQ2
  narrows to "how to repair a CFMNN already in use", and "beats Simpson's contraction" stops being
  the main context. The paper names the baselines and datasets.
- **None is "worse":** CFMNN with repair is never significantly less accurate than a tuned fuzzy
  min-max network in its family. This is the strongest claim for the TFS version. The paper states
  it with the power caveat above, and does not call the models equal.
- **1 to 12 are "worse":** a mixed result. The paper names each comparison where a baseline wins,
  and keeps RQ2's claim to the others.

## The pilot and the Skin amendment

Before any study seed runs, the home PC runs the pilot: seed 194, every dataset and family, with
6 jobs, and the full Skin.

- **Projection:** the pilot's task seconds × 10 seeds ÷ 6 jobs, never less than its longest task
  (`run_fmm_baselines.py --project`).
- **Rule:** if the projection is more than **24 hours**, the study replaces Skin with a stratified
  20% of its rows (`--skin-share 0.2`). The sample is drawn with each seed, before the split. No
  other dataset is cut, whatever the projection. A study that runs longer than 24 hours just takes
  longer.
- **Stopping the pilot early:** Skin's five tasks start first, side by side. While they are still
  running, their elapsed time is a lower bound on their length. If finished tasks plus that lower
  bound already project more than 24 hours, the pilot may stop early and the rule applies
  (`--project --running k --elapsed-hours h`).
- **Record:** the pilot's results and the projection are committed and pushed before any study
  seed runs. The pilot is not analysed.

## Time

A rough estimate from the laptop timings and the repair-at-θ-val run: **about 12–25 hours on the
home PC with 6 jobs, with the full Skin.** Skin's self-repair alone is close to 2 hours per seed
there. The estimate is too rough to choose the Skin amendment, so the pilot decides it (above).

- The pilot itself takes about 2–4 hours. Its longest task is CFMNN-SR-val on Skin.
- Runs P5, then P11 (4 jobs) run beside the pilot, and P6, then P4's check (4 jobs) beside the
  study (runbook, section 13).
- Stop and ask if the study is still running 12 hours past its projection.

## Tests

`tests/test_fmm_baselines_run.py` (7 tests):
- the split equals the paper's unless `--skin-share` is given, and only Skin is sampled;
- the Skin sample is stratified and 20% of the rows;
- CFMNN's θ from `tune_theta` equals the repair run's `theta_val` (Wine, seed 194);
- resume works per (dataset, seed, family);
- Holm across every test, and the reading;
- the projection rule;
- one CFMNN task and one IOL-GFMM task end to end (Wine).

## How to run (home PC, batch 2)

See the runbook, section 13: the pilot, the projection, then

```
.\.venv\Scripts\python -B scripts/run_fmm_baselines.py --jobs 6 *>&1 | Tee-Object results/fmm_baselines/run_log.txt
```

Add `--skin-share 0.2` only if the pilot's projection said so. Commit `results/fmm_baselines/`
unedited as "Run P2: results of the pre-registered run on the home PC".
