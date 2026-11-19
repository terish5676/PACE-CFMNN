# Run O, re-tune θ or repair?: pre-registration

Written on 2026-11-15 and pushed, with the code it describes, before the run.
The code is `scripts/run_theta_churn.py` (tests in `tests/test_theta_churn.py`). It reuses
`prepare`, `box_size`, `compare` and `tally` from `scripts/run_readable_baselines.py`,
`theta_val`, `readable_row` and `CHECK_TOL` from `scripts/run_repair_theta.py`, and
`audit.prediction_churn`. No shared module changes. Runbook: `research_audit/HOME_PC_RUNBOOK.md`,
section 9.

## What ran before this was written

- **Committed runs on the same split and seeds (70–89).** The readable-baselines run
  (`results/readable_baselines/`) gives CFMNN's test accuracy at θ = 0.3 and at the θ chosen by
  3-fold validation from the grid 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05. The repair-at-θ run
  (`results/repair_theta/`) gives B0-0.3, SR-0.3, B0-val and SR-val. Each SR row holds its test
  predictions changed, fixed and broken against the B0 at the same θ, and `summary.md` lists their
  means per dataset. So **repair's side of the primary is known before this run; re-tuning's side
  is not.** Choosing θ by validation beat SR-0.3 on accuracy on 7 of 13 datasets, with 6–23
  times as many boxes. SR-val against B0-val was better on 3 and level on 10.
- **While planning** I printed one SR-0.3 row (Skin, seed 71): 1,433 of 49,012 test predictions
  changed, 1,378 fixed and 55 broken.
- **A smoke slice, seed 193** (not a study seed, never analysed), on the laptop, written outside the
  repository: Wine, Breast Cancer, Page Blocks and Thyroid, 24 s with 4 jobs. I read its whole
  summary: accuracy, boxes, NFR, PFR, training errors and macro-F1 at each θ. Re-tuning broke some
  predictions the θ = 0.3 model got right at most θ on 3 of the 4 datasets. P1 was not computed
  (no repair rows for seed 193). Nothing in the design changed after it, except that a per-task
  timer (`task_s`) and the fix share were added to the output.
- **Tests while building:** `tests/test_theta_churn.py` (12 tests) and
  `tests/test_speed_source.py` (Run P) pass on the laptop. They fit B0 at θ = 0.3 on Wine, seed 70,
  whose accuracy the readable-baselines run already holds. No flip between two study-seed models
  was computed.

## The question

Choosing θ by validation is more accurate than self-repair at θ = 0.3 on 7 of 13 datasets. The
paper's only stated reason to repair instead is size. Repair and re-tuning differ in another way:
repair is a set of gated edits to the boxes the model already has, while re-tuning builds a new
model. **How many test predictions that the explained model (B0 at θ = 0.3) got right does each
update get wrong?** In studies of model updates this is the negative flip rate (Yan and colleagues,
2021); people who work with a model notice such flips (Bansal and colleagues, 2019).

A second, descriptive question: how does accuracy grow with boxes across the θ grid, and how many
boxes does CFMNN need, without repair, to match self-repair's accuracy?

## Setup

- **Split:** the readable-baselines split. The paper's 80/20 split, the seed's row order and
  min-max scaling fitted on the training rows (`run_readable_baselines.prepare`).
- **Seeds:** 70–89, the study seeds of the two runs above, reused on purpose. B0 is deterministic,
  so its predictions can be paired with the stored repair results, and no new study seed is needed.
- **Data:** the 13 accuracy datasets.
- **Model:** CFMNN with γ = 4 (B0), fitted at each θ in the grid.
- **θ-val:** read from `results/readable_baselines/runs.csv` (`theta_val`).
- **Check (a task fails otherwise):** B0 at θ = 0.3 and at θ-val must reproduce the
  readable-baselines accuracy and macro-F1 within 1e-9.

## Measures

Per dataset, seed and θ:
- test accuracy and macro-F1;
- boxes and conditions per box (`box_size`);
- training errors;
- seconds.

Against B0 at θ = 0.3 on the same test rows (`audit.prediction_churn`):
- predictions changed;
- fixed (wrong → right);
- broken (right → wrong).

The rates are shares of the test rows, in %:
- **NFR** = broken ÷ test rows;
- **PFR** = fixed ÷ test rows;
- **fix share** = fixed ÷ changed (as in RQ4).

Self-repair's flips are read from `results/repair_theta/runs.csv` (SR-0.3 against B0-0.3, SR-val
against B0-val), not recomputed.

## Primary

**P1:** the NFR of repair (B0-0.3 → SR-0.3) against the NFR of re-tuning (B0-0.3 → B0 at θ-val).
- Paired two-sided Wilcoxon over the 20 seeds, Holm across the 13 datasets (`compare`, with the
  verdict reversed because lower is better).
- The verdict is for repair: better (lower NFR), level or worse.

**Descriptive (no test):**
- PFR, changed share and fix share of both updates;
- accuracy, boxes, NFR, PFR, training errors and macro-F1 at each θ;
- "boxes needed to match repair": the fewest mean boxes at which B0's mean accuracy reaches
  SR-0.3's, with that θ, or "not reached";
- SR-val's flips against B0-val.

**Not measured:** "re-tune, then repair" against B0-0.3. SR-val's test predictions were not
stored, and recomputing SR-val is the 15-hour part of the repair-at-θ run.

## How the result will be read (written now, before the run)

- **Repair better on k of 13:** the paper reports k, with the table. Where repair is better, the
  paper says repair keeps more of the model's correct answers than re-tuning θ. That is an
  advantage beside size, for a model already in use.
- **Repair level or worse on most datasets:** the paper keeps size as repair's only advantage over
  re-tuning, and says so.
- **Either way:** the accuracy-against-boxes table and "boxes needed to match repair" go into the
  RQ2 section, whatever they show.

## Time

The readable-baselines run's validation tuning did comparable work per task (21 fits on two
thirds of the training rows, with predictions on the third left out). It took 20,777 s of CPU on
the laptop with 6 jobs on 4 cores. Skin is the longest task: 465 s per seed there.

Estimate on the home PC with 12 jobs: **about 15–45 minutes**. The smoke slice below measures it.
Stop and ask if it projects more than 90 minutes.

## Tests

`tests/test_theta_churn.py`:
- the flip counts and rates on hand-made predictions, including a change between two wrong
  classes, which is neither a fix nor a break;
- the grid lookup;
- the reproduction check (it passes on a committed value and raises on a wrong one);
- the lower-is-better verdicts and the "boxes needed" rule on a made-up frame;
- the summary's primary table on a made-up frame;
- one end-to-end task on Wine, smoke seed 193.

## How to run (home PC, after the full test suite passes)

```
.\.venv\Scripts\python -B scripts/run_theta_churn.py --seeds 193 --jobs 12 --out $env:TEMP\theta_churn_smoke
```

The smoke slice: all 13 datasets on seed 193. Seed 193 is not a study seed, so θ-val is tuned there
by 3-fold validation, which the study seeds skip. Project the run from the summary's task seconds:
20 × (sum of `task_s` over the datasets) ÷ 12. Then run:

```
.\.venv\Scripts\python -B scripts/run_theta_churn.py --jobs 12 *>&1 | Tee-Object results/theta_churn/run_log.txt
.\.venv\Scripts\python -B scripts/run_theta_churn.py --jobs 12 --resume *>&1 | Tee-Object -Append results/theta_churn/run_log.txt   # after a stop
```

Commit `results/theta_churn/` unedited (`runs.csv`, `summary.md`, `run_log.txt`) as
"Run O: results of the pre-registered run on the home PC".
