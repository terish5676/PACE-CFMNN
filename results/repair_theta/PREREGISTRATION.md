# Self-repair at the θ chosen by validation: pre-registration

Written on 2026-11-07 and pushed, with the code it describes, before the run.
The code is `scripts/run_repair_theta.py` (tests in `tests/test_repair_theta.py`).
It uses `run_self_repair` from `pace_cfmnn/active.py` and the edits in
`pace_cfmnn/refine.py`. Neither file has changed since RQ3's pre-registration
(commit 1cf58c9).

## What ran before this was written

All of it was thrown away.

- **A timing check on seed 70.** Seed 70 is a study seed. It covered Wine,
  Thyroid, Page Blocks, Landsat, Shuttle, MAGIC, Letter and Skin:
  - it fitted CFMNN at both θ and counted the training errors;
  - it ran self-repair with 20 edits in 2 rounds, not the study's 300 in 10;
  - it printed those test accuracies.
- **A timing check on seed 99,** not a study seed, on S1, S2, S3, Digits and
  Breast Cancer. It printed no test accuracies.
- **A smoke run of the script** on seed 99: Wine, Thyroid, S2, Page Blocks and
  Landsat, with 5 processes. It printed seed-99 accuracies.
- **The tests.** On a study seed they compute only the two models without
  repair, on Wine, seed 70. Those models are already in the readable-baselines
  run.

No setting changed because of any of these. The self-repair settings are
RQ3's. The run size (20 seeds, overnight) was chosen from the times alone.

What the timing check showed, as counts and times only:

| | θ = 0.3 | θ chosen by validation |
|---|---|---|
| Training errors, seed 70 | MAGIC 3,986; Letter 3,474; Skin 5,494; Landsat 580 | MAGIC 7; Letter 6; Skin 594; Landsat 5 |
| Errors self-repair can cut (inside the winning box of a wrong class) | all checked | all checked |
| Self-repair, 300 edits, one core, per seed | Letter ≈ 21 min, Skin ≈ 7 min | Skin ≈ 76 min, Letter ≈ 7 min |

Every check of the gate reads every training row. On Skin at θ = 0.05 that is
196,000 rows against about 2,000 boxes, which is why Skin is slow there. The
whole run is about 43 h on one core.

## Why

The readable-baselines run (`results/readable_baselines/`, seeds 70–89) found:

- **Validation beat θ = 0.3.** θ chosen by validation was more accurate than
  θ = 0.3 on 7 of 13 datasets, by up to 20 points (Letter).
- **But with far more boxes.** On 6 datasets it mostly chose θ = 0.05, with 6–23
  times as many boxes.

Self-repair at θ = 0.3 gained 1.5–6.0 points of macro-F1 in RQ3. That ran on
other seeds, another split and another measure. So reviewers may ask: why
repair at θ = 0.3, instead of choosing a smaller θ?

This run puts the two on the same rows, and asks whether they add up.

θ = 0.3 stays the setting for every RQ1–RQ4 result. Earlier results stay
untouched. Everything new goes to `results/repair_theta/`.

## Setup

- **Datasets:** all 13: S1, S2, S3, Wine, Breast Cancer, Digits, Thyroid,
  Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks.
- **Seeds:** 70–89 (20). These are the seeds of the readable-baselines run, on
  purpose: each seed's chosen θ, and the models without repair, are shared with
  it.
- **Data:** the readable-baselines split (`prepare` in
  `scripts/run_readable_baselines.py`):
  - the paper's 80/20 split;
  - the training rows in the seed's order;
  - min-max scaling fitted on the training rows.
- **θ chosen by validation (θ-val):**
  - read from the CFMNN-val row of `results/readable_baselines/runs.csv` (read
    only). It was chosen there by stratified 3-fold cross-validation on the
    training rows, from {0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05}, with the larger
    θ winning an exact tie;
  - for a seed not in that file (smoke runs and tests only), it is chosen the
    same way.

## Models

| Name | Model |
|---|---|
| **B0-0.3** | CFMNN, θ = 0.3, γ = 4: the model the paper explains |
| **SR-0.3** | B0-0.3, then self-repair |
| **B0-val** | CFMNN, θ-val, γ = 4 |
| **SR-val** | B0-val, then self-repair |

**Self-repair** is RQ3's SR arm (`scripts/run_refinement_v2.py`), called the same way:
`run_self_repair(model, X_train, y_train, X_train, y_train, chooser="volume",
budget=300, n_rounds=10, seed=seed, gate_labels_are_new=False)`.

- It tries up to 300 of the model's own training errors, in a seeded random
  order, in 10 rounds.
- For each error, it cuts the wrong-class box that wins on that row. The cut
  is on the feature where it removes the smallest share of the box's width, so
  the least volume.
- The gate reads the training rows. It keeps an edit only if:
  - macro-F1 falls by at most 0.5 points;
  - no class's recall falls by more than 1 point;
  - the boxes stay at most 1.1 times the start.
- It uses no label beyond the training rows.

The budget is RQ3's: 300 edits (RQ3 used 273 on Page Blocks, where its pool
was smaller). At θ = 0.3 it reaches only 300 of the training errors. On seed 70
there were 3,474 on Letter and 3,986 on MAGIC. Repairing every error would take
about 4 h per seed on Letter, so it is not tried. The write-up will say so.

**Check.** For every dataset and seed, B0-0.3 and B0-val must reproduce the
accuracy and macro-F1 of the readable-baselines run to within 1e-9. If one does
not, that dataset-seed pair fails and is reported. The summary states how many
rows passed.

## Recorded

For each dataset, seed and model:

- θ, and where θ-val came from;
- test accuracy and macro-F1, in %;
- boxes, and conditions per box: the features whose interval is narrower than
  [0, 1];
- training errors. Of these, how many lie inside the winning box of a wrong
  class (the ones self-repair can cut), and how many also have membership 1 in a
  box of their own class;
- for the repaired models:
  - edits tried, skipped because the row was already right, and accepted;
  - test predictions changed, fixed (wrong → right) and broken (right → wrong),
    against that θ's B0;
- seconds: fitting for B0, repair for SR. Six runs share the CPU, so times are
  compared within a dataset only.

## Comparisons

Each comparison is paired over the 20 seeds. It uses a two-sided Wilcoxon
signed-rank test, Holm-corrected across the 13 datasets of that comparison.
When all 20 differences are zero, p = 1.

- The first model is **better on a dataset** if the Holm p is below 0.05 and
  the mean difference favours it.
- **Worse** is the same, in the other direction.
- Otherwise the two are **level**.

On a seed where θ-val is 0.3, the two θ give the same model, so the
difference there is 0.

| Code | Comparison | Measure | Question |
|---|---|---|---|
| P1 (primary) | SR-0.3 vs B0-val | accuracy | Does repair at θ = 0.3 match choosing θ by validation? |
| P2 (primary) | SR-val vs B0-val | accuracy | Does repair still help at the θ validation chose? |
| S1 | SR-0.3 vs B0-0.3 | accuracy | Does repair help on this split? |
| S2 | SR-val vs SR-0.3 | accuracy | With repair on both, which θ is better? |
| S3 | P1, P2, S1 and S2 | macro-F1 | Do the results hold for macro-F1, the measure of RQ3? |

- The two primary comparisons answer separate questions, so they are not
  corrected for each other.
- Accuracy is primary because θ-val was chosen by accuracy, and the
  readable-baselines run used it.
- No direction is predicted.

**Descriptive:**

- boxes, and the ratio between the two θ;
- training errors, and the share that self-repair can cut, at both θ;
- edits accepted;
- test predictions fixed and broken;
- time;
- the gap to the better of the tuned kNN and the Random Forest, from the
  readable-baselines run on the same seeds.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.
- Every number the paper takes from this run is recomputed from `runs.csv` by
  a read-only script.

## How to run

From the repository root, with the laptop kept awake:

```
python -B scripts/run_repair_theta.py --jobs 6
```

- **Processes:** six, on a laptop with 4 cores and 8 threads. Skin, Letter and
  MAGIC start first.
- **Expected time:** about 11 h, overnight.
- **Output:**
  - `runs.csv`, which gets the rows of each dataset-seed pair as it
    finishes. A stopped run continues with `--resume`;
  - `summary.md`. `--summary-only` rebuilds it from `runs.csv`.
