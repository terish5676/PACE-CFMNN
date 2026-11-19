# Run A, repair budgets, gates and certificates: pre-registration

Written on 2026-11-11 and pushed, with the code it describes, before the run.
The code is `scripts/run_repair_sensitivity.py` (tests in
`tests/test_repair_sensitivity.py`). It uses `run_self_repair` from
`pace_cfmnn/active.py`, and adds `certify_centroid` and `explanation_certificate`
to `pace_cfmnn/audit.py` (nothing existing in that file changed). Plan:
`research_audit/LIMITATIONS_PLAN.md`, run A.

## What ran before this was written

- **Run E** (`results/fast_gate_check/`, results commit 5144051) finished first. Its
  summary says **Verdict: PASS**: all 60 replays equal RQ3, and the standard gate
  equals the cached gate on seed 50. So this run uses the cached gate
  (`fast_gate=True`). The script reads that verdict from E's summary and records
  `gate_impl` in every row.
- **Tests while building.**
  - The suite ran with the new certificate code and its tests in place.
  - One of the new tests failed once, on a bug in the test itself (after an
    `add_points` edit it indexed the feature rows without the added row), and was
    fixed.
  - A first summary of the smoke slice below crashed on a pivot that dropped B0's
    empty column; it was fixed (`reindex(index=..., columns=...)`).
  - Neither changed the design below.
- **A smoke slice, seed 120** (never analysed), written outside the repository:
  - Thyroid and Page Blocks;
  - then Letter, Skin, MAGIC and Landsat, for timing.
  - Seed 120 is not a study seed. The runs printed its test macro-F1 and its
    counts of accepted edits.
  - The only things taken from them are the times and that the code runs. No
    arm, gate, budget or certificate definition changed after them.
  - On this slice no certificate missed.
- **Timing.** One seed-120 task (all arms, with the certificate hook on
  SR-default) took, summed over its arms and with 4 tasks sharing the CPU:
  - Thyroid and Page Blocks: a few seconds;
  - Skin, MAGIC and Landsat: about 40–60 s;
  - Letter: about 1.5 min (SR-default with certificates 50 s, the all-errors arm
    16 s).

  The full run (60 tasks, 12 processes) projects to well under the plan's 2 h.

## The question

RQ3's self-repair uses one budget (300 edits) and one gate (macro-F1 may fall at
most 0.5 points, no class recall more than 1 point). Is the result sensitive to
either? And can a repair's effect on predictions and explanations be bounded
exactly, more tightly than the existing certificates?

## Setup

- **Data and model:** RQ3's split (`run_refinement.split`) and fitted model
  (`CFMNN()`: θ = 0.3, γ = 4), on Thyroid, Landsat, Page Blocks, MAGIC, Letter
  and Skin.
- **Seeds:** 110–119, the study seeds of A–D. No earlier run used them.
- **Repair:** RQ3's SR arm: the volume cut (`chooser="volume"`), 10 rounds, a
  seeded random order of the model's own training errors (seed = the study seed),
  gated on the training rows, so no label beyond them is used
  (`gate_labels_are_new=False`). The box cap stays 1.1 times the starting number
  of boxes in every arm.
- **Test rows:** the dataset's own test role, as in RQ3.

## Arms

All arms start from the same fitted model, so B0 is shared.

| Arm | Budget (edits tried) | Gate: macro-F1 / recall limit (points) |
|---|---|---|
| B0 | none: no repair | none |
| **SR-default** | `min(300, len(pool) // 4)` (273 on Page Blocks in RQ3) | 0.5 / 1.0 |
| SR-b100 | 100 | 0.5 / 1.0 |
| SR-b1000 | 1,000 | 0.5 / 1.0 |
| SR-ball | the number of training rows, i.e. every training error | 0.5 / 1.0 |
| SR-gate-none | default | both infinite (the box cap remains) |
| SR-gate-strict | default | 0 / 0 |
| SR-gate-loose | default | 1.0 / 2.0 |
| SR-gate-cumulative | default | 0.5 / 1.0, each edit compared with the model before any edit (`gate_reference="start"`), so the total loss is bounded |

On a dataset with fewer training errors than a budget, the arms with that budget
and the default budget tried the same errors, and so can be identical. That is
reported, not corrected.

## Measures

For each dataset, seed and arm:

- test macro-F1, accuracy and lowest class recall (%);
- boxes and training error, at the start and the end;
- edits accepted; edits rejected by reason (macro-F1, class recall, box cap);
  rows skipped (already right; not inside the winning box);
- test predictions fixed and broken against B0 (`audit.prediction_churn`);
- seconds. Several tasks share the CPU, so times are compared within a dataset
  only.

## Certificates (SR-default only)

On 100 test rows (a seeded draw of the test rows), for every accepted edit, the
model before and after is compared by brute force:

- the prediction (the predicted box class);
- PACE-flip's score (`PaceFlip`, background = the first 200 training rows, 32
  quantiles, as in RQ3);
- PACE-decomposition's contributions φ (the same background).

A row's explanation counts as changed if any entry differs by more than 1e-12
(RQ3's explanation-churn definition). Each certificate is scored for **misses**
(rows that changed but were not flagged), **size** (share of the 100 rows flagged
per edit) and **precision** (share of flagged rows that changed).

| Certificate | For | Definition |
|---|---|---|
| loose, tight | predictions | `audit.certify`, as in RQ3 |
| **centroid** (new) | predictions | the tight certificate, minus the rows where the unchanged boxes of the tied top group are the same before and after and every changed box that is in a tied group at least as near as the nearest unchanged tied box is of that box's class |
| **PACE-flip** (new) | PACE-flip | a row is certified unchanged if no changed box, before or after the edit, can matter: none is kept by PACE-flip's pruning bound (the box's ceiling is below the floor of its group's strongest box, minus `tie_tol` and 1e-12), none is in the tied top group at x, and some other box of its group reaches its membership at x |
| **PACE-decomposition** (new) | PACE-decomposition | the PACE-flip set, minus the rows where the same unchanged box wins before and after, with no changed tied box at least as near |

PACE-SHAP has no certificate: its sampled values have no such bound.

**Drop rule.** A certificate that misses in the tests or in this run is dropped,
and the summary says so. A certificate that never misses is reported with its size
and precision.

## Primary and pass rule

**Primary:** each arm vs SR-default on test macro-F1.

- Paired two-sided Wilcoxon over the 10 seeds, Holm-corrected across the
  datasets, α = 0.05.
- When all 10 differences are zero, p = 1.
- The first arm is **better** on a dataset if the Holm p is below 0.05 and the
  mean difference favours it, **worse** in the other direction, otherwise
  **level**.

No direction is predicted. There are 7 primary comparisons (SR-b100, SR-b1000,
SR-ball and the four gate arms, each against SR-default); they answer separate
questions and are not corrected for each other.
Everything else (accuracy, recall, boxes, edits, fixed and broken, times) is
descriptive.

## Defaults

The defaults stay whatever the results: 300 edits, gate 0.5 / 1.0. Sensitivity
results are reported, never used to pick settings.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.

## How to run

```
python -B scripts/run_repair_sensitivity.py --jobs 12
python -B scripts/run_repair_sensitivity.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_repair_sensitivity.py --summary-only
```

Outputs: `results/repair_sensitivity/runs.csv` (one row per dataset, seed and arm,
the same columns on every row; NaN where a column does not apply), `summary.md`
and `run_log.txt`. Expected: 6 datasets × 10 seeds × 9 arms = 540 rows.
