# Run C, RQ1 with 200 points and tie-aware outputs: pre-registration

Written on 2026-11-11 and pushed, with the code it describes, before the run.
The code is `scripts/run_more_points.py` (tests in `tests/test_more_points.py`). It
imports protocol v3's methods and tests from `scripts/run_faithfulness.py` and the
tie rules of `scripts/run_review_rq1.py`; no library file changed for this run.
Plan: `research_audit/LIMITATIONS_PLAN.md`, run C.

## What ran before this was written

- **Tests.** The full suite (240 tests) passed, including this run's tests:
  - the method list is v3's, in v3's order, plus the added arm;
  - on Wine (6 points) every v3 method reproduces `run_faithfulness.run(..., "v3")`
    for the deletion, insertion, |attr| deletion and class-kept AUCs to 12 decimals;
  - the tie columns are consistent;
  - the summary's verdicts on a made-up table.
- **A smoke slice, seed 120** (never a study seed, never analysed), written outside
  the repository: S1, Page Blocks and Letter, 200 points, 3 processes (with other
  jobs also running). It printed the test AUCs; the only things taken from it are
  that the code runs and its times: Page Blocks 105 s (162 of 200 points tied),
  Letter 5,157 s (117 tied), S1 7,192 s (none tied). Nothing in the design changed
  because of it.

The times project to roughly 100 CPU-hours (S1, S2 and S3 about 2 h per seed on one
core, Letter about 1.5 h, the rest less), about 8 h with 12 processes, inside the
plan's ~9 h.

## The question

Limitation 4a: the deletion and insertion tests use 50 points per dataset (36 on
Wine) and 10 seeds. Do RQ1's conclusions about PACE-SHAP and KernelSHAP hold with
200 points per seed, on seeds no earlier run used? Limitation 4c: tied points
(margin 0) get a margin curve of signs, which ignores the centroid tie-break (55–79%
of test points are tied on Letter, MAGIC and Page Blocks). Do the conclusions hold on
untied points, on the raw margin, and on the class kept?

## Setup

- **Protocol:** v3 of `run_faithfulness.py`, with all its methods and settings:
  - signed ranking, margin targets, 10 repetitions;
  - evaluation rows and explainer rows are two disjoint sets of 200 training rows
    (or half the training set, if smaller);
  - PACE-SHAP on 200 explainer rows (exact for d ≤ 12, otherwise 32 sampled orders
    per row); PACE-SHAP-20 and KernelSHAP on the same 20 rows; KernelSHAP-strong
    (2,048 samples, no l1 selection; no stability re-run); LIME on the training set;
    PACE-flip, PACE-multi, PACE-contrastive, PACE-decomposition and Random.
- **Datasets:** S1, S2, S3, Wine, Breast Cancer, Thyroid, Shuttle, MAGIC, Letter,
  Landsat, Page Blocks (11).
- **Seeds:** 110–119.
- **Points:** 200 test points per seed (the first 200 of the seed's shuffled test
  rows); Wine keeps all 36 of its test rows, and any other dataset with fewer than
  200 test rows keeps all of them.
- **Added arm, KernelSHAP-same:** KernelSHAP (margin target) on PACE-SHAP's own
  explainer rows (`run_faithfulness._same_rows`, 200 rows) instead of 20: the
  like-for-like arm. It runs last, so every v3 method keeps its random streams.

## Tie-aware outputs

As in `run_review_rq1.py` (ties at margin ≤ 1e-12, near-ties at margin < 0.01):

- tied and near-tied counts per dataset and seed;
- the AUCs on untied points only;
- the raw (unnormalised) margin AUCs (`with_raw=True`);
- the class-kept AUCs.

## Primary and pass rule

**Primary:** PACE-SHAP vs KernelSHAP, at the usual setting (KernelSHAP on 20 rows,
v3's) and on the same rows (KernelSHAP-same), by deletion and by insertion: four
comparisons, on the deletion margin AUC (lower is more faithful) and the insertion
margin AUC (higher), as in v2 and v3.

- Paired over the 10 seeds, two-sided Wilcoxon signed-rank test, Holm across the
  datasets, α = 0.05.
- PACE-SHAP is **better** on a dataset if the Holm p is below 0.05 and the mean
  difference favours it, **worse** in the other direction, otherwise **level**.

No direction is predicted. The four comparisons are not corrected for each other.
Secondary, reported with the same test: the four comparisons on untied points only
(a dataset enters if every seed has at least 10 untied points, `run_review_rq1`'s
rule), on the raw margin AUCs, and on the class-kept AUCs. Everything else (all
methods' AUCs, stability, ground truth, time) is descriptive.

## Defaults

The defaults stay whatever the results: v3's settings, 200 explainer rows, 20 rows
for KernelSHAP. Nothing here is used to choose a setting.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.

## How to run

```
python -B scripts/run_more_points.py --jobs 12
python -B scripts/run_more_points.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_more_points.py --summary-only
```

Outputs: `results/rq1_more_points/runs.csv` (one row per dataset, seed and method),
`summary.md` and `run_log.txt`. Expected: 11 datasets × 10 seeds × 11 methods =
1,210 rows.
