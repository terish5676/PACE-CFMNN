# Run B, counterfactual baselines and robustness: pre-registration

Written on 2026-11-11 and pushed, with the code it describes, before the run.
The code is `scripts/run_cf_robustness.py` (tests in `tests/test_cf_robustness.py`).
It adds `brute_force_counterfactual` and `replacement_values` to
`pace_cfmnn/counterfactual.py`, and a `method=` keyword (default `"random"`, so
today's output is unchanged) to `dice_counterfactual` in
`pace_cfmnn/xai_baselines.py`. Plan: `research_audit/LIMITATIONS_PLAN.md`, run B.

## A deviation from the plan, made before the run

The plan leaves only DiCE genetic out of r1 (input sensitivity), for cost. A smoke
slice showed DiCE kdtree is just as costly: about 25 s per call even on Wine
(DiCE random about 6 s), and r1 needs 6 calls per method per point. With kdtree
in r1 the run projected to 15–25 h, over twice the plan's ~5 h. The user was
asked and chose to **leave DiCE kdtree out of r1 as well**. Everything else is as
planned: all 7 methods, 30 points, 10 seeds, and r2, r3 and r4 for every method.
r1 is therefore measured for 5 methods: PACE, PACE-sparse, DiCE random, the
nearest unlike neighbour and the brute-force search.

## What ran before this was written

- **Tests.** The suite and the 11 tests of this run pass (the DiCE calls in the
  end-to-end test are stubbed; `TestDiceMethod` calls the real DiCE).
- **DiCE timing on a toy problem** (synthetic, 200 rows, 4 features): random about
  5 s per call, kdtree about 13 s, genetic about 0 s (one call each, or two).
- **A first smoke run** of the whole script, seed 120 (never a study seed), on Wine,
  Thyroid, Letter and Shuttle. It was stopped after about 40 minutes with no task
  finished, because of the DiCE cost above.
- **Per-method timing on seed 120,** one point on Wine (fit 0.0 s; PACE 0.0 s,
  PACE-sparse 0.01 s, DiCE random 6.3 s, genetic 0.07 s, kdtree 25.3 s, nearest
  unlike neighbour 0.0 s, brute force 0.1 s).
- **Three-point timing runs on seed 120:** Wine, Letter and MAGIC (stopped; they
  were slowed by kdtree in r1 and by other jobs) and then, with the final
  configuration, Wine (99 s for 3 points) and Thyroid (68 s for 3 points). They
  printed the share of points with an answer and r1 medians (seed 120, not
  analysed). Nothing in the design changed because of them except the deviation
  above.

About 20–30 s per point on the small datasets projects to roughly 5–8 h for the
full run with 12 processes.

## The question

The paper's counterfactuals are compared with one DiCE setting, and not for
robustness. Is PACE's counterfactual steadier than DiCE's, and do other
counterfactual methods change the picture?

## Setup

B1's design (`scripts/run_counterfactuals.py`, `run_one`, extended):

- **Datasets:** S1, S2, S3, Wine, Breast Cancer, Thyroid, Shuttle, MAGIC, Letter,
  Landsat, Page Blocks (11).
- **Seeds:** 110–119. No earlier run used them.
- **Points:** 30 test points per seed (the paper's split, the seed's row order,
  min-max scaling fitted on the training rows, test rows clipped to [0, 1]). A
  point with no runner-up class is skipped.
- **Target:** the runner-up class of the model's prediction.
- **Locked features:** Thyroid's features 0 and 1, as in B1. Every method
  except the nearest unlike neighbour respects them (as in B1).
- **Model:** `CFMNN()` (θ = 0.3, γ = 4), fitted on the training rows.

## Methods

| Method | What it is |
|---|---|
| PACE | `counterfactual` |
| PACE-sparse | `sparse_counterfactual` |
| DiCE random | `dice_counterfactual`, B1's v2 setting: 5 candidates, DiCE's own sparsity output, the nearest valid one (seeded) |
| DiCE genetic, DiCE kdtree | the same call and setting with `method="genetic"` and `method="kdtree"` (the global numpy seed is set to the study seed first) |
| Nearest unlike neighbour | the nearest training point (L1) that the model predicts as the target. It is the same rule as `xai_baselines.nearest_unlike_neighbour`, with the model's predictions on the training rows computed once, as in B1 |
| Brute force (new) | `brute_force_counterfactual`: change 1 or 2 mutable features, each to one of up to 16 values that occur in the training data (nearest-rank quantiles at (k + 0.5) / 16), keep the smallest-L1 answer the model confirms, or none |

## Measures

Per dataset, seed, point and method:

- the existing ones: found, valid (the model predicts the target), features
  changed, L1, plausibility (L1 to the nearest training point of the target class),
  milliseconds, model calls;
- **r1, input sensitivity.** Draw 5 perturbed copies x′ of x (`Perturber`, κ = 0.05,
  as RQ2; seeded by the study seed and the point). Keep the copies the model
  predicts like x. For each copy with x′ ≠ x, recompute the method's counterfactual
  for the same target; r1 is the median over copies of L1(cf′, cf) ÷ L1(x′, x). A
  copy whose counterfactual is missing is left out; `r1_copies` counts those used.
  Lower is steadier. Not measured for DiCE genetic and DiCE kdtree (see above);
- **r2:** the share of 20 perturbed copies of the answer (κ = 0.05) that the model
  predicts as the target;
- **r3:** the share of 4 CFMNNs, fitted on the same training rows in other orders
  (random permutations seeded by the study seed), that predict the target for the
  answer;
- **r4:** the target margin at the answer (`xai_baselines.margin_score`).

r1–r4 are missing where a method found no answer.

## Primaries and pass rule

PACE and PACE-sparse vs DiCE random, on r1 and r3: four comparisons.

- For each dataset and seed, take the mean of the measure over the points.
- Paired two-sided Wilcoxon over the 10 seeds, Holm across the 11 datasets, α = 0.05.
  Seeds where either method has no value are dropped; when all differences are zero,
  p = 1.
- The first method is **better** on a dataset if the Holm p is below 0.05 and the
  mean difference favours it (r1: lower; r3: higher), **worse** in the other
  direction, otherwise **level**.

No direction is predicted. The four comparisons answer separate questions and are
not corrected for each other. Everything else (validity, features changed, L1,
plausibility, time, r2, r4, the other methods) is descriptive.

## Defaults

The defaults stay whatever the results (B1's setting, κ = 0.05, the DiCE setting).
Nothing here is used to choose a setting.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.

## How to run

```
python -B scripts/run_cf_robustness.py --jobs 12
python -B scripts/run_cf_robustness.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_cf_robustness.py --summary-only
```

Outputs: `results/cf_robustness/runs.csv` (one row per dataset, seed, point and
method, the same columns on every row, NaN where a column does not apply),
`summary.md` and `run_log.txt`. Expected: 110 tasks (11 datasets × 10 seeds),
up to 30 points × 7 methods each.
