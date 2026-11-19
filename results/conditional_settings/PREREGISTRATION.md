# Run D, the conditional test at more settings: pre-registration

Written on 2026-11-11 and pushed, with the code it describes, before the run.
The code is `scripts/run_conditional_settings.py` (tests in
`tests/test_conditional_settings.py`). It uses `PaceShap`, `kernel_shap_game`,
`cond_curves`, `deletion_auc`, `insertion_auc` and `realism` as they are; no library
file changed for this run. Plan: `research_audit/LIMITATIONS_PLAN.md`, run D.

## What ran before this was written

- **Tests.** The full suite (240 tests) passed, including this run's tests: the
  effective-rows measure, that `PaceShap(sigma=...).explain` equals the shared game
  table of `explain_games`, a small end-to-end run on S1 (4 points, 2 settings) and
  the summary's verdicts on a made-up table.
- **A smoke slice, seed 120** (never a study seed, never analysed), written outside
  the repository: S1, Page Blocks and Landsat, all four settings, 50 points. Seed 120
  is not a study seed. It printed deletion AUCs; the only things taken from it are
  that the code runs and its times: Page Blocks 79 s, Landsat 1,179 s, S1 4,356 s
  (one seed, all four settings, 3 tasks sharing the CPU while other jobs ran).
  Nothing in the design changed because of it.

The times project to roughly 25–35 CPU-hours, about 3 h with 12 processes, inside
the plan's ~6 h.

## The question

B3 (`results/extensions_v1/conditional`) ran the conditional Shapley game at one
setting: σ = 0.1 with 200 explainer rows. At that setting, once a coalition holds
most of the features, a few rows carry nearly all the weight. Does the comparison of
PACE-SHAP-cond with KernelSHAP-cond depend on σ and on the number of rows?

## Setup

- **Datasets:** S1, Shuttle, Letter, Landsat, MAGIC and Page Blocks.
- **Seeds:** 110–119. No earlier run used them.
- **Points:** 50 test points per dataset and seed, B3's split, models, evaluation
  rows and tests (`run_faithfulness.setup`, `backgrounds_v3`).
- **Settings (σ, explainer rows):** (0.05, 200), (0.2, 200), (0.4, 200), (0.2, 800).
  The existing B3 run covers (0.1, 200).
- **Explainer rows:** B3's own 200 rows followed, for 800, by further training rows,
  never the evaluation rows (`posthoc_rows.explainer_rows`).
- **Weights' scale:** the per-feature scale of B3's first 200 rows, kept for every
  setting (as `posthoc_rows.py` does), so that only σ and the number of rows change.

## Explainers

Both on the same rows and the same conditional game, whose row weights are
exp(−D_S² / (2σ²)):

- **PACE-SHAP-cond:** `PaceShap(model, rows, n_permutations=32, seed=seed, sigma=σ,
  scale=scale)`. Exact where d ≤ 12 (S1, Shuttle, MAGIC, Page Blocks), 32 sampled
  orders per row otherwise (Letter, Landsat).
- **KernelSHAP-cond:** `kernel_shap_game(..., nsamples=200, sigma=σ, scale=scale)`:
  200 coalitions per point, as in B3.

## Measures

Per dataset, seed, setting and method:

- the four tests of B3: interventional deletion and insertion margin AUCs, and
  conditional deletion and insertion margin AUCs (`cond_curves`);
- **effective rows:** Kish (Σw)² / Σw² of the conditional weights, median over the
  points and over 100 random coalitions per point, separately for coalitions with at
  least half the features and for smaller ones. This is sampled, not the full
  enumeration `posthoc_sigma.effective_rows` uses, because Landsat has 36 features;
- **off the data:** the share of the game's inputs that lie off the data
  (`pace_cfmnn/realism.py`, B3's measure, 200 inputs per point);
- ground truth (top 3 on S1), and milliseconds per explanation.

## Primary and pass rule

**Primary:** the −cond pair at each setting: PACE-SHAP-cond vs KernelSHAP-cond on
conditional deletion margin AUC (lower is more faithful) and conditional insertion
margin AUC (higher), as B3's primaries. That is 8 comparisons (4 settings × 2
tests).

- Paired over the 10 seeds, two-sided Wilcoxon signed-rank test, Holm across the 6
  datasets, α = 0.05.
- PACE-SHAP-cond is **better** on a dataset if the Holm p is below 0.05 and the
  mean difference favours it, **worse** in the other direction, otherwise **level**.

No direction is predicted. The 8 comparisons are not corrected for each other.
Everything else (the interventional tests, effective rows, off the data, ground
truth, time) is descriptive.

## Defaults

σ = 0.1 and 200 rows stay B3's settings whatever the results. These results are
sensitivity results, reported and never used to pick a setting.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.

## How to run

```
python -B scripts/run_conditional_settings.py --jobs 12
python -B scripts/run_conditional_settings.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_conditional_settings.py --summary-only
```

Outputs: `results/conditional_settings/runs.csv` (one row per dataset, seed, setting
and method), `summary.md` and `run_log.txt`. Expected: 6 datasets × 10 seeds × 4
settings × 2 methods = 480 rows.
