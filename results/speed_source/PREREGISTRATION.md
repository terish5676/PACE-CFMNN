# Run P, where PACE-SHAP's speed comes from: pre-registration (descriptive)

Written on 2026-11-15 and pushed, with the code it describes, before the run.
The code is `scripts/run_speed_source.py` (tests in `tests/test_speed_source.py`).
- It uses the existing `PaceShap(prune=False)` option and `xai_baselines.exact_shap`.
- It uses the second explanation run's set-up (`run_faithfulness.setup`, `backgrounds_v3`,
  `SHAP_V3_R`), as `scripts/exact_baseline.py` did for the exact check.
- No shared module changes.

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 9.

## What ran before this was written

- **The exact check** (`results/review_v1/exact/`, seeds 30–32, 10 points each, laptop):
  - PACE-SHAP's exact mode equals shap's ExactExplainer to ≤ 3.3e-16;
  - it is 16–30 times faster on the 6 datasets with at most 12 features.
  This run splits that speed-up in two.
- **Smoke slices on the laptop, seed 194** (not a study seed, never analysed), 1 point each,
  written outside the repository. Milliseconds per explanation for PACE-SHAP, PACE-SHAP without
  pruning and ExactExplainer:
  - Shuttle: 18, 29 and 1,637;
  - Page Blocks: 62, 124 and 1,093;
  - S1: 2,391, 4,443 and 57,303.

  Pruning skipped 64–76% of the (row, box) pairs. Every largest difference was ≤ 9e-17, and the
  check printed PASS. So the likely direction of the result is known: most of the speed-up comes
  from the structure, and pruning adds up to about 2 times. The design did not change after it,
  except for a grammar fix in the summary.
- **Tests while building:** `tests/test_speed_source.py` (6 tests) and `tests/test_theta_churn.py`
  (Run O) pass on the laptop.

## The question

A reviewer can say that exact Shapley values were to be expected, because a box's membership is a
mean of one-feature terms. But the margin is not additive: it is a maximum over boxes, then over
classes, plus the tie-break. **How much of PACE-SHAP's speed over a general exact explainer comes
from the per-feature structure, and how much from pruning boxes that cannot matter?**

## What the code does (cost per explained point)

With R explainer rows, H boxes and d features:

- **ExactExplainer** calls the model on 2^d × R mixed inputs. Each call evaluates H × d ramp terms,
  so the cost is O(2^d · R · H · d).
- **PACE-SHAP without pruning:**
  - computes each feature's term once per (row, box), as an R × H × d array of changes;
  - gives every coalition's membership with one matrix product per kept (row, box) pair (K × d
    times d × 2^d), then takes each class's maximum and the margin per row;
  - cost: O(K · d · 2^d), with K = R · H.
- **PACE-SHAP** first drops every (row, box) pair that can never be its class's strongest box
  (Proposition 3), so K is smaller.

The two methods have the same order of cost, exponential in d. PACE-SHAP saves a constant factor,
not the 2^d: it adds precomputed terms instead of evaluating ramps, in large vectorised blocks,
and pruning lowers K. This run measures the two factors.

## Setup

- **Datasets:** the 6 RQ1 datasets with at most 12 features (S1, S2, S3, Shuttle, MAGIC and Page
  Blocks; the script refuses a dataset with more).
- **Seeds:** 240–242 (new).
- **Model and rows:** each seed's model and split from the second explanation run's set-up, with
  200 explainer rows.
- **Points:** 2 test points per seed. The time per point varies little, and ExactExplainer takes
  about a minute per point on S1–S3.
- **Order:** one task at a time, with an untimed warm-up of PACE-SHAP on one point first. Then
  PACE-SHAP, PACE-SHAP without pruning and ExactExplainer, always in that order.
- **What is timed:** each explainer's set-up is included, as in the exact check.

## Measures

Per dataset, seed and arm:
- milliseconds per explanation;
- the largest absolute difference from PACE-SHAP's values;
- the largest absolute value, for scale;
- d and boxes;
- for PACE-SHAP, the share of (row, box) pairs pruning skipped (`stats_["share_of_boxes_skipped"]`).

Summary per dataset (means over seeds):
- **structure** = ExactExplainer ÷ no pruning;
- **pruning** = no pruning ÷ PACE-SHAP;
- **total** = ExactExplainer ÷ PACE-SHAP.

**Check (the script exits 1 if it fails; stop and ask):** PACE-SHAP without pruning agrees with
PACE-SHAP within 1e-12, and ExactExplainer within 1e-9, on every point. The 1e-9 allows for shap's
`np.isclose` shortcut, which caused differences near 1e-10 in Runs I and M.

**No primary test.** This run is descriptive: it explains a speed-up already reported.

## How the result will be read (written now, before the run)

The paper's paragraph on why exactness is not free will give, per dataset, the structure factor,
the pruning factor and the share of pairs skipped, whatever they are. It will say that the cost
stays exponential in d, which is why exact mode stops at 12 features.

## Time

On the laptop, ExactExplainer took about 57 s per point on S1; on Shuttle and Page Blocks, about
1–2 s. The 36 points are estimated at about 20–25 minutes on the laptop, and **about 10–30 minutes
on the home PC**. Stop and ask if a smoke slice projects more than 60 minutes.

## Tests

`tests/test_speed_source.py`:
- on a small model, pruning on and off give the same values, and both match ExactExplainer;
- pruning skips some pairs, and no pruning skips none;
- the check passes within the limits and fails outside them;
- the summary's three ratios are computed as described;
- one end-to-end task on Shuttle, smoke seed 194.

## How to run (home PC, alone, after Run O has finished)

```
.\.venv\Scripts\python -B scripts/run_speed_source.py --datasets S1 shuttle --seeds 194 --n 1 --out $env:TEMP\speed_source_smoke
.\.venv\Scripts\python -B scripts/run_speed_source.py *>&1 | Tee-Object results/speed_source/run_log.txt
.\.venv\Scripts\python -B scripts/run_speed_source.py --resume *>&1 | Tee-Object -Append results/speed_source/run_log.txt   # after a stop
```

Commit `results/speed_source/` unedited (`runs.csv`, `summary.md`, `run_log.txt`) as
"Run P: results of the pre-registered run on the home PC".
