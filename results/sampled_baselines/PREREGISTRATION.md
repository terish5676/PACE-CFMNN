# Run J, modern sampled baselines above 12 features: pre-registration

Written on 2026-11-14 on the laptop and pushed, with the code it describes, before the run.
The run itself is on the home PC (i5-14400F).

- **The code:** `scripts/run_sampled_baselines.py` (tests in `tests/test_sampled_baselines.py`).
- **New module:** `pace_cfmnn/sampled_shap.py`: the two baselines and the model call they
  share.
- **Used as it is:** `pace_cfmnn/explain.py` (PACE-SHAP), `scripts/run_faithfulness.py` (the
  second explanation run's model, split, rows and its deletion and insertion tests), and
  `xai_baselines.shapley_kernel_coalitions` and `kernel_solve` (the paired sampler and the
  solver the conditional run already uses). No existing file was changed for this run.

## Why

Above 12 features PACE-SHAP is no longer exact: it samples feature orders. The second
explanation run (`results/faithfulness_v3/`, seeds 30–39) compared it there with shap's
KernelExplainer on 20 background rows and 200 samples, which is an old estimator on a much
smaller budget. The peer review (item 11) asks for "a modern sampled baseline above 12
features at equal evaluation budget (shap.PermutationExplainer; paired-sampling KernelSHAP of
Covert and Lee)".

The second half of item 11 (DiCE's genetic method in the counterfactual table) needs no run
and is left for the write-up.

## What ran before this was written

- **Unit tests** on made-up data, while building.
- **A count of the rows shap really sends to the model.** Given several points in one call,
  shap 0.54's PermutationExplainer also computes "main effects" (d + 1 more masks) for every
  point after the first, although they are not asked for and not used. The baseline therefore
  explains one point per call, which keeps it inside the budget. This was found on made-up
  data, before any real dataset was run.
- **Smoke slices on seed 190** (not a study seed, never analysed), on the laptop, written
  outside the repository:
  - the full task on Wine and Breast Cancer;
  - the task with 10 points on Thyroid, Letter and Landsat (3 jobs);
  - the timing pass alone, 10 points, on all five datasets.

  They printed, for seed 190, each method's deletion and insertion margin AUC, stability,
  error against exact values (Wine, Letter) and milliseconds. So these were seen: on that
  seed the three estimators' deletion AUCs were within 0.012 of each other on every dataset,
  PACE-SHAP's error against exact values was the smallest (Wine 0.008 against 0.029 and
  0.022; Letter 0.012 against 0.045 and 0.045), and PACE-SHAP was 3 to 15 times faster. One
  seed is not evidence; only seeds 160–169 are.
- **What changed because of the smoke slices:**
  - the baselines' model call was rewritten to do the same arithmetic in fewer passes
    (values equal to 1e-15), after the first slices showed the baselines taking 10 to 29 s
    per explanation on Letter and Landsat. This makes the baselines faster, not PACE-SHAP.
    The error figures above come from the slices before the rewrite;
  - the timing pass was cut from 50 points to 10 per seed, to keep it near 20 minutes.

  No dataset, method, budget, measure or comparison changed.

## Setup

- **Datasets:** the explanation datasets (RQ1) with more than 12 features. The feature
  counts were read from the data: **Wine 13, Letter 16, Thyroid 21, Breast Cancer 30,
  Landsat 36**. The other six have 9 or 10.
- **Seeds:** 160–169 (10). No earlier run or smoke slice used them.
- **Model, split, points, rows:** those of the second explanation run
  (`run_faithfulness.setup`, `backgrounds_v3`):
  - the paper's 80/20 split, training rows in the seed's order, min-max scaling fitted on
    them, `CFMNN()` (θ = 0.3, γ = 4);
  - **50 explained test points** per seed (Wine has 36 test points; all are used);
  - two disjoint sets of 200 training rows (71 each on Wine, which has 142): the
    **evaluation rows**, from which the deletion and insertion tests copy values, and the
    **explainer rows**, the only rows an explainer sees.
- **The game:** every method estimates the same thing: the interventional Shapley values of
  the predicted class's margin, with the R explainer rows as reference (R = 200; 71 on
  Wine). So the methods differ only in how they sample.

## Methods

| Name | What it is |
|---|---|
| **PACE-SHAP** | `PaceShap(model, rows, n_permutations=32)`: sampled mode, 32 feature orders per background row (16 random orders and their reverses), read from the boxes without calling the model. The call of the second explanation run (tested) |
| **PermutationSHAP** | `shap.PermutationExplainer` (shap 0.54), with an Independent masker holding all R rows. It runs random feature orders, each forward and backward, the same orders for every row |
| **KernelSHAP-paired** | KernelSHAP with paired sampling: coalitions drawn from the Shapley kernel together with their complements, then least squares under the efficiency constraint. I. Covert and S.-I. Lee, "Improving KernelSHAP: Practical Shapley Value Estimation Using Linear Regression", AISTATS 2021, PMLR 130:3457–3465 (the arXiv version, 2012.01536, is titled "… via Linear Regression"). Implemented in this repository (`sampled_shap.paired_kernel_shap`); no outside package |
| Random | a random ranking: the floor, as in the second explanation run |

Both baselines are black boxes: they see the model only through calls on hybrid inputs (the
explained point on a coalition's features, one background row elsewhere). Their model call
(`sampled_shap.MarginModel`) returns the same margin as the model and counts the rows.

**Not included, on purpose:** the "unbiased KernelSHAP" variant of the same paper (the
paper finds that the original form has the lower variance, and that pairing lowers it
further), and the shap
KernelExplainer setting of the earlier run (already reported there).

## The equal budget

**One model evaluation = the margin on one hybrid input.** The budget per explanation is

> **N = R × P × (d + 1)**, with P = 32,

which is what PACE-SHAP's sampled mode amounts to: on each of the R rows, each of its 32
orders walks through d + 1 coalitions (from the background row to the explained point).

| Dataset | d | R | N per explanation |
|---|---|---|---|
| Wine | 13 | 71 | 31,808 |
| Letter | 16 | 200 | 108,800 |
| Thyroid | 21 | 200 | 140,800 |
| Breast Cancer | 30 | 200 | 198,400 |
| Landsat | 36 | 200 | 236,800 |

- **PermutationSHAP** gets `max_evals = P × (d + 1)` masks, each evaluated on the R rows: at
  most N evaluations. shap turns that into 16 orders, each forward and backward (32
  passes), when all d features vary. shap skips a row whose masked input did not change, so
  it can use fewer than N.
- **KernelSHAP-paired** gets `P × (d + 1) − 2` sampled coalitions, plus the empty and the
  full coalition, each on the R rows: N − R + 1 evaluations.
- The rows each baseline really sent to the model are counted and written to `runs.csv`
  (`rows_per_explanation`). The script stops if a method goes over N.
- **What "equal" does not cover:** PACE-SHAP's N values are read from the model's terms, not
  computed by calling the model, so its evaluations are cheaper. That is the method's
  advantage, and it shows in the time per explanation, not in the budget.
- **A choice that could favour either side:** every method spreads its budget over all R
  rows. A baseline could instead use fewer rows and more coalitions, but it would then
  estimate a different game. Keeping the game the same was chosen before any study seed ran.

## Measures

For each dataset, seed and method (`runs.csv`):

- **deletion margin AUC** (lower is more faithful) and **insertion margin AUC** (higher is
  more faithful): `deletion_auc(signed=True)` and `insertion_auc` of the second explanation
  run, on the evaluation rows, 10 repetitions;
- **error against exact values, where d ≤ 16** (Wine and Letter): exact values from
  PACE-SHAP's exact mode on the same rows (`PaceShap(exact_max_d=16)`; exact mode equals
  shap's ExactExplainer to 1e-16 at 9 and 10 features in `results/review_v1/exact/`), for the **first 10**
  explained points of each seed. Reported: relative L1 error (sum of |estimate − exact| over
  features, divided by the sum of |exact|, mean over points), the largest absolute error, and
  the top-3 overlap with the exact values. Above 16 features an exact reference costs 2^d
  coalitions per row and is not computed (2^21 for Thyroid);
- **top-3 stability under a new seed:** the Jaccard overlap of the top 3 features by
  absolute value between the run with the study seed and a second run with seed + 1000, mean
  over points, as in the second explanation run;
- **milliseconds per explanation,** twice: during the run (several tasks share the CPU, so
  compare within a dataset only), and in a **timing pass run alone** after it (`--timing`:
  one pass per method, the first 10 explained points, seeds 160–162). The timing pass is the
  one the paper quotes;
- model rows per explanation, boxes, d, R.

## Primary and pass rule

Two primary comparisons, both on **deletion margin AUC**:

- **P1:** PACE-SHAP against PermutationSHAP;
- **P2:** PACE-SHAP against KernelSHAP-paired.

Each: a paired two-sided Wilcoxon test over the 10 seeds (one value per seed: the mean over
its explained points), Holm-corrected across the 5 datasets, α = 0.05. When all 10
differences are zero, p = 1. PACE-SHAP is **better** on a dataset if the Holm p is below 0.05
and its mean AUC is lower, **worse** if it is higher, otherwise **level**. No direction is
predicted. P1 and P2 are not corrected for each other.

**Secondary:** the same two pairs on insertion margin AUC (higher is better), the same test.
**Descriptive:** error against exact values, stability, time, rows used.

"Level" is not "equal": no equivalence test is made. With 10 seeds the smallest two-sided
p is 0.002, so a Holm p below 0.05 across 5 datasets is reachable.

## What the result changes

- Results are reported whichever way they go.
- PACE-SHAP's settings (32 orders per row, 200 rows) stay as they are in every committed
  result, whatever this run says.
- If PACE-SHAP is level with both baselines on faithfulness, the paper says so, and its case
  above 12 features rests on what the descriptive measures show (error, stability, time),
  labelled as descriptive.
- If a baseline is better on some dataset, the paper reports it in the main explanation
  table for that dataset.
- Any change to this plan after the push is a dated amendment (before the run) or a reported
  deviation (after it). Any analysis not listed here is labelled post hoc.

## Time

From the laptop's timing pass on seed 190 (alone, 10 points), seconds per explanation:

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired |
|---|---|---|---|
| Wine | 0.02 | 0.24 | 0.12 |
| Breast Cancer | 0.17 | 1.61 | 1.56 |
| Thyroid | 0.15 | 0.51 | 1.10 |
| Letter | 1.07 | 5.22 | 6.48 |
| Landsat | 1.37 | 9.20 | 9.63 |

Each method runs twice per task (the second run is the stability test), on 50 points. That
gives, per seed, about 34 minutes for Landsat, 21 for Letter plus 12 for its exact values
(70 s per point), 6 for Breast Cancer, 3 for Thyroid and under 1 for Wine: about 13 hours of
one core for the 50 tasks. With 12 jobs on the home PC: about **2 hours**, between 1.5 and
3. The timing pass adds about 20 minutes, alone.

If the home PC's smoke slice (below) takes more than twice its expected time, or the run
projects to more than 6 hours, stop and ask before the full run.

## Tests

`tests/test_sampled_baselines.py` checks that:

- the baselines' model call equals `xai_baselines.margin_score` and counts every row;
- **the paired sampler converges to exact values on a small model:** with every coalition it
  equals the exact Shapley values to 1e-12; with sampled coalitions its error falls as the
  coalitions grow (240 against 30) and is small at 240;
- it is seeded, adds up to f(x) minus the mean over the rows, uses k × R + R + 1 rows, draws
  coalitions in complementary pairs, and gives the same values as the conditional run's
  `kernel_shap_game` with the same seed;
- PermutationSHAP converges to the exact values, is seeded, adds up, stays within
  `max_evals × R` rows and leaves numpy's global random state as it found it;
- the budget is as above, and each method keeps to it on a 13-feature toy model;
- PACE-SHAP here is the second explanation run's call, value for value;
- the comparison reads a lower deletion AUC as better;
- one task writes one row per method with the same columns.

No shared module was changed, so no committed result needed re-checking.

## How to run

Smoke slice first (seed 190, written outside the repository; about 4 minutes on the laptop):

```
python -B scripts/run_sampled_baselines.py --datasets wine breast_cancer --seeds 190 --out <scratch folder> --fresh
```

Then the run, and after it the timing pass, alone:

```
python -B scripts/run_sampled_baselines.py --jobs 12
python -B scripts/run_sampled_baselines.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_sampled_baselines.py --timing               # nothing else running
python -B scripts/run_sampled_baselines.py --summary-only
```

Outputs in `results/sampled_baselines/`: `runs.csv` (one row per dataset, seed and method; 5
× 10 × 4 = **200 rows**), `timing.csv` (5 datasets × 3 seeds × 3 methods = **45 rows**) and
`summary.md`. `--timing` rewrites `summary.md` with the times measured alone.
