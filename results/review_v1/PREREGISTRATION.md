# Review fixes v1: pre-registration

Written on 2026-11-01 and pushed (with the code it describes) before any run on
the seeds below. Smoke runs used seed 0 only, with 2 or 3 points, to check that
each script works:

- Test 1 on S1, Page Blocks, Wine and Letter;
- Test 2 on S1, Wine and Thyroid;
- Test 3 on S1 and Digits;
- Test 4 on S1, Wine, Digits and Page Blocks.

Their output was thrown away, and no setting was changed because of their numbers.
One change was made during the smoke runs, for a reason unrelated to their
numbers: the bagged ensemble's bootstrap rows now keep the model's own row order
(Test 3). Before, they came out shuffled, so the bagged members also differed
in row order, which is what the order ensemble varies.

## Why

A pre-submission review of the paper (2026-11-01) asked for four checks. The
codes are the review's issue numbers.

1. **Exact Shapley values computed without CFMNN's structure** (B2, M1). The
   paper says PACE-SHAP is exact and fast, but never compares it with a
   general-purpose exact method. → Test 1.
2. **Stability, ties and LIME in RQ1** (M2, M3, M5). → Test 2.
   - v3's stability test re-drew KernelSHAP's 20 background rows on every
     re-run, but kept PACE-SHAP's 200. So it mixed two things: new rows and
     new random sampling.
   - v3 divides each point's margin curve by the point's own margin. For a
     tied point (margin 0) that turns the curve into signs.
   - v3 gave LIME the whole training set, although v3's pre-registration said
     every explainer sees only the explainer rows.
3. **Generic ensembles in RQ2** (M7). The order ensemble's gain may come from
   any ensemble, not from CFMNN's dependence on the order of the training rows.
   → Test 3.
4. **The accuracy of the explained model** (m3, m1). The paper never reports
   it. → Test 4.

Tests 2 and 3 are confirmatory. Tests 1 and 4 are descriptive: they measure
something, and nothing is tested.

Earlier results stay untouched. Everything new goes to `results/review_v1/`.

## Common settings

- **Models:** the paper's 80/20 split and the main model (fixed centroid +
  search expansion), as in every RQ1 benchmark. RQ2 keeps its own splits.
- **Seeds:** new for the confirmatory tests: 50–59 (Test 2) and 60–69 (Test 3).
  Test 1 uses v3's timing seeds (30–32) and Test 4 uses G0's (0–4), so their
  numbers line up with the earlier results.
- **Tests:** each comparison is paired over the 10 seeds. It uses a two-sided
  Wilcoxon signed-rank test, Holm-corrected across the datasets (or units) of
  that comparison.
  - A method is **better on a dataset** if the Holm p is below 0.05 and the mean
    difference favours it.
  - **Worse** is the same, in the other direction.
  - Otherwise the two are **level**.
- **Reporting:** results are reported whichever way they go.
  - Any change to this plan is reported as a deviation.
  - Any analysis not listed here is labelled post hoc.
- **Running:** every script saves its rows after each dataset-seed pair, so a
  stopped run continues with `--resume`. Each writes a `runs.csv` and a
  `summary.md`. Test 1 measures time, so it runs alone.

## Test 1 (descriptive): PACE-SHAP vs exact Shapley values computed in general ways

Script: `scripts/exact_baseline.py`. Output: `results/review_v1/exact/`.

v3's models and v3's explainer rows (the first 200), seeds 30–32.

**(a) Exact mode,** on the six datasets with at most 12 features (S1, S2, S3,
Shuttle, MAGIC, Page Blocks; 9 or 10 features each), 10 points per seed. Three
ways to compute the same interventional Shapley values of the margin, on the
same rows:

- PACE-SHAP's exact mode;
- shap's `ExactExplainer` (shap 0.54.1), with an `Independent` masker that holds
  all 200 rows (`xai_baselines.exact_shap`);
- this repository's KernelSHAP solver given every coalition
  (`kernel_shap_game` with 2^d coalitions).

Reported: the largest absolute difference from PACE-SHAP, and milliseconds per
explanation. Expected: differences at rounding level (below 1e-6), and PACE-SHAP
faster than both.

**(b) Sampled mode.** PACE-SHAP's sampled mode with 8, 16, 32 and 64 orders per
row (v3 uses 32), against exact values from PACE-SHAP's exact mode run past
its usual 12-feature limit (`exact_max_d=16`):

- Wine (13 features): all 36 test points, seeds 30–32, 71 rows;
- Letter (16 features): 5 points, seed 30, 200 rows.

Reported: the relative L1 error and the top-3 overlap (as in
`faithfulness_v3/settings.md`). Expected: the error falls as the number of
orders grows, roughly as one over its square root.

## Test 2 (pre-registered): stability, ties and LIME in RQ1

Script: `scripts/run_review_rq1.py`. Output: `results/review_v1/faithfulness/`.

**Setup.** Seeds 50–59 and the 11 RQ1 datasets. v3's models, splits,
evaluation rows and explained points: 50 per dataset and seed, 36 on Wine. v3's
deletion and insertion code, with 10 repetitions.

**Methods.**

- **PACE-SHAP:** v3's setting. It uses the 200 explainer rows (71 on Wine). It
  is exact up to 12 features (S1–S3, Shuttle, MAGIC, Page Blocks), and uses 32
  orders per row otherwise.
- **KernelSHAP:** v3's setting. It uses 20 of the explainer rows, picked with the
  sampling seed, and 200 coalitions with shap's default regularisation. It
  explains the margin of the predicted class.
- **LIME:** 1,000 samples, explaining the margin, as in v3. Its training
  statistics come from the explainer rows, as v3's pre-registration said. v3
  used the whole training set.
  - *LIME:* no discretisation, as in v3.
  - *LIME-default:* LIME's default quartile bins. The attribution is the weight
    of the point's own bin.
- **PACE-flip** and **Random,** as in v3.

**Stability.** Each method runs once with sampling seed s, then twice more with
sampling seed s + 1000:

- **rows kept:** each method keeps its own rows. PACE-SHAP, LIME and PACE-flip
  keep the same 200; KernelSHAP keeps the same 20.
- **new rows:** every method gets 200 new training rows, from outside both of
  v3's sets, drawn with seed s + 2000. KernelSHAP picks 20 of them, as in v3.
  - Wine and Breast Cancer have fewer than 600 training rows, so there are not
    enough unused rows. There, the new rows are a bootstrap resample of the
    explainer rows.

Stability is v3's measure: the Jaccard overlap of the top 3 features by
|attribution|, averaged over points. It compares the first run with each re-run.

**Faithfulness,** on the first run:

- v3's normalised margin AUCs (deletion in signed order, and insertion);
- **raw margin AUCs:** the same curves, not divided by the point's margin and
  not clipped;
- each one on untied points (margin above 1e-12) and on tied points;
- the number of tied points, and of points with a margin below 0.01.

**Comparisons,** each paired over seeds with Holm across the datasets:

| Code | Comparison | Measure | Expected |
|---|---|---|---|
| S1 (primary) | PACE-SHAP vs KernelSHAP | stability, rows kept | PACE-SHAP higher: it is exact on 6 datasets, and KernelSHAP samples coalitions on all 11 |
| S2 (primary) | PACE-SHAP vs KernelSHAP | stability, new rows | PACE-SHAP higher: 200 rows against 20 |
| F1a, F1b | PACE-SHAP vs KernelSHAP | raw deletion and insertion margin AUCs | PACE-SHAP better or level on every dataset |
| F2a, F2b | PACE-SHAP vs KernelSHAP | normalised AUCs on untied points; datasets with at least 10 untied points in every seed | as F1 |
| F3a, F3b | PACE-SHAP vs LIME (explainer rows, v3's setting) | normalised deletion and insertion margin AUCs | PACE-SHAP better on most datasets |
| Secondary | PACE-SHAP vs KernelSHAP | v3's normalised AUCs | as v3: better on about 6 of 11, never worse |

**Descriptive:**

- LIME-default vs LIME;
- means for every method;
- the share of tied points;
- milliseconds per explanation. These are measured while runs share the CPU,
  so they are compared within a dataset only.

## Test 3 (pre-registered): generic ensembles in RQ2

Script: `scripts/run_error_detection_v2.py --protocol ensembles`. Output:
`results/review_v1/ensembles/`.

**Setup.** Seeds 60–69. v2's splits, eligibility and combiner:

- **Eligibility:** at least 50 errors in every seed.
- **Combiner:** logistic regression on standardised signals, fitted on the
  inner rows.
- **Units:** the synthetic family (S1, S2, S3, S2-imbalanced, S2-label-noise,
  S2-shifted), Digits, Thyroid, Skin, MAGIC, Letter, Landsat and Page Blocks.
  Eligibility is checked again on the new seeds.
- **Not run:** Wine, Breast Cancer and Shuttle. They fell below 50 errors in v2
  (lowest counts 2, 20 and 37).

**Signals.** Each ensemble signal is the share of its 5 members whose
prediction differs from the model's.

- **Order ensemble** (v2): 5 CFMNNs, each trained on the training rows in a
  different random order.
- **Bagged ensemble** (new): 5 CFMNNs, each trained on a bootstrap sample of
  the training rows, in the model's own order.
- **SGD order ensemble** (new): 5 logistic regressions. Each is trained by one
  pass of stochastic gradient descent over the training rows in a different
  random order (scikit-learn `SGDClassifier`, log loss, one pass, no
  shuffling). This is an order ensemble that is not built from CFMNN.

**Combiners:**

- kNN + Trust Score (TS);
- + order ensemble;
- + bagged ensemble;
- + order + bagged ensembles;
- + SGD order ensemble.

**Comparisons,** by AUPRC, each paired over seeds with Holm across the
eligible units:

| Code | Comparison | Question | Expected |
|---|---|---|---|
| T3a (primary) | kNN + TS + order vs kNN + TS + bagged | Is the order ensemble better than a generic ensemble of the same size? | order at least level |
| T3b | kNN + TS + order + bagged vs kNN + TS + bagged | Does the order ensemble add anything once bagging is in? | adds on some units |
| T3c | kNN + TS + SGD order vs kNN + TS | Does an order ensemble that is not built from CFMNN add anything? | a smaller gain than CFMNN's order ensemble |
| T3d | kNN + TS + order vs kNN + TS | Does v2's finding replicate on new seeds? | as v2 |

## Test 4 (descriptive): accuracy

Script: `scripts/accuracy_table.py`. Output: `results/review_v1/accuracy/`.

**Setup.** All 13 datasets, the paper's 80/20 split and seeds 0–4 (as G0).
Min-max scaling is fitted on the training rows, as in RQ1 and G0.

**Models,** all trained on the same rows in the same order:

- CFMNN (the main model);
- 10-nearest neighbours (scikit-learn defaults otherwise);
- a Random Forest with 200 trees.

**Reported:** test accuracy, as the mean ± standard deviation over seeds.

## How to run

From the repository root, one test at a time, with the laptop kept awake:

```
python -B scripts/accuracy_table.py
python -B scripts/exact_baseline.py
python -B scripts/run_review_rq1.py --jobs 6
python -B scripts/run_error_detection_v2.py --protocol ensembles --jobs 4
```

After a stop, run the same command with `--resume`. To rebuild a summary from
saved rows, add `--summary-only`.
