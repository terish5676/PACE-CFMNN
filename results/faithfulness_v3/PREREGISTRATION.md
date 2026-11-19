# Pre-registration: RQ1 v3 (PACE-SHAP)

Written and pushed to GitHub before the run started. The commit that adds this file fixes the code
(`pace_cfmnn/explain.py` `PaceShap`, `scripts/run_faithfulness.py --protocol v3`). Nothing is changed
after the run starts. Any bug fix found later is reported here with its effect on the results.

## Why this run

Under the fair v2 protocol (seeds 0–9, `results/faithfulness/`), KernelSHAP had a lower deletion margin
AUC than PACE-flip on 6 of 11 datasets (S1–S3, Wine, MAGIC, Letter). PACE-flip was better on Breast
Cancer, Thyroid and Landsat; Shuttle and Page Blocks were level. The likely reason: PACE-flip ranks
features by how often changing *one* feature alone flips the class, while the tests remove features
*together* and measure the margin. KernelSHAP estimates the Shapley value of that margin, but by
sampling (200 coalitions, 20 background rows).

PACE-SHAP computes the same Shapley value from CFMNN's own per-feature membership terms: exactly when
d ≤ 12 (all 2^d coalitions), otherwise by antithetic permutation sampling. It is a model-specific
explainer (like TreeSHAP for trees). It is not fitted to any dataset or to any test result, and the
CFMNN model is unchanged.

## What was known before this run

- The v2 results above (seeds 0–9).
- Correctness checks on small random models (unit tests): PACE-SHAP equals brute-force Shapley values
  of the margin to 1e-12, and equals KernelSHAP with every coalition enumerated on the same rows.
- A check on 3 Page Blocks points (seed 100, attributions only): PACE-SHAP-20 equals brute force to
  5e-17. KernelSHAP with 2,048 samples on the same rows differed by up to 1e-4 because of shap's default
  l1 feature selection; with `l1_reg=False` it matched to 1e-17. KernelSHAP-strong therefore uses
  `l1_reg=False`; plain KernelSHAP keeps v2's default so it stays comparable with v2.
- Timings and the "share of boxes skipped" on seed 99 test points (no faithfulness computed).
- The settings study `settings.md` (seed 99, approximation quality only, no faithfulness computed). Its
  selection rule was written in `scripts/pace_shap_settings.py` before it ran.
  No setting met the rule on every dataset (Thyroid's top-3 overlap stayed below 0.90 at R = 200,
  P = 16), so the rule's fallback applies: **R = 200, P = 32**. The study also showed that with only 20
  background rows (the number KernelSHAP uses) the Shapley values differ from the 200-row values by
  17–76% (relative L1, mean per dataset).
- Speed (seed 99, not faithfulness): at R = 200, PACE-SHAP takes about 2.5–3 s per point on S1–S3
  (KernelSHAP took about 1 s in v2) and is also slower than KernelSHAP on Landsat; with 20 rows it is
  about 0.3 s on S1–S3. Speed is reported as measured, not claimed as an advantage in advance.
- No faithfulness number for PACE-SHAP was computed on any seed before this file was pushed.
- After pushing, a smoke run (seed 100, 5 points, S1 and Wine; excluded from all analysis) checks that
  the script runs end to end. Its numbers are not used, and no setting is changed because of them.

## Fixed settings

- **Seeds:** 30–39 (never used before).
- **Model:** CFMNN, fixed centroid rule + search expansion, θ = 0.3, γ = 4 (the main model).
- **Datasets (11):** S1, S2, S3, Wine, Breast Cancer, Thyroid, Shuttle, MAGIC, Letter, Landsat,
  Page Blocks. Splits as in v2 (`data.paper_split`), 50 test points per dataset and seed.
- **Backgrounds:** two disjoint sets of training rows, 200 each (half the training set if it has fewer
  than 400 rows, i.e. Wine). The deletion and insertion tests copy values only from the *evaluation*
  set; every explainer uses only the *explainer* set.
- **Tests:** as in v2. Features ranked by signed attribution (most supportive of the predicted class
  first); deletion and insertion of the normalised margin, 10 repetitions, each copying from one random
  evaluation row.

## Methods

| Method | Setting |
|---|---|
| PACE-SHAP | first R = **200** explainer rows (71 for Wine); exact when d ≤ 12, otherwise P = **32** orders per row (antithetic pairs), seed = run seed |
| PACE-SHAP-20 | the same 20 rows KernelSHAP draws for the seed; otherwise as PACE-SHAP |
| KernelSHAP | margin target, 20 explainer rows, 200 samples (as in v2) |
| KernelSHAP-strong | as KernelSHAP but 2,048 samples and no l1 feature selection (`l1_reg=False`); shap then enumerates every coalition when d ≤ 11 and is exact; no stability re-run |
| LIME | margin target, 1,000 samples (as in v2) |
| PACE-flip, PACE-multi, PACE-contrastive, PACE-decomposition | as in v2, with the explainer rows |
| Random | uniform scores |

Occlusion (brute force) is dropped: it equals PACE-flip and exists only for timing.

## Hypotheses and tests

All comparisons are paired over the 10 seeds per dataset, two-sided Wilcoxon signed-rank tests, Holm
correction across the 11 datasets within each comparison, α = 0.05.

- **H1 (primary):** PACE-SHAP has a lower deletion margin AUC than KernelSHAP.
  Reported as the number of datasets where PACE-SHAP is significantly better, significantly worse, and
  not different.
- **H2:** PACE-SHAP has a higher insertion margin AUC than KernelSHAP.
- **H3:** PACE-SHAP-20 vs KernelSHAP (deletion and insertion): the same background rows, so any
  difference comes from KernelSHAP's sampling.
- **H4:** PACE-SHAP vs KernelSHAP-strong (deletion and insertion).
- **Secondary, descriptive:** PACE-SHAP vs LIME and vs PACE-flip; stability (top-3 overlap when re-run
  with another seed); milliseconds per explanation (`--timing`, run alone, seeds 30–32); ground-truth
  top-3 on S1–S3.

## Expected outcomes and how they will be reported

- Exact mode makes PACE-SHAP's stability 1.00 (the background rows are fixed for a seed). In sampled
  mode stability is below 1 and is reported as measured.
- Shapley values are not built to give the best deletion order, so PACE-SHAP may still lose to
  KernelSHAP or LIME on some datasets. Every result is reported, whichever way it goes, and the paper's
  RQ1 claim will follow these results only.
- PACE-SHAP-20 and KernelSHAP-strong should agree closely when d ≤ 11 (both are exact on the same rows);
  a large gap would point to a bug and would be investigated and reported here.
