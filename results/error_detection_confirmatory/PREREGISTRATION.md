# Pre-registration: RQ2 confirmatory run

Written and pushed to GitHub before the run started. The commit that adds this file fixes the code
(`pace_cfmnn/signals.py`, `pace_score.py`, `neighborhood.py`, `trust_baselines.py` and
`scripts/run_error_detection.py --protocol confirmatory`). Nothing is changed after the run starts. Any
bug fix found later is reported here with its effect on the results.

## Why a second run

The exploratory run (seeds 0–9, `results/error_detection/`) had three weaknesses:

- **U\* was designed after seeing the data.** We added it after gate G1 showed that the planned U ranks
  tied points too low, and tested it on the same seeds.
- **The baselines were weak.** Only the plain margin and entropy.
- **Few real datasets.** Only Thyroid and Skin had enough errors to test. The 6 synthetic sets share one
  generator.

This run tests the same definitions on fresh seeds, more real datasets and stronger baselines.

## What was known before this run

- **The exploratory results.** EI did not improve the combined score. The planned U failed on Thyroid.
  PD + U\* was the best score on 7 of 8 eligible datasets and beat the plain margin on 7 of 8.
- **A smoke test on seed 100** (excluded from all analysis), run to check the code and the run time. Its
  log printed headline AUPRC values for Wine, Digits, Page Blocks, Landsat, MAGIC, Letter, Skin and S1.
  No definition was changed after it.

## Fixed settings

- **Seeds:** 10–19 (never used before).
- **Model:** CFMNN, fixed centroid rule + search expansion, θ = 0.3, γ = 4 (the main model).
- **Datasets:**
  - Synthetic: S1, S2, S3, S2-imbalanced, S2-label-noise, S2-shifted.
  - Real: Wine, Breast Cancer, Digits, Thyroid, Shuttle, Skin, MAGIC Gamma Telescope, Letter
    Recognition, Landsat Satellite, Page Blocks.
- **Splits:**
  - Datasets under 2,000 rows (Wine, Breast Cancer, Digits) use stratified 5-fold cross-validation
    (blueprint §9.1). In each fold, 20% of the training part is the inner set; the model trains on the
    rest. Out-of-fold scores are pooled.
  - All other datasets use the role split 50/20/10/10/10: train on the training rows, fit the PRS ranks
    on the inner rows, score all other rows. At most 20,000 evaluation rows per seed (a fixed random
    subset).
- **Perturbations:** κ = 0.05, K = 50; discrete features (at most 10 values) are redrawn with
  probability 0.1.
- **Eligibility:** a dataset is tested only if every seed has at least 50 errors among its evaluation
  rows.

## Scores compared (all: higher = more suspicious)

**PACE signals.** None of these reads labels.

| Score | Definition |
| --- | --- |
| PD | Share of the K perturbations whose predicted class differs |
| EI | Instability of the contrastive explanation for the original class pair (blueprint §6.2) |
| U | Blueprint §6.3: η = 0.5, ties graded by the centroid-distance ratio |
| U\* | Ties first: (1 + ratio)/2 for tied points, (1 − margin)/2 otherwise |
| PRS | Mean inner-validation percentile rank of PD, EI and U |
| PRS\* | The same with U\* |
| PD + U | Mean percentile rank of PD and U |
| PD + U\* | Mean percentile rank of PD and U\* |

**Simple references.** Plain margin = 1 − (s′₁ − s′₂); entropy of s′; random.

**Standard baselines.** These read the training labels.

| Baseline | Definition |
| --- | --- |
| Trust Score | Jiang et al., NeurIPS 2018; k = 10, α = 0.1, Euclidean |
| Order ensemble | Share of 5 CFMNNs, trained on the same rows in other random orders, that disagree with the main model |
| kNN disagreement | Share of the 10 nearest training rows with a different label |

## Endpoints

**Metric:** AUPRC.

**Unit of analysis:**
- **The synthetic family:** per seed, each score's mean AUPRC over the 6 synthetic sets, counted as one
  unit.
- **Each eligible real dataset.**

**Test:** a paired two-sided Wilcoxon signed-rank test over the 10 seeds per unit, with Holm correction
across units within each comparison. Significance level 0.05.

**Primary endpoints:**
- **P1.** PD + U\* vs the plain margin. Hypothesis: PD + U\* is higher.
- **P2.** PRS\* vs PD + U\*. Hypothesis: EI adds no value, so PRS\* is not higher. Support = no unit
  where PRS\* is significantly higher.

**Secondary endpoints:**
- PD + U\* vs Trust Score, vs order-ensemble disagreement, vs kNN disagreement (two-sided, no
  direction predicted).
- U\* vs U on real datasets with at least 1% tied points. Hypothesis: U\* is higher.
- AUROC and precision when the top 10% are flagged (descriptive).

## How results will be reported

Every endpoint is reported per unit, whether it supports the hypothesis or not. Per-set synthetic
results are descriptive only. Excluded datasets are listed with their error counts. The exploratory
results stay as they are and are reported as exploratory.
