# RQ1 extensions v1: pre-registration

Written on 2026-10-31 and pushed (with the code it describes) before any run on
the confirmatory seeds 40–49. Smoke runs used seed 0 only, with few points, to
check that each script works: B1 on S1, Wine and Thyroid; B2 on S1 and Wine; B3
on S1c and Wine; B4 on S1, S3 and S1c. Their output was thrown away, and no
setting was changed because of their numbers.

## Why

The guide asked for three improvements to RQ1:

1. **Conditional feature replacement**, to make explanations more realistic (B3).
2. **Sparse counterfactuals**, to address PACE's weakness against DiCE (B1).
3. **Better global aggregation**, to strengthen dataset-level explanations (B2).

Item 3 also asked for 10 seeds, confidence intervals and paired significance
tests. Those are already in the results on `rq1-loose-ends` (PR #12). Here they
are used for every new comparison.

B4 adds pairwise interaction values, which come from feedback on PACE-SHAP.

Earlier results (`faithfulness_v3`, `counterfactuals`, `global_importance`)
stay untouched. Everything new goes to `results/extensions_v1/`.

## Common settings

- **Models:** the paper's 80/20 split and the main model (fixed centroid + search
  expansion), as in every RQ1 benchmark.
- **Seeds:** 40–49 (never used before), 10 per dataset.
- **Datasets:** the 11 RQ1 datasets (S1, S2, S3, Wine, Breast Cancer, Thyroid,
  Shuttle, MAGIC, Letter, Landsat, Page Blocks). B3 adds S1c and S2c. B4 uses the
  datasets with at most 12 features.
- **Tests:** each comparison is paired over the 10 seeds. It uses a two-sided
  Wilcoxon signed-rank test, Holm-corrected across the datasets of that
  comparison.
  - A method is **better on a dataset** if the Holm p is below 0.05 and the mean
    difference favours it.
  - **Worse** is the same, in the other direction.
  - Otherwise the two are **level**.
  - With 10 seeds the smallest possible p is 0.002, so a dataset can reach
    significance after Holm.
- **Reporting:** results are reported whichever way they go.
  - Any change to this plan is reported as a deviation.
  - Any analysis not listed here is labelled post hoc.
- **Running:** each script runs with `--jobs 6`, never during a timing run. Times
  per explanation or counterfactual are measured while runs share the CPU. They
  are compared within a dataset only.

## B1. PACE-sparse counterfactuals

**Method** (`counterfactual.sparse_counterfactual`): find the fewest changed
features that make the model predict the target class.

- **Candidate target-class boxes:** the 10 nearest by L1 (as PACE uses) plus the
  10 with the fewest features outside their intervals. Boxes that exclude a
  locked feature's value are skipped.
- **New values:** for each box and each step s in (0, 0.25, 0.5, 0.75, 1), a
  mutable feature's new value is x clipped into the box, then moved a share s
  towards the box's centroid.
- **Search:** subsets of moved features are scored exactly from CFMNN's own
  terms (memberships and centroid distances are sums over features), smallest
  subsets first. Among winning subsets of the smallest size, the smallest L1
  change wins.
  - The search is exhaustive while a size has at most 65,536 subsets.
  - Beyond that it uses greedy forward selection, then tries dropping each
    feature.
- **Checks:** every answer is verified on the model.
- **Fallback:** if PACE's own answer changes fewer features, or the same number
  with a smaller L1, PACE's answer is returned. So PACE-sparse never changes more
  features than PACE.

**Run:** `python scripts/run_counterfactuals.py --extended --jobs 6`

- 11 datasets, seeds 40–49, 30 test points per seed.
- The target is the runner-up class.
- Thyroid locks features 0 and 1 (age and sex).
- DiCE uses the fair v2 setting: 5 candidates, its own sparse output, the
  nearest valid one.
- Methods: PACE, PACE-sparse, DiCE, nearest unlike neighbour.

**Primary:** **features changed**, PACE-sparse minus DiCE.

- For each seed: the mean difference over the points where both answers are
  valid.
- The test is over seeds, Holm across 11 datasets.
- Reported: the number of datasets where PACE-sparse is better, worse or level.

**Secondary:**

- L1 distance and plausibility (L1 to the nearest training point of the target
  class), PACE-sparse vs DiCE;
- features changed and L1, PACE-sparse vs PACE;
- the share of valid answers, PACE-sparse vs DiCE;
- time per counterfactual;
- how PACE-sparse found its answer (exact search, greedy search or PACE's
  answer).

**Expected:**

- PACE-sparse changes fewer features than PACE (by construction, never more).
- Against DiCE we make no prediction: earlier runs found DiCE sparser on some
  real datasets.
- PACE-sparse may move further in L1 than PACE, because it changes fewer
  features by larger amounts.

## B2. Better global aggregation

**Run:** `python scripts/run_global_importance.py --extended --jobs 6`, on 11
datasets with seeds 40–49. It keeps the five v2 rankings and adds three:

- **mean |PACE-SHAP|:**
  - exact Shapley values of the predicted class's margin;
  - on the same 50 test points as mean |KernelSHAP|;
  - with the same 200 background rows PACE-flip uses;
  - exact for d ≤ 12, otherwise 32 antithetic orders per row.
- **mean |PACE-SHAP-20|:** the same, on KernelSHAP's own 20 rows.
- **PACE-flip, normalised:** each point's PACE-flip scores are scaled to sum 1
  before averaging, so points where many features can flip the decision do not
  dominate.

**Primary:** agreement with **permutation importance** (Spearman rho), mean
|PACE-SHAP| minus mean |KernelSHAP|. The test is paired over seeds, Holm across
11 datasets.

**Secondary:**

- mean |PACE-SHAP-20| vs mean |KernelSHAP|;
- PACE-flip normalised vs PACE-flip averaged;
- PACE-flip averaged vs mean |KernelSHAP| (a replication of PR #12's comparison
  on fresh seeds);
- ground-truth top 3 on S1–S3;
- seconds per ranking.

**Note:** permutation importance is the model-based reference, not the truth. It
measures the drop in accuracy, while SHAP-style rankings measure the margin.

## B3. Interventional vs conditional replacement

There are two ways to fill in the features an explainer removes. They answer
different questions, so they are **reported in separate tables and never
combined into one ranking**.

- **Interventional:** values come from any background row. This is what the
  model does on such inputs, even unrealistic ones.
- **Conditional:** values come from rows that resemble the point on the features
  it keeps. This stays near real data.

**Run:** `python scripts/run_faithfulness.py --protocol cond --jobs 6`

- 13 datasets: the 11, plus S1c and S2c. These are S1 and S2 where feature 3 is
  a noisy copy of feature 0 that the true rule does not use.
- Seeds 40–49, 50 test points per seed.
- Every explainer uses the same explainer rows: the first 200 of the v3
  explainer background, or half the training set if that is smaller.
- The tests copy values from the disjoint v3 evaluation rows.

**The 8 explainers:**

| Explainer | Game | Settings |
|---|---|---|
| PACE-SHAP | interventional Shapley game of the margin | exact for d ≤ 12; otherwise 32 antithetic orders per row |
| PACE-SHAP-cond | conditional game: rows weighted by exp(−D²/(2σ²)), with D² the mean squared standardised difference on the kept features (Aas, Jullum and Løland 2021, diagonal covariance) | σ = 0.1; exact for d ≤ 12; otherwise 32 antithetic orders shared by all rows |
| PACE-flip | single-feature replacement by the 32 background quantiles | as in v3 |
| PACE-flip-cond | single-feature replacement by feature i of the 32 rows nearest to the point on its other features | exact (equals brute-force occlusion with the same values) |
| KernelSHAP | interventional game | our implementation (`kernel_shap_game`); 200 coalitions per point in complementary pairs drawn from the Shapley kernel; efficiency constraint; no l1 selection |
| KernelSHAP-cond | conditional game | the same implementation and settings; only the row weights differ (σ = 0.1) |
| LIME | its own Gaussian samples | margin target; its statistics come from the same rows |
| Random | none | uniform scores |

With every coalition, `kernel_shap_game` equals PACE-SHAP (or PACE-SHAP-cond)
exactly, and equals shap's KernelExplainer. Both are tested.

**The two tests** use signed ranking and margin AUC, as in v3:

- **Interventional (the v3 tests):** removed features come from a random
  evaluation row (10 repetitions).
- **Conditional:** removed features come from the q-th nearest evaluation row to
  the point, measured on the features still taken from x (repetition q = 0..9).
  When none are taken, a random row is used.
- Each test is run as deletion (lower AUC = more faithful) and insertion (higher
  AUC = more faithful).

**Realism measure: the off-manifold share.**

- An input is off-manifold if its nearest training row is farther away than the
  95th percentile of the training rows' own nearest-neighbour distances
  (standardised features). By this rule about 5% of real points are off.
- For each replacement scheme, 200 of the inputs it evaluates are sampled per
  point, and the off share is averaged over points. The schemes:
  - the interventional game;
  - the conditional game;
  - PACE-flip's quantile values;
  - PACE-flip-cond's neighbour values;
  - LIME's samples, weighted by its kernel;
  - both tests;
  - the explained points themselves, as a reference.
- Shapley coalitions are drawn as the first t features of a random order, with t
  uniform on 0..d.

**Primary:**

1. **Faithfulness, like for like:** PACE-SHAP-cond minus KernelSHAP-cond on the
   **conditional** test (deletion margin AUC and insertion margin AUC). The test
   is paired over seeds, Holm across 13 datasets.
2. **Realism:** off-manifold share, conditional game minus interventional game,
   Holm across 13 datasets.

**Secondary:**

- PACE-SHAP vs KernelSHAP on the interventional test (same rows and sampler as
  the conditional pair);
- PACE-SHAP-cond vs PACE-SHAP on both tests;
- PACE-flip-cond vs PACE-flip on both tests;
- realism of PACE-flip-cond vs PACE-flip, and of the conditional vs the
  interventional test;
- on S1c and S2c, the share of attribution given to x3, with the model's own
  reliance on x3 (its share of PACE-flip's decision flips);
- ground-truth top 3 on the synthetic sets;
- time per explanation.

**Expected:**

- Each explainer does best on the test that matches its own game.
- Conditional schemes are less often off-manifold on real data with correlated
  features, and about the same on S1–S3, whose features are independent.
- On S1c and S2c, conditional explainers give x3 more credit than interventional
  ones: x3 picks up credit through its correlation with x0, even where the model
  barely uses it.
- PACE-SHAP-cond beats KernelSHAP-cond where it is exact (d ≤ 12), for the same
  reason PACE-SHAP beat KernelSHAP in v3. Where it is sampled, it may not.

**Known limits:**

- The covariance is diagonal.
- σ is fixed at Aas et al.'s 0.1 and not tuned.
- For d > 12 the conditional game shares 32 orders across rows, which is noisier
  than PACE-SHAP's 32 orders per row.
- The conditional test uses the same nearest-row idea as the conditional
  explainers, so it may favour them. This is why both tests are reported.

## B4. PACE-SHAP interaction values

**Method** (`PaceShap.interactions`): exact SHAP interaction values, from the
same v(S) table as PACE-SHAP. The off-diagonal entries are half the Shapley
interaction index. The diagonal holds the main effects, and each row sums to
the Shapley value. A pair's strength is the mean |interaction| over points.

**Run:** `python scripts/run_interactions.py --jobs 6`

- Datasets with d ≤ 12: S1, S2, S3, S1c, S2c, Shuttle, MAGIC, Page Blocks.
- Seeds 40–49, 50 test points, 200 explainer rows, interventional game.

**Primary:** on S3 (class 1 if x0 > 0.5 and x1 < 0.4, or if x2 > 0.8), the
share of seeds where (x0, x1), the AND pair, is the strongest pair. The claim
is supported if this holds in at least 8 of 10 seeds.

**Secondary (descriptive):**

- the share of pair strength among x0–x2 on the synthetic sets;
- the most frequent top pair on each dataset.

## Order and rough time

1. B4: minutes.
2. B2: 1–2 hours.
3. B1: about 2 hours.
4. B3: 8–12 hours. KernelSHAP and KernelSHAP-cond are the slowest, at 40,000
   model calls per point each (about 17–19 s per point on S1c in the smoke run,
   with three runs sharing the CPU).
