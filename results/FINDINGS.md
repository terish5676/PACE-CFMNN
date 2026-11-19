# PACE-CFMNN: findings (explanations, trust signals and repair)

Main model: CFMNN with the fixed centroid rule and search expansion (θ = 0.3, γ = 4).
Throughout, "no difference", "level" and "the same" mean that the paired test was not significant after
Holm correction, not that the values are equal.
Raw tables: [`g0/summary.md`](g0/summary.md), [`faithfulness/summary.md`](faithfulness/summary.md),
[`faithfulness_v3/summary.md`](faithfulness_v3/summary.md),
[`faithfulness_v3/timing.md`](faithfulness_v3/timing.md), [`speed/summary.md`](speed/summary.md),
[`speed_uci/summary.md`](speed_uci/summary.md), [`error_detection/summary.md`](error_detection/summary.md),
[`error_detection_confirmatory/summary.md`](error_detection_confirmatory/summary.md),
[`counterfactuals/summary.md`](counterfactuals/summary.md),
[`global_importance/summary.md`](global_importance/summary.md), the RQ1 extensions in
[`extensions_v1/`](extensions_v1/PREREGISTRATION.md) (Section 4.2) and the repair runs in
[`refinement_confirmatory/`](refinement_confirmatory/summary_v1.md),
[`refinement_v2_explore/`](refinement_v2_explore/summary.md) and
[`refinement_v2_confirmatory/`](refinement_v2_confirmatory/summary.md) (Section 5), the readable
baselines in [`readable_baselines/`](readable_baselines/summary.md) (Section 9), and repair at the θ
chosen by validation in [`repair_theta/`](repair_theta/summary.md) (Section 10).

## 1. Base model (G0)

5 seeds, the paper's protocol (stratified 80/20 split, min-max scaling fitted on training rows).

- **Parity:** our CFMNN matches a line-by-line transcription of the paper's pseudocode exactly: the
  same boxes, centroids, predictions and winning boxes on randomised tests.
- **Close to the paper:** with the paper's rules, our single model is within 2 points of the paper's
  Spark ensemble (MRCFMNN, Max_Occurrences) on 3 of its 4 datasets.

  | Dataset | CFMNN, paper's rules (ours) | Paper | Majority class |
  |---|---|---|---|
  | Thyroid | 95.04 | 96.90 | 92.57 |
  | Shuttle | 93.33 | 94.04 | 78.59 |
  | Skin | 96.88 | 97.63 | 79.25 |
  | Sepsis | 48.65 | 68.76 | 92.65 |

  The paper used an ensemble and we use one model, so small gaps are expected. Box counts can't be
  compared: the paper's are totals over an unknown number of partitions.
- **Centroid fix:** Shuttle goes from 93.3% to 99.8%. The fix changes 6.5% of predictions.
  - On Shuttle, 99.5% of training points land inside an existing box.
  - The paper's rule skips those points, so each centroid comes only from the few points that created
    or grew its box.
  - Those centroids then decide the 12.6% of test points where boxes tie.
- **Search expansion helps when there are many features:** Digits 81.7 → 95.3, Thyroid 95.0 → 97.2 and
  S1 78.9 → 88.7, with 2.5–13× fewer boxes.
  - With many features the best box is usually already at its size limit, so under the paper's rule
    most new points become single-point boxes (S1: 3,748 boxes from 4,000 points).
  - It makes no difference on Shuttle (9 features) and hurts on the 3-feature sets (Skin 97.0 → 94.7).
- **Sepsis is dropped.**
  - Only 975 distinct inputs exist, and 91.5% of rows share their input with a row of the opposite
    outcome.
  - So the best possible accuracy is 92.65%, the same as always predicting "alive".
  - Details are in [`data/README.md`](../data/README.md).

## 2. Explanations (RQ1)

**Three benchmark versions:**
- **v1** ([`faithfulness_v1/`](faithfulness_v1/summary.md), withdrawn) had two problems, which a later
  audit found:
  - **It ranked features by |attribution|.** KernelSHAP and LIME give signed values, and many of
    theirs are negative (the feature argues *against* the predicted class): 4–41% of KernelSHAP's
    values and 25–52% of LIME's on the 7 datasets of that run. Deleting those early raises the margin,
    so the test penalised them. PACE-flip's scores are never negative, so it was not affected.
  - **KernelSHAP and LIME explained a different quantity:** a pseudo-probability, not the margin the
    test measures.

  Under v1 PACE-flip looked like the most faithful explainer on all 7 datasets; that claim is
  withdrawn.
- **v2** ([`faithfulness/`](faithfulness/summary.md), the fair protocol, seeds 0–9): PACE-flip against
  KernelSHAP and LIME on 11 datasets (Section 2.1).
- **v3** ([`faithfulness_v3/`](faithfulness_v3/summary.md), pre-registered, seeds 30–39): adds
  PACE-SHAP, a stronger KernelSHAP, and separate background rows for the explainers and the tests
  (Section 2.2).

**The tests (v2 and v3):**
- 10 seeds × 50 test points (36 for Wine).
- Features are ranked by signed attribution, most supportive of the predicted class first.
- Every explainer explains the predicted class's margin:
  - KernelSHAP uses 20 real background rows and 200 coalition samples;
  - LIME's output is its surrogate's contributions w_i·(x_i − mean_i)/std_i.
- **Deletion** replaces the top features first (lower AUC = more faithful). **Insertion** starts
  from a real training row and puts the top features back (higher AUC = more faithful). Both copy
  values from one random real row, 10 repetitions.
- Comparisons are paired over the 10 seeds (two-sided Wilcoxon), Holm-corrected across the 11
  datasets.

### 2.1 PACE-flip (v2, seeds 0–9)

Margin AUC, deletion / insertion (mean over 10 seeds; the best of the four explainers in bold).
Occlusion gives PACE-flip's numbers:

| Dataset | PACE-flip | PACE-multi | KernelSHAP | LIME | Random |
|---|---|---|---|---|---|
| S1 | −0.088 / 0.624 | −0.015 / 0.546 | **−0.128** / **0.639** | −0.115 / 0.612 | 0.286 / 0.279 |
| S2 | −0.117 / 0.575 | −0.026 / 0.480 | **−0.159** / **0.597** | −0.131 / 0.572 | 0.248 / 0.248 |
| S3 | 0.005 / 0.619 | 0.082 / 0.547 | **−0.035** / **0.653** | 0.021 / 0.586 | 0.325 / 0.337 |
| Wine | −0.045 / 0.505 | −0.013 / 0.466 | **−0.068** / **0.548** | −0.035 / 0.511 | 0.249 / 0.248 |
| Breast Cancer | **0.149** / **0.708** | 0.159 / 0.689 | 0.193 / 0.652 | 0.160 / 0.700 | 0.424 / 0.432 |
| Thyroid | **0.549** / **0.917** | 0.578 / 0.901 | 0.577 / 0.891 | 0.580 / 0.881 | 0.742 / 0.737 |
| Shuttle | 0.344 / **0.724** | 0.344 / 0.683 | **0.331** / 0.712 | 0.346 / 0.693 | 0.507 / 0.520 |
| MAGIC | 0.029 / 0.236 | 0.067 / 0.179 | **−0.021** / **0.268** | −0.001 / 0.224 | 0.138 / 0.102 |
| Letter | −0.656 / −0.056 | −0.562 / −0.256 | **−0.690** / **−0.025** | −0.663 / −0.114 | −0.399 / −0.413 |
| Landsat | −0.260 / **0.143** | −0.213 / 0.082 | −0.233 / 0.119 | **−0.296** / 0.142 | −0.046 / −0.045 |
| Page Blocks | 0.064 / 0.267 | 0.072 / 0.258 | **0.055** / **0.270** | 0.068 / 0.260 | 0.178 / 0.135 |

- **KernelSHAP vs PACE-flip:**
  - KernelSHAP is more faithful by both tests on 6 datasets: S1–S3, Wine, MAGIC and Letter (at least
    9 of 10 seeds; Holm p = 0.021).
  - PACE-flip is more faithful by both tests on Breast Cancer and Thyroid, and by deletion on Landsat
    (its insertion gain there, +0.024, is not significant after Holm).
  - No difference on Shuttle and Page Blocks.
- **LIME vs PACE-flip:** PACE-flip is more faithful by deletion on Breast Cancer and Thyroid, and by
  insertion on S1, S3, Thyroid, Shuttle and Letter. LIME is more faithful by deletion on S1, MAGIC and
  Landsat. The other comparisons show no significant difference.
- **PACE-flip vs PACE-multi:** PACE-flip is more faithful by insertion on 10 of 11 datasets (not Page
  Blocks), and by deletion on 8 (no difference on Breast Cancer, Shuttle and Page Blocks).
- **The by-construction caveat:** PACE-flip scores single-feature replacements with the same margin
  and the same replacement distribution the tests use, so it is close to the greedy first step of
  the deletion test. Even so, KernelSHAP matches or beats it on 8 of 11 datasets.
- **Why it falls short:** PACE-flip ranks features by what changing *one* feature alone does, while
  the tests remove features *together*. The Shapley value of the margin, which KernelSHAP estimates,
  accounts for that. This is what PACE-SHAP computes exactly (2.2).

### 2.2 PACE-SHAP (v3, pre-registered, seeds 30–39)

**What it computes:** the Shapley value of the predicted class's margin (the quantity KernelSHAP
estimates), computed from CFMNN's own per-feature membership terms:
- a box's membership when some features come from the point and the rest from a background row is
  the background row's membership plus a sum of per-feature differences, so every coalition is
  evaluated without re-running the model;
- exact (all 2^d coalitions) when d ≤ 12; otherwise antithetic permutation sampling, 32 orders per
  background row;
- averaged over 200 explainer-background rows (71 for Wine);
- an exact rule skips boxes that can never be their class's maximum.

It is model-specific, like TreeSHAP for trees: nothing is fitted to a dataset or to a test result,
and the CFMNN model is unchanged. Unit tests check it against brute-force Shapley values (to 1e-12)
and against KernelSHAP with every coalition enumerated on the same rows.

**Safeguards** (written in [`PREREGISTRATION.md`](faithfulness_v3/PREREGISTRATION.md) and pushed
before the run):
- new seeds, 30–39;
- the deletion and insertion tests copy values from one set of 200 training rows, and every explainer
  uses a second, disjoint set, so no explainer is graded on rows it saw;
- PACE-SHAP's settings were chosen beforehand from approximation quality only
  ([`settings.md`](faithfulness_v3/settings.md), seed 99);
- two extra comparisons:
  - **PACE-SHAP-20:** KernelSHAP's own 20 rows, so any difference from KernelSHAP is KernelSHAP's
    estimation error;
  - **KernelSHAP-strong:** 2,048 samples and no feature selection, so it enumerates every coalition
    and is exact when d ≤ 11.

Margin AUC, deletion / insertion (mean over 10 seeds; the best explainer in bold):

| Dataset | PACE-SHAP | PACE-SHAP-20 | PACE-flip | KernelSHAP | KernelSHAP-strong | LIME | Random |
|---|---|---|---|---|---|---|---|
| S1 | **−0.139** / **0.645** | −0.131 / 0.636 | −0.089 / 0.617 | −0.133 / 0.635 | −0.131 / 0.636 | −0.109 / 0.609 | 0.271 / 0.296 |
| S2 | **−0.132** / **0.609** | −0.128 / 0.593 | −0.076 / 0.561 | −0.125 / 0.592 | −0.128 / 0.593 | −0.095 / 0.565 | 0.256 / 0.257 |
| S3 | **−0.055** / **0.639** | −0.044 / 0.627 | 0.006 / 0.595 | −0.041 / 0.626 | −0.044 / 0.627 | 0.012 / 0.569 | 0.318 / 0.314 |
| Wine | **−0.076** / **0.533** | −0.070 / 0.526 | −0.055 / 0.489 | −0.068 / 0.524 | −0.070 / 0.526 | −0.039 / 0.492 | 0.236 / 0.239 |
| Breast Cancer | **0.111** / **0.699** | 0.122 / 0.687 | 0.137 / 0.685 | 0.174 / 0.634 | 0.122 / 0.687 | 0.141 / 0.680 | 0.398 / 0.414 |
| Thyroid | 0.549 / 0.915 | 0.562 / 0.902 | **0.544** / **0.918** | 0.575 / 0.892 | 0.562 / 0.903 | 0.571 / 0.874 | 0.739 / 0.736 |
| Shuttle | 0.370 / 0.744 | **0.365** / 0.747 | 0.374 / **0.750** | **0.365** / 0.747 | 0.366 / 0.747 | 0.381 / 0.721 | 0.552 / 0.536 |
| MAGIC | **−0.052** / **0.290** | −0.038 / 0.279 | 0.015 / 0.234 | −0.035 / 0.272 | −0.038 / 0.279 | −0.011 / 0.235 | 0.140 / 0.093 |
| Letter | **−0.690** / **−0.000** | −0.684 / −0.021 | −0.644 / −0.072 | −0.678 / −0.038 | −0.684 / −0.022 | −0.656 / −0.136 | −0.381 / −0.423 |
| Landsat | **−0.329** / **0.231** | −0.309 / 0.206 | −0.255 / 0.134 | −0.217 / 0.077 | −0.303 / 0.203 | −0.295 / 0.126 | −0.050 / −0.057 |
| Page Blocks | 0.050 / 0.204 | **0.042** / **0.212** | 0.051 / 0.202 | 0.046 / 0.205 | 0.043 / **0.212** | 0.049 / 0.196 | 0.128 / 0.095 |

Number of datasets where the first method is significantly better / not different / significantly
worse (Holm across 11, α = 0.05):

| Comparison | Deletion | Insertion |
|---|---|---|
| PACE-SHAP vs KernelSHAP (H1, primary; H2) | 6 / 5 / 0 | 7 / 4 / 0 |
| PACE-SHAP-20 vs KernelSHAP (H3) | 4 / 7 / 0 | 6 / 5 / 0 |
| PACE-SHAP vs KernelSHAP-strong (H4) | 3 / 8 / 0 | 4 / 7 / 0 |
| PACE-SHAP vs LIME | 9 / 2 / 0 | 10 / 1 / 0 |
| PACE-SHAP vs PACE-flip | 8 / 3 / 0 | 8 / 3 / 0 |

- **H1 (primary), deletion vs KernelSHAP:** PACE-SHAP is more faithful on S3, Wine, Breast Cancer,
  Thyroid, Letter and Landsat.
  - On S1, S2 and MAGIC its mean is also lower, but not significantly after Holm (raw p 0.027–0.049).
  - On Shuttle and Page Blocks there is no difference (KernelSHAP's mean is 0.004–0.005 lower).
- **H2, insertion vs KernelSHAP:** more faithful on S1, S2, Wine, Breast Cancer, Thyroid, Letter and
  Landsat. Higher but not significant on S3 and MAGIC (Holm p 0.078 and 0.082); no difference on
  Shuttle and Page Blocks.
- **H4, vs KernelSHAP-strong:** more faithful by deletion on Breast Cancer, Thyroid and Landsat, and
  by insertion on S1, S3, Breast Cancer and Letter. Never worse.
- **vs PACE-flip:** more faithful by both tests on 8 of 11 datasets. No difference on Thyroid,
  Shuttle and Page Blocks; on Thyroid PACE-flip's mean is slightly better (deletion 0.544 vs 0.549).
- **vs LIME:** more faithful by deletion on 9 datasets and by insertion on 10. No difference on Page
  Blocks (and on Shuttle by deletion).
- **Best mean:** PACE-SHAP has the best mean by both tests on 8 of 11 datasets. On Thyroid, Shuttle
  and Page Blocks another method is ahead by at most 0.009, and none of those differences is
  significant.

**Where the gain over KernelSHAP comes from:**
- **KernelSHAP's estimation error (H3).** On the same 20 rows, the exact values beat KernelSHAP's
  estimates on Breast Cancer, Thyroid, Letter and Landsat by both tests, and on MAGIC and Page Blocks
  by insertion. Those four datasets have 16–36 features, where 200 samples cover a small part of the
  coalitions.
- **shap's default feature selection.** shap 0.54 keeps at most 10 features
  (`l1_reg="num_features(10)"`) and gives every other feature exactly zero. On Breast Cancer (30
  features) and Landsat (36), most features are then tied at zero, and their deletion order is
  arbitrary. This is most of KernelSHAP's large loss there: PACE-SHAP's deletion gain is 0.063 and
  0.112 over KernelSHAP, but 0.011 and 0.026 over KernelSHAP-strong.
- **More background rows.** When d ≤ 11, KernelSHAP-strong and PACE-SHAP-20 compute the same exact
  values on the same 20 rows (checked below). PACE-SHAP's remaining gain over both comes from
  averaging over 200 rows: the settings study found that 20-row values differ from 200-row values by
  17–76%. KernelSHAP would give the same values with the same 200 rows and every coalition, but its
  cost grows in proportion to the number of rows.

**The pre-registered bug check:** PACE-SHAP-20 and KernelSHAP-strong must agree where both are exact
(d ≤ 11).
- Their AUCs agree to 4 decimals in every seed on S1–S3 and MAGIC.
- On Shuttle (2 seeds) and Page Blocks (1 seed) they differ by up to 0.005. Recomputing those seeds
  showed that the attributions agree to 3e-16, and only features with equal values (mostly zeros) are
  ordered differently. So this is a tie-break, not a bug.

**Stability** (top-3 overlap when the explainer is re-run with another seed):
- PACE-SHAP: 1.00 where it is exact (S1–S3, Shuttle, MAGIC, Page Blocks), 0.94–0.99 where it samples
  (Wine, Breast Cancer, Thyroid, Letter, Landsat).
- KernelSHAP 0.15–0.73; LIME 0.52–0.90; PACE-flip and the other PACE methods 1.00.
- PACE-SHAP-20 (0.48–0.74) is about as unstable as KernelSHAP when d ≤ 11. There, most of
  KernelSHAP's instability comes from drawing a new set of 20 background rows for each seed; 200 rows
  remove it.

**Speed (pre-registered timing run, measured alone).** Each method explained the same batch of test
points (50, or 36 on Wine) once, with nothing else running, on seeds 30–32
([`faithfulness_v3/timing.md`](faithfulness_v3/timing.md)). Milliseconds per explanation, mean over
the 3 seeds:

| Dataset | PACE-SHAP | PACE-SHAP-20 | KernelSHAP | KernelSHAP-strong | LIME | PACE-flip | PACE-SHAP time ÷ KernelSHAP time |
|---|---|---|---|---|---|---|---|
| S1 | 2,631 | 271 | 1,177 | 5,605 | 272 | 1.15 | 2.2 |
| S2 | 2,336 | 227 | 1,005 | 5,069 | 280 | 0.87 | 2.3 |
| S3 | 2,245 | 210 | 999 | 5,042 | 249 | 0.84 | 2.2 |
| Wine | 14.5 | 4.70 | 34.0 | 332 | 11.3 | 0.10 | 0.43 |
| Breast Cancer | 142 | 9.32 | 72.2 | 757 | 21.6 | 0.22 | 2.0 |
| Thyroid | 104 | 7.04 | 66.4 | 627 | 19.2 | 0.20 | 1.6 |
| Shuttle | 12.6 | 1.19 | 17.3 | 36.1 | 7.29 | 0.04 | 0.73 |
| MAGIC | 316 | 22.9 | 206 | 1,005 | 54.1 | 0.60 | 1.5 |
| Letter | 920 | 81.4 | 756 | 7,603 | 188 | 1.38 | 1.2 |
| Landsat | 1,092 | 49.2 | 359 | 3,648 | 94.2 | 0.74 | 3.0 |
| Page Blocks | 64.9 | 3.61 | 27.4 | 119 | 10.4 | 0.09 | 2.4 |

- **PACE-SHAP (200 rows) vs KernelSHAP:** PACE-SHAP took 1.2–3.0× as long on 9 of 11 datasets. On
  S1–S3 it took 2.2–2.3×, because exactness there means 1,024 coalitions × 200 rows. It was faster
  on Wine (2.4×) and Shuttle (1.4×). Every dataset went the same way in all 3 seeds.
- **PACE-SHAP-20 vs KernelSHAP:** PACE-SHAP-20 was 4.3–14× faster on every dataset and seed.
- **KernelSHAP-strong:** the slowest Shapley method on every dataset. It took 1.8–23× as long as
  PACE-SHAP and 21–93× as long as PACE-SHAP-20.
- **PACE-flip:** 0.04–1.4 ms per explanation, 311–1,183× faster than KernelSHAP and 90–322× faster
  than LIME. This is close to the v2 timing below.
- **Consistency with earlier estimates:** these confirm the rough times from the main run (6 runs
  sharing the CPU, kept in `runs.csv`). A laptop sleep paused 6 Thyroid jobs there, so those
  KernelSHAP-strong times are too high; no faithfulness number is affected. The pre-registration's
  seed-99 estimate for S1–S3 (2.5–3 s per point) was slightly high: the measured time is 2.2–2.6 s.

PACE-flip's timings (fair-protocol explainers, measured alone, 3 seeds; batch of 50 points, 36 for
Wine; [`faithfulness/timing.md`](faithfulness/timing.md)):

| Dataset | PACE-multi | PACE-flip | Occlusion | PACE-flip vs occlusion | KernelSHAP | LIME |
|---|---|---|---|---|---|---|
| S1 | 0.47 | 0.89 | 78.6 | 88× | 1,068 | 265 |
| S2 | 0.47 | 0.89 | 84.4 | 95× | 1,049 | 281 |
| S3 | 0.47 | 0.90 | 84.0 | 93× | 1,031 | 257 |
| Wine | 0.03 | 0.11 | 5.6 | 49× | 33 | 12 |
| Breast Cancer | 0.05 | 0.25 | 22.8 | 91× | 84 | 24 |
| Thyroid | 0.04 | 0.23 | 14.7 | 65× | 78 | 23 |
| Shuttle | 0.02 | 0.05 | 3.7 | 80× | 18 | 8 |
| MAGIC | 0.11 | 0.64 | 18.4 | 29× | 234 | 57 |
| Letter | 0.31 | 1.37 | 103 | 76× | 794 | 196 |
| Landsat | 0.17 | 0.82 | 116 | 141× | 384 | 108 |
| Page Blocks | 0.02 | 0.11 | 4.0 | 38× | 34 | 13 |

- **PACE-flip vs the others:** 29–141× faster than occlusion (25–134× when occlusion runs in one
  batch; see §8.6), 89–316× faster than LIME and 292–1,194× faster than KernelSHAP.
  - On S1–S3, KernelSHAP is about twice as slow as in v1, because it now uses 20 real background rows
    instead of 10 k-means centres.
- **PACE-multi vs PACE-flip:** PACE-multi is still 1.9–6.0× faster in batches. With one point per
  call, PACE-flip is faster ([`speed/summary.md`](speed/summary.md)): for example S1 1.1 vs 1.2 ms, and
  Wine 0.36 vs 0.62 ms.
- **Fair timing:** CFMNN processes points in cache-sized chunks (`chunk_elems = 65,536`; results
  unchanged), so every method that calls the model benefits equally.
- **Timing noise:** times vary between runs on this machine. Compare methods measured together.

KernelSHAP is now repeatable.
- **Why it wasn't:** shap 0.54 samples from numpy's global random state, so the seed passed to it did
  not fix its sampling.
- **The fix:** `kernel_shap` now seeds that state for the call and restores it afterwards (tested).
  Re-running a benchmark row now gives the same numbers.
- **Before the fix:** two full runs differed by up to 0.02 AUC, and no conclusion changed.

**Ground truth** (synthetic sets, v3; ties broken at random). The share of a method's top 3 that are
truly relevant features:
- LIME 0.89–0.92;
- PACE-SHAP 0.79–0.85;
- KernelSHAP 0.72–0.81;
- PACE-flip 0.49–0.55.

The model's own single-feature decision flips come only 42–47% from the relevant features. On 70–83%
of points, replacing some irrelevant feature can flip the decision.
- So PACE-flip reports what single-feature changes do to this model.
- The Shapley methods and LIME spread credit over combinations of features and rank the relevant
  features higher. PACE-SHAP ranks them higher than KernelSHAP, but LIME ranks them highest.
- Ground truth is about the data, not the model: LIME recovers the relevant features best while being
  less faithful to the model than PACE-SHAP on all three synthetic sets.
- The earlier reading "LIME and SHAP describe the data rather than the model" is withdrawn. It rested
  on an unsourced number (0.49–0.59 "decision sensitivity"). The methods measure different things.

## 3. Trust signals (RQ2)

Question: do PD (prediction disagreement), EI (explanation instability) and U (membership uncertainty)
rank the model's errors first, and does EI add anything beyond PD + U?

There are two runs:
- **Exploratory** (seeds 0–9, Section 3.1). U\* was found here, so this run cannot confirm it.
- **Confirmatory** (seeds 10–19, Section 3.2). Its rules were pushed to GitHub before it ran. It uses
  fresh seeds, 4 more real datasets and 3 standard baselines. Cite these results.

### 3.1 Exploratory run (seeds 0–9)

- **Protocol:** 10 seeds; the blueprint's 50/20/10/10/10 role split.
  - The model trains on the 50% training rows.
  - PRS ranks are fitted on the 10% inner-validation rows.
  - Errors are scored on all the other rows (40%). The signals never read labels.
  - Perturbations: κ = 0.05, K = 50.
- **Eligible datasets:** a dataset needs at least 50 errors in every seed. S1, S2, S3, three S2
  variants (90/10 imbalance, 5% label noise, shifted test set), Thyroid and Skin qualify. Wine, Breast
  Cancer, Digits and Shuttle are excluded.

AUPRC: higher is better. Read it against the base error rate (what a random ranking scores).
Mean over 10 seeds; 95% intervals are in `error_detection/summary.md`.

| Dataset | Base rate | Margin | PD | EI | U | PD + U | PRS | U* | PD + U* | PRS* |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.118 | 0.388 | 0.395 | 0.133 | 0.283 | 0.359 | 0.274 | 0.388 | **0.424** | 0.329 |
| S2 | 0.204 | 0.398 | 0.371 | 0.220 | 0.325 | 0.375 | 0.331 | 0.398 | **0.416** | 0.376 |
| S3 | 0.139 | 0.385 | 0.379 | 0.157 | 0.291 | 0.365 | 0.297 | 0.385 | **0.412** | 0.352 |
| S2, imbalanced | 0.096 | 0.370 | 0.300 | 0.129 | 0.265 | 0.317 | 0.239 | 0.370 | **0.388** | 0.291 |
| S2, label noise | 0.258 | 0.398 | 0.370 | 0.273 | 0.337 | 0.371 | 0.347 | 0.398 | **0.409** | 0.383 |
| S2, shifted | 0.204 | 0.391 | 0.363 | 0.224 | 0.317 | 0.368 | 0.326 | 0.391 | **0.409** | 0.371 |
| Thyroid | 0.031 | 0.213 | 0.234 | 0.220 | 0.071 | 0.098 | 0.087 | **0.323** | 0.210 | 0.208 |
| Skin | 0.041 | 0.199 | 0.365 | 0.159 | 0.097 | 0.187 | 0.186 | 0.376 | **0.416** | 0.378 |

Margin = 1 − (s′₁ − s′₂). EI alone is scored only where it exists (coverage was 1.00 everywhere).
Starred methods use **U\***, a corrected U added after gate G1. It is reported as a separate variant,
because the blueprint's U was the planned analysis.

- **EI adds no value (primary endpoint).**
  - PRS has a lower AUPRC than PD + U on all 6 synthetic sets: 10 of 10 seeds each, Holm-corrected
    p = 0.016. There is no difference on Thyroid or Skin.
  - The same holds with U\*.
  - Following the blueprint (§13), EI becomes an audit-only diagnostic, and the recommended score is the
    simpler PD + U\*.
- **EI alone depends on box overlap.**
  - It is near random where boxes of different classes do not overlap: AUROC 0.52–0.56 on the
    synthetic sets.
  - It is strong where they do: Thyroid 0.89, Skin 0.87.
  - There it largely repeats PD (Spearman ρ = 0.47 and 0.41), so it adds nothing in the combination.
- **The blueprint's U fails gate G1.**
  - On Thyroid its AUROC is 0.32, worse than random.
  - The cause: when the top two classes tie (both boxes contain the point), U grades the tie by
    centroid distance on a scale that puts ties *below* most untied points.
  - Yet ties are where CFMNN errs most. On Thyroid 4.9% of points tie, with a 33% error rate. On Skin
    21% tie, with a 20% error rate, which covers nearly all of Skin's errors.
  - **U\*** ranks every tie first and grades ties by the same centroid ratio.
- **PD + U\* is the best of these label-free scores.** This run had no standard baselines; see 3.2.
  - It has the best AUPRC on 7 of 8 eligible datasets.
  - It beats the plain margin on all 6 synthetic sets and on Skin (10 of 10 seeds, Holm p = 0.016). The
    largest gain is on Skin: 0.416 vs 0.199.
  - On Thyroid it ties with the margin. There U\* alone has the best AUPRC (0.323) and PD alone the best
    AUROC (0.89).
- **PD alone** has the highest precision when the top 10% are flagged, on every eligible dataset except
  Skin.
- **Sensitivity** (S2 and Thyroid, 5 seeds, κ ∈ {0.02, 0.1}, K ∈ {20, 100}): the conclusions do not
  change.
  - EI's AUROC stays at 0.51–0.54 on S2 and 0.87–0.89 on Thyroid.
  - The blueprint's U stays at 0.32 on Thyroid.
  - A larger κ helps PD on S2 and hurts it on Thyroid.
- **Perturbation plausibility (gate G1):**
  - On 10 of 12 datasets, 93–96% of perturbations are as close to the training data as 95% of real
    points.
  - Skin: 67–71%. Its values are whole numbers and 78% of real points duplicate a training row.
  - Shuttle: 0%. Real Shuttle points lie extremely close to training rows (95th percentile distance
    0.0006), while the nudges move about 0.014. Shuttle is not in RQ2 anyway.
- **Cost:** computing PD, EI and U takes about 0.3–13 ms per point. It grows with the number of boxes
  (S1–S3 have about 1,000 boxes). These times were measured while other runs shared the machine.

### 3.2 Confirmatory run (seeds 10–19, pre-registered)

- **Rules:** [`PREREGISTRATION.md`](error_detection_confirmatory/PREREGISTRATION.md), committed (fd271ed)
  and pushed before the run started.
  - Nothing was changed after the run started, and no bug fix was needed.
  - Two rows (Page Blocks and Thyroid, seed 10) were re-run and reproduced exactly.
  - Two separate checks recomputed the summary's means, seed counts and p-values from `runs.csv`, and
    all of them match.
- **What is new compared with 3.1:**
  - Fresh seeds 10–19.
  - 4 more real datasets: MAGIC Gamma Telescope, Letter Recognition, Landsat Satellite and Page Blocks.
  - Datasets under 2,000 rows use stratified 5-fold cross-validation (blueprint §9.1). That gives
    Digits enough errors to test.
  - At most 20,000 evaluation rows per seed.
  - The 6 synthetic sets share one generator, so they count as one unit: per seed, each score's mean
    over the 6 sets.
  - Three standard baselines. Unlike PD, EI and U, all three read the training labels:
    - **Trust Score** (Jiang et al., NeurIPS 2018);
    - **Order ensemble:** the share of 5 CFMNNs, trained on the same rows in other random orders, that
      disagree with the main model;
    - **kNN disagreement:** the share of the 10 nearest training rows with another label.
- **Tested units (8):** the synthetic family, Digits, Thyroid, Skin, MAGIC, Letter, Landsat and Page
  Blocks.
  - Excluded for too few errors in some seed: Wine (3), Breast Cancer (20) and Shuttle (39).
- **Test:** paired two-sided Wilcoxon over the 10 seeds, Holm-corrected across the 8 units.
  - With 10 seeds the smallest possible p is 0.002. After Holm that becomes 0.016 for the 8-unit
    comparisons and 0.012 for U\* vs U, which has 6 units.
  - So these are the strongest results this design can give.

AUPRC (mean over 10 seeds; 95% intervals are in
[`error_detection_confirmatory/summary.md`](error_detection_confirmatory/summary.md)):

| Unit | Base rate | Margin | U | U\* | PD + U\* | PRS\* | Trust Score | Order ensemble | kNN disagreement |
|---|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.171 | 0.389 | 0.307 | 0.389 | 0.409 | 0.349 | 0.551 | 0.652 | **0.694** |
| Digits | 0.044 | 0.386 | 0.384 | 0.394 | 0.397 | 0.189 | **0.857** | 0.601 | 0.825 |
| Thyroid | 0.033 | 0.261 | 0.067 | 0.344 | 0.229 | 0.227 | 0.114 | **0.426** | 0.094 |
| Skin | 0.046 | 0.221 | 0.080 | 0.392 | 0.423 | 0.383 | 0.976 | 0.751 | **0.991** |
| MAGIC | 0.276 | 0.328 | 0.274 | 0.417 | 0.439 | 0.403 | 0.694 | 0.586 | **0.722** |
| Letter | 0.259 | 0.413 | 0.265 | 0.571 | 0.586 | 0.557 | 0.968 | 0.785 | **0.970** |
| Landsat | 0.154 | 0.316 | 0.283 | 0.373 | 0.400 | 0.362 | **0.737** | 0.539 | 0.727 |
| Page Blocks | 0.141 | 0.189 | 0.157 | 0.367 | 0.445 | 0.367 | 0.852 | 0.624 | **0.862** |

**Primary endpoints.**
- **P1, PD + U\* beats the plain margin: supported on 6 of 8 units.**
  - It is higher in 10 of 10 seeds, with Holm p = 0.016, on the synthetic family (+0.020), Skin
    (+0.202), MAGIC (+0.111), Letter (+0.173), Landsat (+0.084) and Page Blocks (+0.256).
  - Not on Digits: +0.012, higher in 9 of 10 seeds, Holm p = 0.074.
  - Not on Thyroid: −0.031, higher in only 2 of 10 seeds, p = 0.11. As in 3.1, adding PD hurts on
    Thyroid. There U\* alone (0.344) beats the margin (0.261); this is descriptive, not tested.
  - The gain is large where ties are common (21–72% of points on Skin, MAGIC, Letter, Landsat and Page
    Blocks). Outside ties U\* ranks points exactly like the margin. The synthetic sets have almost no
    ties, so their small gain comes from PD.
- **P2, EI adds no value: supported.**
  - PRS\* is never significantly higher than PD + U\*.
  - It is significantly lower on 6 of 8 units (higher in 0 of 10 seeds, Holm p = 0.016).
  - There is no significant difference on Thyroid (−0.002) or Skin (−0.040, lower in 9 of 10 seeds,
    Holm p = 0.13).
  - Two descriptive exceptions lie outside this endpoint:
    - With the planned U, PRS beats PD + U on Letter (0.546 vs 0.397), because U is near random there.
    - By AUROC on Thyroid, PRS\* beats PD + U\* (0.881 vs 0.777), and EI alone has the best AUROC of
      all scores (0.891).

**Secondary endpoints.**
- **U\* beats the blueprint's U on all 6 real datasets with at least 1% tied points** (Thyroid, Skin,
  MAGIC, Letter, Landsat and Page Blocks). Digits, with 0.4% ties, is outside this test. It is higher in 10 of 10 seeds on each, with Holm p = 0.012 (6 units),
  and the gain is 0.09–0.31 AUPRC.
  - The blueprint's U is no better than random on MAGIC (0.274 against a base rate of 0.276) and Letter
    (0.265 against 0.259), and not significantly better on Page Blocks (0.157 against a random
    ranking's 0.140, Holm p = 0.17).
  - So the fix found in 3.1 holds on fresh seeds and new datasets.
- **The standard baselines are far stronger than PD + U\*. This is the main negative result.**
  - Trust Score and kNN disagreement beat PD + U\* on 7 of 8 units, by 0.14–0.57 AUPRC. Each is higher
    in 10 of 10 seeds, Holm p = 0.016.
  - The order ensemble beats PD + U\* on all 8 units, by 0.14–0.33 AUPRC (Holm p = 0.016), including
    Thyroid (0.426 vs 0.229).
  - **Thyroid is the exception for the other two.** There PD + U\* beats Trust Score (+0.115, 9 of 10
    seeds) and kNN disagreement (+0.135, 10 of 10 seeds), both Holm p = 0.016.
- **One likely reason (a post-hoc check, not pre-registered):** a plain 10-nearest-neighbour
  classifier is more accurate than the CFMNN on 12 of the 13 tested datasets
  ([`knn_reference.md`](error_detection_confirmatory/knn_reference.md); same splits, seeds 10–19).
  - Examples: Letter 92.2% vs 74.1%, Page Blocks 94.7% vs 85.9%, MAGIC 83.0% vs 72.4% and Skin 99.9%
    vs 95.4%.
  - So kNN disagreement and Trust Score mostly flag points where a more accurate model disagrees with
    the CFMNN. On Letter, 10-NN predicts another class for 92% of the CFMNN's errors.
  - Thyroid is the one dataset where the CFMNN is more accurate (96.7% vs 93.8%). There 10-NN
    disagrees with only 21% of the CFMNN's errors, and PD + U\* beats both of these baselines.
  - PD and U\* see only the CFMNN, so they cannot know what a better model would predict.
  - This does not explain the order ensemble. It is made of CFMNNs, yet it beats PD + U\* on all 8
    units, including Thyroid.
- **Cost:** the PACE signals are not cheaper either. Computing PD, EI and U took longer than all three
  baselines together on 14 of 16 datasets; Letter, for example, took 87 s against 9 s per seed. The
  exceptions are Skin (20 s vs 23 s) and Wine, where both take under 0.05 s.
  - The signal time covers PD, EI and U on the inner and evaluation rows; the baseline time covers the
    evaluation rows only. PD + U\* does not need EI, so on its own it would cost less than shown.
- **AUROC and precision at the top 10%** (descriptive) show the same broad pattern, with these
  differences. See the summary file.
  - All three baselines are ahead of PD + U\* everywhere except Thyroid, where only the order ensemble
    is. The one other exception: by AUROC the order ensemble is below PD + U\* on Digits (0.900 vs
    0.942).
  - On Thyroid, PD + U\* is slightly above the margin by both metrics (AUROC 0.777 vs 0.757; precision
    0.220 vs 0.215), unlike AUPRC.
  - On Digits it is slightly below the margin by both (AUROC 0.942 vs 0.943; precision 0.320 vs 0.323).
  - On Thyroid, PRS\* and EI alone lead PD + U\* by AUROC (see P2).
- **What still favours PACE's signals:** they need no labels and no retraining. PD uses the training
  features but not their labels, and U\* uses the model alone. They come from the same boxes that the
  PACE explanations use. But as error detectors they beat the model's own margin on 6 of 8 units.
  They beat Trust Score and kNN disagreement only on Thyroid, and never the order ensemble.

### 3.3 Why the blueprint's signals could not compete (review after 3.2)

- **EI could hardly add information for CFMNN.**
  - Inside the winning box A, the contrastive vector is (2/d)·min(δ_i, 0.25), where δ_i is the
    point's distance outside the runner-up's best box on feature i. That is just the capped distance
    to that box.
  - Under small nudges it changes mainly near the same box faces that drive PD, plus two artifacts:
    - the top-q feature set is padded by feature index when fewer than q terms are non-zero;
    - the best box can switch with no change in the prediction.
  - Result in 3.2:
    - near chance where there are no ties (AUROC 0.52–0.57 on the synthetic sets);
    - on most tie-heavy datasets it tracks PD (ρ 0.42–0.77 on Landsat, Letter, Page Blocks, Skin and
      Thyroid; 0.29 on MAGIC);
    - it beats PD alone on only 4 of 16 datasets by AUROC (2 of 16 by AUPRC).
- **The plain margin already is PACE's exact distance to the decision boundary.** For an untied point
  inside A, s′_A − s′_B = (4/d)·Σ min(δ_i, 0.25). So a "counterfactual distance" signal would add
  little. U\*'s gain comes from ranking ties first.
- **The winning baselines bring information a single CFMNN does not have:** other training orders, or
  labelled neighbours, in effect a more accurate classifier (10-NN beats CFMNN on 12 of 13 datasets).
- **What follows:** CFMNN's own signals should be judged on whether they *add* to the data-based ones.
  That is RQ2 v2: `results/error_detection_v2/` (pre-registered), designed on seeds 0–9 in
  `results/error_detection_v2_design/`.

### 3.4 RQ2 v2: CFMNN's own uncertainty added to data-based trust (seeds 20–29, pre-registered)

- **Rules:** [`error_detection_v2/PREREGISTRATION.md`](error_detection_v2/PREREGISTRATION.md).
  - The draft, with the design rule for PD, was pushed before the design run's results were seen
    (0e2c940).
  - The final version was pushed before this run (38eb8e3).
  - Design run: seeds 0–9, [`error_detection_v2_design/`](error_detection_v2_design/summary.md).
- **Setup:** same datasets, splits, eligibility and 8 units as 3.2.
- **Signals:**
  - CFMNN-native: U\*, margin, tie, box evidence (distance to the nearest training row the winning
    box holds), the box's member count, the order ensemble.
  - Data-based: kNN disagreement, Trust Score.
- **Combiner:** a logistic regression over the signals, fitted on the inner rows **with their labels**
  (a labelled calibration set, unlike PRS). PD is not used: in the design run it never added 0.01.

AUPRC (mean over 10 seeds):

| Unit | Base | kNN | Trust Score | Order ensemble | LR: kNN + TS | LR: kNN + TS + order ens. | LR: native | LR: all |
|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.170 | 0.697 | 0.558 | 0.652 | 0.733 | **0.774** | 0.706 | 0.771 |
| Digits | 0.046 | 0.834 | **0.877** | 0.606 | 0.873 | 0.853 | 0.698 | 0.842 |
| Thyroid | 0.030 | 0.100 | 0.109 | 0.429 | 0.118 | 0.458 | 0.494 | **0.502** |
| Skin | 0.048 | 0.996 | 0.994 | 0.828 | 0.998 | **0.998** | 0.949 | 0.998 |
| MAGIC | 0.262 | 0.696 | 0.671 | 0.545 | 0.738 | 0.749 | 0.607 | **0.749** |
| Letter | 0.262 | 0.969 | 0.969 | 0.790 | 0.982 | 0.983 | 0.908 | **0.984** |
| Landsat | 0.154 | 0.729 | 0.736 | 0.535 | 0.777 | **0.787** | 0.619 | 0.783 |
| Page Blocks | 0.103 | 0.816 | 0.807 | 0.474 | 0.866 | **0.870** | 0.634 | 0.858 |

The tests below are paired Wilcoxon over 10 seeds, Holm across the 8 units.
- **Q1 (primary), adding the order ensemble to kNN + Trust Score: supported on 5 of 8 units.**
  - The synthetic family (+0.041), Thyroid (+0.341), MAGIC (+0.010), Letter (+0.001) and Landsat
    (+0.010): 10 of 10 seeds each, Holm p = 0.016.
  - No difference on Skin (+0.000) or Page Blocks (+0.004, Holm p = 0.21).
  - **Lower on Digits** (−0.020, Holm p = 0.029).
- **Q1b, the geometry signals add little beyond that** (U\*, margin, tie, box evidence, box count): significantly only on Letter, by 0.0015 (Section 8.5).
  The only significant difference is +0.001 on Letter.
- **Q1c, all signals vs kNN + Trust Score (the draft's primary):** supported on 4 of 8 units: the
  synthetic family, Thyroid, MAGIC and Letter.
- **Q2, the strongest detector overall:** LR(kNN + TS + order ens.).
  - It beats kNN disagreement on all 8 units (Holm p = 0.016; Digits 8 of 10 seeds).
  - It beats Trust Score on 7 of 8. On Digits, Trust Score is ahead by 0.024.
  - It has the highest mean AUPRC on 4 of 8 units.
  - Only on Digits is another detector clearly better (Trust Score, and LR(kNN + TS)). On Letter,
    LR(all) is ahead by 0.0015. On Thyroid, LR(all) and LR(native) are higher, but not
    significantly.
- **Q3, CFMNN's own signals alone:** they beat kNN only on Thyroid (+0.394) and the synthetic family
  (+0.010), and are lower on the other 6 units.
- **Reading:**
  - CFMNN's training-order sensitivity carries information the data-based signals lack. The gain is
    largest where CFMNN is more accurate than kNN (Thyroid).
  - The geometry signals, U\* included, add nothing once the data-based signals are present.
  - On Digits, with only about 11 errors in each cross-validation fold's inner rows, the extra signals
    overfit the combiner.

## 4. Counterfactuals, global importance and extensions (RQ1, continued)

### 4.1 Counterfactuals and global importance (first runs)

Same models and splits as Section 2.1 (the paper's 80/20 split, main model), 11 datasets.
Raw tables: [`counterfactuals/summary.md`](counterfactuals/summary.md) and
[`global_importance/summary.md`](global_importance/summary.md).

**Counterfactuals (3 seeds).**
- **Setup:**
  - 30 test points per dataset and seed; the target is the runner-up class.
  - Every counterfactual is checked on the model.
  - On Thyroid, features 0 and 1 are locked. These are age and sex in the standard attribute order;
    feature 0 is continuous and feature 1 binary.
- **Plausibility** = L1 distance to the nearest training point of the target class (lower is better).
- **DiCE is set up fairly** (the original setup, in [`counterfactuals_v1/`](counterfactuals_v1/summary.md),
  was weak):
  - it is asked for 5 candidates;
  - its own post-hoc sparsity output is used;
  - the nearest candidate that the model predicts as the target is kept.

| Dataset | Valid (%), PACE / DiCE | L1 distance, PACE / DiCE | Features changed, PACE / DiCE | Plausibility, PACE / DiCE | ms, PACE / DiCE |
|---|---|---|---|---|---|
| S1 | 100 / 100 | 0.41 / 0.42 | 1.70 / **1.31** | **1.16** / 1.26 | 2.7 / 2,501 |
| S2 | 100 / 100 | 0.35 / 0.37 | 1.46 / 1.37 | **1.12** / 1.19 | 3.2 / 3,998 |
| S3 | 100 / 100 | 0.37 / 0.42 | 1.51 / **1.29** | **1.12** / 1.23 | 2.9 / 2,858 |
| Wine | 100 / 100 | **0.69** / 0.81 | 3.44 / **2.23** | **1.37** / 1.62 | 1.6 / 248 |
| Breast Cancer | 100 / 100 | **1.22** / 1.56 | 9.68 / **4.21** | **1.75** / 2.88 | 4.3 / 487 |
| Thyroid | 98.9 / 98.9 | **0.17** / 0.30 | 1.81 / **1.48** | **0.24** / 0.54 | 1.6 / 383 |
| Shuttle | 100 / 100 | **0.10** / 0.24 | 3.27 / **1.31** | **0.15** / 0.29 | 2.7 / 438 |
| MAGIC | 100 / 100 | **0.22** / 0.30 | 2.26 / **1.47** | **0.25** / 0.43 | 2.8 / 1,214 |
| Letter | 100 / 98.9 | **0.21** / 0.34 | 2.74 / **1.47** | **0.73** / 0.97 | 12.2 / 5,084 |
| Landsat | 100 / 98.9 | 1.17 / 1.17 | 11.23 / **3.26** | **1.63** / 2.55 | 11.5 / 1,435 |
| Page Blocks | 100 / 100 | 0.29 / 0.34 | 1.80 / **1.38** | **0.21** / 0.51 | 2.9 / 231 |

The bold entries are the significant ones: a paired Wilcoxon test over the 88–90 points per dataset
where both are valid (3 seeds pooled), Holm-corrected across the 11 datasets (the test is now in
the summary). That test treats points from the same model as independent, so it overstates
significance. Section 8.5 repeats the comparison over B1's 10 seeds, and the paper's table now comes
from there.
- **Both are valid on the model** in 99–100% of cases.
  - With the original setup DiCE was only 87% valid on Thyroid and 96% on Shuttle; that was the
    setup, not DiCE.
  - The one PACE miss (Thyroid) is a coverage limit: no target box admits the locked age and sex.
- **Size of change:**
  - PACE's change is smaller on 6 of the 8 real datasets (Wine, Breast Cancer, Thyroid, Shuttle, MAGIC,
    Letter): 42–86% of DiCE's L1 distance, and smaller for 63–86% of the points.
  - There is no difference on S1–S3, Landsat and Page Blocks.
- **Sparsity:** DiCE changes fewer features on 10 of 11 datasets (all but S2).
- **Plausibility:** PACE lands closer to real data of the target class on all 11.
- **Speed:** PACE is about 80–1,200× faster than DiCE (median time per counterfactual).
- **Both respect the locked features** (100%).
- **The nearest unlike neighbour** is an existing training point, so it is "plausible" by
  construction. But it changes almost every feature (for example 10 of 10 on S1) and keeps the locked
  age and sex in only 6% of Thyroid cases.

**Global importance (10 seeds).**
- **Rankings compared:**
  - PACE data-based (mean contrastive term);
  - PACE model-only (from the boxes alone, no data);
  - PACE-flip averaged over the test points;
  - permutation importance (the accuracy drop when a feature is shuffled; the model-based reference);
  - mean absolute KernelSHAP over 50 points (the fair setting: margin target, 20 real background rows).
- **Synthetic recovery:** all five rankings put x0–x2 in their top 3 on S1–S3 (1.00), so this test
  does not separate them.
- **Agreement with permutation importance** (Spearman), PACE-flip averaged vs mean abs KernelSHAP:
  - PACE-flip's mean is higher on 10 of 11 datasets; MAGIC is the exception (0.47 vs 0.68, not
    significant).
  - Significantly higher (paired over seeds, Holm across 11) on 3: Shuttle 0.97 vs 0.92, Letter 0.95
    vs 0.87 and Landsat 0.84 vs 0.49.
  - Large but not significant after Holm on Wine (0.53 vs 0.23) and Page Blocks (0.63 vs 0.31);
    small elsewhere.
  - **Not like for like** (found in the review of Section 4.2): PACE-flip averaged over every test
    point, as permutation importance does, while mean abs KernelSHAP used 50 points (its cost limit).
    On fresh seeds, limiting PACE-flip to the same 50 points cut about a quarter of its lead on Page
    Blocks (Section 4.2, B2).
- **Cost:** the PACE rankings take 0.001–11 s, against 0.3–350 s for permutation importance and
  2.7–176 s for KernelSHAP. Several runs shared the CPU, so compare these within a dataset only.
- **A fix:** the summary script skipped its own paired test because a loop variable overwrote the
  setting that switches it on. Fixed; the table is unchanged and the test is now in the summary.

### 4.2 Extensions v1 (pre-registered, seeds 40–49)

These answer the guide's three requests (sparse counterfactuals, conditional replacement and better
global aggregation) and add pairwise interaction values.
- **Pre-registration:** [`extensions_v1/PREREGISTRATION.md`](extensions_v1/PREREGISTRATION.md) was
  pushed with the code before any run on seeds 40–49, which were never used before.
- **Setup:** the same models and splits as above.
- **Tests:** every comparison is paired over the 10 seeds (two-sided Wilcoxon on per-seed means) and
  Holm-corrected across datasets.
  - **Better** or **worse** means Holm p < 0.05 in that direction; otherwise the two are **level**.
- **Times:** several runs shared the CPU, so compare them within a dataset only.
- **Raw tables:**
  - [`extensions_v1/counterfactuals/summary.md`](extensions_v1/counterfactuals/summary.md);
  - [`extensions_v1/global_importance/summary.md`](extensions_v1/global_importance/summary.md);
  - [`extensions_v1/conditional/summary.md`](extensions_v1/conditional/summary.md);
  - [`extensions_v1/conditional/posthoc_sigma.md`](extensions_v1/conditional/posthoc_sigma.md) (post hoc: B3 at other values of σ);
  - [`extensions_v1/conditional/posthoc_rows.md`](extensions_v1/conditional/posthoc_rows.md) (post hoc: B3 at σ = 0.1 with more rows);
  - [`extensions_v1/interactions/summary.md`](extensions_v1/interactions/summary.md).

**B1. PACE-sparse counterfactuals (primary: features changed, PACE-sparse vs DiCE).**
- **What it does:** it looks for the fewest features to change so that the model predicts the
  runner-up class, among a fixed set of moves towards boxes of the target class.
  - **The moves:** for each of up to 20 target boxes, a feature's new value is x put inside the box's
    range, then moved 0, 25, 50, 75 or 100% of the way to the box's centre. New values never leave the
    target box's range.
  - It scores every set of moved features exactly from CFMNN's per-feature terms, trying the smallest
    sets first.
  - It checks its answer on the model.
  - It never changes more features than PACE.
  - **Exact only within these moves.** A unit test checks it against brute force over the same moves.
    A change outside them can need fewer features; Shuttle is such a case (below).
- **Setting:** 30 test points per dataset and seed; DiCE in the fair setting of Section 4.1.

| Dataset | Features changed, PACE / PACE-sparse / DiCE | L1 distance, PACE-sparse / DiCE | Plausibility, PACE / PACE-sparse / DiCE | ms, PACE-sparse / DiCE |
|---|---|---|---|---|
| S1 | 1.58 / **1.03** / 1.32 | **0.25** / 0.41 | 1.16 / 1.27 / **1.23** | 129 / 6,784 |
| S2 | 1.56 / **1.05** / 1.32 | **0.22** / 0.39 | 1.11 / 1.23 / **1.19** | 130 / 6,996 |
| S3 | 1.44 / **1.03** / 1.36 | **0.23** / 0.39 | 1.14 / 1.24 / 1.24 | 124 / 6,700 |
| Wine | 3.44 / **1.89** / 2.35 | **0.59** / 0.80 | 1.32 / **1.60** / 1.67 | 44 / 560 |
| Breast Cancer | 10.96 / **3.36** / 4.51 | **1.19** / 1.66 | 1.68 / **2.61** / 3.01 | 4,362 / 1,243 |
| Thyroid | 1.91 / **1.48** / 1.60 | **0.14** / 0.29 | 0.25 / **0.29** / 0.51 | 16 / 959 |
| Shuttle | 3.29 / 2.31 / **1.37** | **0.10** / 0.25 | 0.15 / **0.17** / 0.28 | 13 / 1,045 |
| MAGIC | 2.56 / **1.08** / 1.41 | **0.11** / 0.33 | 0.26 / **0.28** / 0.46 | 47 / 2,966 |
| Letter | 2.68 / **1.16** / 1.45 | **0.11** / 0.32 | 0.72 / **0.76** / 0.92 | 99 / 12,446 |
| Landsat | 10.63 / **2.81** / 3.32 | **0.81** / 1.10 | 1.69 / **2.19** / 2.53 | 1,302 / 3,770 |
| Page Blocks | 2.17 / **1.04** / 1.37 | **0.09** / 0.35 | 0.23 / **0.29** / 0.51 | 25 / 556 |

Means over seeds of per-seed means (time: mean of per-seed medians). Bold marks the significant
winner of PACE-sparse vs DiCE. Plausibility is the L1 distance to the nearest training point of the
target class; lower is better.
- **Sparsity (primary):** PACE-sparse changes fewer features than DiCE on 10 of 11 datasets, by
  0.13–1.14 features per counterfactual.
  - On Shuttle it changes more (+0.94).
  - On S1–S3, MAGIC, Letter and Page Blocks it changes a single feature for 85–97% of points; DiCE
    does so for 56–68%.
  - In Section 4.1, DiCE was sparser than PACE on 10 of 11. PACE-sparse closes that gap on 10 of 11.
- **Size of change:** PACE-sparse's L1 distance is smaller than DiCE's on all 11 datasets. It is also
  smaller than PACE's on all 11. The pre-registration expected that it might move further than PACE;
  it did not.
- **Plausibility:**
  - Against DiCE, PACE-sparse lands closer to real target-class data on the 8 real datasets. It is
    farther on S1 and S2 and level on S3.
  - Against PACE (descriptive), it is 5–55% farther from real data on every dataset. Changing fewer
    features leaves the point less like real points of the target class.
- **Validity:** level with DiCE on all 11 datasets (99.3–100% against 97.3–100%).
- **Against PACE:** fewer features on all 11, by 0.42 (S3) to 7.8 (Landsat).
- **How it found its answers:**
  - exact search for 100% of points on 8 datasets and 99% on Thyroid, where the other 1% have no
    answer (the same coverage limit as PACE);
  - on Breast Cancer 71% exact and 29% greedy; on Landsat 83% exact and 17% greedy;
  - PACE's own answer was never sparser.
- **Speed:**
  - median 13–130 ms per counterfactual on 9 datasets, 1.3 s on Landsat and 4.4 s on Breast Cancer;
  - faster than DiCE on 10 of 11 datasets (2.9–126×), but 3.5× slower on Breast Cancer;
  - 1.7–460× slower than PACE (3–31 ms).
- **Why Shuttle is different (post hoc, not pre-registered):** checked on seed 40's 30 points with
  [`scripts/posthoc_shuttle_sparse.py`](../scripts/posthoc_shuttle_sparse.py); output in
  [`extensions_v1/counterfactuals/posthoc_shuttle.md`](extensions_v1/counterfactuals/posthoc_shuttle.md).
  DiCE was sparser on 22 of the 30.
  - **Every one of the 22 starts inside a box of its own class.** On Shuttle that box overlaps the
    target class's boxes. So moving into a target box leaves the point inside its own box as well: a
    tie. After the nearest-centroid tie-break, the point was still not predicted as the target in 21
    of 22.
  - **PACE-sparse then needs a second feature** to leave its own box. This is still true when every
    target box is a candidate, so the 20-box limit is not the cause.
  - **DiCE leaves the target box's range alone.** It moves one feature that is already inside the
    target box's range out of the point's own box (for example x0 from 0.14 to 0.54, past the own
    box's edge at 0.41). Its answer is inside no target box: the target class's membership is just
    under 1 (at least 0.97), but the own class's falls further.
  - **So the gap is in what PACE-sparse searches.** Its new values lie on the line from the target
    box's edge to its centre, at 5 fixed steps. On these 22 points no single-feature move among those
    steps made the target win, even with every target box as a candidate, so DiCE's winning values are
    not among them. Also trying a value just past the edge of the point's own box could close the gap.
    That is not tested here.
- **Reading:** the guide's concern, that DiCE was sparser than PACE, is answered on 10 of 11 datasets.
  The price is lower plausibility than PACE (still better than DiCE's on real data) and more time.

**B2. Better global aggregation (primary: agreement with permutation importance, mean |PACE-SHAP| vs
mean |KernelSHAP|).**
- **Primary:** level on all 11 datasets.
  - The largest gap is on Letter (+0.04, Holm p = 0.43).
  - Exact Shapley values did not give a better global ranking than KernelSHAP's estimates.
  - Two likely reasons, not tested:
    - averaging over 50 points smooths out much of KernelSHAP's per-point noise;
    - Spearman agreement over all features is dominated by the many features of near-zero importance,
      whose order is mostly noise, so this test has little power to show a difference.
- **Mean |PACE-SHAP-20| vs mean |KernelSHAP|:** level on all 11. The two rankings agree with each
  other at 0.91–1.00, as they should for the same game on the same 20 rows.
- **PACE-flip, normalised vs averaged** (each point's scores scaled to sum to 1 before averaging):
  better on MAGIC (0.65 vs 0.52) and Letter (0.96 vs 0.94), level on the other 9, worse on none.
- **Replication of Section 4.1 on fresh seeds (PACE-flip averaged vs mean |KernelSHAP|):**
  - better on Letter (0.94 vs 0.85), Landsat (0.87 vs 0.61) and Page Blocks (0.51 vs 0.18);
  - level on the other 8;
  - higher mean on 10 of 11; MAGIC is again the exception (0.52 vs 0.61, not significant).
  - Seeds 0–9 found Shuttle, Letter and Landsat. Letter and Landsat replicate. Shuttle does not
    (0.94 vs 0.89, Holm p = 0.96). Page Blocks is new.
  - **Not like for like:** PACE-flip averages over every test point, as permutation importance does,
    while mean |KernelSHAP| uses 50 points (its cost limit). Section 4.1 had the same design.
  - **Post hoc, on Page Blocks** ([`scripts/posthoc_review.py`](../scripts/posthoc_review.py),
    [`extensions_v1/global_importance/posthoc_points_page_blocks.md`](extensions_v1/global_importance/posthoc_points_page_blocks.md)):
    - limiting PACE-flip to the same 50 points lowers its agreement from 0.51 to 0.43;
    - that is still above KernelSHAP's 0.18 (higher in 8 of 10 seeds);
    - so about a quarter of its lead there comes from using more points;
    - the other datasets were not checked.
- **Ground truth:** every ranking puts x0–x2 in its top 3 on S1–S3 (1.00).
- **Cost:**
  - mean |PACE-SHAP| took 0.7–2.5× as long as mean |KernelSHAP|;
  - mean |PACE-SHAP-20| was 5–14× faster than KernelSHAP;
  - the PACE-flip rankings took 0.01–17 s.

**B3. Interventional vs conditional replacement (primary: PACE-SHAP-cond vs KernelSHAP-cond on the
conditional test, and the realism of the conditional vs the interventional game).**
- **What it is:** an explainer "removes" a feature by filling in values from other rows.
  - **Interventional** (PACE-SHAP, KernelSHAP and PACE-flip, as before): values come from any row, so
    some inputs combine feature values that never occur together in real data.
  - **Conditional:** values come from rows that resemble the point on the features it keeps.
    PACE-SHAP-cond and KernelSHAP-cond weight the 200 explainer rows by exp(−D²/(2σ²)) with σ = 0.1
    (Aas, Jullum and Løland 2021, diagonal covariance). PACE-flip-cond takes feature i's values from
    the 32 rows nearest to the point on its other features.
  - The two answer different questions, so they are reported in separate tables, as pre-registered.
- **Setting:**
  - 13 datasets: the 11, plus S1c and S2c, where x3 is a noisy copy of x0 that the true rule does not
    use;
  - 50 test points per seed (36 on Wine);
  - two tests. The interventional one is v3's: removed features come from a random evaluation row.
    The conditional one takes them from the q-th nearest evaluation row on the features still taken
    from x (q = 0..9). Each test is run as deletion (lower AUC = more faithful) and insertion (higher
    = more faithful).
- **Realism measure:** an input is off the data if its nearest training row is farther away than the
  95th percentile of the training rows' own nearest-neighbour distances (standardised features).
  200 inputs per point and scheme.
- **Checks:** the explained points themselves are off the data 3.2–6.4% of the time (mean 5.0%), as
  the 95th-percentile threshold intends. The restarted run equals the stopped run on all 75 results
  both logs hold (see the deviations below).

Conditional test (mean margin AUC over seeds). Bold marks the significant winner of PACE-SHAP-cond
vs KernelSHAP-cond (Holm across 13). The last column compares PACE-SHAP-cond with PACE-SHAP on this
test (deletion / insertion).

| Dataset | PACE-SHAP exact? | Deletion, PACE-SHAP-cond / KernelSHAP-cond | Insertion, PACE-SHAP-cond / KernelSHAP-cond | PACE-SHAP, deletion / insertion | PACE-SHAP-cond vs PACE-SHAP |
|---|---|---|---|---|---|
| S1 | yes | -0.131 / -0.120 | **0.738** / 0.729 | -0.184 / 0.778 | worse / worse |
| S2 | yes | -0.110 / -0.092 | 0.660 / 0.649 | -0.172 / 0.713 | worse / worse |
| S3 | yes | **-0.039** / -0.023 | **0.711** / 0.692 | -0.089 / 0.752 | worse / worse |
| S1c | yes | -0.052 / -0.042 | 0.787 / 0.780 | -0.094 / 0.812 | worse / worse |
| S2c | yes | -0.105 / -0.091 | 0.706 / 0.701 | -0.157 / 0.743 | worse / worse |
| Wine | no | 0.148 / 0.163 | 0.857 / 0.861 | 0.080 / 0.891 | worse / worse |
| Breast Cancer | no | **0.360** / 0.471 | **0.915** / 0.898 | 0.249 / 0.933 | worse / worse |
| Thyroid | no | **0.556** / 0.569 | 0.875 / 0.868 | 0.528 / 0.907 | worse / worse |
| Shuttle | yes | **0.452** / 0.469 | **0.771** / 0.758 | 0.482 / 0.773 | better / level |
| MAGIC | yes | **0.047** / 0.060 | 0.174 / 0.164 | 0.050 / 0.187 | level / level |
| Letter | no | -0.525 / -0.517 | -0.057 / -0.060 | -0.600 / 0.079 | worse / worse |
| Landsat | no | **0.265** / 0.411 | **0.598** / 0.550 | 0.160 / 0.597 | worse / level |
| Page Blocks | yes | 0.114 / 0.117 | **0.253** / 0.244 | 0.110 / 0.253 | level / level |

- **Primary 1, faithfulness like for like:** on the conditional test, PACE-SHAP-cond is more faithful
  than KernelSHAP-cond on 6 of 13 datasets by deletion and on 6 by insertion, and never significantly less faithful.
  - Better by deletion: S3, Breast Cancer, Thyroid, Shuttle, MAGIC and Landsat. Better by insertion:
    S1, S3, Breast Cancer, Shuttle, Landsat and Page Blocks.
  - Its mean is better on all 13 datasets by deletion and on 12 by insertion (not Wine).
  - The pre-registration expected it to win where it is exact (d ≤ 12), and perhaps not where it
    samples. It won on 3 (deletion) and 4 (insertion) of the 8 exact datasets, and also on 3 and 2 of
    the 5 sampled ones.
  - The largest gaps are on two sampled datasets, Breast Cancer (−0.111) and Landsat (−0.146), where
    KernelSHAP-cond's 200 coalitions are few for 30–36 features.
- **The conditional explainers were not more faithful than the interventional ones, even on the
  conditional test** (secondary):
  - PACE-SHAP-cond is less faithful than PACE-SHAP on the conditional test on 10 of 13 datasets by
    deletion and on 9 by insertion. It is better only on Shuttle, by deletion. On the interventional
    test it is less faithful on all 13, as expected.
  - PACE-flip-cond vs PACE-flip on the conditional deletion test: level on 9, worse on 3 (S1c, S2c,
    Breast Cancer), better on 1 (Landsat). On the interventional deletion test: worse on 8, level on 5.
  - So the expectation that each explainer does best on the test that matches its own game is not
    supported. Nor did the known limit appear: the conditional test did not favour the conditional
    explainers.
  - A likely reason: at σ = 0.1 the conditional game rests on about one row once half the features
    are kept (see "very few rows" below), which adds noise. Two observations fit this:
    - On S1–S3 the features are independent, so in theory the two games are the same and any gap is
      estimation error. There PACE-SHAP-cond is less faithful on both tests and finds fewer of the
      true features (top 3: 0.67–0.72, against 0.81–0.86 for PACE-SHAP).
    - Post hoc and descriptive: on the 7 datasets where the conditional game rests on about one row
      (S1–S3, S1c, S2c, Wine, Letter), PACE-SHAP-cond is worse than PACE-SHAP on the conditional test
      by both measures. On the 6 where it has more rows (1.7–18), it is worse by both on only 2
      (Breast Cancer, Thyroid).
  - **Post hoc check of this reason** ([`scripts/posthoc_sigma.py`](../scripts/posthoc_sigma.py),
    [`extensions_v1/conditional/posthoc_sigma.md`](extensions_v1/conditional/posthoc_sigma.md)):
    - **Setting:** PACE-SHAP-cond recomputed at σ = 0.2, 0.3, 0.5 and 1 with B3's own models, points,
      rows and tests. It covers the 6 datasets where PACE-SHAP is exact (S1, S1c, S2c, Shuttle, MAGIC,
      Page Blocks) and seeds 40–44. At σ = 0.1 and ∞ it gives B3's numbers exactly. Descriptive only:
      5 seeds are too few for a test.
    - **Rows:** wider weights use more rows. For large coalitions the effective rows rise from about
      1 at σ = 0.1 to about 10 at σ = 0.3, 55 at 0.5 and 170 at 1 on the synthetic datasets. On the
      real ones they reach 90–143 at σ = 0.5.
    - **Synthetic datasets:**
      - the loss to PACE-SHAP on the conditional test shrinks steadily, from 0.04–0.06 by deletion and
        0.03–0.04 by insertion at σ = 0.1 to under 0.01 at σ = 1;
      - the top-3 recovery rises to PACE-SHAP's (S1: from 0.71 to 0.87);
      - it never clearly passes PACE-SHAP. On S1 the two games are the same in theory, so the loss
        there was noise from too few rows.
    - **Real datasets:** wider weights make it more faithful than PACE-SHAP on the conditional test:
      - on Shuttle by deletion at every σ (5 of 5 seeds);
      - on Shuttle and MAGIC by both measures at σ = 0.5 and 1 (3–5 of 5 seeds);
      - Page Blocks stays within 0.007 at every σ.
    - **Interventional test:** it stays less faithful than PACE-SHAP, as expected. Only at σ = 1 does
      the gap fall to 0.013 or less.
    - **The price is realism:** the inputs drift off the data as σ grows.
      - On MAGIC 2% of them are off at σ = 0.1, 5% at 0.5 and 12% at 1, against 20% for the
        interventional game.
      - On S1c the figures are 2%, 6% and 11%, against 14%.
    - **Reading:**
      - This supports the few-rows reason. With more rows the conditional game catches up with
        PACE-SHAP on the synthetic data and gets slightly ahead on the conditional test on Shuttle and
        MAGIC.
      - But the σ values were chosen after seeing B3's results, so B3's pre-registered conclusions
        stand.
      - Keeping σ = 0.1 and adding rows instead is the next check.
  - **Post hoc check with more rows instead** ([`scripts/posthoc_rows.py`](../scripts/posthoc_rows.py),
    [`extensions_v1/conditional/posthoc_rows.md`](extensions_v1/conditional/posthoc_rows.md)):
    - **Setting:** σ = 0.1 and B3's weight function are kept, with 800 explainer rows instead of 200
      (B3's 200 followed by 600 more training rows). PACE-SHAP-cond and PACE-SHAP are computed on the
      same rows, so the game stays fixed and only the number of rows changes. It covers S1, Shuttle,
      MAGIC and Page Blocks, seeds 40–44. With 200 rows it gives B3's numbers exactly. Descriptive
      only.
    - **Rows:** 4 times the rows raise the effective rows far less than wider weights do: from 1.1 to
      1.5 on S1, 2.3 to 5.5 on MAGIC, 3.4 to 9.6 on Shuttle and 15 to 54 on Page Blocks. With 10
      independent features, few rows are close enough to count.
    - **S1** (the two games are the same in theory):
      - PACE-SHAP-cond improves in 5 of 5 seeds on all four measures;
      - its loss to PACE-SHAP on the conditional test halves, from 0.061 to 0.028 by deletion and
        from 0.036 to 0.018 by insertion;
      - its top-3 recovery rises from 0.71 to 0.80 (PACE-SHAP 0.87).
      - With the game fixed, this confirms that the loss on S1 is noise from too few rows.
    - **Real datasets:** the gaps change little. On the conditional test:
      - Shuttle stays ahead by deletion and level by insertion;
      - MAGIC stays level by deletion and behind by insertion;
      - Page Blocks stays within 0.007.
    - **Realism:** the conditional inputs get slightly closer to the data, since the nearest rows are
      now closer (S1 1.3% to 0.8% off, MAGIC 2.2% to 1.5%, Page Blocks 5.5% to 3.7%).
    - **Cost:** about 4 times the time.
    - **Reading:** more rows reduce the noise, but slowly. Wider weights reach more effective rows
      at no extra cost: σ = 0.2 with 200 rows already gives more than σ = 0.1 with 800 on all four
      datasets.

Interventional test (mean margin AUC over seeds). Bold marks the significant winner of PACE-SHAP vs
KernelSHAP (Holm across 13); the conditional explainers are shown for reference.

| Dataset | Deletion, PACE-SHAP / KernelSHAP | Insertion, PACE-SHAP / KernelSHAP | PACE-SHAP-cond, deletion / insertion | KernelSHAP-cond, deletion / insertion |
|---|---|---|---|---|
| S1 | -0.126 / -0.125 | 0.662 / 0.661 | -0.088 / 0.627 | -0.080 / 0.618 |
| S2 | -0.142 / -0.140 | 0.616 / 0.613 | -0.089 / 0.569 | -0.078 / 0.556 |
| S3 | **-0.053** / -0.046 | 0.651 / 0.649 | -0.006 / 0.615 | 0.006 / 0.598 |
| S1c | -0.063 / -0.062 | 0.702 / 0.702 | -0.030 / 0.675 | -0.024 / 0.667 |
| S2c | -0.129 / -0.127 | 0.661 / 0.659 | -0.092 / 0.627 | -0.078 / 0.619 |
| Wine | -0.072 / -0.073 | 0.542 / 0.541 | -0.000 / 0.454 | 0.000 / 0.447 |
| Breast Cancer | **0.135** / 0.137 | **0.709** / 0.706 | 0.222 / 0.601 | 0.271 / 0.559 |
| Thyroid | 0.508 / 0.508 | 0.893 / 0.892 | 0.543 / 0.853 | 0.557 / 0.844 |
| Shuttle | 0.310 / 0.311 | 0.706 / 0.706 | 0.341 / 0.671 | 0.355 / 0.650 |
| MAGIC | -0.047 / -0.050 | 0.268 / 0.267 | 0.055 / 0.158 | 0.070 / 0.146 |
| Letter | -0.703 / -0.702 | -0.015 / -0.016 | -0.607 / -0.210 | -0.606 / -0.219 |
| Landsat | **-0.330** / -0.285 | **0.249** / 0.213 | -0.153 / 0.047 | -0.090 / -0.003 |
| Page Blocks | 0.041 / 0.041 | 0.276 / 0.273 | 0.077 / 0.226 | 0.088 / 0.216 |

- **PACE-SHAP vs KernelSHAP on the interventional test** (secondary): better on 3 datasets by deletion
  (S3, Breast Cancer, Landsat) and on 2 by insertion (Breast Cancer, Landsat), level elsewhere, never
  worse.
  - Here KernelSHAP is our own implementation on the same 200 rows (200 coalitions, no feature
    selection), so the only difference left is sampling.
  - The gap is therefore smaller than in v3 (6–7 of 11), where KernelSHAP used 20 rows and shap's
    feature selection.

Realism (% of the evaluated inputs that are off the data, mean over seeds; lower = more realistic).

| Dataset | Explained points | Interventional game | Conditional game | PACE-flip values | PACE-flip-cond values | LIME samples | Interventional test | Conditional test |
|---|---|---|---|---|---|---|---|---|
| S1 | 4.4 | 3.8 | 1.2 | 4.9 | 5.0 | 13.0 | 3.9 | 2.0 |
| S2 | 6.0 | 4.2 | 1.3 | 5.2 | 5.4 | 12.8 | 4.2 | 2.2 |
| S3 | 6.0 | 4.1 | 1.4 | 5.5 | 5.5 | 13.3 | 4.0 | 2.2 |
| S1c | 5.2 | 13.7 | 1.6 | 11.0 | 7.9 | 35.2 | 13.8 | 3.1 |
| S2c | 4.6 | 13.9 | 1.5 | 11.5 | 8.2 | 35.8 | 14.0 | 3.3 |
| Wine | 4.7 | 16.5 | 1.4 | 8.1 | 6.4 | 48.1 | 16.1 | 2.6 |
| Breast Cancer | 3.2 | 12.7 | 1.2 | 4.0 | 3.3 | 62.4 | 12.7 | 1.8 |
| Thyroid | 4.8 | 8.9 | 2.6 | 6.1 | 5.6 | 100.0 | 8.8 | 4.0 |
| Shuttle | 5.0 | 70.6 | 56.7 | 66.1 | 61.3 | 100.0 | 70.4 | 57.4 |
| MAGIC | 4.0 | 20.3 | 2.4 | 13.9 | 5.9 | 71.2 | 20.1 | 3.7 |
| Letter | 5.2 | 62.7 | 10.1 | 31.9 | 21.7 | 99.5 | 62.6 | 26.2 |
| Landsat | 6.4 | 67.1 | 4.6 | 26.3 | 9.6 | 100.0 | 66.8 | 8.4 |
| Page Blocks | 5.4 | 29.6 | 6.0 | 23.5 | 12.6 | 99.7 | 30.1 | 9.3 |

- **Primary 2, realism:** the conditional game's inputs are off the data less often than the
  interventional game's on all 13 datasets (lower in 10 of 10 seeds each). Outside Shuttle they are
  off at most 10% of the time, against 4–67% for the interventional game; on Shuttle, 57% against 71%.
  - This was expected on real data with correlated features. On S1–S3, whose features are
    independent, the two were expected to be about the same, but the conditional game is lower there
    too (1.2–1.4% against 3.8–4.2%).
  - Part of this is copying. The explainer rows are training rows, and for large coalitions the
    conditional game mostly copies one of them. On 10 of 13 datasets (all but Shuttle, Letter and
    Page Blocks) its inputs are off less often than the explained test points themselves.
  - Shuttle is off for every scheme (57–100%). Its training rows are packed so densely that the
    threshold is tiny (0.05 in standardised units, against 0.5–4.6 on the other datasets).
- **Realism, secondary:**
  - PACE-flip-cond's values are off less often than PACE-flip's on 10 of 13 datasets and level on
    S1–S3, as expected.
  - The conditional test's inputs are off less often than the interventional test's on all 13.
  - The interventional test itself evaluates the model mostly off the data on Shuttle, Letter and
    Landsat (63–70%). So on these datasets the v3 tests mostly measure the model on inputs unlike real
    data.
- **Correlated copy (S1c, S2c):**
  - The model does use x3: it causes 19% of PACE-flip's decision flips on S1c and 13% on S2c, against
    7–8% for the unused, independent x3 on S1 and S2.
  - The pre-registration expected the conditional explainers to give x3 more credit than the
    interventional ones. They gave it less. All three conditional explainers lower x3's share of
    |attribution| by 0.02–0.05 on both datasets, in 9–10 of 10 seeds (descriptive, no planned test).
  - Interventional explainers give x3 0.17–0.21 of the credit, conditional ones 0.13–0.17, and random
    scores 0.10.
  - Not supported at σ = 0.1. A likely reason: the noise above spreads credit over all features.
  - Post hoc, wider weights reverse this, as that reason predicts. At σ = 1 PACE-SHAP-cond gives x3
    slightly more credit than PACE-SHAP (S1c 0.203 vs 0.197, S2c 0.186 vs 0.177), in 5 of 5 seeds.
    This is descriptive; see the check above.
- **Ground truth top 3 (S1–S3, S1c, S2c):**
  - PACE-SHAP 0.69–0.86 and KernelSHAP 0.69–0.85; LIME 0.72–0.93;
  - lower for the conditional versions (PACE-SHAP-cond 0.62–0.72, KernelSHAP-cond 0.59–0.69);
  - PACE-flip and PACE-flip-cond 0.47–0.57.
- **Time per explanation** (several runs shared the CPU; compare within a dataset only):
  - PACE-SHAP-cond adds 0–13% to PACE-SHAP's time;
  - KernelSHAP, on the same 200 rows with 200 coalitions, took 3.5–10× as long as PACE-SHAP;
  - PACE-flip-cond took 1.8–51 ms, 3.9–13× as long as PACE-flip.
- **Reading:**
  - Conditional replacement does what the guide asked for realism: the explainers' inputs stay near
    real data, though partly by copying training rows.
  - Like for like, PACE-SHAP-cond is the better conditional explainer (6 of 13 datasets on each
    measure, never worse).
  - But at σ = 0.1 the conditional values were not more faithful than the interventional ones on
    either test, and they did not credit the correlated copy x3 more.
  - Post hoc, a larger σ gives the conditional game more rows to average over. It then:
    - catches up with PACE-SHAP on the synthetic data;
    - gets slightly ahead on the conditional test on Shuttle and MAGIC;
    - credits x3 slightly more.
  - But its inputs move away from the data. Claiming this would need a σ fixed in advance.
  - Post hoc, 4 times as many rows at σ = 0.1 also reduce the noise: S1's loss halves. But they
    change little on the real datasets and take 4 times as long.

**B4. PACE-SHAP interaction values (primary: is the AND pair the strongest pair on S3?).**
- **What they are:** for each pair of features, how much the two add together beyond what each adds
  alone.
  - They are exact and come from the same table as PACE-SHAP.
  - A pair's strength is its mean |interaction| over 50 points.
  - The run covered the 8 datasets with at most 12 features.
- **Primary:** supported. On S3 (class 1 if x0 > 0.5 and x1 < 0.4, or if x2 > 0.8), the AND pair
  (x0, x1) is the strongest pair in 10 of 10 seeds; at least 8 were needed.
  - Its strength is 0.0078, against 0.0034–0.0037 for the other two relevant pairs.
  - The strongest pair with an irrelevant feature averages 0.0032.
- **S1 and S2:** (x0, x1) is also the strongest pair in 10 of 10 seeds. Their true rule adds the
  features up (2·x0 + 1.5·x1 − x2 above a threshold), so it has no interaction.
  - This interaction comes from the model. A likely reason, not tested: axis-aligned boxes
    approximate the sloped boundary as a staircase, so x0 and x1 act together.
  - Interaction values describe the model, not the data.
  - **So the pre-registered test was weak.** S1 and S2 are a negative control it lacked: the AND pair
    would have come out on top even without a true interaction.
  - **Post hoc** ([`scripts/posthoc_review.py`](../scripts/posthoc_review.py), output in
    [`extensions_v1/posthoc_review.md`](extensions_v1/posthoc_review.md)):
    - on S3, (x0, x1) is 2.06 times as strong as the next pair (1.58–2.80 over seeds);
    - on S1 it is 1.62 times (1.16–2.01), and on S2 1.35 times (1.29–1.53);
    - so it stands out more where the interaction is real. S3's range overlaps S1's but not S2's.
- **S1c and S2c:** the strongest pair is (x0, x3) in 10 of 10 seeds. Here x3 is a noisy copy of x0
  that the true rule does not use.
- **Interaction is spread across all features:**
  - Pairs among x0–x2 hold only 6–13% of total pair strength on the synthetic sets.
  - That is about 1–2 times their share by count (3 of 45 pairs).
  - CFMNN's boxes make every feature interact a little.
- **Real datasets:**
  - Shuttle has a stable strongest pair, (x0, x6), in 10 of 10 seeds;
  - MAGIC has (x0, x2) in 5 of 10;
  - Page Blocks has none (its most frequent strongest pair wins in 2 of 10).
- **Cost:** about 8–10 s per point on the 10-feature synthetic sets, 1.3 s on MAGIC, 0.25 s on Page
  Blocks and 0.05 s on Shuttle (shared CPU).

**Limitations found in the review (after the runs):**
- **Counterfactual robustness was not measured.** PACE-sparse prefers the smallest move that wins.
  When that move just clips features into the box, the answer lies exactly on the box's edge, so a
  tiny change back can undo it. PACE's answers and DiCE's sparsified ones can sit near the edge too.
  A measure such as the target's margin at the answer, or validity under small noise, would be
  needed.
- **DiCE is run with its random search method.** A stronger sparse baseline was not tried, such as
  DiCE's genetic method with a high sparsity weight, or a brute-force search over 1–2 features. So
  "fewer features than DiCE" means fewer than this DiCE setting.
- **The global-importance test has little power** (B2 above), and its PACE-flip comparison uses more
  points than KernelSHAP.
- **The conditional game uses very few rows** (applies to B3). At σ = 0.1 with 200 explainer rows,
  once a coalition holds half the features or more, the Gaussian weights' effective number of rows
  (Kish) is:
  - about 1 (1.0–1.2) on S1–S3, S1c, S2c, Wine and Letter;
  - 1.7–2.6 on Breast Cancer and MAGIC;
  - 2.3–3.9 on Shuttle and Landsat;
  - 4–18 on Thyroid and Page Blocks, whose many repeated values give exact ties.

  So for large coalitions the conditional game mostly copies the nearest row on the kept features.
  Details: [`extensions_v1/posthoc_review.md`](extensions_v1/posthoc_review.md). B3's results fit this:
  the conditional explainers were less faithful than the interventional ones, most clearly where the
  game rests on about one row (B3 above). Two post hoc checks support this (B3 above):
  - wider weights shrink the loss or turn it into a small lead, while the inputs drift off the data;
  - 4 times as many rows at σ = 0.1 halve the loss on S1, but raise the effective rows only slowly.
- **PACE-sparse is slow when many features are inside the box** (4.4 s on Breast Cancer). For the 4
  steps towards the centre it searches every feature, not only those outside the box. Exact pruning
  would cut this without changing any answer: skipping a set size when even its best moves cannot win,
  and trying boxes with fewer features outside first. Not implemented.

**Deviations from the pre-registration:**
- B4 took 65 minutes with 6 workers, not the "minutes" estimated.
- **B3 was stopped after about 3 hours and restarted with shared model calls.** By then 6 of its 130
  dataset-seed jobs were done (S1, seeds 40–45) and 6 more were under way. The two KernelSHAPs took
  about 50 s per point each on S1, most of each job's time, so the run would have taken far longer
  than the 8–12 hours estimated.
  - KernelSHAP and KernelSHAP-cond use the same coalitions and model calls and differ only in how the
    rows are weighted. So do PACE-SHAP and PACE-SHAP-cond where they are exact (d ≤ 12). Each pair
    now makes its model calls once (`xai_baselines.kernel_shap_games`, `PaceShap.explain_games`).
  - The attributions are unchanged. Tests compare them exactly with running the two methods apart
    and with copies of the earlier code. On seed 40's first test points of S1, Page Blocks and Wine,
    every array from the new code equals the earlier code's. B3's whole routine, run with both
    versions on S1 and Wine (2 points), gives identical results except for the times.
  - A paired method's time per explanation is now its shared part plus its own part, an estimate of
    its time alone.
  - Nothing else changed: same seeds, settings, points and tests. The stopped run's log is kept in
    [`extensions_v1/conditional/stopped_run_log.txt`](extensions_v1/conditional/stopped_run_log.txt).
  - The restarted run took 9 h 53 min (10:57–20:50). Its results equal the stopped run's on all 75
    dataset-seed-method results that both logs hold.

## 5. Repair (RQ3, RQ4)

Four runs, in order: an exploratory pilot (5.1), the first pre-registered run (5.2), exploratory checks
that led us to reframe the questions (5.3), and the second pre-registered run, which tests RQ3 and RQ4
as the paper states them (5.4).

### 5.1 Pilot (Phase 5, exploratory, seeds 0–2)

**Questions:**
- **RQ4:** when a labelled point is misclassified inside a box of the wrong class, does choosing which
  face of that box to cut with PACE's contrastive explanation fix the model better than choosing by
  geometry or at random?
- **RQ3:** does it matter which points are labelled?

**Setup:**
- Role split. The pool's labels are hidden and revealed only for selected points.
- Budget: min(300, a quarter of the pool) labels in 10 rounds. Seeds 0–2.
- Each labelled point gets one edit, checked on the acceptance rows and rolled back if it fails:
  - macro-F1 may fall by at most 0.5 points;
  - no class's recall may fall by more than 1 point;
  - boxes may grow to at most 1.1 × the start.
- Raw tables: [`refinement_pilot/summary.md`](refinement_pilot/summary.md).

These results are exploratory. The confirmatory run (seeds 10–19) is pre-registered in
[`refinement_confirmatory/PREREGISTRATION.md`](refinement_confirmatory/PREREGISTRATION.md).

**Edits:**
- **Add:** CFMNN's own training rule (absorb or new box). `add_points` reproduces a full refit exactly
  (tested).
- **Explanation shrink (C):** cut the wrong box just past the point on the feature with the largest
  contrastive term.
- **Controls (C′):** a random feature, or the feature where the cut removes the smallest share of the
  box's width ("smallest cut").

Change in test macro-F1 against no refinement (points, mean of 3 seeds). Selection is by the order
ensemble except in the B9 column:

| Dataset | B9: refit on all pool labels | Add only | Explanation shrink | Random shrink | Smallest-cut shrink |
|---|---|---|---|---|---|
| S2 | 0.0 | +0.1 | +0.1 | +0.1 | +0.1 |
| Thyroid | +0.4 | +2.1 | +3.2 | +5.0 | +6.1 |
| Landsat | −0.1 | −0.4 | +0.5 | +0.8 | +0.6 |
| Page Blocks | −6.2 | −1.8 | +2.4 | +3.4 | +1.2 |
| MAGIC | +0.6 | +0.2 | +3.5 | +3.8 | +5.0 |

- **Shrinking is what helps, not more labels.**
  - The smallest-cut shrink beat add-only in 12 of the 12 real dataset × seed runs; the explanation
    shrink did so in 9 of 12.
  - Even refitting on the whole labelled pool (B9) gained at most 0.6 points, and lost 6.2 on Page
    Blocks.
- **The explanation did not choose better.**
  - The explanation shrink was below the smallest-cut control in all 3 seeds on Thyroid (−2.9 points
    on average) and MAGIC (−1.5).
  - It was mixed on Landsat (−0.1) and Page Blocks (+1.2).
  - Its cuts are deeper: 16–31% of the box's width on average, against 0–7% for the smallest cut.
  - With 3 seeds none of this is tested.
- **With add-only edits, the choice of points made little difference.** Random, margin and PD + U\*
  changed macro-F1 by −0.4 to +1.3 points. The order ensemble picks far more of the model's errors
  (for example 226 of 300 on MAGIC, against 74 for random), which matters only when a shrink can use
  them.
- **The shrink can only act on certain datasets.** On S2 no misclassified pool point lies inside a
  wrong-class box, so the shrink never applies.
  - The same holds for every synthetic set; Shuttle has only 18–20 such points
    ([`refinement_pilot/eligibility.md`](refinement_pilot/eligibility.md)).
  - These sets are not in the confirmatory run.
- **The gate matters on small datasets.** For example, Page Blocks: 53 shrinks tried per seed,
  12 accepted. MAGIC: 164 tried, 157 accepted.
- **Cost** (one dataset × seed, all 9 arms):
  - 1–8 minutes on Thyroid, Landsat, Page Blocks and S2;
  - about 21 minutes on MAGIC, of which PD + U\* selection takes about 9.
  - `--jobs 4` runs 4 at a time with identical results, about 2.2× faster.

**Known issues found before the confirmatory run.** The first confirmatory run (5.2) went ahead as
pre-registered. The second run (5.4) deals with these issues: it counts the acceptance rows as labels,
adds self-repair (a gated contraction at the model's own training errors), and adds retraining,
10-NN and a Random Forest given the same labels.
- **No contraction baseline.** CFMNN has no contraction step by design, while Simpson's FMNN (1992)
  removes class overlaps without new labels. Without that reference, the gains may simply be
  "contraction put back".
- **The explanation-guided rule is mis-aimed.**
  - 64–89% of the repairable points lie in an overlap (tie mode; Thyroid, Landsat, Page Blocks, MAGIC,
    Letter). This comes from a one-off check of the pilot seeds' starting models, not saved as a results
    file: it took the pool points each model gets wrong inside the winning wrong-class box and read
    their contrastive mode.
  - There the rule picks the feature on which the point is nearest the wrong box's centre, so it cuts
    through the box's core.
  - Pooled over the pilot, its cuts take 27% of the box's width (smallest cut: 5%) and remove twice as
    many training rows (32 vs 16).
  - The smallest-cut control follows the minimal-disturbance idea behind Simpson's contraction.
- **The label cost is understated.** The gate reads the labelled acceptance rows: 2.0–6.3× the label
  budget (MAGIC: 1,902 rows against 300 labels).
- **Planned for the extension:**
  - Simpson contraction as a reference, with and without the gate;
  - a refit with the acceptance rows;
  - label cost reported as budget + acceptance rows.

### 5.2 First pre-registered run (seeds 10–19)

Rules: [`refinement_confirmatory/PREREGISTRATION.md`](refinement_confirmatory/PREREGISTRATION.md).
Results: [`refinement_confirmatory/summary_v1.md`](refinement_confirmatory/summary_v1.md), recomputed
from the teammate's `runs.csv` by `scripts/summarise_refinement_v1.py`. Six datasets with at least 2,000
rows and enough repairable points: Thyroid, Landsat, Page Blocks, MAGIC, Letter and Skin. Test
macro-F1; paired Wilcoxon over the 10 seeds, Holm across the 6 datasets.

- **R1 (primary): the explanation's cut is no better than the thinnest cut.** No significant
  difference on any dataset (mean −1.98 to +2.00 points). The first-posed hypothesis is not supported.
- **R2: shrinking behind the gate beats adding points only,** on 6 of 6 datasets (+0.74 to +8.61).
- **R3: the explanation's cut against a random feature:** worse on Letter (−0.86), no significant
  difference elsewhere.
- **S1: order-ensemble selection beats random selection** on 3 of 6 (Page Blocks, MAGIC, Skin).
- The teammate's run also has two arms added by a protocol amendment
  ([`PROTOCOL_AMENDMENT_1.md`](refinement_confirmatory/PROTOCOL_AMENDMENT_1.md)): a Simpson-style
  contraction and a "C-minimal" cut. They were not pre-registered, their code is not in this
  repository, and the paper does not use them. `summary_v1.md` reports them in an appendix.

### 5.3 Exploratory checks (seeds 0–2)

Results: [`refinement_v2_explore/summary.md`](refinement_v2_explore/summary.md). Not pre-registered.

- **Every training error is a tie.** Each training row CFMNN gets wrong lies inside a box of its own
  class and a box of the wrong class (membership 1 in both), and the nearest-centroid rule picks the
  wrong one. A post hoc check in the final review confirmed this for 100% of training errors on all 6
  datasets at seeds 0, 10 and 50 (66 to 10,980 errors per case):
  [`final_review/tie_errors/summary.md`](final_review/tie_errors/summary.md).
- **Self-repair with no new labels** (shrinking the wrong box at the model's own training errors) was
  within half a point of buying 300 labels on 5 of 6 datasets (MAGIC: 2.44 points behind).
- **The gate's acceptance rows are labels too:** 548–24,506 per dataset, which the first run did not
  count.
- **Retraining CFMNN on the same labels gained nothing:** −4.3 to +0.7 points against no update.
- **Which feature to cut no longer matters much:** trying every feature and keeping the cut the gate
  scored best gained only −0.57 to +1.13 points over the thinnest cut.
- **The certificates held:** no prediction changed outside an edit's certificate in 69,488 edits.

We then wrote RQ3 and RQ4 as in the paper and pre-registered a second run.

### 5.4 Second pre-registered run (seeds 50–59): RQ3 and RQ4

Rules: [`refinement_v2_confirmatory/PREREGISTRATION.md`](refinement_v2_confirmatory/PREREGISTRATION.md).
Results: [`refinement_v2_confirmatory/summary.md`](refinement_v2_confirmatory/summary.md), written by
`scripts/run_refinement_v2.py`.

- **Split:** training rows 50%, unlabelled pool 20%, acceptance rows 10%, test rows 10%, rows held
  back 10%.
- **Points tried:** up to 300 per method, or a quarter of the pool if smaller.
- **Labels counted:** the methods that buy labels read 821–24,806 labels beyond the training rows
  (bought labels plus acceptance rows).
- **Tests:** paired Wilcoxon over 10 seeds, Holm across the 6 datasets. "No worse than" tests are
  one-sided, with margins of 1 macro-F1 point and 2 points of the lowest class recall.

Test macro-F1 (%), mean over 10 seeds:

| Method | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Starting model (B0) | 84.0 | 82.0 | 67.6 | 72.6 | 74.3 | 93.1 |
| Self-repair (SR, no new labels) | 90.0 | 83.5 | 70.1 | 75.9 | 76.6 | 98.8 |
| Active learning (AL) | 90.3 | 83.9 | 72.4 | 77.7 | 76.3 | 99.2 |
| SR then AL | 90.5 | 84.5 | 72.4 | 77.9 | 77.3 | 99.3 |
| CFMNN retrained on the same labels, shuffled | 85.7 | 82.9 | 64.2 | 71.1 | 74.7 | 91.2 |
| 10-NN, same labels | 58.4 | 87.9 | 68.2 | 80.2 | 93.5 | 99.9 |
| Random Forest, same labels | 97.7 | 89.6 | 82.4 | 86.8 | 95.9 | 99.9 |

**RQ3: with every label counted, do bought labels add to self-repair?**

- **Self-repair helps on every dataset** (S3a): +1.5 to +6.0 macro-F1 points over the starting model,
  significant on all 6. It closes 10–83% of the gap to the Random Forest.
- **Bought labels add less, on 3 of 6** (P3, primary). SR then AL beat SR on Landsat (+1.08), MAGIC
  (+2.03) and Letter (+0.73). The Landsat and Letter results are only just significant (Holm p = 0.049).
- **"SR no worse than AL, within 1 point"** (S3b) was shown only on Letter. On Page Blocks and MAGIC
  SR trailed by 2.3 and 1.8 points.
- **Lowest class recall** (S3c, S3d): SR then AL kept it within 2 points of the start on 4 of 6
  (Thyroid +20.5, Skin +17.9), but not on Page Blocks or MAGIC. **SR alone lowered MAGIC's lowest
  class recall by 2.7 points, in 9 of 10 seeds.**
- **Retraining:**
  - SR then AL beat shuffled retraining by 1.7–8.2 points on all 6 (S4c).
  - Against the starting model, shuffled retraining changed macro-F1 by −3.4 to +1.8 points, never
    significantly (post hoc).
  - Appended retraining changed it by −2.6 to +0.9 points, significantly only on Letter (+0.85, post
    hoc).
- **The gap to other models remains:** the Random Forest stays 5–19 points ahead on 5 of 6.

**RQ4: certificates and targeted change.** Certificates were computed on up to 2,000 test rows (548
to 2,000 per dataset), and PACE-flip's explanations on 100 of them.

| Measure (mean over 10 seeds) | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Fix share, SR then AL (%) | 85.3 | 64.2 | 81.9 | 64.8 | 66.2 | 94.1 |
| Fix share, retrained, shuffled (%) | 51.8 | 52.4 | 42.6 | 47.2 | 50.9 | 40.5 |
| Broken, SR then AL (% of test rows) | 0.19 | 2.36 | 0.82 | 6.85 | 3.06 | 0.22 |
| Broken, retrained, shuffled (% of test rows) | 1.03 | 6.09 | 6.81 | 15.60 | 10.48 | 3.83 |
| Broken, retrained, appended (% of test rows) | 0.39 | 1.77 | 3.23 | 2.97 | 2.33 | 0.49 |
| Tight certificate per edit, SR then AL (% of test rows) | 1.2 | 1.9 | 34.6 | 7.9 | 1.5 | 4.1 |
| Certified rows that changed, SR then AL (%) | 0.7 | 3.4 | 0.2 | 1.9 | 3.7 | 0.7 |

- **No prediction changed outside its certificate** (C4): 0 in 46,153 accepted edits, for both the
  tight and the plain certificate.
- **The certificates are loose:**
  - an edit's tight certificate held 1.2–35% of the test rows;
  - only 0.2–3.7% of those rows changed;
  - over a whole repair, 5–74% of rows fell inside some edit's certificate.
- **They do not bound explanation changes:** among the explained rows outside every tight
  certificate, PACE-flip's top 3 still changed on up to 51% (MAGIC).
- **Repair's changes are mostly corrections** (P4):
  - the fix share beat shuffled retraining on all 6, by 12–54 points;
  - repair also broke fewer predictions (S4a);
  - it changed fewer explanations (S4b; Skin left out, as planned).
- **The order of retraining matters:**
  - shuffled retraining changed 2.4–30% of test predictions, appended retraining 0.7–8%;
  - against appended retraining, repair still had the higher fix share on all 6 datasets, but **broke
    more predictions on Landsat, MAGIC and Letter.**
- SR then AL changed PACE-flip's top 3 on 26–78% of the explained rows (Skin left out).

### 5.5 Self-repair as the method (written after the results; no new run)

Self-repair (SR) is the repair method. Active learning (AL) and SR then AL are kept as the
comparison that answers RQ3: do bought labels add anything? Everything below comes from saved
results: RQ3's `runs.csv` for the metrics, labels, churn and certificates, and Run T
([`repair_timing/`](repair_timing/README.md), §12) for the time. Tables:
[`sr_method/tables.md`](sr_method/tables.md), written by `scripts/sr_method_tables.py`. Checked by
`research_audit/scripts/claims_sr_method.py` (51 checks, all OK). Paper: §9 "Self-repair as the
method", §10 "Self-repair alone (post hoc)" and the conclusion (doc rev 453).

Gain over the starting model (points, mean of paired differences over seeds 50–59). `*`: a
pre-registered two-sided test, Holm p < 0.05 (S3a for SR over B0, P3 for SR then AL over SR). The
other cells are means and were not tested.

| Macro-F1 | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Starting model (%) | 84.0 | 82.0 | 67.6 | 72.6 | 74.3 | 93.1 |
| SR | +6.0* | +1.5* | +2.6* | +3.3* | +2.2* | +5.7* |
| AL | +6.4 | +1.9 | +4.9 | +5.1 | +1.9 | +6.1 |
| SR then AL | +6.6 | +2.6 | +4.9 | +5.3 | +3.0 | +6.2 |
| SR then AL, over SR | +0.6 | +1.1* | +2.3 | +2.0* | +0.7* | +0.6 |

| Lowest class recall | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Starting model (%) | 59.2 | 47.3 | 52.4 | 71.9 | 49.7 | 80.9 |
| SR | +19.4 | +6.4 | +0.1 | −2.7 | +2.9 | +16.7 |
| AL | +19.2 | +8.0 | −1.9 | +0.4 | +1.9 | +17.4 |
| SR then AL | +20.5 | +9.1 | −1.3 | −1.2 | +3.8 | +17.9 |

| Cost | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Labels beyond the training rows, AL and SR then AL | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| Seconds, SR (cached gate; Run T, seed 50) | 0.2 | 1.4 | 0.8 | 2.1 | 4.7 | 2.5 |
| Seconds, SR then AL (Run T, seed 50) | 6.8 | 17.1 | 4.6 | 33.3 | 117.5 | 145.0 |

Accuracy gains are in `sr_method/tables.md` (SR +1.1 to +3.9, SR then AL +1.2 to +5.9).

- **The trade-off:** buying labels on top of SR added +0.6 to +2.3 macro-F1 points, significantly on
  3 of 6 (Landsat, MAGIC, Letter), at a cost of 821–24,806 labels and 5.5–57 times SR's time.
  Leaving AL out saves 82–98% of the time and every bought label.
- **Against SR as the method (stated as plainly):**
  - AL alone was ahead of SR alone by 2.3 points on Page Blocks and 1.8 on MAGIC (S3b, "SR no worse
    than AL within 1 point", passed only on Letter);
  - SR alone lowered MAGIC's lowest class recall by 2.7 points (9 of 10 seeds); AL alone raised it
    by 0.4.
- **Why not a smarter order inside SR?** Run H (§12): greedy is better on MAGIC and Letter, but at
  36–46 times the random order's time there (4.5–132 times over the 6 datasets) and with no gain on
  the other 4. SR keeps the random order.

**RQ4 for SR alone (post hoc).** RQ3's P4, S4a and S4b tests (paired two-sided Wilcoxon over seeds,
Holm across datasets) with SR in place of SR then AL. Not pre-registered; P4's pre-registered result
for SR then AL (above) still stands. The shuffled retrain read every label SR then AL read; SR read
none.

| SR against shuffled retraining | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Fix share, SR (%) | 86.4 | 59.0 | 69.4 | 60.3 | 65.9 | 87.5 |
| Fix share, retrain (%) | 51.8 | 52.4 | 42.6 | 47.2 | 50.9 | 40.5 |
| Fix share, SR − retrain (points) | +34.6* | +6.6 | +26.8 | +13.2* | +15.0* | +47.0* |
| Broken, retrain − SR (points of the test rows) | +0.8* | +3.7* | +6.1* | +8.4* | +8.1* | +3.3* |
| PACE-flip top 3 changed, retrain − SR (points of the explained rows) | +36.3* | +64.5* | +58.1* | +22.9* | +52.6* | – |

- SR's fix share beat retraining on every dataset, significantly on 4 of 6 (Holm p = 0.074 on
  Landsat and Page Blocks; on Page Blocks SR made only 12.5 accepted edits per seed, and one seed's
  fix share was 0%).
- SR broke fewer predictions on all 6 and changed fewer explanations on all 5 tested (Skin has 3
  features, so its top 3 never changes).
- Certificates: 0 misses, tight or plain, over SR's 6,233 accepted edits.

## 6. What we can claim

- **PACE-SHAP computes the Shapley value of CFMNN's margin exactly when d ≤ 12 (otherwise by sampling
  with small, measured error), and is the most faithful explainer in the pre-registered v3 benchmark.**
  - It is more faithful than KernelSHAP on 6 of 11 datasets by deletion (the primary test) and on 7 by
    insertion, and never significantly worse.
  - Against a stronger KernelSHAP (2,048 samples, no feature selection) it is more faithful on 3
    datasets by deletion and 4 by insertion, and never worse.
  - It has the best mean by both tests on 8 of 11 datasets, and is level with the best on the other 3.
  - It is stable: 1.00 when exact, 0.94–0.99 when sampled (KernelSHAP 0.15–0.73, LIME 0.52–0.90).
  - It equals brute-force Shapley values and KernelSHAP with every coalition (unit tests), and the
    pre-registered bug check passed.
  - Its gains come from computing the same quantity better: no estimation error, no feature
    selection, and 200 background rows instead of 20.
- **PACE-flip computes brute-force occlusion exactly, in closed form, and is fully stable.**
  - Under the fair protocol (v2, 10 seeds), it is as faithful as KernelSHAP or better on 5 of 11
    datasets: better on Breast Cancer and Thyroid (both tests) and Landsat (deletion), level on Shuttle
    and Page Blocks. KernelSHAP is more faithful on S1–S3, Wine, MAGIC and Letter.
  - PACE-flip is more faithful than PACE-multi by insertion on 10 of 11 datasets (and by deletion on
    8), and more faithful than LIME by insertion on 5 of 11.
  - **Same as occlusion up to rounding:** the scores agree to floating-point rounding (tested to 1e-9).
    On Thyroid, Letter and Page Blocks, where many scores are near ties, rounding reorders a few
    features, so per-seed AUCs differ by up to 0.002. On the other 8 datasets the results are
    identical.
  - **Speed:** 0.05–1.4 ms per explanation when measured alone, which is:
    - 29–141× faster than occlusion (25–134× when occlusion runs in one batch; §8.6);
    - 89–316× faster than LIME;
    - 292–1,194× faster than KernelSHAP.
  - **How:**
    - each box's terms at the 32 replacement values are tabulated once;
    - an exact bound skips boxes that cannot change the decision or the margin (98% of boxes on S1–S3,
      71–92% on the 8 real datasets);
    - all points and features run in one batched pass.
  - **Same output:** bit for bit identical to the original loop (tested on all 11 datasets × 3 seeds;
    [`speed/summary.md`](speed/summary.md) and [`speed_uci/summary.md`](speed_uci/summary.md)), so
    every faithfulness, stability and ground-truth number is unchanged. It is just 5–28× faster (5× on
    MAGIC, 8–11× on Letter, Landsat and Page Blocks, 12–28× on the other 7).
- **PACE-multi** is faster than PACE-flip in batches (0.02–0.47 ms) and perfectly stable, but less
  faithful than PACE-flip under the fair protocol (by insertion on 10 of 11 datasets). (PACE-contrastive
  and the decomposition are faster than PACE-flip, but less faithful.)
- **PACE decomposition** equals exact Shapley values of the winning box's membership (tested), but only
  for that box's score, not for the final prediction. (PACE-SHAP is the Shapley value of the final
  margin.)
- **Ground truth:** PACE-flip reports which single-feature changes move this model's decision. Only
  42–47% of those decision flips come from the truly relevant features. The Shapley methods and LIME
  rank the relevant features higher: LIME 0.89–0.92, PACE-SHAP 0.79–0.85, KernelSHAP 0.72–0.81,
  PACE-flip 0.49–0.55 (v3).
- **Trust signals** (from the pre-registered confirmatory run):
  - PD + U\* ranks CFMNN's errors better than the plain margin on 6 of 8 units: the synthetic family,
    Skin, MAGIC, Letter, Landsat and Page Blocks. The gain on Digits is not significant (+0.012, Holm
    p = 0.074), and on Thyroid the margin is ahead (−0.031, not significant).
  - U\* beats the blueprint's U on all 6 real datasets with at least 1% tied points. Points tied
    between boxes of two classes are far more often wrong than other points (Thyroid: 38%, against
    3.3% overall), so a useful uncertainty score must rank them first.
  - EI does not improve PD + U\* (the pre-registered AUPRC test). On its own (descriptive) it is
    informative on some datasets (AUROC 0.89 on Thyroid, 0.87 on Skin) and near random on others
    (0.53 on MAGIC, 0.54 on the synthetic family).
  - These signals need no labels and no retraining.
  - **RQ2 v2 (pre-registered, seeds 20–29):**
    - CFMNN's training-order sensitivity (the order ensemble) adds to kNN + Trust Score on 5 of 8
      units: a lot on Thyroid, a little elsewhere, and lower on Digits.
    - The combination is the strongest detector overall: it beats kNN on all 8 units and Trust Score
      on 7. Only on Digits is another detector clearly better.
    - It needs a labelled calibration set.
- **Counterfactuals** (DiCE set up fairly, 11 datasets; tested over B1's 10 seeds, Section 8.5):
  - The model confirms 99–100% of PACE's counterfactuals and 97–100% of DiCE's.
  - PACE's counterfactuals land closer to real data of the target class on all 11 datasets.
  - They make a smaller change than DiCE on 5 of the 8 real datasets (Wine, Thyroid, Shuttle, MAGIC
    and Letter), with no significant difference on the other 6.
  - They are about 70–1,000× faster.
  - DiCE changes fewer features on 10 of 11 (no significant difference on S3).
- **PACE-sparse counterfactuals** (pre-registered, seeds 40–49, Section 4.2):
  - they change fewer features than DiCE on 10 of 11 datasets;
  - they make a smaller change than DiCE on all 11, and than PACE on all 11;
  - they land closer to real target-class data than DiCE's on the 8 real datasets;
  - they are as often valid as DiCE's (level on all 11);
  - they are faster than DiCE on 10 of 11 datasets.
- **Global importance** (10 seeds): PACE-flip averaged over points agrees with permutation importance
  better than mean |KernelSHAP| does on 3 of 11 datasets (Shuttle, Letter, Landsat; Holm p < 0.05),
  with a higher mean on 10 of 11, at a small fraction of the cost.
  - On fresh seeds 40–49 (Section 4.2) it is better on 3 again (Letter, Landsat, Page Blocks), with
    a higher mean on 10 of 11.
  - PACE-flip averaged over every test point and KernelSHAP over 50 (its cost limit). This helps
    PACE-flip: on Page Blocks, at the same 50 points, about a quarter of its lead goes (post hoc).
  - Permutation importance also changes one feature at a time, as PACE-flip does, so part of the
    agreement comes from how the two are built.
  - Normalising each point's PACE-flip scores before averaging improves agreement on MAGIC and Letter
    and lowers it nowhere.
- **Interaction values** (pre-registered, Section 4.2): PACE-SHAP's exact interaction values name
  S3's true AND pair (x0, x1) as the strongest pair in 10 of 10 seeds.
  - They name the same pair on S1 and S2, which have no true interaction.
  - On S3 the pair stands out more: 2.06 times the next pair, against 1.62 on S1 and 1.35 on S2
    (post hoc). S3's range over seeds overlaps S1's but not S2's.
- **Conditional replacement** (pre-registered, seeds 40–49, 13 datasets, Section 4.2):
  - the conditional game's inputs are off the data less often than the interventional game's on all
    13 datasets (outside Shuttle, at most 10% against up to 67%);
  - like for like, on the conditional test, PACE-SHAP-cond is more faithful than KernelSHAP-cond on 6
    of 13 datasets by deletion and on 6 by insertion, and never significantly less faithful;
  - on the same 200 rows, PACE-SHAP is more faithful than our KernelSHAP on the interventional test on
    3 datasets by deletion and 2 by insertion, and never significantly less faithful.
- **Repair** (pre-registered, seeds 50–59, 6 datasets, Section 5.4):
  - every CFMNN training error we checked is a cross-class tie (100% on 6 datasets at seeds 0, 10 and
    50; post hoc), and on all 13 datasets at both θ over seeds 70–89 (Section 10);
  - self-repair, with no new labels, raises test macro-F1 on all 6 datasets (+1.5 to +6.0 points);
  - bought labels add a further 0.7–2.0 points on 3 of 6 (Landsat, MAGIC, Letter);
  - no prediction changed outside an edit's certificate in 46,153 accepted edits;
  - repair's changes are corrections more often than shuffled retraining's on all 6 (fix share
    64–94% against 41–52%).
- **Against other readable models, only modest claims hold (Section 9, pre-registered).** With θ
  chosen by validation, CFMNN is more accurate than logistic regression on 7 of 13 datasets (less on
  4) and than a decision tree on 5 (less on 8). θ chosen by validation is never less accurate than
  θ = 0.3.
- **Repair at θ = 0.3 on a new split (Section 10, pre-registered).** Self-repair at θ = 0.3 raises
  accuracy on 7 of 13 datasets and is never worse. By macro-F1 it is better on all 6 of RQ3's
  datasets (+1.7 to +5.6 points). It never added a box.
- **Self-repair against contraction (Section 13, pre-registered, seeds 120–129).** Self-repair is
  more accurate by macro-F1 than Simpson's contraction on every overlap on all 6 datasets (+2.4 to
  +22.5 points), and than the same contraction at the same training errors, with no gate, on all
  6 (+1.7 to +9.0). On Thyroid Simpson's test finds no overlap, so there the primary comparison is
  self-repair against the starting model.
- **Proposition 1 at other γ and on Covertype (Sections 15 and 17).** At γ = 2, 4 and 8, 0 of
  569,239 training errors are not ties inside a box of both classes; on Covertype, all 1,128,010
  are.
- **PACE-SHAP on contracted boxes (Section 15).** On CFMNN's boxes after Simpson's contraction it
  matches shap's ExactExplainer to within 1.6e-10 (the plan's limit: 1e-9). Post hoc, the
  differences come from ExactExplainer, not from PACE-SHAP: it skips a model call when a
  background value is within `np.isclose`'s tolerance of the point's (Section 15).
- **Above 12 features at an equal budget (Section 16, pre-registered, seeds 160–169).** By
  deletion, PACE-SHAP is better on 1 dataset (Landsat), level on 4 and worse on 0 against
  PermutationSHAP, and the same against paired KernelSHAP. Its error against exact values is
  28–40% of theirs, and, timed alone, the two samplers take 3.8–9.6 times as long.
- **PACE-SHAP on Simpson's classifier trained from scratch (Section 19, a check).** It matches
  shap's ExactExplainer to within 1.4e-10 (the plan's limit: 1e-9) on 6 datasets. Post hoc, the
  one difference above 3.1e-16 is the same shap shortcut as in Section 15.
- **Covertype at a smaller θ (Section 18, descriptive, a 10% share of the rows).** θ by validation
  (0.05, the end of the grid) raised accuracy by +20.4 to +24.8 points over θ = 0.3, with 34–42
  times as many boxes; self-repair then added +0.1 to +0.6, and +0.5 to +5.3 at θ = 0.3, on
  every seed, without adding a box.
- **Covertype among the samplers (Section 20, descriptive).** At 54 features PACE-SHAP was within
  0.0067 of PermutationSHAP and paired KernelSHAP by deletion on every seed, and, timed alone,
  they took 4.6 and 13.9 times as long.

## 7. What we must not claim

- That PACE-flip is the most faithful explainer. Under the fair protocol KernelSHAP is more faithful
  on S1–S3, Wine, MAGIC and Letter. The v1 "most faithful on all 7" result came from ranking signed
  attributions by |attribution|.
- That PACE-SHAP is the most faithful on every dataset. On Thyroid, Shuttle and Page Blocks it is level
  with the best (another method's mean is ahead by at most 0.009, not significantly), and on S1, S2 and
  MAGIC its deletion advantage over KernelSHAP is not significant.
- That PACE-SHAP measures something KernelSHAP cannot. Both target the same Shapley value; KernelSHAP
  with the same 200 rows and every coalition would give the same numbers, only more slowly.
- That PACE-SHAP is faster than KernelSHAP. With 200 rows it took 1.2–3.0× as long on 9 of 11
  datasets in the timing run measured alone. PACE-SHAP-20 is the fast variant (4.3–14× faster than
  KernelSHAP), and it is never significantly less faithful than KernelSHAP.
- That PACE-SHAP's sampled mode is exact. When d > 12 it samples (stability 0.94–0.99).
- That PACE-flip is more faithful than occlusion. It computes the same scores, up to floating-point
  rounding; its contribution is exact speed.
- That PACE-flip equals occlusion bit for bit. Bit-for-bit equality holds against the original
  PACE-flip loop only; against occlusion it holds up to rounding (see above).
- That PACE-flip is faster than PACE-multi in batches. There PACE-multi is still 1.9–6.0× faster per
  point; PACE-flip is faster only when points are explained one per call.
- That every method beats the random ranking by a wide margin. On Thyroid and Page Blocks the
  decomposition's deletion margin AUC is within the seed-to-seed spread of random (v2: 0.725 vs 0.742
  and 0.145 vs 0.178).
- That LIME and SHAP "describe the data rather than the model". KernelSHAP explains the same model's
  margin and is more faithful than PACE-flip on the synthetic sets; the methods measure different
  things.
- Absolute milliseconds as general facts. They depend on the machine; compare methods measured
  together.
- That PACE recovers the data's true features better than LIME. LIME's top 3 are most often the truly
  relevant features (0.89–0.92, against 0.79–0.85 for PACE-SHAP). PACE-SHAP's lead over KernelSHAP
  here (0.79–0.85 vs 0.72–0.81) is descriptive, not tested.
- That SHAP equivalence holds for the final prediction.
- That results hold on Sepsis-like data. Sepsis is dropped because its labels are not predictable from
  its 3 features.
- That EI or the full PRS improves on PD + U\*. In the confirmatory run PRS\* has a lower mean AUPRC
  on all 8 units, significantly on 6, and is never significantly higher. Two descriptive exceptions
  are given in 3.2.
- That PACE's signals are competitive error detectors on their own. Trust Score, kNN disagreement and
  the order ensemble, which all read the training labels, beat PD + U\* by wide margins on 7 or 8 of 8
  units. PD + U\* beats Trust Score and kNN disagreement only on Thyroid, and never the order ensemble.
- That the geometry signals (U\*, box evidence) improve a combined detector. In RQ2 v2 they add nothing
  once kNN and Trust Score are present.
- That the combined detector is label-free. It is fitted on labelled calibration rows.
- That the signals are cheap. PD, EI and U together took longer than the three baselines together on
  14 of 16 datasets.
- That U\* was in the blueprint. It was designed in the exploratory run after the G1 check. Only the
  confirmatory run tests it as planned.
- That any signal is a probability of error, or detects distribution shift in general. They only rank
  points, and the shifted set is one synthetic shift.
- That the RQ2 results cover Wine, Breast Cancer or Shuttle. They have too few errors to test. (Digits
  is covered in the confirmatory run, through 5-fold cross-validation.)
- That PACE counterfactuals are sparser than DiCE's. DiCE changes fewer features than PACE on 10 of 11
  datasets. PACE-sparse is the sparse variant.
- That PACE-sparse is sparser than DiCE everywhere. On Shuttle DiCE changes fewer features (1.37 vs
  2.31).
- That PACE-sparse finds the fewest features possible. It is exact only within its own moves towards
  target boxes; on Shuttle, DiCE found single-feature answers outside them.
- That PACE-sparse is sparser than DiCE in general. DiCE ran with its random search method; stronger
  sparse searches were not tried.
- That PACE-sparse costs nothing. Its answers are farther from real target-class data than PACE's on
  every dataset (5–55%). It is 1.7–460× slower than PACE, and on Breast Cancer 3.5× slower than DiCE.
- That mean |PACE-SHAP| ranks features globally better than mean |KernelSHAP|. They are level on all 11
  datasets.
- That interaction values show the data's true interactions. On S1 and S2, whose true rule has none,
  the strongest pair still interacts in the model. They describe the model's boxes. So the S3 test
  does not show by itself that they detect true interactions.
- That conditional explanations are more faithful than interventional ones. At σ = 0.1, PACE-SHAP-cond
  was less faithful than PACE-SHAP on the conditional test on 10 of 13 datasets by deletion (9 by
  insertion), and on the interventional test on all 13.
- That conditional explainers give a correlated feature more credit. On S1c and S2c they gave the
  noisy copy x3 less credit than the interventional explainers did. Only post hoc, at wider σ, did
  PACE-SHAP-cond give it slightly more (by under 0.01).
- That the conditional game's realism is all genuine. For large coalitions it mostly copies one
  training row, so its inputs are near the data partly by construction: on 10 of 13 datasets they are
  off less often than real test points.
- That PACE-SHAP-cond beats KernelSHAP-cond because it is exact. It was also better on sampled
  datasets, with the largest gaps there (Breast Cancer, Landsat), and level on 5 of the 8 exact
  datasets by deletion.
- That the B3 results hold for other settings. Only σ = 0.1 was pre-registered, with a diagonal
  covariance and 200 rows. A post hoc check shows the results depend on σ. With wider weights the
  conditional explainer catches up with PACE-SHAP, and on Shuttle and MAGIC it gets slightly ahead on
  the conditional test. On S1 they also depend on the number of rows: 4 times as many halve the loss.
- That a wider σ makes conditional explanations better. The check has three limits:
  - it chose σ after seeing B3's results;
  - it used 6 datasets, 5 seeds and no tests;
  - wider weights move the inputs off the data.
- That the interventional faithfulness tests measure realistic behaviour everywhere. On Shuttle, Letter
  and Landsat, 63–70% of the inputs they evaluate are off the data.
- That DiCE's counterfactuals are often invalid on this model. That came from its original setup; set
  up fairly it is 97–100% valid (B1's 10 seeds).
- That PACE's change is smaller everywhere. Tested over seeds, it is not significantly different from
  DiCE's on S1–S3, Breast Cancer, Landsat and Page Blocks.
- That PACE's global importance agrees significantly better with permutation importance than SHAP does
  in general. With 10 seeds the difference is significant on 3 of 11 datasets; on MAGIC SHAP's mean is
  higher.
- That PACE-flip's global ranking beats mean |KernelSHAP| at equal numbers of points. PACE-flip used
  every test point and KernelSHAP 50; on Page Blocks about a quarter of the lead came from that.
- That the explanation-guided cut beats simpler cuts. In the first pre-registered repair run it was
  no better than the thinnest cut on any dataset (Section 5.2).
- That bought labels are not worth buying, or that self-repair is as good as active learning. Bought
  labels added significantly on 3 of 6 datasets, and self-repair was shown no worse than active
  learning only on Letter (Section 5.4).
- That self-repair never hurts a class. It lowered MAGIC's lowest class recall by 2.7 points.
- That the certificates are tight or cover explanations. They are exact but loose (0.2–3.7% of
  certified rows changed), and explanations changed outside them.
- That repair breaks fewer predictions than any retraining. It broke more than appended retraining on
  Landsat, MAGIC and Letter.
- That the order ensemble's gain is specific to CFMNN. A bagged ensemble of CFMNNs gave the same
  gain on all 8 units (Section 8.3).
- That PACE-SHAP uses no random sampling. Above 12 features it samples orders; at 32 orders its error
  against exact values is 1–2% (Section 8.1).
- That PACE-SHAP beats KernelSHAP when both use the same rows. With the same 200 rows it is better on
  3 of 13 datasets by deletion and 2 by insertion, level on the rest and never worse (Section 8.5).
- That CFMNN is as accurate as standard models. It is less accurate than the better of 10-NN and a
  Random Forest on all 13 datasets (Section 8.4).
- That CFMNN is as accurate as other readable models. At θ = 0.3, an EBM with main effects only is
  more accurate on 12 of 13 datasets and a decision tree on 10. With θ tuned, they are still more
  accurate on 9 and 8 (Section 9).
- That θ = 0.3 is the best setting. θ chosen by validation is more accurate on 7 of 13 datasets, by
  up to 20 points. On 6 of them it needs 6–23× as many boxes (Section 9).
- That self-repair makes up for a smaller θ. θ chosen by validation, without repair, is more
  accurate than repair at θ = 0.3 on 7 of 13 datasets, by up to 17.5 points (Letter). Repair at
  θ = 0.3 is more accurate only on Thyroid (Section 10).
- That repair adds much at the θ chosen by validation. It is more accurate there on 3 of 13
  datasets, by at most 1.6 points, and level on the rest (Section 10).
- That CFMNN's rules are fewer or shorter than a tree's. The tuned tree has fewer leaves than
  CFMNN at θ = 0.3 has boxes on 10 of 13 datasets. Each leaf tests 2.3–9.3 features on average,
  while each box limits 79–100% of them (Section 9).
- "Never less faithful" for a comparison with level results. Write "never significantly less faithful".
- That self-repair is more accurate than GFMM. Against GFMM trained from scratch it is better on 3
  datasets (Thyroid, MAGIC, Skin), level on 2 and worse on Letter (76.9 against 82.3). Its wins over
  GFMM's rule applied to CFMNN's boxes are not wins over GFMM (Run F's Amendment 1; Section 13).
- That self-repair keeps more training rows inside boxes than contraction. It evicts more than
  Simpson's contraction on every overlap on Thyroid, Landsat and Letter, and fewer on MAGIC and
  Skin (Section 13).
- That CFMNN + PACE-SHAP is a better system than an EBM or a decision tree with its own exact
  explanation. It is less accurate on 10 and 9 of 11 datasets and slower to explain on all 11. By
  the lead it is better on 2, level on 6 and worse on 3 against the EBM, and better on 3, level on
  4 and worse on 4 against the tree; by the undivided gain it is worse on 10 and 9 (Section 14).
  The paper's claim is for hyperbox models.
- That the lead verdicts on Letter, MAGIC and Page Blocks are settled. Letter's depends on the tied
  points (worse on all points, better on untied points); MAGIC and Page Blocks have too few untied
  points to check (Section 14).
- That PACE has been tested in full on Simpson's classifier. On CFMNN's boxes after Simpson's
  contraction (Section 15) and on Simpson's classifier trained from scratch (Section 19), only
  PACE-SHAP's exact values were checked, on 6 datasets with at most 12 features; no faithfulness,
  trust or repair test ran on either. Our Simpson's classifier also uses CFMNN's size limit and
  expansion.
- That Simpson's classifier is less accurate than CFMNN in general. It was lower on 4 datasets,
  higher on 3 and within 0.3 points on 6, with no test (Section 19).
- That γ = 4 is the most accurate γ. γ = 2 is more accurate on S1 (+2.1) and Digits (+1.0); γ stays
  4 because the plan fixed it (Section 15).
- That PACE-SHAP's sampled mode is more faithful than modern samplers. By deletion it is better on
  1 dataset (Landsat) and level on 4, and the differences outside Landsat are at most 0.0006
  (Section 16).
- That CFMNN or PACE scales to large data. One large dataset was tested: on Covertype CFMNN reaches
  51.4% test accuracy, against 92.7% for 10-NN and 95.6% for a Random Forest (Section 17). On a
  10% share, θ = 0.05 reached 72.9%, with about 3,010 boxes and 9.3 s per PACE-SHAP explanation
  (Section 18).
- That self-repair is the fix on Covertype. At θ = 0.05 it added +0.1 to +0.6 points, against
  +20.4 to +24.8 from choosing θ (Section 18).
- That PACE-SHAP is more faithful than the samplers at 54 features. Run N is descriptive and its
  per-seed differences go both ways (Section 20).

## 8. Review fixes (pre-submission review, 2026-11-01)

A pre-submission review of the paper asked for four new tests and a re-analysis of existing results.
The tests' rules were pushed before any run: [`review_v1/PREREGISTRATION.md`](review_v1/PREREGISTRATION.md).
Tests 2 and 3 are confirmatory, Tests 1 and 4 descriptive, and the re-analysis is post hoc. As
everywhere else, tests are paired over seeds (Wilcoxon), with Holm across datasets.

### 8.1 Exact Shapley values computed in general ways (Test 1, descriptive, seeds 30–32)

[`review_v1/exact/summary.md`](review_v1/exact/summary.md)

- **The same values.** On the six datasets with at most 12 features, PACE-SHAP's values equal shap's
  ExactExplainer and this repository's KernelSHAP solver given every coalition. The largest difference
  is 3.3e-16, which is rounding error.
- **16–30× faster** per explanation, measured alone.
  - S1: 2.5 s against 53.7 s for ExactExplainer.
  - MAGIC: 0.38 s against 11.4 s.
  - Shuttle: 19 ms against 478 ms.
  - Page Blocks: 68 ms against 1.1 s.
  - The KernelSHAP solver with every coalition is about as slow as ExactExplainer (16–27×).
- **The sampled mode is close to exact.** At v3's 32 orders per row, the relative L1 error against
  exact values is 0.010 on Wine (108 points) and 0.017 on Letter (5 points). The top-3 overlap is 0.995
  and 1.0.
  - From 8 to 64 orders, the error falls from 0.020 to 0.007 on Wine, and from 0.028 to 0.010 on
    Letter.
  - On Letter, exact values take 41.7 s per explanation and the sampled mode 0.9 s.

### 8.2 Stability, ties and LIME (Test 2, pre-registered, seeds 50–59)

[`review_v1/faithfulness/summary.md`](review_v1/faithfulness/summary.md). PACE-SHAP is never worse in any
comparison.

- **Stability (both primary).** PACE-SHAP is more stable than KernelSHAP on 11 of 11 datasets when each
  method keeps its rows. It is more stable on 10 of 11 when every method gets new rows (level on Page
  Blocks). So v3's stability gap did not come only from re-drawing KernelSHAP's rows.
- **Raw margin AUCs** (not divided by each point's own margin): better on 10 of 11 by deletion (level on
  Page Blocks) and on 11 of 11 by insertion.
- **Untied points only** (the 9 datasets with at least 10 in every seed): better on 7 by deletion (level
  on Wine and Shuttle) and on 6 by insertion (level on S2, Wine and Shuttle).
- **LIME on the explainer rows,** as v3's pre-registration said: PACE-SHAP is better on 9 of 11 by
  deletion (level on Shuttle and Page Blocks) and on 11 of 11 by insertion.
- **v3's comparison on the new seeds:** better on 6 of 11 by deletion and by insertion (S1, S3, Breast
  Cancer, Thyroid, Letter and Landsat), and level on the rest. This repeats v3.

### 8.3 Generic ensembles in RQ2 (Test 3, pre-registered, seeds 60–69, 8 units)

[`review_v1/ensembles/summary.md`](review_v1/ensembles/summary.md). AUPRC, with each ensemble added to
kNN + Trust Score.

- **Primary: the order ensemble is no better than a bagged ensemble of CFMNNs** (level on all 8 units).
  Its gain is not specific to CFMNN's sensitivity to the order of the training rows.
- **It still adds beyond bagging on 4 of 8 units** (the synthetic family, Thyroid, MAGIC and Letter).
- **v2's finding repeats on 5 units** (the synthetic family, Thyroid, MAGIC, Letter and Landsat). It is
  worse on Digits (−0.016, lower in 10 of 10 seeds) and level on Skin and Page Blocks.
- **An order ensemble of single-pass logistic regressions** helps a lot on the synthetic family
  (+0.083), where a linear model is far more accurate than CFMNN, and nowhere else. It is slightly
  worse on Skin (−0.001).

### 8.4 Accuracy of the explained model (Test 4, descriptive, seeds 0–4)

[`review_v1/accuracy/summary.md`](review_v1/accuracy/summary.md)

- CFMNN is less accurate than the better of 10-NN and a Random Forest (200 trees) on all 13 datasets.
- The gap is small on Shuttle (0.2 points), Wine (1.7), Breast Cancer (1.9), Thyroid (2.5) and Digits
  (2.7).
- It is large on Letter (21.5), MAGIC (15.8), S3 (13.0), Page Blocks (11.0), S1 (10.0) and S2 (9.5).
  Landsat (7.4) and Skin (5.2) are in between.
- So PACE explains a transparent model that is often clearly less accurate than standard ones.

### 8.5 Re-analysis of existing results (post hoc)

[`review_v1/reanalysis.md`](review_v1/reanalysis.md), made by `scripts/review_reanalysis.py`.

- **KernelSHAP on the same 200 rows** (B3's data). PACE-SHAP is better on 3 of 13 datasets by deletion
  (S3, Breast Cancer and Landsat) and on 2 by insertion (Breast Cancer and Landsat). It is level on the
  rest, and never worse. KernelSHAP took 3.5–10× as long on every dataset (Section 4.2; timed while
  other runs shared the CPU; `research_audit/scripts/check_matched_speed.py`). Most of v3's lead came from KernelSHAP's 20 rows and its feature selection.
- **Absolute AUCs and effect sizes** for every method and dataset are in the re-analysis. The paper's
  table after Figure 2 gives the mean deletion AUCs.
- **Figure 2's intervals.** With Bonferroni-adjusted intervals, the deletion intervals that exclude
  zero are exactly the Holm-significant datasets.
  - The unadjusted intervals also exclude zero on S2 and MAGIC, where Holm finds no difference.
  - By insertion, S2 and Wine are significant by Holm, but their Bonferroni intervals reach zero. The
    t-interval and the Wilcoxon test differ.
- **Stability with 20 rows on both sides.** PACE-SHAP-20, which re-draws 20 rows as KernelSHAP does,
  is more stable than KernelSHAP on 5 of 11 datasets and level on 6.
- **Tied points** (margin 0) among v3's explained points: Page Blocks 79%, MAGIC 75%, Letter 55%,
  Landsat 25%, Shuttle 11%, Thyroid 5%, Breast Cancer 1%, and the rest 0.
  - With a margin below 0.01: Shuttle 94%, Page Blocks 90%, MAGIC 89%, Thyroid 79%.
- **Counterfactuals tested over seeds** (B1's data):
  - PACE's change is smaller than DiCE's on 5 of the 8 real datasets (Wine, Thyroid, Shuttle, MAGIC
    and Letter).
  - DiCE changes fewer features on 10 of 11.
  - PACE lands closer to real data on 11 of 11.
  - PACE is 70–1,043× faster (median time per counterfactual; several runs shared the CPU, so
    compare within a dataset only).
  - The model confirms 99–100% of PACE's counterfactuals and 97–100% of DiCE's.
  - The paper's counterfactual table now comes from this comparison:
    [`review_v1/counterfactual_table.md`](review_v1/counterfactual_table.md), made by
    `scripts/counterfactual_table_seeds.py`, on the points where both are valid (27–30 of 30 per seed).
- **The geometry signals** add significantly only on Letter (+0.0015).
- **Conditional PACE-SHAP** beats PACE-SHAP on the conditional deletion test only on Shuttle.
- **Figure 4's leads** (as ratios): S1 1.16–2.01, S2 1.29–1.53 and S3 1.58–2.80. S2's and S3's ranges do
  not overlap.
- **Page Blocks' share of errors** differs between the two RQ2 runs because of seed variation.

### 8.6 Occlusion in one batch (post hoc, seeds 0–2)

The occlusion baseline behind PACE-flip's speed-up (`xai_baselines.occlusion`) calls the model once
per point and feature. The review suggested checking whether that per-call overhead inflates the
speed-up. `xai_baselines.occlusion_batched` sends the modified inputs of all points in one call (a
unit test checks it gives the loop's scores). `scripts/time_occlusion_batched.py` times PACE-flip
and both occlusions on the paper's timing setup (11 datasets, seeds 0–2, 50 test points, 36 for
Wine), each alone: [`review_v1/occlusion_batched/summary.md`](review_v1/occlusion_batched/summary.md).

- **Batching saves little:** occlusion becomes 1.0–2.2× faster. Its time goes into the model calls
  themselves, not the per-call overhead.
- **PACE-flip is 25–134× faster than batched occlusion,** and 28–156× faster than point-by-point
  occlusion in the same run (the paper's timing table gave 29–141×).
- **Same scores:** the three differ by at most 5.4e-15.
- **In the paper:** the abstract and conclusion now give the batched range; the Speed section gives
  both.

## 9. Readable baselines and θ chosen by validation (pre-registered, seeds 70–89)

Rules: [`readable_baselines/PREREGISTRATION.md`](readable_baselines/PREREGISTRATION.md), pushed in
`42a71d4` before the run. Results: [`readable_baselines/summary.md`](readable_baselines/summary.md).
Every number below is recomputed from `runs.csv` by `research_audit/scripts/claims_readable.py`,
whose verdict counts match the run's own summary exactly.

The run took 11 h 15 min with six processes, from 2026-11-06 17:13 to 2026-11-07 04:28. It wrote
2,080 rows (13 datasets × 20 seeds × 8 models), with no failures and no deviations from the plan.

**Questions.** Is CFMNN as accurate as other models people can read, when each is tuned the same
way? Was θ = 0.3 a poor choice?

**Setup.**
- **Data:** the paper's 80/20 split, the seed's row order, and scaling fitted on the training rows.
- **Tuned models** pick a setting by stratified 3-fold cross-validation on the training rows. The
  folds keep the row order, and the simplest setting wins ties. The grids:
  - CFMNN-val: θ from 0.5 down to 0.05;
  - the tree: 4–1,024 leaves;
  - logistic regression: C from 0.01 to 100;
  - kNN: k from 20 down to 1.
- **Untuned models:**
  - EBM, with main effects only;
  - GFMM, at θ = 0.3 like the explained model;
  - the Random Forest, with 200 trees.

  EBM's pair terms and GFMM's tuning were dropped before the run because they took too long.
- **Tests:** a paired Wilcoxon test over the 20 seeds, with Holm correction across the 13 datasets.

**Results (accuracy unless stated).** Counts are of datasets, for the first model named.

| Code | Comparison | Better | Level | Worse |
|---|---|---|---|---|
| P1 | CFMNN-val vs tree | 5: Wine, Breast Cancer, Digits, Letter, Landsat | 0 | 8: S1, S2, S3, Thyroid, Shuttle, Skin, MAGIC, Page Blocks |
| P2 | CFMNN-val vs logistic regression | 7: S3, Thyroid, Shuttle, Skin, MAGIC, Letter, Landsat | 2: Wine, Page Blocks | 4: S1, S2, Breast Cancer, Digits |
| P3 | CFMNN-val vs EBM | 2: Skin, Letter | 2: Wine, Landsat | 9: the rest |
| P4 | CFMNN-0.3 vs GFMM-0.3 | 4: S3, Wine, Thyroid, Shuttle | 3: S1, S2, Breast Cancer | 6: Digits, Skin, MAGIC, Letter, Landsat, Page Blocks |
| S1 | CFMNN-0.3 vs tree | 3: Wine, Breast Cancer, Digits | 0 | 10: the rest |
| S1 | CFMNN-0.3 vs logistic regression | 4: S3, Thyroid, Shuttle, Skin | 2: Wine, Landsat | 7: S1, S2, Breast Cancer, Digits, MAGIC, Letter, Page Blocks |
| S1 | CFMNN-0.3 vs EBM | 0 | 1: Wine | 12: the rest |
| S2 | CFMNN-val vs CFMNN-0.3 | 7: S1, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks | 6: the rest | 0 |
| S3 | P1, P2, P3, P4 by macro-F1 | 5, 7, 2, 4 | 0, 2, 3, 5 | 8, 4, 8, 4 |

- **The tree.**
  - Tuned CFMNN is mixed against it: better on 5 datasets, worse on 8.
  - Two of the losses are tiny: Shuttle −0.03 points and Skin −0.15.
  - The others run from 1.5 points (Page Blocks) to 11.9 (S3).
  - At θ = 0.3, CFMNN loses to the tree on 10 of 13.
- **EBM** is the strongest readable model here.
  - Tuned CFMNN loses to it on 9 datasets, by up to 11.9 points.
  - At θ = 0.3 it loses on 12, by up to 14.2 points (MAGIC). It is level only on Wine.
- **Logistic regression.** Tuned CFMNN mostly beats it, on 7 of 13 datasets, by up to 17.8 points
  (Letter).
- **GFMM at the same θ** is mixed: CFMNN is better on 4 datasets (by up to 11.3 points, on Shuttle)
  and worse on 6 (by up to 9.0, on Letter).
- **θ = 0.3 was a poor choice on the large datasets.**
  - θ chosen by validation was better on 7 datasets, by 0.2–20.0 points: Letter 20.0, Page Blocks
    10.3, MAGIC 10.1, Landsat 4.9, Skin 3.8, S1 1.5 and Shuttle 0.2.
  - It was level on the other 6 and never worse.
  - On 6 of the 7, it chose θ = 0.05, the smallest value tried, in 13–20 of the 20 seeds. Those
    models had 6–23× as many boxes: Letter 5,325 against 565, and MAGIC 5,728 against 286.
  - On S1 it chose larger boxes (θ = 0.5 in 17 seeds) and used about half as many.
- **The reference models** are the better of the tuned kNN and the Random Forest, per dataset.
  - Tuned CFMNN trails them on all 13 datasets, by up to 11.9 points (S3).
  - At θ = 0.3, CFMNN ties the Random Forest on Wine (both 98.3%). It trails elsewhere by
    0.2–21.3 points.
- **Size (descriptive).**
  - The tuned tree has fewer leaves than CFMNN at θ = 0.3 has boxes on 10 of 13 datasets. It has
    fewer than tuned CFMNN on all 13.
  - A leaf tests 2.3–9.3 features on average, while a CFMNN box limits 79–100% of the features.
  - EBM has one curve per feature (3–64 terms). That is a different kind of rule, so it is not
    compared directly.
- **Grid edges (descriptive).** Several tuned models often chose the end of their grid:
  - CFMNN-val chose θ = 0.05 on 6 datasets;
  - logistic regression chose C = 100 on 6;
  - kNN chose k = 1 on 4;
  - the tree chose 1,024 leaves on Letter.

  Wider grids might change some verdicts. This was not tested.
- **Time (descriptive, six processes shared the CPU).**
  - EBM was by far the slowest to fit: about 52 min per fit on Letter, 46 on Skin and 43 on
    Shuttle.
  - GFMM took about 10 min per fit on Letter.
  - CFMNN at θ = 0.3 took 2 s.

**What this means.**
- CFMNN is not the most accurate readable model, nor the smallest.
  - With θ tuned, it beats logistic regression on most datasets, splits with the tree, and loses to
    EBM on most.
  - At θ = 0.3, the setting PACE explains, it loses to both the tree and EBM on most datasets.
- Tuning θ buys accuracy on the large datasets, but only with far more boxes. So CFMNN's
  readability gets weaker exactly where its accuracy improves.
- PACE's case therefore cannot rest on CFMNN's accuracy. It rests on what CFMNN's structure
  allows: exact explanations, and repairs whose reach can be stated.
- θ = 0.3 stays the setting for RQ1–RQ4. The paper reports this run in §7, "Base model".

## 10. Self-repair at the θ chosen by validation (pre-registered, seeds 70–89)

Rules: [`repair_theta/PREREGISTRATION.md`](repair_theta/PREREGISTRATION.md), pushed in `22e687d`
before the run. Results: [`repair_theta/summary.md`](repair_theta/summary.md). Every number below
is recomputed from `runs.csv` by `research_audit/scripts/claims_repair_theta.py`, whose verdict
counts match the run's own summary exactly.

The run took 15 h 4 min with six processes, from 2026-11-08 01:06 to 16:10. The plan expected
about 11 h: six processes on four cores ran each one slower than the timing check, which used one.
It wrote 1,040 rows (13 datasets × 20 seeds × 4 models), with no failures and no deviations from
the plan. All 520 models without repair reproduce the accuracy and macro-F1 of the readable-baselines
run exactly.

**Question.** Section 9 found that θ chosen by validation (θ-val) beats θ = 0.3 on 7 datasets.
Self-repair at θ = 0.3 raised macro-F1 in RQ3. So why repair at θ = 0.3 instead of choosing a
smaller θ? And does repair still help at θ-val?

**Setup.**
- **Data and seeds:** those of Section 9, so each seed's θ-val and its models without repair are
  shared with that run.
- **Models:**
  - B0-0.3 and B0-val: CFMNN at θ = 0.3 and at θ-val, without repair;
  - SR-0.3 and SR-val: the same models after self-repair.
- **Self-repair** is RQ3's SR arm, called the same way:
  - 300 edits in 10 rounds;
  - the volume cut;
  - RQ3's gate, which reads only the training rows;
  - no new labels.
- **Tests:** a paired Wilcoxon test over the 20 seeds, with Holm correction across the 13 datasets.
  - P1 and P2 are primary, on accuracy.
  - On the 17 seeds where θ-val was 0.3, the two θ give the same model.

**Results.** Counts are of datasets, for the first model named.

| Code | Comparison | Measure | Better | Level | Worse |
|---|---|---|---|---|---|
| P1 | SR-0.3 vs B0-val | accuracy | 1: Thyroid | 5: S2, S3, Wine, Breast Cancer, Digits | 7: S1, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks |
| P2 | SR-val vs B0-val | accuracy | 3: Thyroid, Skin, Page Blocks | 10: the rest | 0 |
| S1 | SR-0.3 vs B0-0.3 | accuracy | 7: Thyroid, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks | 6: S1, S2, S3, Wine, Breast Cancer, Digits | 0 |
| S2 | SR-val vs SR-0.3 | accuracy | 8: S1, Thyroid, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks | 5: the rest | 0 |
| S3 | P1, P2, S1, S2 | macro-F1 | 1, 3, 7, 7 | 6, 10, 6, 6 | 6, 0, 0, 0 |

Mean test accuracy (%) on the datasets where something changed:

| Dataset | B0-0.3 | SR-0.3 | B0-val | SR-val |
|---|---|---|---|---|
| Thyroid | 97.36 | 98.44 | 97.38 | 98.93 |
| Shuttle | 99.78 | 99.84 | 99.94 | 99.95 |
| Skin | 95.98 | 99.34 | 99.77 | 99.91 |
| MAGIC | 71.64 | 76.71 | 81.76 | 81.76 |
| Letter | 75.15 | 77.59 | 95.12 | 95.12 |
| Landsat | 85.12 | 86.34 | 90.03 | 90.04 |
| Page Blocks | 85.04 | 87.56 | 95.37 | 95.63 |

- **Repair at θ = 0.3 does not make up for a smaller θ (P1).**
  - It is less accurate than θ-val without repair on 7 datasets, by 0.1–17.5 points: Letter 17.5,
    Page Blocks 7.8, MAGIC 5.1, Landsat 3.7, S1 1.5, Skin 0.4 and Shuttle 0.1.
  - It is more accurate only on Thyroid (+1.1). There θ-val chose larger boxes (θ = 0.4 or 0.5 in
    18 of 20 seeds), so a smaller θ was no help.
  - By macro-F1 the counts are 1, 6 and 6. Shuttle becomes level.
- **Repair adds little at θ-val (P2).**
  - It is more accurate on 3 datasets: Thyroid +1.55, Page Blocks +0.26 and Skin +0.14. It is
    level on the other 10 and never worse.
  - By macro-F1 the counts are the same. Thyroid gains 7.9 points there.
  - On Letter, MAGIC and Landsat, θ-val leaves 2–11 training errors on average (0.04–0.07% of the
    rows). Repair has almost nothing to work on.
- **Repair at θ = 0.3 still helps on this split (S1).**
  - It is more accurate on 7 datasets, by 0.1–5.1 points (MAGIC 5.1, Skin 3.4), and never worse.
  - By macro-F1 it is better on all 6 of RQ3's datasets, by 1.7–5.6 points, and on Shuttle. RQ3
    found 1.5–6.0 on other seeds and another split.
  - It changes nothing on S1, S2, S3, Wine and Digits. There the model at θ = 0.3 has less than one
    training error on average.
- **With repair on both, θ-val is better (S2):** on 8 datasets, and never worse. By macro-F1 it is
  better on 7.
- **Size (descriptive).**
  - Repair never added a box, in all 520 repaired models.
  - θ-val was 0.1 or less in most seeds on Shuttle, Skin, MAGIC, Letter, Landsat and Page Blocks.
    There the models had 6.2–23.2 times as many boxes as at θ = 0.3: Letter 5,325 against 565, and
    Skin 2,013 against 87.
- **Training errors (descriptive).**
  - At θ = 0.3, the model gets 26.6% of MAGIC's training rows wrong, 21.3% of Letter's, 14.4% of
    Page Blocks' and 10.4% of Landsat's. At θ-val, 0.04–1.9% on these.
  - Every training error of all 520 models without repair is a cross-class tie. It is inside the
    winning box of a wrong class, and also has a full box of its own class. That is 347,339 errors
    over 353 models that had any. Section 6 found the same on 6 datasets.
- **Budget (descriptive).**
  - At θ = 0.3, all 300 edits were used on every seed of Skin, MAGIC, Letter, Landsat and Page
    Blocks. MAGIC had 4,048 training errors on average, and Letter 3,403. So repair reached only a
    small share of them. Repairing them all was not tried, because of the time.
  - Some errors had already been put right by earlier edits when their turn came: from 43 of 300
    on Letter to 256 of 300 on Skin.
- **Predictions (descriptive).** Against that θ's model without repair, repair at θ = 0.3 fixed more
  test predictions than it broke, on average, on every dataset where it changed any. For example,
  Skin fixed 1,814 and broke 169, and MAGIC fixed 488 and broke 295.
- **The reference models** are the better of the tuned kNN and the Random Forest (Section 9).
  - Repair at θ = 0.3 still trails them by up to 18.9 points (Letter).
  - θ-val, with or without repair, trails them by up to 11.9 (S3).
- **Time (descriptive; six processes shared the CPU).**
  - Repair at θ = 0.3 took up to 61 min per seed (Letter).
  - At θ-val, Skin took 107 min, because each check of the gate reads 196,000 training rows against
    about 2,000 boxes.

**What this means.**
- The reviewers' question has a plain answer on accuracy. Choosing a smaller θ by validation is
  more accurate than repairing at θ = 0.3, on 7 of 13 datasets, by up to 17.5 points.
- The two do not add up much. At θ-val, repair helps on 3 datasets, by at most 1.6 points.
- What θ = 0.3 keeps is size. Repair adds no box, while θ-val needs 6–23 times as many on six
  datasets. So the case for repairing at θ = 0.3 is a small model that is easier to read, not
  accuracy.
- θ = 0.3 stays the setting for RQ1–RQ4.


## 11. Follow-up tests of the limitations (pre-registered, seeds 110–119, home PC)

After a review, five runs tested the limitations stated in the paper (rev 311). Each was planned, coded, tested and pre-registered, and pushed before it ran. They ran on a desktop (Intel Core i5-14400F, Python 3.14.10), following `research_audit/LIMITATIONS_PLAN.md` and `research_audit/HOME_PC_RUNBOOK.md`. The numbers below are rechecked by `research_audit/scripts/claims_followup.py`.

**E: the cached gate** (`results/fast_gate_check/`).
- **Verdict: PASS.** It replayed RQ3's SR arm (seeds 50–59) with `fast_gate=True`. All 14,827 edit decisions and all 60 test macro-F1 values equal the committed run.
- **Speed:** 2.1–21.9× faster on seed 50, and 11.9× faster in total.

**A: budgets, gates and certificates** (`results/repair_sensitivity/`).

Each arm against SR-default (300 edits, gate 0.5/1.0), by macro-F1:
- **Budgets:** 100 edits was worse on Letter only (−1.14). 1,000 edits was better on Letter only (+1.42), and so was every error (+2.09).
- **No gate:** worse on all 6 datasets, by 0.71 points (Landsat) to 10.29 (Page Blocks).
- **Strict gate (0/0):** worse on 3 (Skin, Letter, Thyroid).
- **Loose gate (1/2):** worse on 1 (MAGIC).
- **Cumulative gate** (0.5/1.0 against the starting model): worse on 2 (Skin, Letter).
- No arm was better.

MAGIC's lowest class recall:

| Model | Lowest class recall (%) |
|---|---|
| B0 | 69.6 |
| SR-default | 68.0 |
| Cumulative gate | 70.8 |

Certificates, over 6,272 accepted edits with 100 test rows each. None missed a change.

| Certificate | Rows flagged per edit (%) | Of those, really changed (%) |
|---|---|---|
| Loose | 12.95 | 2.76 |
| Tight | 6.33 | 6.22 |
| Centroid (new) | 0.79 | 38.29 |
| PACE-flip (new) | 55.40 | 23.33 |
| PACE-decomposition (new) | 4.98 | 99.44 |

**B: counterfactual baselines and robustness** (`results/cf_robustness/`).
- **Setup:** 7 methods, 11 datasets, 30 points per seed.
- **Deviation, made before the run:** DiCE kdtree was left out of r1, like DiCE genetic.
- **r1 (how far the answer moves under small nudges), against DiCE random:**
  - PACE was better on 11 of 11 (DiCE's r1 is 2.5–12.8× PACE's);
  - PACE-sparse was better on 11 of 11 (1.5–9.7×).
- **r3 (still valid under 4 refits in other row orders), against DiCE random:**
  - PACE: better on 1 (Shuttle), level on 7, worse on 3 (Wine, Breast Cancer, Page Blocks);
  - PACE-sparse: better on 1, level on 3, worse on 7.
- **What holds up best under refits:** DiCE genetic, DiCE kdtree and the nearest unlike neighbour. These answers are farther from x: genetic and kdtree have a larger L1 than PACE on every dataset.

**C: RQ1 with 200 points per seed, tie-aware** (`results/rq1_more_points/`).

PACE-SHAP against KernelSHAP (better / level / worse):

| Comparison | Deletion | Insertion |
|---|---|---|
| KernelSHAP, usual setting | 7 / 4 / 0 | 8 / 3 / 0 |
| KernelSHAP, same rows | 4 / 7 / 0 | 5 / 6 / 0 |
| Raw curves, usual setting | 11 / 0 / 0 | 11 / 0 / 0 |
| Raw curves, same rows | 5 / 6 / 0 | 6 / 5 / 0 |

- **Untied points only:** the same counts as in the first two rows.
- **Tied shares:** MAGIC 80%, Page Blocks 70%, Letter 56%, Landsat 25%.

**D: the conditional test at more settings** (`results/conditional_settings/`).

PACE-SHAP-cond against KernelSHAP-cond, never worse at any setting:

| Setting | Better by deletion | Better by insertion |
|---|---|---|
| σ 0.05, 200 rows | 4 of 6 | 4 of 6 |
| σ 0.2, 200 rows | 3 of 6 | 3 of 6 |
| σ 0.4, 200 rows | 3 of 6 | 3 of 6 |
| σ 0.2, 800 rows | 3 of 6 | 6 of 6 |

- **Effective rows (large coalitions):** 1.0–2.9 at σ = 0.05, and 27–120 at σ = 0.4.
- **Inputs off the data, from σ = 0.05 to σ = 0.4:** Letter 10% → 34%, Shuttle 49% → 67%.

## 12. Repair order and repair time (Run H pre-registered, seeds 130–139; Run T descriptive, seed 50; home PC)

Checked by `research_audit/scripts/claims_order_timing.py` (30 checks, all OK). Paper: §9, Limitations, Next steps and Appendix C (doc rev 449). The two sentences that close §9's Run H paragraph (doc rev 453: why self-repair keeps the random order) are checked by `research_audit/scripts/claims_sr_method.py`.

**Run H: the order of training errors in self-repair** (`results/repair_order/`). RQ3's SR arm with the cached gate; only the order changes. Test macro-F1, paired two-sided Wilcoxon over 10 seeds, Holm across the 6 datasets.

| Comparison | Better | Level | Worse |
|---|---|---|---|
| greedy-300 vs random-300 (primary) | MAGIC +4.87, Letter +2.98 | Thyroid, Landsat, Page Blocks | Skin −1.08 |
| greedy-300 vs random-300-real (amendment 1, A1) | MAGIC +4.65, Letter +2.67 | the other 4 (Skin −0.16) | none |
| greedy-300 vs random-all | MAGIC +7.97 | the other 5 | none |
| greedy-1000 vs random-1000 | MAGIC, Letter | the other 4 | none |
| box-300 vs random-300 | none | 4 | Skin, Letter |
| box-1000 vs random-1000 | none | 6 | none |
| greedy-300 vs box-300 | MAGIC, Letter | the other 4 | none |

- **Pre-registered reading:** greedy is "a better order" on MAGIC and Letter (primary and A1 both better). Elsewhere it is level, and Skin's primary loss disappears against real tries.
- **Wasted tries:** random-300 spent 108 tries on MAGIC and 260 on Skin on rows already right; greedy-300 spent none on MAGIC and Letter.
- **Skin:** random-300 reached the gate only 39 times; random-300-real and random-all both gave 97.96 against random-300's 98.88 (descriptive).
- **Work (E1, E2):** greedy-300 was slower than random-all on all 6 datasets (Skin 1,160 s against 26.6 s per seed). It checked fewer edits on MAGIC (−526) and Letter (−961) only, so the reading is "fewer edits, more time", not "less work".
- **Time against random-300:** MAGIC 241 s against 5.2 s; Letter 420 s against 11.8 s.

**Run T: time of each repair setup, one task at a time** (`results/repair_timing/`, Verdict PASS: every setup's macro-F1 equals RQ3's).

| Setup | Seconds, 6 datasets | Multiple of SR (cached gate) |
|---|---|---|
| Train CFMNN | 0.04–1.9 | |
| SR, cached gate | 0.2–4.7 | 1 |
| SR, old gate (RQ3's) | 0.8–211 | 2.4–45 |
| AL | 3.9–142 | 4.8–56 |
| SR+AL | 4.6–145 | 5.5–57 |

- Leaving out AL saves 82–98% of SR+AL's time and every bought label. It gives up SR+AL's significant gains over SR on Landsat (+1.1), MAGIC (+2.0) and Letter (+0.7) (RQ3, P3).
- `results/sr_method/tables.md` now includes this time table (regenerated by `scripts/sr_method_tables.py`).

## 13. Self-repair against contraction (Run F, pre-registered, seeds 120–129, home PC)

Checked by `research_audit/scripts/claims_contraction.py` (42 checks, all OK). Source: `results/contraction/runs.csv`. RQ3's six datasets and split; test macro-F1; paired two-sided Wilcoxon over 10 seeds, Holm across the 6 datasets. "Higher" and "lower" are the run's words for a Holm p below 0.05.

The question: classic fuzzy min-max networks fix class overlap by contraction. Is gated self-repair more accurate than that?

| Comparison (first − second, macro-F1 points) | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin | Higher / not significant / lower |
|---|---|---|---|---|---|---|---|
| **P1 (primary): self-repair − Simpson's contraction on every overlap** | +4.5 | +2.4 | +10.0 | +14.9 | +2.6 | +22.5 | 6 / 0 / 0 |
| S1: self-repair − Simpson's contraction at the same training errors, no gate | +4.5 | +1.7 | +5.3 | +5.1 | +2.4 | +9.0 | 6 / 0 / 0 |
| S3: self-repair − GFMM trained from scratch | +14.4 | +1.5 | +2.0 | +2.0 | −5.4 | +1.0 | 3 / 2 / 1 |
| S4a: self-repair − the starting model | +4.5 | +2.0 | +6.3 | +1.9 | +2.4 | +7.6 | 6 / 0 / 0 |

- **Contraction does not help this model.** Simpson's contraction on every overlap is lower than the starting model on MAGIC (−13.0) and Skin (−14.9) and not significantly different elsewhere; at the training errors only, it is lower on MAGIC and not significantly different elsewhere.
- **Against GFMM trained from scratch** self-repair is higher on Thyroid, MAGIC and Skin, not significantly different on Landsat and Page Blocks, and **lower on Letter** (76.9 against 82.3). This, not S2, is the comparison with GFMM.
- **GFMM's rule applied to CFMNN's boxes** (S2a, S2b) loses to self-repair on all 6, but it cuts away most of each box (applied to every overlap it evicts 21–90% of the training rows), so, as Amendment 1 fixed before the run, these wins are not wins over GFMM.
- **Thyroid:** Simpson's test found no overlap on any seed (0 contractions), so there P1 and S1 are the same comparison as self-repair against the starting model.
- **Budget:** 300 tries on every dataset. On Page Blocks this is not RQ3's budget (273), so Page Blocks' self-repair here is not RQ3's arm exactly (Amendment 1).
- **Evicted training rows (% of training rows, same dataset order):** self-repair 59.5, 23.5, 5.8, 32.6, 9.5, 2.8; Simpson on every overlap 0, 12.5, 27.3, 89.3, 0.8, 17.8. Self-repair evicts more on Thyroid, Landsat and Letter and fewer on MAGIC and Skin (S8).
- Also: self-repair is higher on lowest class recall on all 6 (S5) and on accuracy on 5 (S9; Page Blocks not significant). It breaks fewer correct test predictions than Simpson on MAGIC, Page Blocks and Skin and more on Letter (S7).

## 14. CFMNN + PACE-SHAP against models that explain themselves (Run G, pre-registered, seeds 120–129, home PC)

Checked by `research_audit/scripts/claims_systems.py` (57 checks, all OK). Sources: `results/systems/runs.csv` and `timing.csv`. 11 explanation datasets, the paper's 80/20 split, 50 explained points per seed. Each system explains its own predicted-class margin, exactly. Paired two-sided Wilcoxon over 10 seeds, Holm across the 11 datasets.

**Deviation (Amendment 2):** the first attempt failed on 3 of 110 tasks (shap's TreeExplainer "additivity check failed"). shap rounds each tree threshold to the nearest float32 while scikit-learn compares with the float64 threshold, so a point on a rounded threshold took a different branch. The explainer's thresholds are now rounded down (`float32_floor`, with a test), and the whole main run was repeated with the fix; the first attempt's results were not read beyond its first line and were not committed.

| | Against an EBM (main effects only) | Against a tuned tree + TreeSHAP |
|---|---|---|
| **P1 / P2 (primary): normalised deletion lead** | better 2 (S1, Page Blocks), level 6, worse 3 (Breast Cancer, Shuttle, Letter) | better 3 (S1, S2, Page Blocks), level 4, worse 4 (Wine, Breast Cancer, Shuttle, Letter) |
| ... with the near-zero rule (Landsat read from the gain: worse) | 2 / 5 / 4 | 3 / 3 / 5 |
| Deletion gain, not divided | 0 / 1 / 10 | 0 / 2 / 9 |
| Test accuracy | 0 / 1 / 10 | 2 / 0 / 9 (better on Wine, Breast Cancer) |
| Milliseconds per explanation | worse on 11 | worse on 11 |

- **Tied points (Amendment 1).** CFMNN ties on many explained points (MAGIC 77%, Page Blocks 75%, Letter 54%, Landsat 26%); the EBM and the tree almost never do. On untied points only, Landsat's verdict holds; **Letter's flips from worse to better**, so it depends on the tied points; MAGIC and Page Blocks have too few untied points to check, so Page Blocks' "better" is not checked.
- **Accuracy** (mean over seeds, CFMNN / EBM / tree): MAGIC 71.9 / 86.0 / 85.1, Letter 75.6 / 87.8 / 86.7, Page Blocks 83.4 / 97.5 / 96.9, S2 80.1 / 90.6 / 88.4.
- **Size:** CFMNN 14–1,538 boxes (about 1,520 on S1–S3), the EBM 9–36 terms, the tree 4–96 leaves except on Letter (1,024).
- **Time, measured alone (seed 120):** PACE-SHAP 7–1,812 ms per explanation; the EBM at most 0.03 ms; TreeSHAP at most 0.43 ms except Letter (8.5 ms). PACE-SHAP takes 35–69,000 times as long as TreeSHAP.
- **Reading:** CFMNN + PACE-SHAP is not a better system than these readable models on accuracy, size or time, and its lead is mixed. Its explanation is exact for a hyperbox model; that is the claim the paper keeps.

## 15. Generality and the γ sweep (Run I, pre-registered, seeds 150–159, home PC)

Checked by `research_audit/scripts/claims_generality.py` (22 checks, all OK). Sources: `results/generality/runs.csv` and `exact.csv`. 13 accuracy datasets, the paper's 80/20 split. The post hoc check of the exact check is `research_audit/scripts/check_exact_simpson.py`.

- **Proposition 1 holds at γ = 2, 4 and 8:** 0 of 569,239 training errors (every dataset, seed and γ) are not ties inside a box of both classes.
- **γ sensitivity (test accuracy against γ = 4, Holm across 13):** γ = 2 is better on S1 (+2.1) and Digits (+1.0) and level on 11; γ = 8 is worse on S1 (−4.3), S2 (−2.6), S3 (−3.9) and Digits (−0.9) and level on 9. γ stays 4, as the plan fixed.
- **The Simpson model** is CFMNN's boxes after Simpson's overlap test and contraction on every overlap (Run F's arm), not Simpson's classifier trained from scratch. Simpson's test found nothing on Thyroid and Digits (every seed) and on Wine (9 of 10 seeds).
  - Where it contracts, Proposition 1 stops holding, as expected: the share of training errors that are ties falls to 0–8% on S1–S3, Breast Cancer, Shuttle, Skin and MAGIC, 64–70% on Page Blocks and Landsat and 98.6% on Letter.
  - Accuracy falls on Shuttle (99.8 to 71.2), MAGIC (71.2 to 58.3), Page Blocks (87.4 to 80.5) and Skin (94.5 to 89.0); elsewhere it moves by at most 0.5 points.
- **PACE-SHAP carries over to the contracted boxes:** against shap's ExactExplainer on the Simpson model (6 datasets with at most 12 features, 5 points per seed) the largest difference is 1.6e-10, below the plan's limit of 1e-9 (**Check: PASS**). 55 of the 60 tasks agree to 2.7e-16; five (S1 152, S2 159, S3 150 and 159, MAGIC 158) show 7e-11 to 1.6e-10.
- **Post hoc: which side carries those differences.** In three of the five (the two largest, S3 150 and S1 152, and the smallest, MAGIC 158) a brute-force Shapley sum (every coalition on every explainer row evaluated by calling the model, summed with `math.fsum`) agrees with PACE-SHAP to 1.3e-16 and with ExactExplainer only to 7.1e-11 to 1.6e-10 (`research_audit/scripts/check_exact_simpson_out.txt`). The model's output does not depend on how the rows are batched, and no hybrid input lies within 100 times the tie tolerance of a tie. So the differences come from shap's ExactExplainer, not from PACE-SHAP. Post hoc; no committed file was changed.
- **Post hoc: the cause inside shap** (`research_audit/scripts/check_exact_isclose.py`, output beside it). shap 0.54.1's tabular masker treats a background value as unchanged when `np.isclose` says it equals the explained point's (relative tolerance 1e-5, absolute 1e-8). For such a value, ExactExplainer does not call the model again for that background row; it keeps the row's earlier output.
  - Each of the five tasks has exactly one such value, at one point (a gap of 3.3e-6 to 5.7e-6). Of the other 55 tasks, only MAGIC 152 has one (a gap of 2.9e-7), and its committed difference is 1.3e-16.
  - At the point holding the value, ExactExplainer as shipped reproduces each task's committed difference (7.1e-11 to 1.6e-10). With exact equality in place of `np.isclose` (changed in that process only), it matches PACE-SHAP to at most 1.3e-16 on all five.
  - So the 1e-10 differences are a shortcut in shap's ExactExplainer, not an error in PACE-SHAP. Post hoc; no committed file was changed.

## 16. Sampled PACE-SHAP against modern samplers above 12 features (Run J, pre-registered, seeds 160–169, home PC)

Checked by `research_audit/scripts/claims_sampled.py` (18 checks, all OK). Sources: `results/sampled_baselines/runs.csv` and `timing.csv`. Wine (13 features), Letter (16), Thyroid (21), Breast Cancer (30), Landsat (36). Every method estimates the same game on the same 200 explainer rows (71 on Wine), at an equal budget of model evaluations; both baselines stayed within it on every task.

| PACE-SHAP against | Deletion margin AUC (primary) | Insertion margin AUC |
|---|---|---|
| PermutationSHAP (shap 0.54) | better 1 (Landsat, −0.0045), level 4, worse 0 | better 1 (Wine), level 4, worse 0 |
| KernelSHAP with paired sampling (Covert and Lee) | better 1 (Landsat, −0.0057), level 4, worse 0 | better 2 (Thyroid, Landsat), level 3, worse 0 |

- The differences in deletion AUC are tiny: at most 0.0006 outside Landsat.
- **Error against exact values** (relative L1, the first 10 points per seed): Wine 0.010 against 0.029 and 0.025; Letter 0.014 against 0.046 and 0.049. PACE-SHAP's error is 28–40% of the baselines'.
- **Top-3 stability under a new seed:** PACE-SHAP is the steadiest on all 5 datasets (Landsat 0.95 against 0.76 and 0.71).
- **Time, measured alone (seeds 160–162):** PACE-SHAP 7–636 ms per explanation; the baselines took 3.8–9.6 times as long.

## 17. One large dataset: Covertype (Run K, descriptive, seeds 170–174, home PC)

Checked by `research_audit/scripts/claims_covertype.py` (20 checks, all OK). Source: `results/covertype/runs.csv`. UCI Covertype: 581,012 rows, 54 features, 7 classes; the paper's 80/20 split (464,809 training rows); CFMNN at θ = 0.3, γ = 4, untuned. One seed at a time, with nothing else running. Descriptive: no test.

| Model | Test accuracy (%) | Test macro-F1 (%) |
|---|---|---|
| CFMNN | 51.4 ± 2.7 (47.7–55.1 over the seeds) | 39.1 |
| 10-NN | 92.7 | 87.1 |
| Random Forest | 95.6 | 92.7 |

- **CFMNN is far behind on this dataset.** At θ = 0.3 it grows only about 119 boxes, about 3,900 training rows each, and gets 48.5% of its own training rows wrong.
- **The tie finding holds:** all 1,128,010 training errors (pooled over the seeds) are ties inside a box of both classes. 98.9% of the test predictions are decided by the nearest centroid, and they hold 99.9% of the test errors.
- **Cost:** CFMNN trains in 10.3 s; PACE-flip takes 1.1 ms and PACE-SHAP (sampled mode, 54 features) 454 ms per explanation; 2.2 minutes per seed, at most 3.0 GB of memory.
- This is one dataset: it does not show that CFMNN or PACE scales in general.

## 18. Covertype with θ chosen by validation, then self-repair (Run L, descriptive, seeds 210–214, home PC)

Checked by `research_audit/scripts/claims_covertype_theta.py` (36 checks, all OK). Source: `results/covertype_theta/runs.csv`. One task at a time, with nothing else running. Descriptive: no test. The four models of the repair-at-θ run (Section 10): CFMNN at θ = 0.3 (B0-0.3), at the θ with the best 3-fold CV accuracy on the training rows (B0-val, grid 0.5 to 0.05), and each after RQ3's self-repair (300 tries, gate on the training rows, cached gate).

**On a stratified 10% of Covertype per seed, not the full data** (46,480 training rows, 11,621 test rows). Before any study seed, a trial on seed 192 projected the full-data run at about 50 hours, with about 58 GB for the cached gate, so the user chose the 10% share; Run L's pre-registration (`bd0d023`) records it, against `PART_B_PLAN.md`'s full data. These numbers are therefore not directly comparable with Run K's (Section 17).

| Model | Test accuracy (%) | Test macro-F1 (%) | Boxes | Training error (%) | PACE-flip ms | PACE-SHAP s |
|---|---|---|---|---|---|---|
| B0-0.3 | 50.7 | 39.5 | 82.8 | 48.9 | 0.84 | 0.3 |
| SR-0.3 | 53.9 | 41.4 | 78.0 | 45.6 | 0.69 | |
| B0-val (θ = 0.05) | 72.9 | 64.3 | 3,009.6 | 19.6 | 8.66 | 9.3 |
| SR-val | 73.4 | 64.8 | 3,009.6 | 18.9 | 8.94 | |

- **θ by validation chose 0.05, the smallest θ in the grid, on all 5 seeds;** CV accuracy was highest there on every seed (72.4–73.4%), so the grid's end was hit. Over θ = 0.3 it raised test accuracy by +20.4 to +24.8 points per seed, with 34–42 times as many boxes.
- **Self-repair helped at both θ, on every seed, and never added a box.** At θ = 0.3: +3.1 accuracy (+0.5 to +5.3 per seed) and +1.9 macro-F1 (+0.6 to +4.2), with 25.8 of 300 tries accepted (18–39), in 25 s. At θ = 0.05: +0.4 accuracy (+0.1 to +0.6) and +0.5 macro-F1 (+0.1 to +1.4), with 281.4 accepted (272–286), in 448 s.
- **The tie finding holds before repair:** all 113,545 training errors at θ = 0.3 and all 45,542 at θ = 0.05 (pooled) are ties inside a box of both classes. After repair, 96.8% and 95.6% are, because a shrink can leave rows in no box of their class (Section 5). The nearest centroid decided 97.3% of the test predictions at θ = 0.3 and 41.0% at θ = 0.05 (93.8% and 39.3% after repair).
- **The cost of θ = 0.05:** PACE-SHAP takes 9.3 s per explanation instead of 0.3 s, and PACE-flip 8.66 ms instead of 0.84 ms; training takes 8.8 s instead of 0.9 s, and tuning 158–278 s per seed. Each seed took 35.7–40.8 minutes, at most 3,998 MB of memory.
- Run K's 10-NN (92.7%) and Random Forest (95.6%) used the full data and were not rerun on the 10% share, so the gap to them at θ = 0.05 is not measured here.

## 19. Simpson's classifier trained from scratch (Run M, an exact check plus a descriptive comparison, seeds 220–229, home PC)

Checked by `research_audit/scripts/claims_simpson.py` (30 checks, all OK); the cause of the one large exact-check difference by `research_audit/scripts/check_exact_isclose_simpson.py` (post hoc, 5 checks, all OK). Sources: `results/simpson_fmm/exact.csv` and `runs.csv`. 12 jobs at once.

The model, `SimpsonFMM` (`pace_cfmnn/simpson_fmm.py`), trains row by row with CFMNN's membership, size limit (a sum over the box's sides) and expansion, and runs Simpson's overlap test and contraction after every expansion; a tie goes to the box made first, since Simpson's classifier has no centroid. θ = 0.3, γ = 4. **A change before the run:** `PART_B_PLAN.md` asked for a test that no two boxes of different classes overlap after training. That does not hold for the published algorithm, which never tests a box made from a new point, so the pre-registration replaced it with two hand-made tests and a count of the pairs that still overlap.

- **Exact check: PASS.** On S1–S3, Shuttle, MAGIC and Page Blocks (10 seeds, 5 points each), PACE-SHAP matched shap's ExactExplainer to within 1.4e-10, inside the plan's 1e-9. 59 of the 60 tasks agree to 3.1e-16; the other is S1, seed 222. During the run, ExactExplainer took 11–30 times as long as PACE-SHAP.
  - **Post hoc: the same shap shortcut as in Run I (Section 15).** That task has one background value 4.1e-6 from the explained point's, which `np.isclose` treats as equal. As shipped, ExactExplainer reproduces the committed 1.35e-10 at that point; with exact equality it matches PACE-SHAP to 1.0e-16. Two other tasks (S1 229, MAGIC 222) also hold such a value, with differences of at most 3.1e-16. No committed file was changed.
- **Accuracy beside CFMNN (descriptive, no test; means over 10 seeds).** SimpsonFMM was less accurate on Shuttle (85.8 against 99.8, lower on every seed), Letter (67.0 against 75.3, every seed), MAGIC (−1.9) and Thyroid (−1.0), more accurate on Skin (+2.1), Breast Cancer (+1.1) and Page Blocks (+0.7), and within 0.3 points on S1–S3, Wine, Digits and Landsat. By macro-F1: Shuttle −10.5, Letter −8.3, Thyroid −7.1, Skin +3.7.
- **Ties.** CFMNN's training errors are all ties wherever it has any (Proposition 1). SimpsonFMM's mostly are not, as expected with contraction: 0.0% on S2 and Breast Cancer, 0.7% on Shuttle, 2.1% on MAGIC, 37.8% on Skin, 71.4% on Page Blocks, 73.9% on Landsat, 98.7% on Letter, and 100% on Thyroid, where no contraction happened.
- **Post hoc reading of Thyroid.** With no contraction on any seed and the same number of boxes as CFMNN on every seed, the two models differ there only in the tie rule. The nearest centroid is 1.0 accuracy point and 7.1 macro-F1 points ahead of "the box made first"; the tie rule decides 5.4% of the test predictions in both.
- **Contractions and leftover overlaps.** None on Thyroid or Digits; about 22,000 per model on Shuttle and 21,000 on MAGIC. After training, Simpson's test still finds overlapping pairs only on Landsat and Letter (at most 4 in one model).

## 20. Covertype in the sampled-baseline comparison (Run N, descriptive, seeds 230–234, home PC)

Checked by `research_audit/scripts/claims_sampled_covertype.py` (26 checks, all OK). Sources: `results/sampled_baselines_covertype/runs.csv` (5 jobs at once) and `timing.csv` (timing pass alone). Run J's task, unchanged (Section 16), on Covertype: 54 features, CFMNN on all 464,809 training rows (109–121 boxes), 50 points per seed, 200 explainer rows, a budget of 352,000 model rows per explanation. No exact reference (2^54 coalitions). Descriptive: 5 seeds, no test.

| Method | Deletion margin AUC (lower is better) | Insertion margin AUC (higher is better) | Top-3 stability | Model rows per explanation |
|---|---|---|---|---|
| PACE-SHAP | −0.2151 | 0.0568 | 0.980 | 352,000 |
| PermutationSHAP | −0.2153 | 0.0560 | 0.944 | 114,777 |
| KernelSHAP-paired | −0.2128 | 0.0514 | 0.978 | 351,801 |
| Random | −0.093 | | | 0 |

- **Faithfulness: close, both ways.** By deletion, per seed, PACE-SHAP was ahead of PermutationSHAP on 1 seed and behind on 4, and ahead of paired KernelSHAP on 3 and behind on 2; no difference was larger than 0.0067, and the mean differences were +0.0002 and −0.0023 (negative favours PACE-SHAP). By insertion it was ahead of PermutationSHAP on 4 seeds and behind on 1, and ahead of paired KernelSHAP on 4 and level on 1.
- **Budget:** every method kept to it. PermutationSHAP evaluated about a third of it (32.6%, 114,777 rows on average); the other two used all of it.
- **Stability:** PACE-SHAP's top 3 under a new seed agreed 0.980 of the time, against 0.944 and 0.978.
- **Time, measured alone** (seeds 230–232, 10 points each): PACE-SHAP 434 ms per explanation; PermutationSHAP took 4.6 times as long and paired KernelSHAP 13.9 times (4.4–4.7 and 13.5–14.8 per seed). On Run J's five datasets the range was 3.8–9.6.

## 21. Re-tune θ or repair? (Run O, pre-registered, seeds 70–89, home PC)

Checked by `research_audit/scripts/claims_theta_churn.py` (all checks OK). Sources: `results/theta_churn/runs.csv` (B0 at each θ of the grid 0.5 to 0.05, 12 jobs) and, for self-repair's flips, `results/repair_theta/runs.csv` (Section 10), as the plan says. The readable-baselines split, the 13 accuracy datasets, γ = 4. Every flip is counted against B0 at θ = 0.3 on the same test rows: **broken** = right before, wrong after; **fixed** = wrong before, right after. NFR (negative flip rate) = broken ÷ test rows, in %. **Check:** all 503 B0 rows at θ = 0.3 or θ-val reproduce the readable-baselines accuracy and macro-F1 exactly.

- **P1 (primary): repair breaks fewer of the model's correct predictions than re-tuning θ on 10 of 13 datasets, and more on 3** (Shuttle +0.065, Skin +0.185 and Letter +0.929 points; Holm p ≤ 0.0143 everywhere). Re-tuning broke 0.02–10.5% of the test predictions; repair broke 0–7.8%.
  - On S1–S3, Wine and Digits, repair changes no test prediction at all (no training error to repair, or none it can fix), while re-tuning breaks 0.8–10.5%.
  - On MAGIC both break many (7.8% and 8.9%).
- **Re-tuning is the bigger update.** Where it is more accurate (7 of 13, from Section 10), it also fixes more (PFR) and has a higher fix share on Shuttle, Skin, MAGIC, Letter, Landsat and Page Blocks (Letter 85.3% of its changes are fixes, against 44.3% for repair). It changes up to 27.9% of the test predictions; repair up to 20.6%. On those datasets, apart from S1, it uses 6.3–23.2 times as many boxes as repair.
- **Boxes needed to match repair (descriptive):** without repair, B0 reaches SR-0.3's accuracy at a smaller θ only, with 1.8–23.2 times the boxes at θ = 0.3 on Breast Cancer, Shuttle, Skin, MAGIC, Letter, Landsat and Page Blocks (Skin 23.2 at θ = 0.05; Shuttle 6.2; MAGIC 4.9 at θ = 0.15; Letter 2.4 at θ = 0.2). On Thyroid no θ in the grid reaches it. On S1–S3 a larger θ (fewer boxes) reaches it; on Wine and Digits θ = 0.3 already does.
- **θ-val chose the grid's smallest value, 0.05, on all 20 seeds** for Shuttle, Skin, MAGIC and Letter.
- **Repair after re-tuning is gentle:** SR-val against B0-val breaks at most 0.41% of the test predictions (Page Blocks; ≤ 0.29% elsewhere). "Re-tune, then repair" against B0-0.3 was not measured (SR-val's test predictions were not stored).
- **How the plan said to read it:** repair better on k = 10, so the paper says repair keeps more of the model's correct answers than re-tuning, an advantage beside size for a model already in use.

## 22. Where PACE-SHAP's speed comes from (Run P, descriptive, seeds 240–242, home PC)

Checked by `research_audit/scripts/claims_speed_source.py` (all checks OK). Source: `results/speed_source/runs.csv`. Run alone, one task at a time. The 6 RQ1 datasets with at most 12 features; the second explanation run's models and 200 explainer rows; 2 points per seed. Each time includes its explainer's set-up. Descriptive: no test.

| Dataset | d | Boxes | ExactExplainer (ms) | PACE-SHAP, no pruning (ms) | PACE-SHAP (ms) | Structure | Pruning | Total | Pairs skipped |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 1,523 | 27,819.8 | 3,823.1 | 1,948.7 | 7.3× | 2.0× | 14.3× | 0.624 |
| S2 | 10 | 1,539 | 28,329.5 | 3,845.7 | 1,957.4 | 7.4× | 2.0× | 14.5× | 0.623 |
| S3 | 10 | 1,523 | 27,695.2 | 3,813.7 | 1,889.5 | 7.3× | 2.0× | 14.7× | 0.635 |
| Shuttle | 9 | 13 | 162.9 | 15.2 | 8.7 | 10.7× | 1.8× | 18.8× | 0.719 |
| MAGIC | 10 | 285 | 5,800.2 | 730.0 | 256.0 | 7.9× | 2.9× | 22.7× | 0.759 |
| Page Blocks | 10 | 27 | 669.0 | 77.7 | 29.4 | 8.6× | 2.6× | 22.7× | 0.681 |

- **Check: PASS.** Without pruning, PACE-SHAP agrees with PACE-SHAP to ≤ 3.1e-17, and ExactExplainer to ≤ 2.5e-16.
- **Most of the speed-up comes from the structure (7.3–10.7×):** PACE-SHAP computes each feature's ramp term once per (row, box) and adds them for every coalition, instead of calling the model on every mixed input. **Pruning adds 1.8–2.9×,** skipping 62–76% of the (row, box) pairs. Together: 14.3–22.7×.
- **The cost is still exponential in d** for both (O(K·d·2^d) for PACE-SHAP against O(R·H·d·2^d) for ExactExplainer, with R rows and H boxes, K ≤ R·H kept pairs): the saving is a constant factor, which is why exact mode stops at 12 features.

## 23. Tough-review runs, batches 1 and 2 (pre-registered, seeds 250–399, home PC)

Nine runs answer the tough review of 2026-11-16 (`research_audit/TOUGH_REVIEW_PLAN.md`). Each plan, script and test was pushed before its run. The home PC ran them on 2026-11-16/18 (`research_audit/HOME_PC_RUNBOOK.md` §12 and §14).

The numbers below are read from each run's `summary.md`, which its script writes from the raw CSV. `research_audit/scripts/claims_tough_batch1.py` and `claims_tough_batch2.py` recompute them from the CSVs. **They are written but not yet run**, because nothing runs on the laptop; the home PC runs them next (runbook §17).

**What did not go as planned (all reported in the paper's Deviations):**

- **Three fixes after a failed first run.** Each was approved by the user. Each first run is kept unedited, each fix has a test that fails on the old code, and each run was repeated in full.
  - **P8** (`results/recurrence/AMENDMENT_1.md`).
    - The check compared ExactExplainer's one Letter point with all ten PACE-SHAP points, through numpy broadcasting.
    - The values agreed within 1.9e-16 on the shared point.
    - Only the comparison changed.
  - **P4, Amendment 1.**
    - The GFMM adapter broke IOL-GFMM's ties at zero membership with Online GFMM's Manhattan rule, not IOL-GFMM's own probability rule. So it predicted a different class from hyperbox-brain on one point in 10 of 60 tasks.
    - PACE-SHAP-min matched the brute force in that run too.
    - P4's plan said a fix would run "as a new variant with its own pre-registration, not in place of this run". The rerun was made under an amendment note instead.
    - The paper reports the first run's FAIL and treats the rerun as the variant.
  - **P4, Amendment 2.**
    - shap's ExactExplainer arm needed 33–36 GiB on S1–S3 and MAGIC, and failed with `MemoryError` on 14 of 18 timing tasks.
    - It now evaluates in blocks. The values are unchanged.
- **Time.**
  - P6 took 29.8 h against 2–4 h. P4's check took 5.3 h against 1–2 h.
  - Both passed their stop-and-ask limits, and the user chose to let them finish.
  - P4's check part also ran beside P6 rather than after it. That part is not timed, so this changed only its speed.

### 23.1 Time to accuracy, and one clean re-time (Run P1, seeds 300–303)

Source: `results/time_to_accuracy/`. Run alone, one task at a time, with v3's models and 200 explainer rows.

**Part (a), primary.** The table gives KernelSHAP's relative L1 error against PACE-SHAP's exact values (median over 30 points) at each budget, on the 6 datasets with at most 12 features. The last column is KernelSHAP's time over PACE-SHAP's at b*.

| Dataset | 100 | 200 | 400 | 700 | every coalition | KernelSHAP ÷ PACE-SHAP at b* (95% interval) |
|---|---|---|---|---|---|---|
| S1 | 13.97% | 9.04% | 6.40% | 4.03% | 0 | 16.9× (15.9–18.1) |
| S2 | 16.53% | 9.62% | 7.53% | 5.76% | 0 | 17.3× (16.0–18.9) |
| S3 | 20.43% | 12.59% | 9.11% | 7.10% | 0 | 18.0× (16.8–19.5) |
| Shuttle | 2.39% | 1.87% | 1.33% | – | 0 | 20.1× (16.8–23.6) |
| MAGIC | 22.59% | 13.51% | 8.95% | 7.25% | 0 | 26.4× (24.6–28.0) |
| Page Blocks | 13.02% | 9.34% | 5.68% | 4.58% | 0 | 20.7× (19.7–21.9) |

- No sampled budget reached a median error of 1% on any dataset, so b* is every coalition everywhere.
- Shuttle has only 510 coalitions, so it has no 700 column.
- At b*, KernelSHAP took **16.9–26.4×** PACE-SHAP's time, and every interval is above 1.
- **Reading (as planned):** PACE-SHAP is faster at b* on every dataset. To match its accuracy within 1%, KernelSHAP needs every coalition and 17–26× the time.

**Part (b), the re-time.**

- Seed 303, 10 points per dataset. Each explainer ran alone on the same 200 rows, with KernelSHAP at 200 coalitions.
- KernelSHAP took **3.3–8.0×** as long as PACE-SHAP's default mode on the 13 like-for-like datasets (median 4.9×). Every interval is above 1.
- This replaces "3.5–10×", which was timed while other runs shared the laptop.
- The faithfulness verdicts are unchanged.

### 23.2 Skin without its duplicate rows (Run P3, seeds 260–269)

Source: `results/skin_dedup/`. `skin_dedup` keeps the first copy of each row: 51,444 of 245,057.

- **Primary.**
  - Self-repair minus the starting model, in macro-F1, was **+6.20 points**, better on 10 of 10 seeds (p = 0.0020).
  - On the original Skin, with the same seeds, it was +7.56 (10 of 10).
  - The paper's gain was +5.66, and the plan's bar was half of it, 2.83.
- **Reading (as planned):** p < 0.05 and the gain is at least 2.83, so **the Skin result stands**. The duplicates did not drive it.
- **Secondary (descriptive).**
  - SR+AL beat SR by +0.62 (10 of 10).
  - Accuracy without duplicates: CFMNN 92.92, 10-NN 99.73, Random Forest 99.80. That is a gap of 6.9 points, against 5.2 on Skin with duplicates.
  - θ by validation chose 0.05 on every seed: 99.43% accuracy with 2,011 boxes, against 92.92% with 89 boxes at θ = 0.3.
  - Negative flip rate, repair against re-tuning θ: repair broke **more** on all 10 seeds, by 0.93 points (1.27% against 0.34%). On Skin with duplicates, repair also broke more, by 0.19 (Section 21). The direction is the same and the gap is larger.

### 23.3 Exact flip sets (Run P7, seeds 340–349)

Source: `results/flip_intervals/`. v3's models and explainer rows, 11 datasets.

- **Check: PASS.**
  - On 17,500 (point, feature) pairs, every value of a 10,001-value grid that changes the class lies in the exact flip set (within 1e-9).
  - Every exact interval wider than the grid step holds a grid value that changes the class.
  - On all 85,680 pairs, the class at PACE-flip's 32 values agrees with the exact set. The 2,414 values within 1e-9 of a breakpoint are left out of this comparison.
- **Secondary.**
  - The 32 values miss **6,356 of 34,832 flipping pairs (18.2%)**.
  - By dataset: S1–S3 1.7–1.9%, Wine 5.4%, Breast Cancer 7.1%, Landsat 8.1%, Letter 8.3%, **MAGIC 13.9%, Thyroid 45.1%, Page Blocks 51.0% and Shuttle 56.2%**.
  - The 32 values call 0–128 points per dataset unflippable that the exact sets can flip (on Thyroid, 128 of 446).
- **Reading (as planned).**
  - Proposition 8 stands, and exact flip sets become an option of PACE-flip.
  - The miss share is above 10% on 4 datasets, so the PACE-flip section says the 32-value grid can miss flips there and points to the exact sets.
  - The RQ1 results, which use the 32 values, are unchanged.

### 23.4 The recurrence exact mode (Run P8, seeds 350–352)

Source: `results/recurrence/`, the rerun after Amendment 1. The first run is in `first_run/`. Run alone.

- **Check: PASS.**
  - The recurrence and the vectorised brute force match the matrix-product mode within 1e-12 on every point.
  - On Wine and Letter, both PACE-SHAP modes match shap's ExactExplainer within 1e-9.
- **Speed.**
  - Matrix product ÷ recurrence: 1.03–1.23× at 9–10 features, 2.13× on Wine and 1.40× on Letter.
  - Brute force ÷ recurrence: 47–112×.
- **Decision rule:** Letter's recurrence takes 26.1 s per explanation, above the 5 s the plan set, so **the exact limit stays at 12 features**.
- **Reading (as planned).**
  - The recurrence is faster everywhere, so it becomes the default exact algorithm. That is a code change, with the full suite rerun.
  - The change waits until batch 3 (P9, P10) has run, so those runs use the code their plans describe.
  - Every timing in the paper uses the matrix product.
  - The brute force is never within 2× of PACE-SHAP.

### 23.5 Tuned fuzzy min-max baselines (Run P2, seeds 250–259)

Source: `results/fmm_baselines/`, with the pilot in `pilot/`. The paper's 80/20 split and the 6 repair datasets, with θ chosen by 3-fold cross-validation for every model. The pilot projected 13.4 h, under the 24 h limit, so Skin ran in full, as the plan's rule says.

- **Primary.**
  - Test macro-F1 of CFMNN with self-repair, at its validated θ, against EFMNN, KNEFMNN, RFMNN and IOL-GFMM (hyperbox-brain 0.1.6), each at its own validated θ.
  - With Holm across all 24 tests: **better 6, no significant difference 17, worse 1**.
  - Better: Thyroid against all four (+14.59 to +29.40 points), and Landsat (+2.34) and Letter (+3.17) against IOL-GFMM.
  - Worse: MAGIC against IOL-GFMM (−1.22, 0 of 10 seeds better, Holm p = 0.0469).
  - With 24 tests, a significant result needs all 10 seeds to agree.
- **Reading (as planned):** between 1 and 12 comparisons are worse, so the result is **mixed**. The paper names the comparison where a baseline wins (IOL-GFMM on MAGIC) and keeps RQ2's claim to the others.
- **Secondary.**
  - Test accuracy gives the same split, 6 / 17 / 1, with the same comparison worse.
  - At θ = 0.3 for every model: 7 better, 11 not significant and 6 worse. The 6 are KNEFMNN and RFMNN on Letter, RFMNN on Skin, and IOL-GFMM on Landsat, MAGIC and Letter.
  - Tuned CFMNN without repair: 5 / 15 / 4.
- **Size.**
  - Tuned CFMNN used fewer boxes than tuned EFMNN, KNEFMNN and RFMNN on all 6 datasets (on Letter, 5,343 against 15,075).
  - Against IOL-GFMM it used fewer on Thyroid, Letter and Skin, and more on Landsat, Page Blocks and MAGIC.
- **θ by validation** chose 0.05 for every model on Letter and Skin, and for most models on MAGIC and Page Blocks. IOL-GFMM chose larger values of θ on Landsat, Page Blocks and MAGIC.

### 23.6 GFMM exactness (Run P4, seeds 310–319)

Source: `results/gfmm_exact/`. The first runs are in `first_run/` and `timing_first_run/`, and the amendments in `AMENDMENT_1.md` and `AMENDMENT_2.md`.

- **First check run: FAIL**, on the adapter's prediction (see above). PACE-SHAP-min still matched the brute force within 7e-16 on every compared point.
- **Check after Amendment 1: PASS.**
  - The adapter predicts like hyperbox-brain on every explained point.
  - PACE-SHAP-min equals the brute force within **6.8e-16** on Online GFMM and IOL-GFMM, over 6 datasets.
  - In the timing part, shap's ExactExplainer agrees within 1e-9.
- **Timing**, after Amendment 2, run alone, one point each.
  - On S1–S3, PACE-SHAP-min was 10.7–10.8× faster than both the brute force and ExactExplainer.
  - On Shuttle, MAGIC and Page Blocks, it was 33.5–94.2× faster than the brute force and 33.4–90.3× faster than ExactExplainer.
- **Reading (as planned).**
  - The check passed (in the amended variant), so the paper states that PACE-SHAP-min gives exact values for GFMM-family models, with these ratios.
  - It also says that the cost still grows as 2^d. No hardness is claimed for the minimum.

### 23.7 Explaining the decision at ties (Run P5, seeds 320–329)

Source: `results/decision/`. MAGIC, Letter and Page Blocks, 50 points per seed, of which 22–46 per seed were tied.

- **Check: PASS.** Where exact (MAGIC and Page Blocks), PACE-decision equals the brute force within 1e-9.
- **Primary.**
  - Decision deletion AUC on tied points, where lower is better.
  - PACE-decision was **better than PACE-SHAP on all 3 datasets, in 10 of 10 seeds each** (Holm p = 0.0059).
  - The differences: MAGIC −0.0506, Letter −0.0161, Page Blocks −0.1085.
- **Secondary.**
  - Decision deletion on untied points: better on all 3.
  - Decision insertion on tied points: better on all 3.
  - **Margin measures on tied points, PACE-SHAP's home ground:** PACE-decision was worse on Letter, by deletion (+0.0268, Holm p = 0.0059) and by insertion (−0.0246, p = 0.0410). It showed no significant difference on MAGIC and Page Blocks.
- **Reading (as planned).**
  - PACE-decision was better on at least 2 of the 3 datasets, so the paper offers it as the explanation at ties.
  - The paper fixes the margin sentence (S6) and reports the margin measures beside it.
  - As the plan's bias note says, each explainer leads on the quantity it explains, so neither is called better overall.

### 23.8 The PACE-SHAP certificate (Run P6, seeds 330–339)

Source: `results/shap_certificate/`. Self-repair at θ = 0.3 on the refinement split, with 100 explained rows per edit.

- **Check: PASS.** Over 6,085 accepted edits and 608,500 (edit, row) pairs, no row outside the certificate changed by more than 1e-12.
- **Certified share.**
  - The certificate held **99.1–100%** of the explained rows per edit.
  - The rows whose values really changed were 31.7% on Letter, 46.2% on Skin and 80.3–82.7% elsewhere.
- **Relative change** of the rows that changed:
  - median from 2.0e-5 (Letter) to 2.0e-3 (Page Blocks);
  - largest from 1.5 to 23 times the row's largest value before the edit.
- **Reading (as planned).**
  - The check passed and the certified share is at least 90% on every dataset, so the paper states Proposition 7 as **sound but loose**.
  - After an edit almost every row may change, so re-explaining every row costs about the same.

### 23.9 Edits that provably cannot hurt (Run P11, seeds 390–399)

Source: `results/cannot_hurt/`. The refinement split, with self-repair gated on the training rows.

- **Check: PASS.** No edit meeting condition S broke a correct training prediction. There were 116 such edits.
- **Share meeting S.**
  - 116 of 6,110 accepted edits met S: **1.9%** overall, from 0% on Thyroid to 8.0% on Page Blocks.
  - The gate rejected none of them.
- **Why S fails:** 90–100% of accepted edits cut off a training row of the box's own class, which breaks clause (i).
- **Second stage: not needed.** No dataset reached 10%, and the gate rejected no edit meeting S, so skipping the gate for those edits would give the same model.
- **Reading (as planned).**
  - The paper keeps Proposition 6 as a safety guarantee and says that S rarely holds.
  - It gives the fast path as a "Next steps" line only.
