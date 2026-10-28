# PACE-CFMNN: findings so far (Phases 1–2)

Main model: CFMNN with the fixed centroid rule and search expansion (θ = 0.3, γ = 4).
Raw tables: [`g0/summary.md`](g0/summary.md), [`faithfulness/summary.md`](faithfulness/summary.md)
and [`speed/summary.md`](speed/summary.md).

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

Each dataset uses 3 seeds × 50 test points. Features are deleted in each method's ranked order by
copying values from real data rows. *Margin AUC* measures how fast the predicted class's lead
shrinks; lower is more faithful.

| Dataset | PACE-flip | PACE-multi | Occlusion (brute force) | KernelSHAP | LIME | Random |
|---|---|---|---|---|---|---|
| S1 | **−0.107** | 0.070 | −0.107 | 0.051 | 0.074 | 0.275 |
| S2 | **−0.110** | 0.135 | −0.110 | 0.083 | 0.119 | 0.271 |
| S3 | **0.040** | 0.189 | 0.040 | 0.174 | 0.182 | 0.341 |
| Wine | **−0.047** | 0.027 | −0.047 | 0.002 | 0.037 | 0.255 |
| Breast Cancer | **0.158** | 0.176 | 0.158 | 0.198 | 0.231 | 0.421 |
| Thyroid | **0.534** | 0.569 | 0.534 | 0.606 | 0.646 | 0.741 |
| Shuttle | **0.250** | 0.254 | 0.250 | 0.293 | 0.297 | 0.433 |

Milliseconds per explanation, batch of 50 points:

| Dataset | PACE-multi | PACE-flip | PACE-flip, original loop | Occlusion | PACE-flip vs occlusion | KernelSHAP | LIME |
|---|---|---|---|---|---|---|---|
| S1 | 0.50 | 1.11 | 14.5 | 86.2 | 78× | 548 | 266 |
| S2 | 0.52 | 0.97 | 14.2 | 86.1 | 89× | 549 | 275 |
| S3 | 0.49 | 0.95 | 12.1 | 85.3 | 90× | 537 | 270 |
| Wine | 0.03 | 0.11 | 1.9 | 5.7 | 51× | 28 | 13 |
| Breast Cancer | 0.05 | 0.26 | 4.4 | 21.5 | 84× | 61 | 24 |
| Thyroid | 0.05 | 0.23 | 3.8 | 15.4 | 67× | 53 | 25 |
| Shuttle | 0.04 | 0.07 | 1.5 | 4.9 | 70× | 62 | 18 |

- **Sources:** the original-loop column is from `speed/summary.md` (same models and points); the rest
  are from `faithfulness/summary.md`.
- **One point per call:** PACE-flip is faster than PACE-multi on every dataset, for example S1 1.1 vs
  1.2 ms and Wine 0.36 vs 0.62 ms.
- **Fair timing:** CFMNN now processes points in cache-sized chunks (`chunk_elems = 65,536`; results
  unchanged).
  - This made every method that calls the model faster on S1–S3: KernelSHAP 1,697 → 548 ms, LIME
    820 → 266 ms, occlusion 264 → 86 ms.
  - So all methods are compared on equal terms.
- **Timing noise:** times vary between runs on this machine. In the previous run every Shuttle time
  was 1.5–2.7× lower. Compare methods within one run; `speed/summary.md` uses the median of 5 repeats
  for the fast methods.

Stability is the top-3 overlap when the explainer is re-run with another seed:

| Method | Stability |
|---|---|
| PACE (all versions) | 1.00 |
| KernelSHAP | 0.60–0.99 |
| LIME | 0.70–1.00 |

KernelSHAP is now repeatable.
- **Why it wasn't:** shap 0.54 samples from numpy's global random state, so the seed passed to it did
  not fix its sampling.
- **The fix:** `kernel_shap` now seeds that state for the call and restores it afterwards (tested).
  Re-running a benchmark row now gives the same numbers.
- **Before the fix:** two full runs differed by up to 0.02 AUC, and no conclusion changed.

Ground truth is the share of a method's top 3 that are truly relevant features (synthetic sets):

| Method | Ground truth | Note |
|---|---|---|
| PACE-flip | 0.51–0.57 | |
| The model's own decision sensitivity | 0.49–0.59 | with fair tie-breaking |
| KernelSHAP | 0.68–0.80 | above what the model's decisions rely on |
| LIME | 1.00 | above what the model's decisions rely on |

## 3. What we can claim

- **PACE-flip** is the most faithful explainer on all 7 datasets, and fully stable. It equals
  brute-force occlusion exactly (tested), but computes it in closed form from the additive box terms.
  - **Speed:** about 0.05–1.1 ms per explanation. That is 51–90× faster than occlusion, 95–280×
    faster than LIME and 230–890× faster than KernelSHAP, in the same run.
  - **How:**
    - each box's terms at the 32 replacement values are tabulated once;
    - an exact bound skips boxes that cannot change the decision or the margin (98% of boxes on S1–S3,
      71–81% on the real datasets);
    - all points and features run in one batched pass.
  - **Same output:** bit for bit identical to the original loop (tested on 7 datasets × 3 seeds), so
    every faithfulness, stability and ground-truth number is unchanged. It is just 12–28× faster.
- **PACE-multi** is the instant option (0.02–0.5 ms): perfectly stable, and second to PACE-flip on
  faithfulness on the real datasets.
- **PACE decomposition** equals exact Shapley values of the winning box's membership (tested), but only
  for that box's score, not for the final prediction.
- **Ground truth:** PACE reports what this model actually uses. LIME and SHAP rank the true features
  higher than the model's own decisions do, so they partly describe the data rather than the model.

## 4. What we must not claim

- That PACE-flip is more faithful than occlusion. It is identical; its contribution is exact speed.
- That PACE-flip is faster than PACE-multi in batches. There PACE-multi is still 2–5× faster per
  point; PACE-flip is faster only when points are explained one per call.
- Absolute milliseconds as general facts. They depend on the machine; compare methods measured
  together.
- That PACE recovers the data's true features better than LIME or SHAP.
- That SHAP equivalence holds for the final prediction.
- That results hold on Sepsis-like data. Sepsis is dropped because its labels are not predictable from
  its 3 features.
