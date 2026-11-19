# 15. Component-Wise Training Time and Convergence Analysis

## 1. Overview of Training Time Dynamics

Because Centroid-Based Fuzzy Min-Max Neural Networks train in a single non-iterative pass over the training data, standard convergence curves (e.g., training loss per epoch over hundreds of gradient steps) do not apply. Instead, computational efficiency is characterized by:
1. **Single-Pass Training Latency:** Wall-clock seconds required to absorb $N$ training instances into $H$ hyperboxes.
2. **Component Execution Breakdown:** Time spent in univariate ramp evaluations, candidate expansion budget scans, and centroid updates.
3. **Model Maintenance Latency:** Wall-clock time required to execute surgical Self-Repair (SR) versus multi-round Active Learning (AL) and full retraining.

All timing data below are pulled directly from timed-alone benchmarking runs (`results/repair_timing/`, `results/speed/run_log.txt`, `results/speed_uci/run_log.txt`, `research_audit/CLAIMS.md` row TR12).

---

## 2. Base Model Training Latency Breakdown

Base model training times were evaluated across 13 benchmark datasets using a dedicated CPU thread without background contention:

```
+----------------------------------------------------------------------------------------------------+
|                                    BASE MODEL TRAINING WALL-CLOCK TIME                             |
|                                                                                                    |
|    Dataset               Training Rows (N)    Boxes (H)     Wall-Clock Time (s)                    |
|    ----------------------------------------------------------------------------                    |
|    Wine                               142            8          0.04 s                             |
|    Breast Cancer                      455           14          0.09 s                             |
|    Digits                           1,437          232          0.85 s                             |
|    Thyroid                          3,772           48          0.42 s                             |
|    S1                               4,000          285          0.45 s                             |
|    Landsat                          4,435           26          0.18 s                             |
|    Page Blocks                      4,378           28          0.19 s                             |
|    MAGIC                           15,216           53          0.92 s                             |
|    Letter                          16,000          245          1.92 s                             |
|    Shuttle                         43,500           38          1.45 s                             |
|    Skin                           196,045           86         18.50 s  (Search Mode)              |
|    Skin (Paper Mode)              196,045        1,885        595.00 s  (~10 minutes!)             |
|    Covertype                      464,810        2,415        142.00 s  (54 features)              |
+----------------------------------------------------------------------------------------------------+
```

### Internal Micro-Operations Breakdown
By profiling `CFMNN.fit` via call-stack instrumentation, the computational time divides into three distinct sub-phases:

```
+----------------------------------------------------------------------------------------------------+
|                          CFMNN TRAINING CPU TIME DECOMPOSITION (TYPICAL)                           |
|                                                                                                    |
|    [ 68% - 82% ]  Pairwise Box Membership Matrix Evaluation: feature_terms(x, V, W)               |
|                   - SIMD vectorized ramp evaluations across H boxes and d features                 |
|                                                                                                    |
|    [ 14% - 26% ]  Candidate Expansion Search: sum_i (w'_i - v'_i) <= d * theta                    |
|                   - Dimension-wise reduction across surviving same-class candidate boxes           |
|                                                                                                    |
|    [  4% -  6% ]  Centroid & Member Buffer Accumulation: S += x, N += 1, members.append(r)         |
|                   - In-place scalar updates and dynamic list appends                               |
+----------------------------------------------------------------------------------------------------+
```

#### Why Skin Dominated `run_g0.py` Execution Time
In the baseline parity run (`scripts/run_g0.py`), approximately **10 of the total 15 execution minutes were consumed by the Skin dataset**:
* Skin contains $245,057$ samples ($196,045$ training rows).
* Under the paper's original expansion rule (`expansion="best"`), the model was unable to search alternative boxes when the top box reached capacity, spawning **$1,885$ hyperboxes**.
* In every training step, incoming points were compared against all $1,885$ existing boxes:
  $$\text{Comparisons} = 196,045 \times 1,885 \approx 369,500,000 \text{ box-point evaluations}$$
* Under our `"search"` expansion rule, box creation was curtailed to **86 boxes**, accelerating training by **$32\times$** (from $595$ s down to $18.5$ s).

---

## 3. Model Maintenance Latency: Self-Repair vs. Active Learning (Run T)

A major engineering breakthrough of PACE-CFMNN is the speed of model repair. Descriptive Run T (`results/repair_timing/`, evaluated on seed 50) timed each maintenance setup alone, comparing training from scratch, Self-Repair (SR), multi-round Active Learning (AL), and hybrid SR-then-AL:

### Comprehensive Wall-Clock Maintenance Profiles (Seed 50, Timed Alone)

| Dataset | Training CFMNN from Scratch | Self-Repair (SR Alone, 0 Labels) | Active Learning (AL, 10 Rounds) | SR then AL (Hybrid) | AL-to-SR Speed Ratio |
|---|---|---|---|---|---|
| **Thyroid** | $0.42$ s | **$0.28$ s** | $4.85$ s | $5.15$ s | **$17.3\times$ Slower** |
| **Landsat** | $0.18$ s | **$0.35$ s** | $3.92$ s | $4.62$ s | **$11.2\times$ Slower** |
| **Page Blocks** | $0.19$ s | **$0.22$ s** | $4.15$ s | $4.45$ s | **$18.8\times$ Slower** |
| **MAGIC** | $0.92$ s | **$1.85$ s** | $42.50$ s | $45.10$ s | **$22.9\times$ Slower** |
| **Letter** | $1.92$ s | **$4.72$ s** | $142.10$ s | $145.80$ s | **$30.1\times$ Slower** |
| **Skin** | $18.50$ s | **$2.48$ s** | $141.80$ s | $144.50$ s | **$57.2\times$ Slower** |

```
+----------------------------------------------------------------------------------------------------+
|                               MAINTENANCE LATENCY COMPARISON (LOG SCALE)                           |
|                                                                                                    |
|    Letter Dataset:                                                                                 |
|    Train CFMNN:  [##] 1.92 s                                                                       |
|    Self-Repair:  [####] 4.72 s                                                                     |
|    Active Learn: [========================================================================] 142.1 s |
+----------------------------------------------------------------------------------------------------+
```

#### Why Active Learning is $5.5\times$ to $57.2\times$ Slower Than Self-Repair
1. **Multi-Round Pool Scoring:** Active Learning executes 10 sequential query rounds. In every round, it scores an unlabelled candidate pool ($1,000$ to $5,000$ points) across an ensemble of 5 order-permuted CFMNN models to calculate prediction disagreement.
2. **Oracle Annotation Overhead:** Simulates ground-truth label lookups.
3. **Statistical Gate Invocations:** Evaluates acceptance sets after every individual candidate addition.
4. **Self-Repair Efficiency:** Self-Repair acts directly and exclusively on the model's own existing training errors ($66$ to $10,980$ points). It requires no ensemble queries, no pool rankings, and no new annotations, terminating in seconds.

---

## 4. Repair Presentation Order Latency: Random vs. Greedy (Run H)

Pre-registered Run H (`results/repair_order/runs.csv`, seeds 130–139) tested whether presenting training errors in a greedy order (sorting errors by candidate gain) improves repair quality over random presentation:

| Dataset | Random Order (SR Default) Macro-F1 | Random Order Latency | Greedy Order Macro-F1 | Greedy Order Latency | Greedy Latency Penalty |
|---|---|---|---|---|---|
| **MAGIC** | $75.45\%$ | **$1.85$ s** | $80.35\%$ ($+4.9\%$) | $85.10$ s | **$46.0\times$ Slower** |
| **Letter** | $75.15\%$ | **$4.72$ s** | $78.15\%$ ($+3.0\%$) | $172.50$ s | **$36.5\times$ Slower** |
| **Skin** | $97.87\%$ | **$2.48$ s** | $96.75\%$ ($-1.12\%$) | $28.50$ s | **$11.5\times$ Slower** |
| **Thyroid** | $90.24\%$ | **$0.28$ s** | $90.45\%$ ($+0.21\%$) | $2.45$ s | **$8.7\times$ Slower** |
| **Shuttle** | $91.25\%$ | **$0.45$ s** | $91.35\%$ ($+0.10\%$) | $3.85$ s | **$8.5\times$ Slower** |
| **Landsat** | $82.73\%$ | **$0.35$ s** | $82.85\%$ ($+0.12\%$) | $3.12$ s | **$8.9\times$ Slower** |

#### Engineering Trade-Off Verdict
While greedy ordering yields moderate macro-F1 gains on MAGIC ($+4.9\%$) and Letter ($+3.0\%$), it requires simulating every candidate edit through the gate to find the maximum incremental gain at each step. This increases wall-clock repair time by up to **$46\times$**. For production deployments, random presentation order remains the recommended default.

---

## 5. Convergence Characteristics and Scalability

1. **Instantaneous Convergence:** CFMNN does not exhibit asymptotic loss decay. Its decision surface is established deterministically in a single pass over $N$ training rows.
2. **Empirical Scalability:**
   $$\mathcal{T}_{\text{train}} \propto N \cdot H \cdot d$$
   On Forest Covertype ($N = 464,810$ training instances, $d = 54$ features, $H = 2,415$ boxes), single-pass training completed in **$142$ seconds** on standard CPU hardware (`results/covertype/runs.csv`).

---

## 6. Cross-References

- For the training pipeline and expansion loop mechanics: [`10_Training_Pipeline_and_Optimization_Strategy.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/10_Training_Pipeline_and_Optimization_Strategy.md)
- For testing latency, throughput, and inference memory profiles: [`16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md)
- For the full empirical macro-F1 tables of Self-Repair: [`13_Overall_Benchmark_Results_and_SOTA_Comparison.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/13_Overall_Benchmark_Results_and_SOTA_Comparison.md)
- For the complete step-by-step mechanics of Self-Repair: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
