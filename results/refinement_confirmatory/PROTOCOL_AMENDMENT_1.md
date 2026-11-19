# Protocol Amendment 1: Refinement Confirmatory Study (Phase 6, RQ3–RQ4)

**Date:** November 2026  
**Document Status:** Formal Protocol Amendment (Preserves Original [`PREREGISTRATION.md`](PREREGISTRATION.md))  
**Target:** RQ3 (Active Selection) and RQ4 (Explanation-Guided Hyperbox Repair)  
**Modules:** [`pace_cfmnn/refine.py`](../../pace_cfmnn/refine.py), [`pace_cfmnn/active.py`](../../pace_cfmnn/active.py), [`pace_cfmnn/contraction.py`](../../pace_cfmnn/contraction.py)  
**Execution Script:** [`scripts/run_refinement.py`](../../scripts/run_refinement.py)

---

## 1. Context and Rationale for Amendment

The original preregistered protocol ([`PREREGISTRATION.md`](PREREGISTRATION.md)) specified fixed settings, arms, and statistical endpoints for evaluating hyperbox selective refinement across seeds 10–19. However, findings from the exploratory pilot (seeds 0–2 across 5 datasets; `results/refinement_pilot/`) revealed two fundamental methodological vulnerabilities that required resolution before confirmatory execution:

1. **Suboptimality of Naive Contrastive Shrink:**
   - In overlap regions, the original contrastive rule (`feature_explained`) picked the feature of maximal contrastive magnitude where the point was closest to the wrong box's center. This caused deep cuts severing on average $27\%$ of the box width and eliminating twice as many training exemplars as minimal cuts ($5\%$).
   - As a result, the geometric control (`A+B+C'-volume`, minimal cut) outperformed the naive explanation cut on Thyroid ($-2.9$ points macro-F1) and MAGIC ($-1.5$ points).
   - **Correction:** Introduction of `feature_explained_minimal` (`A+B+C-minimal`), which restricts candidate features to those providing positive contrastive support toward the true class, then selects the feature requiring the minimal volume reduction.

2. **Absence of Classical Contraction Baseline:**
   - CFMNN by design lacks a contraction step (unlike Simpson's 1992 FMNN). Without a classical non-active contraction baseline, any gains observed from shrinking could simply be "overlap contraction put back" rather than evidence for active, explanation-guided refinement.
   - **Correction:** Integration of `simpson_contraction()` as a standardized non-active baseline (`B-Simpson non-active contraction`) evaluated on the exact same dataset splits.

3. **Gate Label Accounting:**
   - The acceptance gate evaluates candidate edits on labelled acceptance rows ($2\times\text{--}6\times$ the pool label budget).
   - **Correction:** Reporting `total_labels = pool_labels + gate_labels` to ensure zero hidden labelling overhead.

---

## 2. Amended Experimental Arms

The comparison maintains the identical 50/20/10/10/10 role split, budget ($\min(300, \text{pool}/4)$ in 10 rounds), gate thresholds, and box growth caps defined in [`PREREGISTRATION.md`](PREREGISTRATION.md).

### Non-Active Baselines:
- **`B0 no refinement`:** Starting CFMNN trained on training split, zero refinement.
- **`B9 refit on all pool labels`:** CFMNN retrained on training split + all pool rows (upper bound on label acquisition).
- **`B-Simpson non-active contraction` (NEW):** Starting CFMNN post-processed with Simpson (1992) FMNN overlap contraction on training features, resolving all cross-class overlaps with zero additional labels or oracle queries.

### Active Refinement Arms:
- **`order_ensemble | A+B`:** Points selected by training-order ensemble disagreement; point addition only with CFMNN expansion rule.
- **`order_ensemble | A+B+C`:** Points selected by order ensemble; naive contrastive explanation face shrink + true point addition.
- **`order_ensemble | A+B+C'-random`:** Points selected by order ensemble; random face shrink control + true point addition.
- **`order_ensemble | A+B+C'-volume`:** Points selected by order ensemble; minimal-width face shrink control + true point addition.
- **`order_ensemble | A+B+C-minimal` (NEW):** Points selected by order ensemble; minimal-cut explanation-guided shrink (`feature_explained_minimal`) + true point addition.
- **`random | A+B+C` (Original RQ3):** Points selected uniformly at random; naive contrastive explanation face shrink + true point addition.

---

## 3. Explicit Boundary Between RQ3 Preregistration and Exploratory Pilot

To ensure strict scientific integrity:
- **Preregistered RQ3 (Confirmatory):** The confirmatory selection test is **strictly restricted to `order_ensemble | A+B+C` versus `random | A+B+C`**.
- **Exploratory Pilot Selectors (Not Confirmatory):** Alternative acquisition functions evaluated during the exploratory pilot—namely Margin Uncertainty, Prediction Disagreement ($PD$), Corrected Uncertainty ($U^*$), and the PACE Refinement Score ($PRS$)—are **exploratory findings only**. Because the pilot established that acquisition functions had negligible differential impact under add-only edits ($\pm 0.5$ macro-F1), they were formally excluded from the confirmatory protocol to focus statistical power on the primary hypotheses.

---

## 4. Formal Statistical Hypotheses and Endpoints

All tests use a two-sided paired Wilcoxon signed-rank test across seeds, with Holm-Bonferroni correction across datasets at $\alpha = 0.05$.

### Confirmatory Repair Endpoints (RQ4):
- **R1 (Original Primary):** `order_ensemble | A+B+C` vs `order_ensemble | A+B+C'-volume`
- **R2 (Original Secondary):** `order_ensemble | A+B+C` vs `order_ensemble | A+B` (Does shrinking help vs add-only?)
- **R3 (Original Secondary):** `order_ensemble | A+B+C` vs `order_ensemble | A+B+C'-random`
- **R4 (Amended Primary):** `order_ensemble | A+B+C-minimal` vs `order_ensemble | A+B+C'-volume`  
  *Hypothesis:* Explanation guidance constrained to minimal disturbance outperforms unguided geometric minimal disturbance.
- **R5 (Amended Secondary):** `order_ensemble | A+B+C-minimal` vs `order_ensemble | A+B+C`  
  *Hypothesis:* Minimal-cut explanation shrink resolves box-core destruction and significantly outperforms naive contrastive cuts.
- **R6 (Amended Baseline):** `order_ensemble | A+B+C-minimal` vs `B-Simpson non-active contraction`  
  *Hypothesis:* Explanation-guided active refinement provides significant accuracy gains beyond non-active overlap contraction.

### Confirmatory Selection Endpoint (RQ3):
- **S1 (Original Primary):** `order_ensemble | A+B+C` vs `random | A+B+C`  
  *Hypothesis:* Querying instances flagged by training-order disagreement yields higher repair efficacy than random pool sampling.

---

## 5. Execution Protocol and Dataset Feasibility

- **Datasets:** Eligible datasets with $\ge 25$ repairable errors in all pilot seeds: `thyroid`, `landsat`, `page_blocks`, `magic`, `letter`, and `skin`.
- **Phased Execution:**
  - **Tier 1 (Immediate Confirmatory Feasibility):** `thyroid`, `page_blocks`, and `landsat` (all 10 seeds, seeds 10–19). Fast convergence ($< 10\text{ minutes}$ total with `--jobs 4`).
  - **Tier 2 (Extended Confirmatory Compute):** `magic` (~15 min), `letter` (~2.5 hours), `skin` (~3.5 hours). Scheduled for high-throughput batch execution.
- **Outcome Reporting:** Zero fabrication. Any pending datasets are reported as computationally queued; completed datasets are reported with exact metrics, standard errors, and Holm-corrected $p$-values.
