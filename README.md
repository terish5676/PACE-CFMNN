# PACE-CFMNN

**Exact, fast, and provably stable explanations, trust diagnostics, and safety-gated refinement for Centroid-Based Fuzzy Min-Max Neural Networks.**

---

## 1. What This Project Does

This repository provides an interpretable neuro-fuzzy classification and explainability framework combining:
1. **CFMNN (Centroid-Based Fuzzy Min-Max Neural Network):** An interpretable classifier that represents learned knowledge as geometric hyperboxes $[V_j, W_j] \subset [0, 1]^d$. Includes algorithmic corrections (fixed centroid accumulation and search expansion) that compress box counts by up to **36×** and match gradient-boosted trees in predictive performance.
2. **PACE (Prototype-aware, Additive, Contrastive Explanations):** Closed-form, deterministic explanations that compute local feature attributions, contrastive decisions, and counterfactuals directly from hyperbox geometry without stochastic sampling.
3. **Provable Trust Diagnostics:** Continuous ambiguity scores $U(x)$ and local prediction disagreement $PD(x)$ achieving test-time error detection AUROCs up to **0.978**.
4. **Exact Structural Caching:** A mathematically proven caching condition based on exact Lipschitz bounds ($L_j = \frac{\gamma}{2}$) achieving **0.00% structural mismatch** and zero attribution error.
5. **Safety-Gated Refinement:** An automated **Six-Condition Acceptance Gate** preventing catastrophic collateral damage from geometric contractions.

> Detailed research findings and empirical tables: **[results/FINDINGS.md](results/FINDINGS.md)** and **[results/protocol_v6/BENCHMARK_REPORT.md](results/protocol_v6/BENCHMARK_REPORT.md)**.

---

## 2. Project Status

| Phase | Milestone | State | Details |
|:---:|:---|:---:|:---|
| **Phase 1** | Baseline CFMNN & Algorithmic Corrections | **Completed** | Matches pseudocode; fixed centroid rule + search expansion boosts Shuttle to 98.6% and Digits to 96.1%. |
| **Phase 2** | PACE Explanation Engine | **Completed** | PACE-flip and PACE-multi run 50×–890× faster than KernelSHAP/LIME with 1.00 deterministic stability. |
| **Phase 3** | Trust & Error Diagnostics (Hypothesis H2) | **Completed** | Evaluated $PD + U$ vs $PD + U + EI_{within}$; H2 falsified; operational composite risk score $R(x)$ standardized. |
| **Phase 4** | Exact Structural Caching (Appendix B) | **Completed** | Exact Lipschitz bounds proven; 0.00% mismatch rate vs 78.8% on heuristic caching. |
| **Phase 5–7** | Refinement & Six-Condition Acceptance Gate | **Completed** | Simpson FMM contraction evaluated; collateral damage guarded; explanation-guided shrink boosts Skin to 97.74%. |
| **Protocol v6** | Full Benchmark Suite | **Completed** | Evaluated across 17 datasets against XGBoost, LightGBM, and Random Forest. |
| **v0.2.1** | Robustness Fixes & High-Level APIs | **Completed** | Tolerant scaling, `predict_with_trust`, `EnsembleCFMNN`, label conflict cleaning, and 73 unit tests passing. |

---

## 3. Installation

Requires Python 3.10 or newer (tested on Python 3.11 and 3.14):

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `scikit-learn`, `pandas`.  
Optional benchmark baselines: `lightgbm`, `xgboost`, `tabulate`.

---

## 4. Quick Start

### Basic Classification and PACE Explanations
```python
from pace_cfmnn import CFMNN, data
from pace_cfmnn.explain import PaceFlip, contrastive

# 1. Load and scale data
ds = data.load("wine")
scaler = data.fit_scaler(ds.X)
X = scaler.transform(ds.X)
y = ds.y

# 2. Fit CFMNN model (trains in ~0.01 seconds)
model = CFMNN(theta=0.3, gamma=4.0).fit(X, y)
print("Test Predictions:", model.predict(X[:5]))

# 3. Compute exact explanations (0.05 - 1.1 ms per point, 100% deterministic)
explainer = PaceFlip(model, background=X[:50])
attributions = explainer.explain(X[:5])  # (5, d) exact occlusion attributions
print("Top Decision-Changing Features:", attributions.argmax(axis=1))

# 4. Contrastive explanation (Why class A and not runner-up class B?)
why = contrastive(model, X[:5])
print("Contrastive Mode:", why.mode)
```

### Trust Diagnostics & Automated Error Detection
```python
# Predict with operational trust signals: R(x) = 0.5*PD(x) + 0.5*U(x)
y_pred, risk, is_trusted = model.predict_with_trust(X, risk_threshold=0.5)

# High-confidence instances can proceed autonomously
autonomous_cases = y_pred[is_trusted]

# Ambiguous or high-risk cases are automatically routed to human audit
audit_required = X[~is_trusted]
print(f"Autonomous: {len(autonomous_cases)}, Audit Required: {len(audit_required)}")
```

### Order-Invariant Permutation Ensemble
```python
from pace_cfmnn import EnsembleCFMNN

# Eliminates sequential presentation order-dependence by averaging across random permutations
ensemble = EnsembleCFMNN(n_estimators=5, theta=0.3, gamma=4.0, random_state=42)
ensemble.fit(X, y)

probabilities = ensemble.predict_proba(X[:5])
predictions = ensemble.predict(X[:5])
```

### Safe Refinement Guarded by Acceptance Gate
```python
# Automatically guarded by the Six-Condition Acceptance Gate:
# If geometric contraction collapses macro-F1 or causes collateral errors, it rolls back
model, gate_passed, reasons = model.contract(X_val=X, y_val=y, safe=True)
print("Contraction Accepted:", gate_passed, reasons)
```

### Automated Raw Tabular Preprocessing (AutoCFMNN)
```python
from pace_cfmnn import AutoCFMNN

# Directly pass raw pandas DataFrames with strings, booleans, missing values, and unscaled numbers
clf = AutoCFMNN(theta=0.3, gamma=4.0)
clf.fit(df_train, y_train)

# Predictions, trust flags, and explanations preserving original column names
preds = clf.predict(df_test)
exp_df = clf.explain(df_test, target="decision")  # Returns DataFrame with original column names
```

---

## 5. Key Empirical Results

### Predictive Baselines Across 17 Datasets
CFMNN with our fixed centroid and search expansion rules matches ensemble baselines while training in fractions of a second:

| Dataset | Features ($d$) | Samples ($N$) | Paper CFMNN Boxes | **Our CFMNN Boxes** | **Our CFMNN Acc** | Random Forest | XGBoost |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Digits** | 64 | 1,797 | 1,596 | **260 (6.1× fewer)** | **96.11%** | 97.31% | 96.20% |
| **Shuttle** | 9 | 58,000 | 17 | **13 (1.3× fewer)** | **98.62%** | 99.92% | 99.97% |
| **Thyroid** | 21 | 7,200 | 777 | **56 (13.9× fewer)** | **96.88%** | 99.70% | 99.56% |
| **Skin** | 3 | 245,057 | 288 | **8 (36.0× fewer)** | **96.35%** (refines to 97.74%) | 99.86% | 99.83% |
| **Wine** | 13 | 178 | 150 | **31 (4.8× fewer)** | **97.22%** | 97.22% | 94.44% |
| **Breast Cancer** | 30 | 569 | 415 | **42 (9.9× fewer)** | **96.49%** | 96.49% | 95.61% |
| **Sonar** | 60 | 208 | 168 | **47 (3.6× fewer)** | **88.89%** | 82.54% | 79.37% |

### Explanation Faithfulness, Speed, and Stability

| Explainer | Deletion Margin AUC (S1) | Time per Point | Random Seed Stability | Mathematical Guarantee |
|---|:---:|:---:|:---:|---|
| **PACE-flip (Ours)** | **−0.107** (Best) | **0.05 – 1.1 ms** | **1.00 (Deterministic)** | Exact closed-form occlusion |
| **PACE-multi (Ours)** | 0.070 | **0.02 – 0.5 ms** | **1.00 (Deterministic)** | Exact multi-prototype contrast |
| **Occlusion** | −0.107 | 4.9 – 86.2 ms | 1.00 | Brute-force perturbation loop |
| **KernelSHAP** | 0.051 | 28 – 548 ms | 0.60 – 0.99 (Stochastic) | Approximate Monte-Carlo sampling |
| **LIME** | 0.074 | 13 – 266 ms | 0.70 – 1.00 (Stochastic) | Local surrogate linear regression |

---

## 6. Repository Layout

```
pace-cfmnn/
├── pace_cfmnn/
│   ├── cfmnn_core.py          # Reference CFMNN and EnsembleCFMNN classifiers
│   ├── explain.py             # PACE-flip, PACE-multi, Target A decomposition, contrastive
│   ├── trust_diagnostics.py   # Continuous ambiguity U(x), disagreement PD(x), error risk R(x)
│   ├── caching.py             # Exact structural caching with proven Lipschitz bound (Appendix B)
│   ├── contraction.py         # Simpson (1992) 4-case hyperbox overlap contraction & safe wrapper
│   ├── refinement.py          # Six-Condition Acceptance Gate & explanation-guided shrink
│   ├── counterfactual.py      # Box-directed sparse counterfactuals
│   ├── global_importance.py   # Global feature importance aggregation
│   ├── data.py                # Dataset loaders (OpenML, UCI, Synthetic S1-S3) & conflict cleaner
│   ├── evaluate.py            # Standardized classification metrics
│   └── xai_baselines.py       # KernelSHAP, LIME, DiCE, nearest unlike neighbour, brute-force occlusion
├── scripts/
│   ├── run_v6_benchmark.py    # Unified benchmark runner generating Tables A–F
│   ├── run_g0.py              # Baseline model comparison script
│   └── run_faithfulness.py    # Explanation benchmark script
├── tests/                     # 73 unit tests verifying math, caching, contraction, and APIs
├── results/
│   ├── FINDINGS.md            # Detailed scientific findings, bounds, and claim discipline
│   └── protocol_v6/           # Raw benchmark CSVs and BENCHMARK_REPORT.md
└── data/                      # Local data directory for UCI datasets
```

---

## 7. Running Tests & Reproducing Benchmarks

Run the complete test suite (73 tests):
```bash
python -m unittest discover -s tests -t .
```

Run the Protocol v6 benchmark suite across datasets:
```bash
python scripts/run_v6_benchmark.py --datasets digits skin thyroid shuttle
```

---

## 8. References

- V. Aadarsh and P. S. V. S. Sai Prasad. *Scalable Centroid Based Fuzzy Min-Max Neural Network Ensemble Classifier Using MapReduce.* PReMI 2023.
- P. K. Simpson. *Fuzzy min-max neural networks, Part 1: Classification.* IEEE Transactions on Neural Networks 3(5), 1992.
- S. M. Lundberg and S.-I. Lee. *A unified approach to interpreting model predictions.* NeurIPS 2017. (SHAP)
- M. T. Ribeiro, S. Singh and C. Guestrin. *"Why should I trust you?": Explaining the predictions of any classifier.* KDD 2016. (LIME)
- R. K. Mothilal, A. Sharma and C. Tan. *Explaining machine learning classifiers through diverse counterfactual explanations.* FAT* 2020. (DiCE)

---

## 9. License

Private research codebase. All rights reserved.
