# 03. Tech Stack, Environment, and Repository Map

## 1. Environment & Technical Specifications

The PACE-CFMNN codebase was designed, implemented, and audited for high-performance scientific reproducibility across both consumer laptop workstations and multi-core research compute nodes.

### 1.1 Hardware & Operating System Baseline

Based on workspace audit logs (`results/speed/run_log.txt`, `results/speed_uci/run_log.txt`, `research_audit/HOME_PC_RUNBOOK.md`, `results/FINDINGS.md`), the experimental suite was evaluated across two primary physical systems:

| System Parameter | Development Workstation (Laptop) | Execution Compute Node (Home PC) |
|---|---|---|
| **Operating System** | Microsoft Windows 11 / Windows 10 | Microsoft Windows 11 / Linux Container |
| **CPU Architecture** | Multi-core x86_64 Intel/AMD | 12-Core / 24-Thread High-Throughput CPU |
| **Python Version** | Python 3.10+ (Developed with Python 3.14.0a) | Python 3.10+ / Python 3.11 CPython |
| **Primary Execution Shell** | PowerShell 7 / Windows Terminal | PowerShell / Bash |
| **Parallel Execution** | `--jobs 2` to `--jobs 6` | `--jobs 6` to `--jobs 12` |
| **RAM Configuration** | 16 GB to 32 GB DDR4/DDR5 | 32 GB to 64 GB High-Speed DDR5 |
| **GPU Requirements** | **None (Pure CPU Analytical Engine)** | **None (Pure CPU Analytical Engine)** |

> [!NOTE]
> PACE-CFMNN relies on closed-form hyperbox geometry and vectorized NumPy/SciPy operations. It requires **no CUDA or GPU acceleration**, allowing complete reproducibility on commodity hardware.

---

### 1.2 Python Runtime & Package Inventory

The core system dependencies are specified in `pyproject.toml` and pinned down to exact patch releases in `requirements-lock.txt`:

```
+----------------------------------------------------------------------------------------------------+
|                                    DEPENDENCY TIER ARCHITECTURE                                    |
|                                                                                                    |
|  [Tier 1: Core Classifier Engine]                                                                  |
|      numpy >= 2.0 (2.5.3)       | scipy >= 1.13 (1.18.1)   | scikit-learn >= 1.5 (1.9.1)            |
|      pandas >= 2.2 (3.0.5)      | setuptools >= 69.0.0     |                                       |
|                                                                                                    |
|  [Tier 2: Baseline Explainability Suite (.[xai])]                                                  |
|      shap >= 0.54 (0.54.1)      | lime >= 0.2 (0.2.0.1)    | dice-ml >= 0.12 (0.12.0)               |
|                                                                                                    |
|  [Tier 3: Readable Baseline Models (.[baselines])]                                                 |
|      interpret-core >= 0.7 (0.7.8 - EBM) | hyperbox-brain >= 0.1.6 (0.1.6 - GFMM)                  |
|                                                                                                    |
|  [Tier 4: Build, Testing & Scripting Tools]                                                        |
|      unittest (Standard Library)| multiprocessing (StdLib) | gzip (External Binary)                |
+----------------------------------------------------------------------------------------------------+
```

#### Exact Version Manifest (`requirements-lock.txt` & `requirements.txt`)
* `numpy==2.5.3`: High-performance multidimensional array processing, broadcasting, and linear algebra routines.
* `scipy==1.18.1`: Statistical distributions, two-sided Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`), and spatial metrics.
* `scikit-learn==1.9.1`: Evaluation metrics (`accuracy_score`, `f1_score`, `roc_auc_score`, `precision_recall_curve`, `auc`), `NearestNeighbors` for plausibility and Trust Score, and base estimators (`ClassifierMixin`, `BaseEstimator`).
* `pandas==3.0.5`: Dataframe manipulation, logging aggregation, and CSV serialization (`runs.csv`).
* `shap==0.54.1`: Sampling-based KernelSHAP benchmark comparator (`shap.KernelExplainer`).
* `lime==0.2.0.1`: Local Interpretable Model-agnostic Explanations comparator (`lime.lime_tabular.LimeTabularExplainer`).
* `dice-ml==0.12.0`: Diverse Counterfactual Explanations comparator (`dice_ml.Dice`).
* `interpret-core==0.7.8`: Explainable Boosting Machine baseline (`interpret.glassbox.ExplainableBoostingClassifier`).
* `hyperbox-brain==0.1.6`: General Fuzzy Min-Max Neural Network baseline (`hbbrain.numerical_data.incremental_learner.onln_gfmm.OnlineGFMM` and `ImprovedOnlineGFMM`).

---

### 1.3 Containerization & Dockerfile Analysis

The repository provides an isolated deployment specification in `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /workspace
# Install system packages (gzip needed for UCI datasets)
RUN apt-get update && apt-get install -y --no-install-recommends gzip git && rm -rf /var/lib/apt/lists/*
# Install pinned dependencies
COPY requirements-lock.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements-lock.txt
# Copy source code and scripts
COPY pace_cfmnn ./pace_cfmnn
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY release ./release
# Install package in editable mode
RUN pip install -e .[all]
```

---

## 2. Annotated Repository Directory Map

The entire workspace directory tree is structured with strict academic rigor, separating production source code, unit tests, reproducibility scripts, raw experimental results, and external audit assets:

```
pace-cfmnn-main/
│
├── .dockerignore                     # Files excluded from container build context
├── .gitignore                        # Git ignore patterns (bytecode, raw data files, checkpoints)
├── Dockerfile                        # Multi-stage production container build specification
├── LICENSE-PENDING.md                # Licensing analysis and GPL-3.0 compatibility notes
├── pyproject.toml                    # Standard Python packaging configuration (PEP 517/518/621)
├── README.md                         # Primary project documentation, CLI reference, and findings summary
├── requirements.txt                  # Minimal third-party runtime dependencies
├── requirements-lock.txt             # Pinned hash-locked dependencies for byte-for-byte reproducibility
│
├── data/                             # Dataset ingestion directory
│   └── README.md                     # Data download instructions, MD5 hashes, and cohort schemas
│
├── docs/                             # Academic documentation and research deep-dives
│   ├── figures/                      # High-resolution conceptual diagrams and RQ3 figures
│   │   ├── rq3_1_membership.png      # Ramp membership function visualization
│   │   ├── rq3_2_order.png           # Training-order sensitivity visualization
│   │   ├── rq3_3_tie_error.png       # Geometry of cross-class tie error
│   │   ├── rq3_4_shrink_steps.png    # Surgical boundary face-shrink progression
│   │   ├── rq3_5_decision_map.png    # 2D decision boundary map before and after repair
│   │   ├── rq3_6_outside_change.png  # Prediction invariance certificate verification
│   │   ├── rq3_7_macro_f1.png        # Macro-F1 trajectories across repair rounds
│   │   ├── rq3_8_fixed_broken.png    # Fixed vs broken prediction breakdown
│   │   └── rq3_numbers.txt           # Machine-generated verification numbers for doc text
│   ├── peer_review_paper_text.md     # Draft paper text and propositions for Phase 1 submission
│   ├── RQ3_self_repair.md            # Comprehensive theoretical and empirical report on RQ3
│   ├── rq3_figures.py                # Standalone script generating all verified RQ3 figures
│   └── rq3_timing.py                 # Benchmarking script for self-repair execution latency
│
├── pace_cfmnn/                       # Core Production Library Source Code
│   ├── __init__.py                   # Package exports (CFMNN, PaceClassifier, PaceShap, PaceFlip)
│   ├── active.py                     # Active learning loop, label budgeting, selection, rollback
│   ├── api.py                        # High-level scikit-learn compatible front-end (`PaceClassifier`)
│   ├── audit.py                      # Proposition 4 & 7 prediction and explanation certificates
│   ├── cfmnn_core.py                 # Core CFMNN implementation, ramp functions, and training
│   ├── contraction.py                # Simpson (1992) FMNN contraction baseline implementation
│   ├── counterfactual.py             # Analytical box-directed and sparse counterfactual generators
│   ├── data.py                       # Synthetic generators (S1-S3), UCI loaders, scalers, splitters
│   ├── decision.py                   # Proposition 10: Exact Shapley values of categorical decisions
│   ├── evaluate.py                   # Evaluation metric computation (accuracy, macro-F1, recall)
│   ├── explain.py                    # PACE explainers (PaceShap, PaceFlip, contrastive, multi)
│   ├── flip_exact.py                 # Proposition 8: Exact piecewise linear single-feature flip sets
│   ├── fmm_baselines.py              # Hyperbox-brain baseline adapter (GFMM, EFMM, IOL-GFMM)
│   ├── gfmm_explain.py               # Proposition 9: PACE-SHAP adapted for min-membership networks
│   ├── global_importance.py          # Data-based and model-only global feature importance
│   ├── hardness.py                   # Proposition 5: #P-hardness reduction from weighted voting games
│   ├── neighborhood.py               # Perturbation engines for prediction disagreement and trust
│   ├── pace_score.py                 # PACE Refinement Score (PRS) rank aggregation
│   ├── realism.py                    # Manifold density scoring and off-manifold evaluation
│   ├── refine.py                     # Surgical face shrink, gate checks, and invariant validators
│   ├── resume.py                     # Atomic disk-backed experiment checkpointing (`--resume`)
│   ├── sampled_shap.py               # Modern permutation sampling engines for SHAP
│   ├── signals.py                    # Trust signals: Prediction Disagreement (PD), EI, and U*
│   ├── simpson_fmm.py                # Standalone reference implementation of Simpson's 1992 FMNN
│   ├── ties.py                       # Numerical tie-breaking utilities and centroid analytics
│   ├── trust_baselines.py            # External trust baselines: Trust Score, kNN disagreement
│   ├── trust_combine.py              # RQ2 v2: Multi-signal logistic regression combiner
│   └── xai_baselines.py              # Wrappers for KernelSHAP, LIME, DiCE, and brute-force occlusion
│
├── release/                          # Release artifacts, OSF bundles, and web visualizers
│   ├── demo/
│   │   └── pace_cfmnn_demo.html      # Interactive 2D client-side browser visualizer
│   └── osf/                          # Open Science Framework (OSF) pre-registration bundle
│       ├── MANIFEST.md               # Cryptographic manifest of pre-registered study protocols
│       └── sha256sums.txt            # SHA-256 hashes of all 38 pre-registration markdown plans
│
├── research_audit/                   # Comprehensive Academic & Systematic Research Audit
│   ├── scripts/                      # Recomputation and claims-verification scripts
│   │   ├── claims_base_rq2.py        # Validates base model and RQ2 trust signals
│   │   ├── claims_figures.py         # Recomputes and validates numbers in paper figures
│   │   ├── claims_repair.py          # Validates RQ3/RQ4 repair and certificate claims
│   │   ├── claims_rq1.py             # Validates RQ1 explanation benchmarks
│   │   ├── check_tie_errors.py       # Validates Proposition 1 across all training errors
│   │   └── check_matched_speed.py    # Recomputes like-for-like explainer execution speedups
│   ├── CHANGELOG.md                  # Complete version history across paper revisions (rev 262 to 657)
│   ├── CLAIMS.md                     # 134-row claims ledger mapping paper claims to CSV data
│   ├── HOME_PC_PROMPT_FABLE.md       # Execution prompt for high-throughput headless compute runs
│   ├── HOME_PC_RUNBOOK.md            # Comprehensive runbook for multi-process experimental suites
│   ├── ISSUES.md                     # Itemized log of paper discrepancies and resolved wording items
│   ├── LIMITATIONS_PLAN.md           # Formal experimental plan addressing reviewer limitations
│   ├── REPORT.md                     # Final audit report certifying empirical integrity
│   ├── SMOKE.md                      # Smoke-test validation protocol across all entry points
│   ├── THEORY.md                     # Mathematical theory draft and proofs for tough review
│   └── TOUGH_REVIEW_PLAN.md          # Implementation checklist addressing tough reviewer feedback
│
├── results/                          # 55 Experimental Result Suites (Committed CSVs, Logs & Summaries)
│   ├── FINDINGS.md                   # Master synthesis of all experimental findings (184 KB)
│   ├── cannot_hurt/                  # Empirical evaluation of Condition S (Proposition 6)
│   ├── cf_robustness/                # Counterfactual stability under model refits and perturbations
│   ├── conditional_settings/         # Sensitivity analysis of Gaussian kernel width sigma and rows
│   ├── contraction/                  # Self-repair vs Simpson (1992) contraction comparison
│   ├── counterfactuals/              # PACE vs DiCE counterfactual benchmark (seeds 40-49)
│   ├── covertype/                    # Large-scale benchmark on Forest Covertype (581,012 rows)
│   ├── decision/                     # Decision-level Shapley evaluation (Proposition 10)
│   ├── error_detection/              # RQ2 Exploratory trust signal evaluation
│   ├── error_detection_confirmatory/ # RQ2 Confirmatory pre-registered trust signal benchmarks
│   ├── error_detection_v2/           # RQ2 v2 Learned multi-signal combiner evaluation
│   ├── extensions_v1/                # Pre-registered RQ1 extensions (sparse CFs, global, conditional)
│   ├── faithfulness/                 # RQ1 Fair benchmark v2 (PACE-flip vs KernelSHAP vs LIME)
│   ├── faithfulness_v3/              # RQ1 Pre-registered benchmark v3 (PACE-SHAP vs KernelSHAP)
│   ├── heldout/                      # Pre-registered generalization test on 4 held-out OpenML datasets
│   ├── readable_baselines/           # Benchmark against EBM, Decision Tree, Logistic Regression, GFMM
│   ├── refinement_confirmatory/      # RQ3 First pre-registered repair experiment (seeds 10-19)
│   ├── refinement_v2_confirmatory/   # RQ3/RQ4 Second pre-registered repair run (seeds 50-59)
│   ├── repair_order/                 # Impact of training error presentation order (greedy vs random)
│   ├── repair_theta/                 # Interaction between hyperbox size budget theta and self-repair
│   ├── review_v1/                    # Pre-submission review experiments, accuracy tables, and checks
│   ├── speed/                        # Explainer timing benchmarks on synthetic and small benchmarks
│   ├── speed_uci/                    # Explainer timing benchmarks on large UCI datasets
│   ├── time_ordered/                 # Streaming time-ordered case study on Electricity (OpenML 151)
│   └── tough_review/                 # Tough review post-hoc re-analysis and Nadeau-Bengio corrections
│
├── scripts/                          # Executable Experiment Entry Points (60+ Standalone CLI Scripts)
│   ├── accuracy_table.py             # Generates multi-model accuracy comparisons across 13 datasets
│   ├── download_data.py              # Downloads, uncompresses, and verifies SHA-256 for UCI datasets
│   ├── exact_baseline.py             # Validates exact PACE-SHAP against brute-force Shapley formulas
│   ├── run_counterfactuals.py        # Executes counterfactual benchmarks comparing PACE and DiCE
│   ├── run_error_detection.py        # Executes RQ2 trust signal error detection pipelines
│   ├── run_faithfulness.py           # Executes RQ1 faithfulness benchmarking (v1, v2, v3, conditional)
│   ├── run_g0.py                     # Executes baseline CFMNN parity and architecture ablation checks
│   ├── run_heldout.py                # Runs generalization evaluation on OpenML held-out benchmarks
│   ├── run_readable_baselines.py     # Evaluates CFMNN against EBM, Decision Trees, and GFMM
│   ├── run_refinement_v2.py          # Executes pre-registered RQ3/RQ4 self-repair pipelines
│   ├── run_repair_order.py           # Evaluates greedy vs random error presentation sequences
│   ├── run_time_ordered.py           # Executes streaming week-by-week repair on Electricity
│   ├── time_pace_flip.py             # Profiles execution latency of PACE-flip vs brute force
│   └── tough_review_reanalysis.py    # Recomputes all comparisons with Nadeau-Bengio corrections
│
└── tests/                            # Comprehensive Unit & Regression Test Suite (501 Tests)
    ├── naive_reference.py            # Line-by-line literal transcription of PReMI 2023 paper pseudocode
    ├── reference_pace_flip.py        # Unoptimized reference implementation of PACE-flip
    ├── test_api.py                   # High-level PaceClassifier scikit-learn API tests
    ├── test_audit.py                 # Proposition 4 certificate non-interference tests
    ├── test_cfmnn_core.py            # CFMNN training, membership, and prediction tests
    ├── test_explain.py               # PACE-SHAP, PACE-flip, contrastive, and decomposition tests
    ├── test_hardness.py              # Proposition 5 #P-hardness reduction tests
    ├── test_propositions.py          # Direct mathematical validation of Propositions 1 and 2
    ├── test_refine.py                # Face shrink, gate acceptance, and rollback tests
    └── test_tough_audit.py           # Proposition 6 & 7 Condition S and SHAP certificate tests
```

---

## 3. Execution Role of Major Modules

### Core Production Module (`pace_cfmnn/`)
1. **`cfmnn_core.py`**: Encapsulates the core `CFMNN` estimator. Implements `ramp()`, `feature_terms()`, `predict_details()`, `add_points()`, and the `"fixed"` vs `"original"` centroid mechanics. Contains zero external dependencies beyond NumPy and scikit-learn.
2. **`explain.py`**: The centerpiece explainer engine. Implements `PaceShap` (with exact dynamic recurrence and antithetic permutation modes), `PaceFlip` (with piecewise quantile bound pruning), `contrastive()`, `contrastive_multi()`, and `decomposition()`.
3. **`refine.py` & `active.py`**: The closed-loop adaptation infrastructure. `refine.py` defines the surgical `shrink_box()` operator, boundary cut calculation `face_cut()`, and the statistical validation `Gate`. `active.py` orchestrates the multi-round active learning budget, sample selection policies, and transactional rollback.
4. **`audit.py`**: Mathematical certification engine implementing Proposition 4 (`certify`, `impact_set`) and Proposition 7 (`pace_shap_certificate`). Computes predictive churn and non-flip rates.
5. **`hardness.py`**: Complexity theory module implementing Proposition 5's reduction from weighted voting games to CFMNN instances, demonstrating #P-hardness.

### Research Audit Module (`research_audit/`)
Maintains an exhaustive, verifiable audit trail connecting every sentence and figure in the academic papers directly to committed raw experimental outputs in `results/`. Contains 134 automated assertion checks in `CLAIMS.md` ensuring 100% data fidelity.

---

## 4. Cross-References

- For the end-to-end system execution architecture: [`04_End_to_End_Workflow_and_System_Architecture.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/04_End_to_End_Workflow_and_System_Architecture.md)
- For dataset schemas, sample distributions, and download mechanics: [`05_Datasets_and_Data_Engineering_Pipeline.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/05_Datasets_and_Data_Engineering_Pipeline.md)
- For step-by-step terminal execution instructions to reproduce the repository: [`20_End_to_End_Reproducibility_Guide_and_Future_Roadmap.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/20_End_to_End_Reproducibility_Guide_and_Future_Roadmap.md)
