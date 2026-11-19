# 20. End-to-End Reproducibility Guide and Future Roadmap

## 1. Reproducibility Philosophy & Environmental Pre-requisites

The PACE-CFMNN framework was constructed from inception with the goal of 100% deterministic reproducibility. Every random operation (dataset splitting, antithetic permutation sampling, neighbor jittering, and active learning order) is parameterized by explicit seed parameters.

### System Prerequisites
* **Operating System:** Microsoft Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+ (Apple Silicon or Intel).
* **Python Runtime:** Python 3.10 or newer (the codebase was developed and verified on Python 3.14.0a and Python 3.11).
* **External System Tools:** `gzip` and `git` must be available on the system PATH (used to decompress UCI archives during automated ingestion).
* **Hardware Requirements:** Commodity multi-core CPU (8 to 16 threads recommended). **No GPU or CUDA drivers are required.**

---

## 2. Step-by-Step Environment Setup

### Step 2.1: Clone & Navigate to the Repository
```bash
git clone https://github.com/anonymous-author/pace-cfmnn.git
cd pace-cfmnn
```

### Step 2.2: Virtual Environment Instantiation
```bash
# Using standard Python venv
python -m venv venv

# Activation on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activation on Linux / macOS (Bash)
source venv/bin/activate
```

### Step 2.3: Deterministic Dependency Installation
To install the exact, byte-for-byte package versions evaluated in the paper, install from the locked dependency manifest:
```bash
# Install exact pinned versions
pip install -r requirements-lock.txt

# Or install editable package with all optional baseline dependencies (XAI + EBM + GFMM)
pip install -e .[all]
```

### Step 2.4: Automated Dataset Ingestion & Cryptographic Verification
The UCI Machine Learning Repository datasets are not stored directly in Git. Download, extract, and verify them via the automated script:
```bash
python -B scripts/download_data.py
```
* **Integrity Guarantee:** The script downloads all zips, extracts the target cohorts, and validates each file against pinned SHA-256 and MD5 hashes (`data/README.md`).

### Step 2.5: Full Unit Test Verification
Execute the test runner across all 501 test cases:
```bash
python -B -m unittest discover -s tests -t .
```
* **Expected Output:** All 501 unit tests pass (1 skipped if `hyperbox-brain` is not installed; total time ~27 minutes on laptop).

---

## 3. Master Reproducibility Execution Matrix

The table below provides the exact terminal command line, destination artifact directories, estimated execution durations, and recommended CPU parallelization flags (`--jobs`) for reproducing every experimental finding:

| Experiment & Research Question | Exact Terminal Command Line | Output Result Directory | Est. Runtime (12 Threads) |
|---|---|---|---|
| **Base Model Parity (G0)** | `python scripts/run_g0.py` | `results/g0/` | ~15 minutes |
| **RQ1 Faithfulness (v2 Fair)**| `python scripts/run_faithfulness.py --jobs 6` | `results/faithfulness/` | ~1 hour |
| **RQ1 Explainer Latency** | `python scripts/run_faithfulness.py --timing` | `results/faithfulness/timing.*` | ~12 minutes |
| **RQ1 PACE-SHAP Pre-Reg (v3)**| `python scripts/run_faithfulness.py --protocol v3 --jobs 6` | `results/faithfulness_v3/` | ~5 hours |
| **RQ1 Pre-Reg Timing (v3)**| `python scripts/run_faithfulness.py --protocol v3 --timing --seeds 30 31 32` | `results/faithfulness_v3/timing.*` | ~2 hours |
| **RQ1 Conditional SHAP (B3)** | `python scripts/run_faithfulness.py --protocol cond --jobs 6` | `results/extensions_v1/conditional/` | ~8 hours |
| **RQ1 Sparse Counterfactuals**| `python scripts/run_counterfactuals.py --extended --jobs 6` | `results/extensions_v1/counterfactuals/`| ~2 hours |
| **RQ1 Global Importance (B2)**| `python scripts/run_global_importance.py --extended --jobs 6` | `results/extensions_v1/global_importance/`| ~2 hours |
| **RQ1 Interaction Values (B4)**| `python scripts/run_interactions.py --jobs 6` | `results/extensions_v1/interactions/` | ~65 minutes |
| **RQ2 Trust Signals (Conf.)** | `python scripts/run_error_detection.py --protocol confirmatory --jobs 4`| `results/error_detection_confirmatory/`| ~45 minutes |
| **RQ2 v2 Learned Combiner** | `python scripts/run_error_detection_v2.py --protocol confirmatory --jobs 6`| `results/error_detection_v2/` | ~10 minutes |
| **RQ3/RQ4 Self-Repair (v2)** | `python scripts/run_refinement_v2.py --jobs 6` | `results/refinement_v2_confirmatory/` | ~10 hours |
| **Self-Repair Order (Run H)** | `python scripts/run_repair_order.py --jobs 12` | `results/repair_order/` | ~1 hour |
| **Self-Repair Timing (Run T)**| `python scripts/time_repair_arms.py` | `results/repair_timing/` | ~20 minutes |
| **Readable Baselines (EBM/GFMM)**| `python scripts/run_readable_baselines.py --jobs 6` | `results/readable_baselines/` | ~9 hours |
| **Repair at Tuned $\theta$** | `python scripts/run_repair_theta.py --jobs 6` | `results/repair_theta/` | ~12 hours |
| **PACE-flip Latency vs Occlusion**| `python scripts/time_pace_flip.py` | `results/speed/` | ~2 minutes |
| **Batched Occlusion Latency** | `python scripts/time_occlusion_batched.py` | `results/review_v1/occlusion_batched/` | ~8 minutes |
| **Tough Review Re-analysis** | `python scripts/tough_review_reanalysis.py` | `results/tough_review/reanalysis.md` | ~10 minutes |
| **OpenML Held-Out Suite (P10)**| `python scripts/run_heldout.py --jobs 6` | `results/heldout/` | ~3 hours |
| **Streaming Electricity (P9)**| `python scripts/run_time_ordered.py --jobs 6` | `results/time_ordered/` | ~4 hours |

---

## 4. Containerized Execution via Docker

To eliminate local environment discrepancies entirely, the repository includes a self-contained `Dockerfile`:

```bash
# 1. Build the production Docker image
docker build -t pace-cfmnn:latest .

# 2. Run the test suite inside the container
docker run --rm pace-cfmnn:latest python -B -m unittest discover -s tests -t .

# 3. Execute any benchmark script inside the container with volume mounting
docker run --rm -v "$(pwd)/results:/workspace/results" pace-cfmnn:latest \
    python scripts/run_faithfulness.py --protocol v3 --jobs 6
```

---

## 5. Prioritized Research and Engineering Roadmap

The development of PACE-CFMNN has established a solid theoretical and systems baseline. We structure the future trajectory into two chronological phases:

```
+----------------------------------------------------------------------------------------------------+
|                                    PRIORITIZED DEVELOPMENT ROADMAP                                 |
|                                                                                                    |
|    PHASE I: IMMEDIATE ENGINEERING UPGRADES (Next 3-6 Months)                                       |
|    ---------------------------------------------------------------------------------------------   |
|    [Priority 1] Default Exact Explainer to Dynamic Recurrence Engine (exact_algo="recurrence")     |
|                 - Formally make recurrence the default in PaceShap for 2.13x speedup.              |
|    [Priority 2] Public Artifact Deposit & DOI Registration                                         |
|                 - Finalize license choice (Apache-2.0 or GPL-3.0 compatibility with hbbrain).       |
|                 - Deposit pre-registration bundles to Open Science Framework (OSF).                |
|    [Priority 3] PyPI Package Distribution                                                          |
|                 - Publish standalone pace-cfmnn binary wheels to Python Package Index.             |
|                                                                                                    |
|    PHASE II: LONG-TERM RESEARCH EXTENSIONS (Next 6-18 Months)                                      |
|    ---------------------------------------------------------------------------------------------   |
|    [Research 1] Hierarchical Deep Hyperbox Networks (Deep-CFMNN)                                   |
|                 - Stack multiple CFMNN hyperbox layers to learn compositional feature abstractions |
|                 - Maintain closed-form additive Shapley evaluation across layer compositions.      |
|    [Research 2] GPU Block-Recurrence Acceleration via JAX / PyTorch                                |
|                 - Port subset sum recurrence to CUDA tensor cores for massive parallel batches.    |
|    [Research 3] Streaming Online Self-Repair with Concept Drift Detection                          |
|                 - Integrate statistical drift monitors (ADWIN) with automated boundary shrinks.    |
|    [Research 4] Certified Continuous Recourse                                                      |
|                 - Formulate non-interference spatial certificates for gradient-updated models.     |
+----------------------------------------------------------------------------------------------------+
```

---

## 6. Cross-References

- For the executive vision and core idea: [`01_Executive_Summary_and_Core_Idea.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/01_Executive_Summary_and_Core_Idea.md)
- For the environment inventory and annotated directory map: [`03_Tech_Stack_Environment_and_Repository_Map.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/03_Tech_Stack_Environment_and_Repository_Map.md)
- For the complete benchmark tables across all suites: [`13_Overall_Benchmark_Results_and_SOTA_Comparison.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/13_Overall_Benchmark_Results_and_SOTA_Comparison.md)
- For the academic review audit and publication blueprint: [`19_Research_Readiness_Audit_and_Paper_Blueprint.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/19_Research_Readiness_Audit_and_Paper_Blueprint.md)
- For the dedicated deep dives into tough review extensions: [`21_Tough_Review_Audit_and_Formal_Proof_Extensions.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/21_Tough_Review_Audit_and_Formal_Proof_Extensions.md)
- For the dedicated deep dive into self-repair: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
