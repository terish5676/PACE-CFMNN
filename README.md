# PACE-CFMNN

Exact, fast and stable explanations for a fuzzy min-max classifier.

This repository contains:

- **CFMNN:** a centroid-based fuzzy min-max neural network classifier, implemented from Aadarsh & Sai
  Prasad (PReMI 2023) as a single model. The paper's MapReduce ensemble is not needed here.
- **PACE (Prototype-aware, Additive, Contrastive Explanations):** explanations computed exactly from
  the model's own hyperboxes, instead of being approximated by sampling.
- **The experiments** behind both, with their results.

**Results so far: [results/FINDINGS.md](results/FINDINGS.md).** The design follows the team's internal
"PACE-CFMNN Blueprint v3.0".

> Private research code. No license is granted: all rights reserved.

## How CFMNN and PACE work

CFMNN covers each class with hyperboxes.
- A point's membership in a box is the average, over features, of a term that is 1 inside the box's
  range and falls linearly to a floor of 0.5 outside it.
- The box with the highest membership wins.
- Ties go to the box whose centroid (the mean of the training points it took in) is nearest.

Because a box's membership is a sum of one-feature terms, the effect of changing a single feature can be
computed exactly and cheaply. PACE uses this to explain predictions without sampling:

| Explainer | Question it answers | Notes |
|---|---|---|
| `pace_flip` / `PaceFlip` (PACE-flip) | Which single-feature changes would change the decision, and how much they shrink the winning margin | Exact occlusion over 32 quantile values per feature. The most faithful method in our benchmark |
| `contrastive_multi` (PACE-multi) | Why the predicted class and not its strongest competitors | The fastest. Weighs competing boxes by how close they are to winning |
| `contrastive` | Why class A and not the runner-up class B, feature by feature | Uses centroid distances when memberships tie |
| `decomposition` | How each feature moves the winning box's membership away from the background average | Equals exact Shapley values of that box's membership, not of the final prediction |
| `counterfactual` | The smallest change, confirmed by the model, that gives a target class | Can lock features such as age or sex. Sparsified |
| `global_importance.data_based` / `model_only` | Which features matter overall | From explanations over a dataset, or from the boxes alone |

`xai_baselines` runs KernelSHAP, LIME, DiCE, nearest unlike neighbour and brute-force occlusion on the
same model, for comparison.

## Status

| Phase | Work | State |
|---|---|---|
| 1 | Baseline CFMNN (gate G0): matches the paper's pseudocode and comes close to its accuracy | Done |
| 2 | PACE explanations and the faithfulness, stability and speed benchmark | Done |
| 3 | Trust signals: prediction disagreement under perturbation (PD), explanation instability (EI) and uncertainty (U), combined into one score (PRS) | Next |
| 4 | First-paper experiments | Planned |
| 5–7 | Explanation-guided selective refinement (active learning): method, experiments, analysis | Planned |
| 8 | Writing | Planned |

## Install

Python 3.10 or newer (developed with 3.14):

```bash
pip install -r requirements.txt
```

shap, lime and dice-ml are needed only for the XAI baselines and the faithfulness benchmark.

## Quick start

```python
from pace_cfmnn import CFMNN, data
from pace_cfmnn.explain import PaceFlip, contrastive
from pace_cfmnn.counterfactual import counterfactual

ds = data.load("S1")                                   # synthetic: only x0-x2 matter
split = data.paper_split(ds.y, seed=0)                 # stratified 80/20
scaler = data.fit_scaler(ds.X[split["train"]])         # min-max, fitted on training rows
X_train = scaler.transform(ds.X[split["train"]])
y_train = ds.y[split["train"]]
X_test = scaler.transform(ds.X[split["test"]])

model = CFMNN().fit(X_train, y_train)                  # inputs must be scaled to [0, 1]
print(model.predict(X_test[:5]))

explainer = PaceFlip(model, background=X_train[:200])  # build once, reuse
score = explainer.explain(X_test[:5])                  # (5, d): higher = more decision-changing
print(score.argmax(axis=1))                            # most decisive feature per point

why = contrastive(model, X_test[:5])                   # why class A, not runner-up B
print(why.mode, why.e.shape)

cf = counterfactual(model, X_test[0], target=1 - model.predict(X_test[:1])[0],
                    X_train=X_train, y_train=y_train)
print(cf.found, cf.n_changed, "features changed")
```

`PaceFlip` rebuilds its lookup tables by itself if the model's boxes change.

## Model options

```python
CFMNN(theta=0.3, gamma=4.0, centroid_rule="fixed", expansion="search", tie_tol=1e-9, chunk_elems=65_536)
```

- **`theta`, `gamma`:** the box size limit and the slope of the membership ramp. These are the paper's
  values.
- **`centroid_rule`:**
  - `"fixed"` (default) counts every point a box accepts in its centroid.
  - `"original"` is the paper as written: a point already inside its best box is skipped, so the
    centroid ignores it.
- **`expansion`:**
  - `"search"` (default) grows the best same-class box that can still grow, as in Simpson's FMNN.
  - `"best"` is the paper: only the highest-membership box is tested, and a new point box is made if
    it would be too big.
- **`tie_tol`:** memberships within this tolerance count as tied. Ties go to the nearest centroid, then
  to the lowest box index.
- **`chunk_elems`:** how many numbers are processed together. The default keeps arrays in the CPU
  cache; results never depend on it.

The defaults (fixed rule, search expansion) are the main model in our experiments: best or tied-best on
7 of 8 datasets in G0. They are not in the paper, so the paper's rules are always reported alongside.

## Reproducing the results

From the repository root:

| Command | Writes | Time |
|---|---|---|
| `python -m unittest discover -s tests -t .` | 54 tests | about 10 s |
| `python scripts/run_g0.py` | `results/g0/` | about 15 min, mostly Skin |
| `python scripts/run_faithfulness.py` | `results/faithfulness/` | about 25 min, mostly KernelSHAP and LIME |
| `python scripts/time_pace_flip.py` | `results/speed/` | a few minutes |

- **UCI data:** download it first (see [Data](#data)).
  - `run_g0.py` skips any UCI dataset that is missing.
  - The other two scripts include Thyroid and Shuttle by default. Use `--datasets` to choose others.
- **Timings:** they depend on the machine, so compare methods within one run.

## Key results

- **Base model:** our single CFMNN with the paper's rules is within 2 points of the paper's ensemble on
  Thyroid, Shuttle and Skin. Counting every accepted point in the centroids raises Shuttle from 93.3%
  to 99.8%.
- **PACE-flip:**
  - It is the most faithful explainer on all 7 benchmark datasets (deletion AUC), with stability 1.00.
  - It equals brute-force occlusion exactly, but is 50–90× faster.
  - It is about 100–900× faster than LIME and KernelSHAP, at roughly 0.05–1 ms per explanation.
- **PACE-multi** is the fastest option (0.02–0.5 ms) and the second most faithful on the real datasets.
- **Synthetic data:** PACE-flip's top features match what the model's own decisions rely on. LIME and
  SHAP rank the truly relevant features higher than the model itself does, so they partly describe the
  data rather than the model.

Full tables, caveats, and what we can and cannot claim: [results/FINDINGS.md](results/FINDINGS.md).

## Data

| Dataset | Source |
|---|---|
| S1, S2, S3 | Generated by `data.make_synthetic`; only features 0–2 decide the class |
| Wine, Breast Cancer, Digits | Bundled with scikit-learn |
| Thyroid, Shuttle, Skin, Sepsis | The paper's UCI datasets. **Not stored in this repository**; download steps are in [data/README.md](data/README.md) |

## Repository layout

| Path | Contents |
|---|---|
| `pace_cfmnn/cfmnn_core.py` | CFMNN: membership (Eq. 1), training (Algorithm 2), prediction with the centroid tie-break (Algorithm 4), class supports, runner-up class |
| `pace_cfmnn/explain.py` | PACE-flip (`PaceFlip`), PACE-multi, contrastive and decomposition explanations |
| `pace_cfmnn/counterfactual.py` | Box-directed counterfactuals, verified on the model |
| `pace_cfmnn/global_importance.py` | Data-based and model-only global importance |
| `pace_cfmnn/xai_baselines.py` | KernelSHAP, LIME, DiCE, nearest unlike neighbour and brute-force occlusion |
| `pace_cfmnn/data.py` | Synthetic and real datasets, splits, scaling fitted on training rows |
| `pace_cfmnn/evaluate.py` | Accuracy, macro-F1, balanced accuracy, per-class recall |
| `scripts/` | The experiments: `run_g0.py`, `run_faithfulness.py`, `time_pace_flip.py` |
| `results/` | Each experiment's `runs.csv`, `summary.md` and log, plus `FINDINGS.md` |
| `tests/` | 54 unit tests (see below) |
| `data/` | Where the UCI files go ([data/README.md](data/README.md)) |

Two test reference files hold the fast code to exact standards:
- `tests/naive_reference.py` transcribes the paper's pseudocode line by line.
- `tests/reference_pace_flip.py` keeps the original PACE-flip loop.

The fast code must match both exactly.

Equation and algorithm numbers refer to the CFMNN paper below.

## References

- V. Aadarsh and P. S. V. S. Sai Prasad. Scalable Centroid Based Fuzzy Min-Max Neural Network Ensemble
  Classifier Using MapReduce. PReMI 2023.
- P. K. Simpson. Fuzzy min-max neural networks, Part 1: Classification. IEEE Transactions on Neural
  Networks 3(5), 1992.
- S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. NeurIPS 2017.
  (SHAP)
- M. T. Ribeiro, S. Singh and C. Guestrin. "Why should I trust you?": Explaining the predictions of any
  classifier. KDD 2016. (LIME)
- R. K. Mothilal, A. Sharma and C. Tan. Explaining machine learning classifiers through diverse
  counterfactual explanations. FAT* 2020. (DiCE)

Dataset sources and citations are in [data/README.md](data/README.md).

## License

No license is granted: all rights reserved. Ask the team before reusing the code.
