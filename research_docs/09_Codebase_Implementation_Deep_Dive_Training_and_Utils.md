# 09. Codebase Implementation Deep Dive: Training and Utilities

## 1. Scope of Training, Quality Assurance, and Utility Modules

While `cfmnn_core.py` and `explain.py` define the internal representation and attribution mathematics, the operational reliability of the PACE-CFMNN system is governed by its auxiliary subsystems:
* **Trust & Error Telemetry:** `signals.py`, `neighborhood.py`, `pace_score.py`, `trust_baselines.py`, `trust_combine.py`.
* **Model Maintenance & Closed-Loop Adaptation:** `refine.py`, `active.py`.
* **Verification, Auditing & Certificates:** `audit.py`.
* **Fault-Tolerant Persistence & User Front-End:** `resume.py`, `evaluate.py`, `api.py`.

```
+----------------------------------------------------------------------------------------------------+
|                               TRAINING & UTILITY MODULE INVENTORY                                  |
+--------------------------+---------------------------------------------------------+---------------+
| Source Code File         | Primary Engineering Function                            | Line Count    |
+--------------------------+---------------------------------------------------------+---------------+
| `signals.py`             | Prediction disagreement (PD), EI, and corrected U*      | 193 lines     |
| `neighborhood.py`        | Continuous and discrete neighbor perturbation generator | 108 lines     |
| `pace_score.py`          | PACE Refinement Score (PRS) rank aggregation            | 53 lines      |
| `trust_baselines.py`     | Trust Score, kNN disagreement, Order Ensemble           | 94 lines      |
| `trust_combine.py`       | Logistic regression multimodal error detector           | 177 lines     |
| `refine.py`              | Surgical boundary shrink operator and validation Gate   | 389 lines     |
| `active.py`              | Budgeted active learning, query selectors, rollback     | 539 lines     |
| `audit.py`               | Propositions 4 & 7 prediction and SHAP certificates     | 354 lines     |
| `resume.py`              | Atomic file checkpointing engine (`--resume`)           | 104 lines     |
| `evaluate.py`            | Comprehensive classification and macro-F1 metrics       | 78 lines      |
| `api.py`                 | Scikit-learn unified estimator interface                | 149 lines     |
+--------------------------+---------------------------------------------------------+---------------+
```

---

## 2. Deep Dive: Trust Telemetry Subsystem

### 2.1 Neighborhood Perturbation Generator (`pace_cfmnn/neighborhood.py`)
To evaluate local prediction stability and explanation invariance without ground-truth labels, `Perturber` creates $K$ localized neighbor inputs ($K = 50$ by default).

#### Implementation Architecture (Lines 30–80)
```python
class Perturber:
    def __init__(self, X_train, scale_ratio: float = 0.05, discrete_threshold: int = 10,
                 redraw_prob: float = 0.1, rng=None):
        self.std_ = np.nanstd(X_train, axis=0)
        self.scale_ = scale_ratio * np.where(self.std_ > 0, self.std_, 1.0)
        ...
```
* **Continuous Feature Noise:** For continuous dimensions, perturbations are sampled from a Gaussian distribution:
  $$x^{(k)}_i = \text{clip}\Big(x_i + \mathcal{N}\big(0, \, (0.05 \cdot \sigma_i)^2\big), \, 0.0, \, 1.0\Big)$$
* **Discrete Feature Handling:** If a feature exhibits $\le 10$ unique values in the training set (e.g., binary indicators or ordinal integers), Gaussian jitter violates the data structure. With probability $0.1$, the value is redrawn directly from the empirical training distribution of that feature, preserving discrete integrity.

---

### 2.2 Intrinsic Trust Signals (`pace_cfmnn/signals.py`)
Computes three distinct, orthogonal dimensions of model certainty:

#### 1. Prediction Disagreement (PD) (Lines 140–155)
Evaluates local decision boundary proximity:
$$\text{PD}(x) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\Big( \hat{y}(x^{(k)}) \ne \hat{y}(x) \Big) \in [0, 1]$$
High PD indicates that $x$ lies in close spatial proximity to a classification boundary.

#### 2. Explanation Instability (EI) (Lines 106–138)
Evaluates local attribution volatility. For each perturbation $x^{(k)}$, computes the contrastive explanation vector $e(x^{(k)})$ between the original predicted class $A$ and original runner-up $B$:
$$\text{EI}(x) = \frac{1}{K} \sum_{k=1}^K \frac{1 - \cos\big(e(x), \, e(x^{(k)})\big)}{2} \in [0, 1]$$
High EI indicates that while the categorical decision may remain constant, the underlying feature logic is turbulent.

#### 3. Corrected Uncertainty ($U^*$) (Lines 64–105)
Resolves a critical flaw in classical margin uncertainty. In hyperbox classifiers, overlapping regions between classes cause ties ($s'_1(x) = s'_2(x)$).
* *Naive Margin Uncertainty:* Evaluates $1 - (s'_1 - s'_2)$, which scores ties as $1.0$, but normalizes them identically to distant points in empty space.
* *Corrected $U^*$ (Proposition 1 Alignment):*
  $$U^*(x) = \begin{cases}
  \frac{1 + \min\left(1.0, \, \frac{\|x - \mu_A\|}{\|x - \mu_B\|}\right)}{2} \in [0.5, 1.0] & \text{if } |s'_1(x) - s'_2(x)| \le 2\tau \quad (\text{Tied Points}) \\
  \frac{\text{clip}(1 - (s'_1(x) - s'_2(x)), \, 0.0, \, 1.0)}{2} \in [0.0, 0.5] & \text{otherwise} \quad (\text{Untied Points})
  \end{cases}$$
By construction, **every tied point is ranked as strictly more uncertain than every untied point**, matching Proposition 1's proof that training errors are ties.

---

### 2.3 The Multimodal Error Combiner (`pace_cfmnn/trust_combine.py`)
Evaluates the learned combination (RQ2 v2) of geometric and empirical signals:
* **Feature Vector per Instance:**
  $$v(x) = \Big[ \text{kNN\_Disagreement}(x), \; \text{Trust\_Score}(x), \; \text{Order\_Ensemble\_Disagreement}(x), \; U^*(x), \; \text{Box\_Evidence}(x) \Big]$$
* **Model:** $\ell_2$-regularized Logistic Regression calibrated on labeled validation rows.
* **Findings:** In pre-registered benchmarks (`results/error_detection_v2/`), the learned combiner beat kNN disagreement on all 8 benchmark units and Trust Score on 7 of 8 units.

---

## 3. Deep Dive: Surgical Self-Repair & Active Learning

### 3.1 The Boundary Face Shrink Operator (`pace_cfmnn/refine.py`)
Lines 60–100 implement the surgical geometric operator that resolves cross-class overlaps:

```python
def face_cut(model: CFMNN, k: int, x: np.ndarray, i: int, margin: float = 1e-3):
    v, w, xi = model.V_[k, i], model.W_[k, i], x[i]
    width = w - v
    if xi - v <= w - xi:
        new, side = xi + margin, "low"
        share = (new - v) / width if width > 0 else 1.0
    else:
        new, side = xi - margin, "high"
        share = (w - new) / width if width > 0 else 1.0
    return side, float(new), float(min(share, 1.0))
```

#### Detailed Transformation Logic
1. Evaluates whether coordinate $x_i$ is closer to the lower boundary $v_{ki}$ or upper boundary $w_{ki}$.
2. If closer to $v_{ki}$: Moves lower boundary up past $x_i$ ($v'_{ki} = x_i + 10^{-3}$).
3. If closer to $w_{ki}$: Moves upper boundary down past $x_i$ ($w'_{ki} = x_i - 10^{-3}$).
4. Updates box membership: Discards absorbed training members that fall outside the new boundary face.
5. Recomputes centroid: $\mu_k = \frac{1}{|N'_k|} \sum x_{\text{members}}$.
6. If all members are cut away: Deletes box $k$ entirely (`model._remove_box(k)`).

---

### 3.2 The Statistical Validation Gate (`Gate`)
Lines 164–215 implement the safety guardrail that prevents model regression:

```python
@dataclass
class Gate:
    X: np.ndarray
    y: np.ndarray
    box_cap: int
    max_f1_drop: float = 0.5
    max_recall_drop: float = 1.0

    def check(self, before: dict, after: dict, model: CFMNN) -> tuple[bool, str]:
        if model.n_boxes_ > self.box_cap:
            return False, "box_cap"
        if after["macro_f1"] < before["macro_f1"] - self.max_f1_drop:
            return False, "macro_f1"
        for c in range(len(before["recall"])):
            if after["recall"][c] < before["recall"][c] - self.max_recall_drop:
                return False, f"recall_class_{c}"
        return True, "ok"
```
* **Guardrail Semantics:** An edit is rejected if:
  - Total boxes exceed capacity budget ($H > \text{box\_cap}$).
  - Overall macro-F1 on acceptance rows falls by more than $0.5$ percentage points.
  - Any individual class recall falls by more than $1.0$ percentage point (preventing minority class starvation).

---

### 3.3 Active Learning Loop & Rollback (`pace_cfmnn/active.py`)
Lines 220–415 manage the multi-round active learning and repair sequences:
* **Transactional State Mutation:** Before applying an edit, records an in-memory snapshot of mutated attributes (`V_[k]`, `W_[k]`, `centroid_[k]`, `members_[k]`).
* **Non-Destructive Rollback:** If the gate rejects the edit, restores the snapshot arrays in place within microsecond latency, avoiding full deep copies of the estimator.

---

## 4. Deep Dive: Invariance & Certification Engine (`pace_cfmnn/audit.py`)

Lines 24–120 implement Proposition 4 and Proposition 7 to construct provable mathematical certificates:

```python
def certify(before: CFMNN, after: CFMNN, tied_before, tied_after, removed: int | None = None):
    ch_b, ch_a = changed_boxes(before, after, removed)
    hit = tied_before[:, ch_b].any(axis=1) | tied_after[:, ch_a].any(axis=1)
    classes = np.zeros((len(hit), len(before.classes_)), dtype=bool)
    for c in range(len(before.classes_)):
        classes[:, c] = (tied_before[:, before.box_class_ == c].any(axis=1)
                         | tied_after[:, after.box_class_ == c].any(axis=1))
    return hit, hit & (classes.sum(axis=1) > 1)
```

#### Certificate Dual-Level Hierarchy
1. **Loose Certificate (`hit`):** Flags any query point $x$ whose tied top group contained a modified box before the edit or contains a modified box after the edit.
2. **Tight Certificate (`hit & (classes.sum(axis=1) > 1)`):** Eliminates points whose top tied group consists exclusively of boxes of a single class $c$ both before and after the edit. Even if box boundaries changed, the prediction is mathematically locked to class $c$ in both models.
3. **Audit Verification:** Across $46,153$ empirical edits evaluated in `results/refinement_v2_confirmatory/`, **zero predictions ever flipped outside the tight certificate**.

---

## 5. Fault-Tolerant Persistence (`pace_cfmnn/resume.py`)

For multi-hour distributed runs (e.g., EBM readable baselines taking 11 hours, or conditional faithfulness taking 10 hours), `resume.py` provides atomic disk-backed persistence:
* Writes intermediate seed results to temporary files (`runs.csv.tmp`) and executes atomic renames.
* Inspects existing CSV records on startup; if a seed has completed, skips execution and resumes from the next task.
* Eliminates wasted compute during cluster restarts or transient failures.

---

## 6. High-Level Front-End API (`pace_cfmnn/api.py`)

Lines 35–145 wrap the underlying mathematical modules into a single, intuitive scikit-learn compatible estimator:

```python
class PaceClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, theta=0.3, gamma=4.0, centroid_rule="fixed", expansion="search",
                 bg_rows: int = 100, exact_max_d: int = 12, seed: int = 0):
        ...
```

#### Methods Exposed
* `fit(X, y)`: Fits CFMNN and caches $100$ background rows for explanations.
* `predict(X)`: Returns predicted class labels.
* `predict_proba(X)`: Returns normalized rescaled class supports $s'_c(x)$.
* `explain(X, method="shap")`: Evaluates PACE-SHAP or PACE-flip attributions.
* `repair(X_train, y_train, certify_on=X_test)`: Executes self-repair on training errors and returns the repaired model accompanied by an invariance certificate.

---

## 7. Cross-References

- For the training pipeline and optimization dynamics: [`10_Training_Pipeline_and_Optimization_Strategy.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/10_Training_Pipeline_and_Optimization_Strategy.md)
- For evaluation metrics, baselines, and statistical test formulations: [`12_Evaluation_Protocol_Metrics_and_Baselines.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/12_Evaluation_Protocol_Metrics_and_Baselines.md)
- For granular training wall-clock time breakdowns: [`15_Component_Wise_Training_Time_and_Convergence_Analysis.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/15_Component_Wise_Training_Time_and_Convergence_Analysis.md)
- For full architectural and empirical details of RQ3 self-repair: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
