# PACE-CFMNN

Exact, fast and stable explanations for a fuzzy min-max classifier, with trust signals and box repair.

This repository contains:

- **CFMNN:** a centroid-based fuzzy min-max neural network classifier, implemented from Aadarsh & Sai
  Prasad (PReMI 2023) as a single model. The paper's MapReduce ensemble is not needed here.
- **PACE (Prototype-aware, Additive, Contrastive Explanations):** explanations computed exactly from
  the model's own hyperboxes, instead of being approximated by sampling.
- **Trust signals** that flag likely errors, and **repair**: shrinking a wrong box at a misclassified
  point, behind a gate that checks labelled rows, with a certificate of which predictions an edit can
  change.
- **The experiments** behind all of these, with their results.

**Results so far: [results/FINDINGS.md](results/FINDINGS.md).** 

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
| `pace_shap` / `PaceShap` (PACE-SHAP) | How much each feature moves the predicted class's margin, with credit shared fairly over all combinations of features (the Shapley value KernelSHAP estimates) | Computed from the boxes' per-feature terms: exact when there are at most 12 features, otherwise by sampling with a fixed seed. In the pre-registered benchmark the most faithful explainer: better than KernelSHAP on 6–7 of 11 datasets, never worse. With 200 background rows it took 1.2–3.0× as long as KernelSHAP on 9 of 11 datasets (timing measured alone); on KernelSHAP's 20 rows it was 4.3–14× faster |
| `PaceShap(sigma=0.1)` (PACE-SHAP-cond) | The same question, with removed features filled in from rows that resemble the point on the features it keeps (the conditional game) | Rows weighted by exp(−D²/(2σ²)) on standardised distances; exact for at most 12 features, otherwise 32 sampled orders shared by all rows. Its inputs stay near real data. On a conditional test it is more faithful than KernelSHAP-cond on 6 of 13 datasets and never less; but at σ = 0.1 it is less faithful than PACE-SHAP on most datasets, even on that test |
| `pace_flip` / `PaceFlip` (PACE-flip) | Which single-feature changes would change the decision, and how much they shrink the winning margin | Occlusion over 32 quantile values per feature, computed exactly (equal to brute-force occlusion up to floating-point rounding). Deterministic and 29–141× faster than brute force (25–134× when brute force runs in one batch); under the fair benchmark less faithful than KernelSHAP on 6 of 11 datasets and more faithful on 3 |
| `pace_flip_cond` (PACE-flip-cond) | Which single-feature changes would change the decision, trying values seen in rows like the point | Feature i takes the values of the 32 rows nearest to the point on its other features; exact. Its inputs are off the data less often than PACE-flip's on 10 of 13 datasets, but it is not more faithful |
| `contrastive_multi` (PACE-multi) | Why the predicted class and not its strongest competitors | Faster than PACE-flip in batches. Weighs competing boxes by how close they are to winning |
| `contrastive` | Why class A and not the runner-up class B, feature by feature | Uses centroid distances when memberships tie |
| `decomposition` | How each feature moves the winning box's membership away from the background average | Equals exact Shapley values of that box's membership, not of the final prediction |
| `PaceShap.interactions` | Which pairs of features act together on the margin, beyond what each does alone | Exact SHAP interaction values from the same table as PACE-SHAP (at most 12 features). On S3 it names the true AND pair as the strongest pair in 10 of 10 seeds |
| `counterfactual` | The smallest change, confirmed by the model, that gives a target class | Can lock features such as age or sex. Sparsified |
| `sparse_counterfactual` (PACE-sparse) | The fewest changed features, among moves towards boxes of the target class, that give that class (confirmed by the model) | Exact search over sets of features within those moves, scored from CFMNN's own terms (greedy beyond 65,536 sets per size). A change outside those moves can need fewer features. Never changes more features than PACE. Changes fewer features than DiCE on 10 of 11 datasets (not Shuttle) |
| `global_importance.data_based` / `model_only` / `normalised_mean` | Which features matter overall | From explanations over a dataset, or from the boxes alone; `normalised_mean` scales each point's scores to sum to 1 before averaging |

`xai_baselines` runs KernelSHAP, LIME, DiCE, nearest unlike neighbour and brute-force occlusion on the
same model, for comparison. Its `kernel_shap_game` is our own KernelSHAP for either game,
interventional or conditional (`sigma`); with every coalition it equals PACE-SHAP exactly.
`kernel_shap_games` computes several games from one set of model calls.

`realism` measures how realistic an explainer's inputs are: the share that lie farther from the
training data than 95% of training rows lie from their nearest neighbour.

## Status

| Phase | Work | State |
|---|---|---|
| 1 | Baseline CFMNN (gate G0): matches the paper's pseudocode and comes close to its accuracy | Done |
| 2 | PACE explanations and the faithfulness, stability and speed benchmark | Done |
| 3 | Trust signals: prediction disagreement under perturbation (PD), explanation instability (EI) and uncertainty (U), combined into one score (PRS) | Done |
| 4 | First-paper experiments: the fair faithfulness benchmark (RQ1, v2) and the pre-registered PACE-SHAP benchmark (RQ1, v3), error detection by the trust signals (RQ2: exploratory run, pre-registered confirmatory run, and RQ2 v2 on learned combinations), counterfactual and global-importance benchmarks, and the pre-registered RQ1 extensions (sparse counterfactuals, global aggregation, conditional replacement, interaction values) | Done |
| 5 | Explanation-guided selective refinement (active learning): the loop, box edits, acceptance gate and rollback; exploratory pilot | Done |
| 6–7 | Repair experiments (RQ3–RQ4) and analysis: the first pre-registered run (seeds 10–19), exploratory checks, and the second pre-registered run (seeds 50–59; see FINDINGS §5) | Done |
| 8 | Writing: one paper covering RQ1–RQ4 | In progress |

## Install

Python 3.10 or newer (developed with 3.14):

```bash
pip install -r requirements.txt
```

shap, lime and dice-ml are needed only for the XAI baselines and the faithfulness benchmark.
interpret-core (EBM) and hyperbox-brain (GFMM) are needed only for the readable-baselines run.

To install the exact versions our results were produced with, use the lock file instead
(`python -B scripts/make_lock.py` regenerates it from the installed packages):

```bash
pip install -r requirements-lock.txt
```

Or install the package itself, so `import pace_cfmnn` works from anywhere (`.[xai]` adds the
XAI baselines, `.[baselines]` EBM and hyperbox-brain, `.[all]` both):

```bash
pip install -e .
```

A `Dockerfile` builds the same environment from the lock file; its header gives the commands to
download the data, run the tests and rerun a table inside it (not yet built by the team).

Then download the UCI and OpenML datasets and check them against the files our runs used:

```bash
python -B scripts/download_data.py
```

The OpenML files are for Runs P9 and P10:
- phoneme, wilt, mfeat-morphological and jungle_chess are the held-out datasets, chosen by the
  rule in `results/heldout/RULE.md`;
- electricity is the time-ordered dataset.

All five are listed as Public on OpenML. Each download is checked against OpenML's MD5 before it
is saved.

## Demo

[`release/demo/pace_cfmnn_demo.html`](release/demo/pace_cfmnn_demo.html) is an interactive 2-D
demo. It trains CFMNN on two features, repairs its training errors with gated shrinks, shades
each edit's certificate, and explains any prediction with exact PACE-SHAP. It runs in the
browser.

It is a separate JavaScript version written for illustration; the Python package is the
reference. The team's hosted copy is a claude.ai artifact (private until shared).

## Quick start

In a few lines, with the front end (`pace_cfmnn/api.py`); features must be scaled to [0, 1]:

```python
from pace_cfmnn import PaceClassifier

model = PaceClassifier(theta=0.3).fit(X_train, y_train)   # a CFMNN that keeps 100 background rows
phi = model.explain(X_test[:5])                            # PACE-SHAP values, exact up to 12 features
rank = model.explain(X_test[:5], method="flip")            # PACE-flip scores
result = model.repair(X_train, y_train, certify_on=X_test) # self-repair: no labels beyond training
print(result.model.score(X_test, y_test), result.certificate.mean())  # share that may have changed
```

The same steps with the classes underneath, and the datasets of the paper:

```python
from pace_cfmnn import CFMNN, data
from pace_cfmnn.explain import PaceFlip, PaceShap, contrastive
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

shapley = PaceShap(model, background=X_train[:200]).explain(X_test[:5])
print(shapley.sum(axis=1))                             # margin minus its background average

why = contrastive(model, X_test[:5])                   # why class A, not runner-up B
print(why.mode, why.e.shape)

cf = counterfactual(model, X_test[0], target=1 - model.predict(X_test[:1])[0],
                    X_train=X_train, y_train=y_train)
print(cf.found, cf.n_changed, "features changed")
```

`PaceFlip` and `PaceShap` rebuild their tables by themselves if the model's boxes change.

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

| Command | Writes | Run time |
|---|---|---|
| `python -m unittest discover -s tests -t .` | nothing; runs the 501 tests | about 27 minutes (on the laptop, 2026-11-16; 1 skipped) |
| `python scripts/run_g0.py` | `results/g0/` | about 15 minutes |
| `python scripts/run_faithfulness.py --jobs 3` | `results/faithfulness/` (fair protocol v2, 10 seeds) | about 1 h 45 min with `--jobs 3` (it shared the CPU with another run) |
| `python scripts/run_faithfulness.py --timing` | `results/faithfulness/timing.*` (milliseconds per explanation; run it alone) | about 12 minutes |
| `python scripts/pace_shap_settings.py` | `results/faithfulness_v3/settings.*` (PACE-SHAP's settings, chosen from approximation quality only, seed 99) | about 20 minutes |
| `python scripts/run_faithfulness.py --protocol v3 --jobs 6` | `results/faithfulness_v3/` (pre-registered PACE-SHAP benchmark; seeds 30–39) | about 7 hours with `--jobs 6`; keep the laptop awake |
| `python scripts/run_faithfulness.py --protocol v3 --timing --seeds 30 31 32` | `results/faithfulness_v3/timing.*` (pre-registered timing run; run it alone) | about 2 hours |
| `python scripts/run_faithfulness.py --protocol v1` | `results/faithfulness_v1/` (the original protocol, kept for traceability) | about 17 minutes |
| `python scripts/run_error_detection.py` | `results/error_detection/` | about 50 minutes |
| `python scripts/run_error_detection.py --protocol confirmatory` | `results/error_detection_confirmatory/` (pre-registered; seeds 10–19, more datasets and baselines) | about 75 minutes (Letter alone about 16) |
| `python scripts/run_error_detection_v2.py --protocol exploratory --jobs 2` | `results/error_detection_v2_design/` (RQ2 v2 design run, seeds 0–9) | about 1 h 50 min with `--jobs 2` (shared CPU) |
| `python scripts/run_error_detection_v2.py --protocol confirmatory --jobs 4` | `results/error_detection_v2/` (pre-registered; seeds 20–29) | about 10 minutes with `--jobs 4` |
| `python scripts/knn_reference.py` | `results/error_detection_confirmatory/knn_reference.*` (post-hoc: CFMNN vs a plain 10-NN classifier on the same splits) | about 5 minutes |
| `python scripts/run_refinement.py` | `results/refinement_pilot/` (exploratory pilot, seeds 0–2) | about 70 minutes one at a time; add `--jobs 4` for about 2.2× faster |
| `python scripts/run_refinement.py --protocol eligibility` | `results/refinement_pilot/eligibility.*` (which datasets a shrink can repair) | about 1 minute |
| `python scripts/run_refinement.py --protocol confirmatory --jobs 4` | `results/refinement_confirmatory/` (first pre-registered repair run; seeds 10–19) | about 7 hours with `--jobs 4` (Skin is about half); keep the laptop awake |
| `python scripts/summarise_refinement_v1.py` | `results/refinement_confirmatory/summary_v1.md` (the first repair run's tests, recomputed from its `runs.csv`) | seconds |
| `python scripts/run_refinement_explore.py --jobs 6` | `results/refinement_v2_explore/` (exploratory repair checks; seeds 0–2) | about 18 CPU-hours in total; divide by `--jobs`; keep the laptop awake |
| `python scripts/run_refinement_v2.py --jobs 6` | `results/refinement_v2_confirmatory/` (second pre-registered repair run, RQ3 and RQ4; seeds 50–59) | about 13 CPU-hours in total; divide by `--jobs`; keep the laptop awake |
| `python scripts/run_counterfactuals.py` | `results/counterfactuals/` (fairer DiCE; `--dice v1` gives the original setup in `counterfactuals_v1/`) | about 1.5 hours (mostly DiCE) |
| `python scripts/run_global_importance.py --jobs 4` | `results/global_importance/` (10 seeds, 11 datasets) | a few hours one at a time (mostly KernelSHAP and permutation importance); `--jobs` runs seeds in parallel |
| `python scripts/time_pace_flip.py` | `results/speed/` (S1–S3, Wine, Breast Cancer, Thyroid, Shuttle) | about 2 minutes |
| `python scripts/time_pace_flip.py --datasets magic letter landsat page_blocks --out results/speed_uci` | `results/speed_uci/` (the other 4 datasets) | about 1 minute |
| `python scripts/run_faithfulness.py --protocol cond --jobs 6` | `results/extensions_v1/conditional/` (B3, interventional vs conditional replacement; 13 datasets, seeds 40–49) | about 10 hours with `--jobs 6` (mostly the two KernelSHAPs); keep the laptop awake |
| `python scripts/run_interactions.py --jobs 6` | `results/extensions_v1/interactions/` (pre-registered extension B4, interaction values; seeds 40–49) | about 65 minutes with `--jobs 6` |
| `python scripts/run_global_importance.py --extended --jobs 6` | `results/extensions_v1/global_importance/` (B2, global aggregation; seeds 40–49) | about 2 hours with `--jobs 6` |
| `python scripts/run_counterfactuals.py --extended --jobs 6` | `results/extensions_v1/counterfactuals/` (B1, PACE-sparse; seeds 40–49) | about 2 hours with `--jobs 6` (mostly DiCE) |
| `python scripts/posthoc_shuttle_sparse.py` | `results/extensions_v1/counterfactuals/posthoc_shuttle.md` (post hoc: why DiCE is sparser on Shuttle) | a few minutes |
| `python scripts/posthoc_review.py --points page_blocks` | `results/extensions_v1/posthoc_review.md` and `results/extensions_v1/global_importance/posthoc_points_page_blocks.md` (post hoc checks from the review) | a few minutes |
| `python scripts/posthoc_sigma.py --jobs 5` | `results/extensions_v1/conditional/posthoc_sigma.md` (post hoc: B3's conditional game at σ = 0.1–1 and ∞; 6 datasets, seeds 40–44) | about 30 minutes with `--jobs 5` |
| `python scripts/posthoc_rows.py --jobs 6` | `results/extensions_v1/conditional/posthoc_rows.md` (post hoc: B3's conditional game at σ = 0.1 with 800 explainer rows instead of 200; S1, Shuttle, MAGIC, Page Blocks, seeds 40–44) | about 50 minutes with `--jobs 6` (mostly S1) |
| `python scripts/accuracy_table.py` | `results/review_v1/accuracy/` (review Test 4: accuracy of CFMNN, 10-NN and Random Forest; 13 datasets, seeds 0–4) | about 10 minutes |
| `python scripts/exact_baseline.py` | `results/review_v1/exact/` (review Test 1: PACE-SHAP vs exact Shapley values computed in general ways; run it alone) | about 3 hours |
| `python scripts/run_review_rq1.py --jobs 6` | `results/review_v1/faithfulness/` (pre-registered review Test 2: stability, ties and LIME; seeds 50–59) | about 4.5 hours with `--jobs 6`; keep the laptop awake |
| `python scripts/run_error_detection_v2.py --protocol ensembles --jobs 4` | `results/review_v1/ensembles/` (pre-registered review Test 3: generic ensembles in RQ2; seeds 60–69) | about 15 minutes with `--jobs 4` |
| `python scripts/review_reanalysis.py --jobs 6` | `results/review_v1/reanalysis.md` (post hoc re-analysis of existing results for the review) | about 10 minutes |
| `python scripts/counterfactual_table_seeds.py` | `results/review_v1/counterfactual_table.md` (post hoc: counterfactual table for the paper, tested over seeds) | seconds |
| `python scripts/time_occlusion_batched.py` | `results/review_v1/occlusion_batched/` (post hoc: PACE-flip against occlusion run in one batch, 11 datasets, seeds 0–2; run it alone, it measures time) | about 8 minutes |
| `python scripts/run_readable_baselines.py --jobs 6` | `results/readable_baselines/` (pre-registered: CFMNN against a tree, logistic regression, EBM and GFMM, and θ chosen by validation; 13 datasets, seeds 70–89; `--resume` continues a stopped run) | about 11 hours with `--jobs 6` (EBM on Letter, Skin and Shuttle is most of it); keep the laptop awake |
| `python scripts/run_repair_theta.py --jobs 6` | `results/repair_theta/` (pre-registered: self-repair at θ = 0.3 and at the θ chosen by validation; 13 datasets, seeds 70–89; needs `results/readable_baselines/runs.csv`; `--resume` continues a stopped run) | about 15 hours with `--jobs 6` (repair on Skin at the chosen θ is most of it); keep the laptop awake |
| `python scripts/time_repair_arms.py` | `results/repair_timing/` (descriptive check after RQ3: seconds of training CFMNN, SR with each gate, AL and SR then AL on seed 50, one task at a time; checks every macro-F1 against RQ3) | about 20 minutes; run it alone, it measures time |
| `python scripts/run_repair_order.py --jobs 12` | `results/repair_order/` (pre-registered Run H: random, box and greedy order of the training errors in self-repair; 6 datasets, seeds 130–139; `--resume` continues a stopped run) | about 1 hour with `--jobs 12` on the home PC (the greedy arms are most of it) |
| `python scripts/sr_method_tables.py` | `results/sr_method/tables.md` (self-repair as the method: time, gains over the starting model, labels, and RQ4 for SR alone, post hoc; from saved results) | seconds |
| `python scripts/run_heldout.py --jobs 6` | `results/heldout/` (pre-registered Run P10: RQ1 and RQ2 on 4 held-out OpenML datasets chosen by a rule; seeds 360–379; `--resume` continues a stopped run) | about 2–4 hours with `--jobs 6` on the home PC (Jungle Chess is most of it) |
| `python scripts/run_time_ordered.py --jobs 6` | `results/time_ordered/` (pre-registered Run P9: a model in use on electricity, week by week; self-repair against the incremental update and re-tuning; seeds 380–389) | about 4–5 hours with `--jobs 5` on the home PC (about 2–2.5 hours per seed on the laptop) |

Where the time goes:

- **`run_g0.py`:** about 10 of the 15 minutes are spent on the Skin dataset.
  - Skin has 245,057 rows, far more than any other dataset, and every model version is trained on it
    once per seed (5 seeds).
  - The two versions that use the paper's expansion rule are the slowest. On Skin they build about
    1,900 boxes, and CFMNN compares every training and test point with every box.
- **`run_faithfulness.py`:** most of the time is KernelSHAP and LIME.
  - To explain one point, KernelSHAP (fair protocol) runs the model on about 4,000 sampled inputs (200
    coalitions × 20 background rows) and LIME on 1,000.
  - Every explainer runs twice; the second run measures stability.
  - The PACE methods read their explanations directly from the boxes and take only seconds in total.
  - In v3, most of the time goes to KernelSHAP-strong (2,048 samples) and PACE-SHAP with 200
    background rows (on S1–S3, 1,024 coalitions × 200 rows per point).
  - In B3 (`--protocol cond`), most of the time goes to the two KernelSHAPs: 200 coalitions × 200
    rows = 40,000 model calls per point, made once and shared by the pair.
- **`time_pace_flip.py`:** about half the time is brute-force occlusion on S1–S3. The script runs it
  only to measure how much faster PACE-flip is.

These times come from one Windows PC. On another machine they will be higher or lower, but the split
should be similar.

- **UCI data:** download it first (see [Data](#data)).
  - `run_g0.py` skips any UCI dataset that is missing.
  - The other two scripts include Thyroid and Shuttle by default. Use `--datasets` to choose others.
- **Benchmark timings** (milliseconds per explanation in `results/`) also depend on the machine, so
  compare methods within one run.

## Key results

- **Base model:** our single CFMNN with the paper's rules is within 2 points of the paper's ensemble on
  Thyroid, Shuttle and Skin. Counting every accepted point in the centroids raises Shuttle from 93.3%
  to 99.8%.
- **PACE-SHAP** (pre-registered benchmark, seeds 30–39, 11 datasets; deletion and insertion tests of
  the predicted class's margin):
  - It is more faithful than KernelSHAP on 6 of 11 datasets by deletion (the primary test) and on 7 by
    insertion, and never significantly worse. Against a stronger KernelSHAP (2,048 samples, no feature
    selection) it is better on 3 and 4, and never worse.
  - It has the best mean on 8 of 11 datasets; on Thyroid, Shuttle and Page Blocks it is level with the
    best.
  - It is exact when there are at most 12 features, and stable (1.00 when exact, 0.94–0.99 when
    sampled; KernelSHAP 0.15–0.73).
  - It computes the same quantity as KernelSHAP, only better: no estimation error, no feature
    selection (shap's default keeps 10 features), and 200 background rows instead of 20.
  - Speed, measured alone: with 200 rows it took 1.2–3.0× as long as KernelSHAP on 9 of 11 datasets
    and was faster on Wine and Shuttle. On KernelSHAP's 20 rows (PACE-SHAP-20) it was 4.3–14× faster
    than KernelSHAP on every dataset.
- **PACE-flip:**
  - It equals brute-force occlusion up to floating-point rounding, gives the same answer every run
    (stability 1.00), and takes 0.05–1.4 ms per explanation. That is 29–141× faster than occlusion
    (25–134× when occlusion runs in one batch), 89–316× faster than LIME and 292–1,194× faster than
    KernelSHAP.
  - Faithfulness, fair protocol (v2, 10 seeds; signed ranking; deletion and insertion; SHAP and LIME
    explaining the same margin):
    - KernelSHAP is more faithful on S1–S3, Wine, MAGIC and Letter;
    - PACE-flip is more faithful on Breast Cancer and Thyroid, and on Landsat by deletion;
    - level on Shuttle and Page Blocks.
  - The original benchmark's "most faithful on all 7 datasets" came from ranking signed attributions
    by their absolute value, and is withdrawn.
- **PACE-multi** is faster than PACE-flip in batches (0.02–0.47 ms) but less faithful.
- **Counterfactuals** (DiCE set up fairly: 5 candidates, its sparse output, the nearest valid one; 11
  datasets; tested over 10 seeds, 40–49):
  - The model confirms 99–100% of PACE's counterfactuals and 97–100% of DiCE's.
  - PACE's land closer to real data of the target class on all 11 datasets, and make a smaller change
    on 5 of the 8 real ones (no significant difference on the other 6).
  - PACE is about 70–1,000× faster.
  - DiCE changes fewer features on 10 of 11.
- **Global importance** (10 seeds): PACE-flip averaged over points agrees with permutation importance
  better than mean |KernelSHAP| on 3 of 11 datasets (significant), with a higher mean on 10. PACE-flip
  used every test point and KernelSHAP 50 (its cost limit); on Page Blocks about a quarter of the lead
  came from that. Permutation importance also changes one feature at a time, as PACE-flip does, so
  part of the agreement is built in.
- **RQ1 extensions** (pre-registered, fresh seeds 40–49):
  - **PACE-sparse counterfactuals:**
    - they change fewer features than DiCE on 10 of 11 datasets (more on Shuttle);
    - they make a smaller change on all 11;
    - they land closer to real target-class data than DiCE's on the 8 real datasets;
    - they are faster than DiCE on 10 of 11.
  - **Global aggregation:**
    - mean |PACE-SHAP| ranks features no better than mean |KernelSHAP| (level on all 11);
    - normalising PACE-flip's scores before averaging helps on 2 datasets and hurts on none;
    - PACE-flip still beats mean |KernelSHAP| on 3 (Letter, Landsat, Page Blocks), using all test
      points against KernelSHAP's 50.
  - **Conditional replacement** (13 datasets; σ = 0.1):
    - the conditional game's inputs are off the data less often than the interventional game's on
      all 13 datasets (outside Shuttle, at most 10% against up to 67%), partly because it copies a
      few training rows;
    - like for like, on a conditional test, PACE-SHAP-cond is more faithful than KernelSHAP-cond on 6
      of 13 datasets (deletion and insertion alike), never less;
    - but the conditional explainers are not more faithful than the interventional ones, even on the
      conditional test (PACE-SHAP beats PACE-SHAP-cond there on 10 of 13 by deletion), and they give
      S1c's and S2c's correlated copy x3 less credit, not more. At σ = 0.1 the conditional game
      rests on about one row for large coalitions, a likely reason. Two post hoc checks support it:
      - with wider weights the conditional explainer catches up with PACE-SHAP on synthetic data,
        gets slightly ahead on the conditional test on Shuttle and MAGIC and credits x3 slightly
        more, but its inputs drift off the data;
      - with 4 times the rows at σ = 0.1 instead, its loss halves on S1, where the two games should
        agree, but the real datasets change little.
  - **Interaction values:** they name S3's true AND pair as the strongest pair in 10 of 10 seeds. The
    same pair is also strongest on S1 and S2, although their true rule has no interaction: the values
    describe the model's boxes, not the data. On S3 the pair stands out more (2.06 times the next
    pair, against 1.62 on S1 and 1.35 on S2; post hoc). S3's range over seeds overlaps S1's but not
    S2's.
- **Trust signals (RQ2), pre-registered confirmatory run:**
  - Prediction disagreement plus a corrected uncertainty (PD + U\*) beats the plain margin on 6 of the
    8 tested units (the 6 synthetic sets count as one unit, plus 7 real datasets). The gain on Digits
    is not significant, and on Thyroid the margin is ahead.
  - The corrected U\* beats the blueprint's U on all 6 real datasets with at least 1% tied points.
  - Explanation instability (EI) does not improve PD + U\* (the pre-registered AUPRC test).
  - Standard baselines that read the training labels (Trust Score, kNN disagreement, an ensemble of
    CFMNNs trained in other orders) are far better error detectors than PD + U\* on 7–8 of 8 units.
    A post-hoc check shows that a plain 10-NN classifier is more accurate than the CFMNN on 12 of the
    13 tested datasets.
- **Trust signals, RQ2 v2 (pre-registered, seeds 20–29):**
  - Combining kNN disagreement and Trust Score with CFMNN's training-order sensitivity (the order
    ensemble), using a logistic regression fitted on labelled calibration rows, gives the strongest
    error detector overall. It beats kNN on all 8 units and Trust Score on 7; only on Digits is another
    detector clearly better.
  - The order signal adds significantly on 5 of 8 units (Thyroid +0.34 AUPRC), and lowers the result
    on Digits.
  - CFMNN's geometry signals (U\*, box evidence) add nothing once the data-based ones are present.
- **Repair, first pre-registered run (seeds 10–19, 6 datasets):**
  - Cutting the wrong box at the explained feature was never better than the thinnest cut (the
    primary test; not supported).
  - Shrinking boxes behind the gate beat adding points alone on 6 of 6 datasets.
- **Repair, exploratory checks (seeds 0–2):** every training error is a tie between a box of the right
  class and a box of the wrong class; repairing the model's own training errors, with no new labels,
  came within half a point of buying 300 labels on 5 of 6 datasets; the gate's acceptance rows
  (548–24,506) are labels too.
- **Repair, second pre-registered run (RQ3 and RQ4, seeds 50–59, every label counted):**
  - Self-repair with no new labels (SR) raised macro-F1 by 1.5–6.0 points on all 6 datasets.
  - Bought labels added to SR on 3 of 6 (Landsat, MAGIC, Letter; +0.7 to +2.0 points).
  - SR lowered MAGIC's lowest class recall by 2.7 points.
  - Retraining CFMNN on the same labels helped significantly only once: with the new rows appended,
    on Letter (+0.85 points). Shuffled retraining changed macro-F1 by −3.4 to +1.8 points, never
    significantly (both post hoc).
  - The Random Forest stays 5–19 points ahead on 5 of 6.
  - Certificates: no prediction changed outside an edit's certificate in 46,153 edits, but they are
    loose (an edit's tight certificate held 1.2–35% of test rows, of which 0.2–3.7% changed) and do
    not bound explanation changes.
  - Repair's fix share beat shuffled retraining on all 6, by 12–54 points. Against retraining with
    the new rows appended, repair still fixed a larger share but broke more predictions on Landsat,
    MAGIC and Letter.
  - How self-repair works, step by step, with diagrams: [docs/RQ3_self_repair.md](docs/RQ3_self_repair.md).
- **Self-repair as the method** (no new run; [results/sr_method/tables.md](results/sr_method/tables.md),
  FINDINGS §5.5):
  - Gains over the starting model, macro-F1: SR +1.5 to +6.0; active learning +1.9 to +6.4; SR then AL
    +2.6 to +6.6.
  - Buying labels on top of SR added +0.6 to +2.3 points, significantly on 3 of 6, at a cost of
    821–24,806 labels and 5.5–57 times SR's time (Run T, below).
  - Against it: active learning alone was ahead of SR alone on Page Blocks (+2.3) and MAGIC (+1.8).
  - RQ4 for SR alone (post hoc): its fix share beat shuffled retraining on all 6 (significantly on 4),
    it broke fewer predictions and changed fewer explanations on every dataset tested, and none of its
    6,233 edits changed a prediction outside its certificate.
- **Order of the training errors and time of each repair setup** (home PC; FINDINGS §12):
  - Run H (pre-registered, seeds 130–139): a greedy order beat the random order on MAGIC (+4.9) and
    Letter (+3.0), was worse on Skin (−1.1) and level on 3, but took 36–46 times as long on MAGIC and
    Letter. SR keeps the random order.
  - Run T (descriptive, seed 50, one task at a time): training CFMNN 0.04–1.9 s, SR 0.2–4.7 s, active
    learning 3.9–142 s, SR then AL 4.6–145 s; every setup's macro-F1 equals RQ3's.
- **Synthetic data:**
  - PACE-flip reports which single-feature changes move this model's decision; only 42–47% of those
    flips come from the truly relevant features.
  - The Shapley methods and LIME rank the relevant features higher. LIME ranks them highest (0.89–0.92
    of its top 3), then PACE-SHAP (0.79–0.85) and KernelSHAP (0.72–0.81), but PACE-SHAP is the most
    faithful to the model on these sets.

- **Review fixes** ([results/review_v1](results/review_v1)):
  - PACE-SHAP's values equal those of general exact methods, and come 16–30× faster.
  - It is more stable than KernelSHAP whether rows are kept or re-drawn.
  - The order ensemble's gain is not specific to CFMNN.
  - CFMNN is less accurate than 10-NN and Random Forests on all 13 datasets, by 0.2–21.5 points.
  - Like for like (KernelSHAP on the same 200 rows, every feature), PACE-SHAP is more faithful on
    3 of 13 datasets by deletion and 2 by insertion, level on the rest and never worse, and KernelSHAP
    took 3.5–10× as long: most of PACE-SHAP's lead over KernelSHAP's usual setting comes from the setup.
  - The paper's counterfactual table is now tested over seeds
    ([counterfactual_table.md](results/review_v1/counterfactual_table.md)).
  - PACE-flip is 25–134× faster than occlusion run in one batch; batching makes occlusion at most
    2.2× faster ([occlusion_batched](results/review_v1/occlusion_batched/summary.md)).

- **Readable baselines** (pre-registered, seeds 70–89, [summary](results/readable_baselines/summary.md)):
  - CFMNN is not the most accurate readable model. At θ = 0.3, an EBM (main effects only) is more
    accurate on 12 of 13 datasets and a decision tree on 10.
  - With θ chosen by validation, CFMNN beats logistic regression on 7 of 13 (loses on 4), splits
    with the tree (5 better, 8 worse), and still loses to EBM on 9.
  - Against GFMM at the same θ = 0.3, CFMNN is better on 4 datasets and worse on 6.
  - θ chosen by validation beats θ = 0.3 on 7 datasets, by up to 20 points (Letter), and is never
    worse. But there it mostly picks θ = 0.05, with 6–23× as many boxes.
  - The tuned tree has fewer leaves than CFMNN at θ = 0.3 has boxes on 10 of 13 datasets.

- **Repair at the θ chosen by validation** (pre-registered, seeds 70–89, [summary](results/repair_theta/summary.md)):
  - Self-repair at θ = 0.3 does not make up for a smaller θ. θ chosen by validation, without
    repair, is more accurate on 7 of 13 datasets, by up to 17.5 points (Letter). Repair at θ = 0.3
    is more accurate only on Thyroid.
  - At the θ chosen by validation, repair helps on 3 datasets, by at most 1.6 points, and is never
    worse.
  - Repair at θ = 0.3 is still more accurate than no repair on 7 datasets, and never adds a box.
    The smaller θ needs 6–23× as many boxes.

Full tables, caveats, and what we can and cannot claim: [results/FINDINGS.md](results/FINDINGS.md).

## Data

| Dataset | Source |
|---|---|
| S1, S2, S3 | Generated by `data.make_synthetic`; only features 0–2 decide the class |
| Wine, Breast Cancer, Digits | Bundled with scikit-learn |
| Thyroid, Shuttle, Skin, Sepsis | The paper's UCI datasets. **Not stored in this repository**; download steps are in [data/README.md](data/README.md) |
| MAGIC Gamma Telescope, Letter Recognition, Landsat Satellite, Page Blocks | UCI datasets added for the RQ2 confirmatory run. Also not stored here; same download steps |

## Repository layout

| Path | Contents |
|---|---|
| `pace_cfmnn/cfmnn_core.py` | CFMNN: membership (Eq. 1), training (Algorithm 2), prediction with the centroid tie-break (Algorithm 4), class supports, runner-up class, and `add_points` to continue training (equal to a full refit) |
| `pace_cfmnn/explain.py` | PACE-SHAP (`PaceShap`, interventional or conditional, with interaction values), PACE-flip (`PaceFlip`), PACE-flip-cond (`pace_flip_cond`), PACE-multi, contrastive and decomposition explanations |
| `pace_cfmnn/counterfactual.py` | Box-directed counterfactuals and PACE-sparse, verified on the model |
| `pace_cfmnn/realism.py` | Off-manifold share of the inputs each explainer or test evaluates |
| `pace_cfmnn/global_importance.py` | Data-based and model-only global importance |
| `pace_cfmnn/xai_baselines.py` | KernelSHAP, LIME (pseudo-probability or margin targets), DiCE (original or fairer setting), nearest unlike neighbour and brute-force occlusion |
| `pace_cfmnn/neighborhood.py` | Perturbations for the trust signals: noise scaled to each feature's spread, discrete features redrawn from training values |
| `pace_cfmnn/signals.py` | Trust signals PD, EI and U, plus the corrected U* and box evidence |
| `pace_cfmnn/pace_score.py` | PACE Refinement Score: percentile ranks fitted on inner-validation rows, averaged |
| `pace_cfmnn/trust_baselines.py` | Standard error-detection baselines: Trust Score, order-ensemble disagreement, kNN disagreement |
| `pace_cfmnn/trust_combine.py` | RQ2 v2: all trust signals per point and a learned combiner (logistic regression on labelled calibration rows) |
| `pace_cfmnn/refine.py` | Refinement edits: explanation-guided and control shrinks, the acceptance gate, structure checks |
| `pace_cfmnn/active.py` | The budgeted active-learning loop: label oracle, selectors, rollback |
| `pace_cfmnn/audit.py` | Certificates for box edits: which test predictions an edit can change, and which did change |
| `pace_cfmnn/resume.py` | Save-as-you-go running for the long repair experiments (`--resume`) |
| `pace_cfmnn/data.py` | Synthetic and real datasets, splits, scaling fitted on training rows |
| `pace_cfmnn/evaluate.py` | Accuracy, macro-F1, balanced accuracy, per-class recall |
| `scripts/` | The experiments: `run_g0.py`, `run_faithfulness.py`, `pace_shap_settings.py`, `time_pace_flip.py`, `run_error_detection.py`, `run_error_detection_v2.py`, `knn_reference.py`, `run_refinement.py`, `summarise_refinement_v1.py`, `run_refinement_explore.py`, `run_refinement_v2.py`, `run_counterfactuals.py`, `run_global_importance.py`, `run_interactions.py`, and the post hoc checks `posthoc_shuttle_sparse.py`, `posthoc_review.py`, `posthoc_sigma.py` and `posthoc_rows.py`, and `run_readable_baselines.py` and `run_repair_theta.py` |
| `docs/` | [RQ3_self_repair.md](docs/RQ3_self_repair.md): how self-repair shrinks boxes, with formulas, diagrams drawn by our code, results and limits. `rq3_figures.py` draws its figures; `rq3_timing.py` times self-repair |
| `results/` | Each experiment's `runs.csv`, `summary.md` and log, plus `FINDINGS.md` |
| `tests/` | 198 unit tests (see below) |
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
- H. Jiang, B. Kim, M. Y. Guan and M. Gupta. To trust or not to trust a classifier. NeurIPS 2018.
  (Trust Score)

Dataset sources and citations are in [data/README.md](data/README.md).

## License

No license is granted: all rights reserved. Ask the team before reusing the code. The team will
choose a licence before release; [LICENSE-PENDING.md](LICENSE-PENDING.md) lists the points to
settle (hyperbox-brain, an optional dependency, is GPL-3.0).
