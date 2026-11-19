# Run I, generality and sensitivity: pre-registration

Written on 2026-11-14 on the laptop and pushed, with the code it describes, before the run.
The run itself is on the home PC (i5-14400F).

- **The code:** `scripts/run_generality.py` (tests in `tests/test_generality.py`).
- **New module:** `pace_cfmnn/ties.py` (the two tie counts). It only reads a fitted model.
- **Used as it is:** `pace_cfmnn/contraction.py` (Run F's code), `pace_cfmnn/explain.py`
  (PACE-SHAP), `xai_baselines.exact_shap` (shap's ExactExplainer), `scripts/accuracy_table.py`
  (the 13 datasets) and `scripts/run_faithfulness.py` (the explanation run's split and rows).
  No existing file was changed for this run.

## Why

The peer review (item 9) asks: "run PACE-SHAP and the tie check on Simpson's FMM (same
membership); a small γ sweep (2, 4, 8) for ties and accuracy." Two claims of the paper are
behind it:

- **PACE carries over.** PACE-SHAP reads Shapley values from the boxes and the ramp
  membership. The paper says this works for any fuzzy min-max model with that membership,
  not only for CFMNN. It has been checked on CFMNN only.
- **Proposition 1** (every training error is a tie inside a box of both classes) follows
  from training by expansion only. It should hold at every γ, and it is not expected to hold
  once boxes are contracted. Every committed run used γ = 4.

## What "Simpson's FMM" means here

**The Simpson model = CFMNN's boxes (θ = 0.3, γ = 4) after Simpson's (1992) overlap test and
contraction on every pair of boxes of different classes**: `contraction.contract_all(model,
X_train)`, with the default rule `"simpson"`.

- **This is Run F's SC-all arm**, the same function with the same arguments. Run F's plan
  (`results/contraction/PREREGISTRATION.md`) does not use the name "Simpson's FMM"; it calls
  the arm "B0, then Simpson's overlap test and contraction on every pair of boxes of
  different classes". This plan uses "the Simpson model" for it.
- **It is not Simpson's classifier trained from scratch.** Simpson's training contracts
  after every expansion; here the contraction is applied once to boxes that CFMNN has already
  grown. And the model still breaks ties by the nearest centroid, which Simpson's classifier
  does not have. What is shared with Simpson's classifier is the membership function (CFMNN
  already uses his ramp) and the overlap test and contraction rule, as published.
- **Where the test finds nothing, the Simpson model is the CFMNN itself.** Simpson's test
  uses strict inequalities, so it misses boxes that share a face (Run F's plan: nothing found
  on Thyroid). The summary reports the contractions per dataset, and the seeds with none, so
  a reader can see where the comparison says nothing new.
- **Evicted rows stay in the count.** After a contraction a changed box keeps only the
  training rows still inside it. A row dropped this way is still a training row: it is
  predicted, and it counts in the training error and the tie share.

## What ran before this was written

- **Unit tests** on made-up data, while building.
- **Smoke slices on seed 190** (not a study seed, never analysed), on the laptop, written
  outside the repository:
  - `models` part on Wine, Breast Cancer, Thyroid, Page Blocks, MAGIC (one process), then S1,
    Shuttle, Skin, Letter, Digits (3 jobs);
  - `exact` part on MAGIC and Page Blocks, then S1 and Shuttle.

  They printed, for seed 190, every model's test accuracy, macro-F1, boxes, training errors
  and tie counts, and the exact check's largest difference. So these were seen:
  - at γ = 2, 4 and 8, every training error was a tie (0 exceptions on the 10 datasets);
  - the Simpson model's tie share was far below 100% on several datasets (for example Shuttle
    48 of 5,860 errors, Skin 2,339 of 23,001, MAGIC 0.3%), and its accuracy was lower than
    CFMNN's on Shuttle, Skin, MAGIC and Page Blocks and higher on Breast Cancer;
  - Wine, Thyroid and Digits had no contraction;
  - the exact check's largest difference was 1.8e-16.

  One seed is not evidence; only seeds 150–159 are.
- **What changed because of the smoke slices:** the model names in `runs.csv` were changed
  from "CFMNN γ=4" to "gamma=4" (a Greek letter in a printed line would stop a worker
  process on Windows). No dataset, model, measure or comparison changed.
- **Times seen (laptop):** the first five datasets took 2 min 51 s in one process. With 3
  jobs, the second five took 6 min 18 s, almost all of it S1's exact check (shap's
  ExactExplainer, 59 s per point).

## Setup

- **Seeds:** 150–159 (10). No earlier run or smoke slice used them.
- **Split:** the paper's 80/20 split (`data.paper_split`), the training rows in the seed's
  order, min-max scaling fitted on the training rows: the accuracy table's setup
  (`scripts/accuracy_table.py`), which the second explanation run also uses.
- **Model:** `CFMNN(theta=0.3, gamma=γ)`, the fixed centroid rule and search expansion.

## Part A: the Simpson model

**(1) PACE-SHAP against shap's ExactExplainer** (`exact.csv`).

- **Datasets:** the explanation datasets with at most 12 features, where PACE-SHAP is exact:
  S1, S2, S3 (10 features), Shuttle (9), MAGIC (10), Page Blocks (10). The feature counts
  were read from the data.
- **Per seed:** the model, test points and explainer rows of the second explanation run
  (`run_faithfulness.setup` and `backgrounds_v3`): the first **5** explained test points and
  the 200 explainer rows.
- **Compared:** `PaceShap(simpson_model, rows).explain(points)` and
  `xai_baselines.exact_shap(simpson_model, points, rows)`. Both compute the Shapley values of
  the predicted class's margin on the same rows. PACE-SHAP reads them from the boxes;
  ExactExplainer calls the model on every coalition.
- **Measure:** the largest absolute difference over every point, feature and seed.
- **Check, fixed now:** PACE "carries over" if the largest difference is at most **1e-9** on
  every dataset (the summary prints PASS or FAIL). On CFMNN itself the same comparison gave
  at most 2.5e-16 (`results/review_v1/exact/summary.md`, seeds 30–32).
- This is a check, not a statistical test.

**(2) Ties and accuracy, next to CFMNN** (`runs.csv`), on the 13 accuracy datasets (S1, S2,
S3, Wine, Breast Cancer, Digits, Thyroid, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks).
For the Simpson model and for the CFMNN it was made from (γ = 4):

- training error; the training errors that are ties (defined below); test accuracy and
  macro-F1; boxes;
- contractions, sweeps, boxes removed, training rows evicted;
- test predictions decided by the nearest centroid, and the test errors among them.

Descriptive. With contraction, Proposition 1 is **not expected to hold**; the share is
reported whatever it is.

## Part B: CFMNN at γ = 2, 4 and 8

On the same 13 datasets and seeds, θ = 0.3. For each γ: training error, the share of training
errors that are ties, the share of test predictions decided by the nearest centroid (and the
test errors among them), test accuracy and macro-F1, boxes.

- γ enters training too (it picks the box that expands), so the boxes can differ between γ.
- **Proposition 1 predicts 100% ties at every γ.** The summary prints the number of training
  errors that are not ties, over every dataset, seed and γ. Any number other than 0 would
  contradict the proposition, and would be reported as such.

## The two tie counts (`pace_cfmnn/ties.py`)

- **A training error is a tie** when the row's own class reaches the winner's membership
  (within the model's tolerance, 1e-9): the membership did not decide, and the nearest
  centroid picked the wrong class. It is a tie **inside both boxes** when both memberships
  are 1: the row lies inside a box of its own class and inside a box of the winning class.
  That second form is what Proposition 1 states, and it is the count used for Part B. For
  the Simpson model both counts are reported.
- **A test prediction is decided by the nearest centroid** when the best box of another
  class has the winner's membership: the rule of `peer_review_reanalysis.tie_rows`
  (`results/peer_review/reanalysis.md`, section 10).
- Shares are pooled over seeds (ties summed, divided by errors summed). A dataset with no
  training error has no share.

## Primary and pass rule

Two primary comparisons, both on **test accuracy**:

- **P1:** γ = 2 against γ = 4;
- **P2:** γ = 8 against γ = 4.

Each: a paired two-sided Wilcoxon test over the 10 seeds, Holm-corrected across the 13
datasets, α = 0.05 (`compare` in `scripts/run_readable_baselines.py`). When all 10
differences are zero, p = 1. The first γ is **better** on a dataset if the Holm p is below
0.05 and its mean accuracy is higher, **worse** in the other direction, otherwise **level**.
No direction is predicted. P1 and P2 are not corrected for each other.

Everything else is descriptive: Part A, the tie shares, macro-F1, boxes, the centroid-decided
shares. "Level" is not "equal": no equivalence test is made.

## What the result changes

- Results are reported whichever way they go.
- **γ stays 4** in the paper and in every committed result, whatever P1 and P2 say. If
  another γ is better on some datasets, the paper says so as a sensitivity result; it does
  not re-run earlier studies with it.
- If the exact check fails, the paper's claim that PACE carries over is withdrawn or
  narrowed to what the numbers support.
- If any training error at any γ is not a tie, Proposition 1's proof is re-examined before
  anything else is written.
- Any change to this plan after the push is a dated amendment (before the run) or a reported
  deviation (after it). Any analysis not listed here is labelled post hoc.

## Time

From the smoke slices: about **30 minutes** with 12 jobs on the home PC, between 15 and 45.
Nearly all of it is ExactExplainer on S1–S3 (about 5 minutes per task, 30 tasks) and MAGIC
(about 2 minutes per task). The `models` part takes a few minutes. If the home PC's smoke
slice (below) takes more than twice its expected time, stop and ask before the full run.

## Tests

`tests/test_generality.py` checks that:

- at γ = 2, 4 and 8 every training error of a toy CFMNN is a tie inside both boxes, and the
  counts equal a brute-force count from the memberships;
- after a contraction (Simpson's rule and GFMM's) the counts still equal brute force, and
  some error is then not a tie;
- the centroid-decided count equals a brute-force count;
- on a contracted model PACE-SHAP equals brute-force Shapley values and shap's
  ExactExplainer;
- the script's setup is as above, each model's row matches a model fitted directly, and the
  summary prints the verdicts, the pooled shares and PASS or FAIL correctly.

No shared module was changed, so no committed result needed re-checking.

## How to run

Smoke slice first (seed 190, written outside the repository; 73 s on the laptop):

```
python -B scripts/run_generality.py --datasets magic page_blocks --seeds 190 --out <scratch folder> --fresh
```

Then the run:

```
python -B scripts/run_generality.py --jobs 12
python -B scripts/run_generality.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_generality.py --summary-only
```

Outputs in `results/generality/`: `runs.csv` (one row per dataset, seed and model: gamma=2,
gamma=4, gamma=8, Simpson; 13 × 10 × 4 = **520 rows**), `exact.csv` (one row per dataset and
seed; 6 × 10 = **60 rows**) and `summary.md`. `--part models` or `--part exact` runs one
part.
