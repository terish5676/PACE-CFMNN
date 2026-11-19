# Run G: CFMNN + PACE-SHAP against other models that explain themselves — pre-registration

Written on 2026-11-13 and pushed (with the code it describes) before any run on the
study seeds, 120–129.

## Why

The peer review (`reviews/fable_peer_review.md`, weakness A1) said the paper never
compares "CFMNN + PACE" with other models that people can read and that already have an
exact explanation. A reader who wants an exact explanation could instead pick:

- an **explainable boosting machine (EBM)** with main effects only, which is a sum of
  one curve per feature, so its own terms are its explanation; or
- a **decision tree**, explained exactly by TreeSHAP.

This run compares the three as whole **systems** (a model plus its explanation): how
accurate the model is, how large it is, how long one explanation takes, and how well the
explanation ranks the features that drive the model's own decision.

We expect CFMNN to lose on accuracy (the readable-baselines run found EBM ahead on 12 of
13 datasets and the tree on 10) and probably on time. The run is meant to show that price
honestly. Results are reported whichever way they go.

## Setup

- **Datasets:** the 11 explanation datasets of RQ1: S1, S2, S3, Wine, Breast Cancer,
  Thyroid, Shuttle, MAGIC, Letter, Landsat, Page Blocks.
- **Seeds:** 120–129 (10). No earlier study used them.
- **Data:** the paper's 80/20 split, the seed's order of the training rows, min-max
  scaling fitted on the training rows (`prepare` in `scripts/run_readable_baselines.py`,
  the same as `setup` in `scripts/run_faithfulness.py`). Every model is trained on the
  same rows in the same order.
- **Explained points:** 50 test points per seed (36 on Wine, whose test set has 36 rows),
  chosen as in RQ1 (`setup`).
- **Two sets of 200 training rows** (or half the training set if smaller), as in RQ1 v3
  (`backgrounds_v3`): the deletion and insertion tests copy values from the first set;
  every explainer sees only the second set. No explanation is graded on rows it saw.
- Accuracy and macro-F1 use the whole test set.

## Systems

| System | Model | Explanation |
|---|---|---|
| **CFMNN + PACE-SHAP** | CFMNN, θ = 0.3, γ = 4 (the explained model; not tuned) | PACE-SHAP on the 200 explainer rows: exact when d ≤ 12, otherwise 32 antithetic feature orders per row (the RQ1 setting) |
| **EBM + terms** | interpret-core 0.7.8, main effects only (`interactions=0`), library defaults otherwise, `random_state` = seed, one process (as in `run_readable_baselines.py`) | its own per-feature terms, minus their mean on the 200 explainer rows |
| **Tree + TreeSHAP** | scikit-learn 1.9.1 CART, max leaves tuned over {4, …, 1024} by stratified 3-fold CV, as in `run_readable_baselines.py` | shap 0.54.1 `TreeExplainer`, `feature_perturbation="interventional"`, all 200 explainer rows |

**What each system explains: its own predicted-class margin.**

- CFMNN: the rescaled support of the predicted class minus the best other class (RQ1's
  margin).
- EBM: the logit of the predicted class minus the best other class's logit (with two
  classes, the logits are 0 and the log-odds).
- Tree: the leaf's share of the predicted class minus the best other class's share.

EBM and TreeSHAP give one value per feature for **each class**. They are turned into the
margin's values with weights fixed at the explained point: +1 on the predicted class and
−1 on the best other class. When several other classes tie for best (a pure tree leaf
gives every other class 0), the −1 is split equally among them, so no class is chosen by
its index. These values add up exactly to the point's margin minus the mean of the same
weighted score over the explainer rows (unit tests in `tests/test_systems.py`). With three
or more classes this explains the margin against the runner-up as seen at the point;
PACE-SHAP explains the margin with the best other class re-chosen in every coalition.
Both are exact for what they explain.

Two implementation details:

- For an additive model, its centred terms are also its exact interventional Shapley
  values, so "EBM + terms" and "EBM + SHAP" are the same thing.
- scikit-learn trees compare float32 inputs; rows are rounded to float32 before TreeSHAP
  so it follows the same paths as the tree.

## Recorded

For each dataset, seed and system (`results/systems/runs.csv`):

- test accuracy and macro-F1 (%);
- model size: rules (CFMNN boxes, EBM terms, tree leaves) and conditions per rule (as in
  `run_readable_baselines.py`); the tree's chosen number of leaves;
- milliseconds per explanation (building the explainer plus explaining all points,
  divided by the number of points), and model fit time in seconds;
- deletion and insertion margin AUC (RQ1 v2/v3 tests: signed ranking, 10 repetitions,
  margin divided by the point's own margin and clipped to [−1, 1]), and the raw-margin
  AUCs;
- the same AUCs for a **random ranking** on the same model and points: the mean of 5
  random rankings;
- the **normalised lead**, larger is better:
  - deletion: (random − method) ÷ |random|;
  - insertion: (method − random) ÷ |random|;
- the **gain**, the same differences without dividing;
- the share of explained points whose margin is zero.

## Comparisons

Each comparison is paired over the 10 seeds, two-sided Wilcoxon signed-rank test,
Holm-corrected across the 11 datasets. When all differences are zero, p = 1. CFMNN +
PACE-SHAP is **better** on a dataset if the Holm p is below 0.05 and the mean difference
favours it, **worse** the other way, otherwise **level**. Every comparison is also
reported with S1–S3 averaged into **one synthetic unit** per seed (9 units, Holm across
the 9).

| Code | Comparison | Measure |
|---|---|---|
| **P1 (primary)** | CFMNN + PACE-SHAP vs EBM + terms | normalised deletion lead |
| **P2 (primary)** | CFMNN + PACE-SHAP vs Tree + TreeSHAP | normalised deletion lead |
| S1 | both | normalised insertion lead |
| S2 | both | test accuracy (and macro-F1) |
| S3 | both | milliseconds per explanation (lower is better) |
| S4 (sensitivity) | both | deletion and insertion gain (not divided) |
| Descriptive | all three | model size, raw AUCs, random AUCs, chosen tree size |

P1 and P2 answer separate questions and are not corrected for each other.

**When the lead is unstable.** The lead divides by the random ranking's AUC. In the seed-99
smoke run, that AUC was close to zero on Landsat (−0.02 for CFMNN), which made the lead
+18.8; on Letter it was negative for all three models. So, decided now:

- the summary counts the runs whose random deletion AUC is within 0.05 of zero;
- on any dataset where at least one seed of a compared pair is flagged, the paper reads
  that dataset's P1/P2 verdict from the **gain** (S4) and reports the lead beside it.
  The primary tally is reported both ways (lead only, and with this rule).

## Expectations

Stated before the run, and reported whichever way it goes:

- **Accuracy:** CFMNN worse than both on most datasets.
- **Time:** EBM's explanation is a table look-up, so it is far faster than PACE-SHAP on
  every dataset. TreeSHAP is expected to be faster than PACE-SHAP on most datasets.
- **Size:** the tree is smaller than CFMNN on most datasets; EBM has d terms.
- **Normalised deletion lead:** no direction predicted. All three explanations are exact
  for their own model, so the lead mostly measures how concentrated each model's
  decision is on a few features.

## Smoke runs

Seed 99 only (not a study seed); output kept outside the repository and thrown away:

- Wine, all three systems (15 s), to check the code end to end;
- Letter, Shuttle, MAGIC, Landsat, Page Blocks, S1, Thyroid, Breast Cancer, `--jobs 4`
  on the laptop, for timing (26 min in all). EBM's fit took about 22 min on Letter and on
  Shuttle; PACE-SHAP took about 7 s per point on S1 and 3.7 s on Letter.

The smoke runs printed accuracy, lead and time. The one change made because of them is
the near-zero rule above, made because of the Landsat value, before any study seed ran.
One bug was fixed: shap keeps only 100 background rows unless told otherwise; TreeSHAP now
uses all 200.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.
- Every number the paper takes from this run is recomputed from `runs.csv` by a read-only
  script.
- Whatever the result, the paper's claim is narrowed to "for hyperbox models".

## How to run

From the repository root, on the home PC (i5-14400F):

```
python -B scripts/run_systems.py --jobs 12
```

- **Expected time:** about 2–3 h. Each dataset-seed pair runs in one process; the
  slowest (Letter and Shuttle, about 25 min each, mostly EBM's fit) start first.
- **Output:** `results/systems/runs.csv` (saved after each dataset-seed pair) and
  `summary.md`. A stopped run continues with `--resume`; `--summary-only` rebuilds the
  summary from the saved rows.

## Amendment 1 (2026-11-14, before the run)

Written and pushed, with its code and tests, before any study seed (120–129) was run.
Nothing above this section was edited; where this section differs, this section holds.

**Why.** A review of the plan and code, before the run, found three gaps. No bug was found
in what the code computes.

1. **Tied points make the lead not comparable across models.** CFMNN can tie: a point whose
   margin is zero. The deletion and insertion curves divide by the point's own margin (at
   least 1e-12), so for a tied point they turn into signs. The EBM and the tree almost never
   have such points. On tie-heavy datasets this pulls CFMNN's random-ranking AUC towards zero
   and raises its normalised lead for a reason that is not the quality of the ranking. The
   plan's near-zero rule catches only part of it.
2. **The times are taken while 12 tasks share the CPU,** and S3 tests them.
3. **A worker's first TreeSHAP call includes importing shap.** That adds seconds to one
   task's tree time per worker process (up to 12 of the 110 tasks).

**What was known when this was written.** The plan above; its seed-99 smoke runs; and smoke
slices on **seed 191** (not a study seed, never analysed), on the laptop, written outside the
repository: Wine, Breast Cancer and Page Blocks (the full task, before and after this change),
MAGIC and Page Blocks (CFMNN only, split into tied and untied points), and the timing pass on
Wine and Page Blocks. They printed seed 191's accuracies, leads and times. What was seen:

| Dataset, seed 191 | Tied points | CFMNN's lead, all points | Tied only | Untied only |
|---|---|---|---|---|
| MAGIC | 35 of 50 | 1.44 | 3.71 | 0.84 |
| Page Blocks | 31 of 50 | 0.60 | 0.93 | 0.48 |

MAGIC's random-ranking AUC was 0.098, so the near-zero rule (0.05) did not flag it. Wine's
tree time was 35 ms per explanation when its task imported shap and 0.06 ms when it did not.
One seed shows the mechanism; it is not evidence about the result. No study seed had been run.

**What is added.**

- **New columns** in `runs.csv`, after the existing ones, for every system: `n_untied`, and
  the deletion and insertion AUC, random AUC, lead and gain on the **untied** explained
  points only (`*_untied`). A point is untied when its own model's margin is above 1e-12
  (the rule `share_zero_margin` already uses). They are empty when no point is untied.
- **A new comparison, A1: CFMNN + PACE-SHAP against each other system on the normalised
  deletion lead of untied points,** the same test as the primary (paired two-sided Wilcoxon
  over the 10 seeds, Holm across the datasets tested). The deletion gain on untied points is
  reported beside it.
  - A1 is tested only on datasets where **every seed of every system has at least 10 untied
    points** (the rule of `scripts/run_review_rq1.py`). Holm is across those datasets.
- **A timing pass, run alone after the main run:** `--timing`, seed 120, one task at a time,
  the same models and explanations, with shap imported before anything is timed. It writes
  `timing.csv` (11 datasets × 3 systems = 33 rows) and a table in `summary.md`. It is one
  seed, so it is descriptive.

**What stays the same.** P1 and P2 (the lead on all points), the near-zero rule, every
secondary, the seeds, datasets, models and explainers. Every value the plan already
recorded is computed exactly as before: on seed 191 the old columns of `runs.csv` are
identical before and after this change (the times apart).

**How the results will be read** (written now, before the run):

- A dataset is **tie-heavy** when CFMNN's share of tied explained points, averaged over the
  seeds, is at least **10%**.
- On a tie-heavy dataset, P1's or P2's verdict is reported as it is, and beside it:
  - **"holds on untied points"** if A1 gives the same word (better, level or worse);
  - **"depends on the tied points"** if A1 gives a different word. The paper then does not
    count that dataset as a win or a loss for CFMNN on the strength of the primary alone; it
    reports both verdicts;
  - **"not checked: too few untied points"** if A1 could not be tested there.
- On other datasets the primary verdict stands as planned.
- The summary prints this table for each pair (`untied_reading` in the script).
- **Times:** S3 is computed and reported as planned. The paper quotes the times of the timing
  pass, not those of the main run. Where S3's direction against the tree disagrees with the
  timing pass on a dataset, both are reported and the main run's time is named as taken under
  load.

**Disclosed, with no change:**

- **Seeds 120–129 are also Run F's seeds.** Both plans say no earlier study used them. The
  two runs use different splits, and neither had run when the other was written, so no
  result of one could shape the other.
- The untied scores drop CFMNN's tied points but keep all of the other systems' points, so
  the systems are then scored on different subsets of the 50 points. Each system is already
  scored on its own model's margin; this is stated so the reader knows.

**Effect on the run.** The main run's time does not change (the new columns reuse curves
already computed). The timing pass adds about 1 to 1.5 hours alone: it refits every model
once, and EBM's fit on Letter and on Shuttle took about 22 minutes each in the seed-99 smoke
run. This estimate is rough; if the pass projects to more than 3 hours, stop and ask.

**Tests** (`tests/test_systems_amendment.py`, run on the laptop):

- the all-point values equal the old `curves` and random means exactly; the untied values
  are the mean over the untied points; with no tied point they equal the all-point values;
  with every point tied they are empty;
- the eligible and tie-heavy datasets, the three readings and the reading table, on made-up
  rows where the lead flips on untied points;
- the summary prints the amendment's sections, and prints none for rows without the new
  columns;
- the timing pass returns one row per system, and its table appears only once `timing.csv`
  exists.

**How to run** (the first line is the plan's; the second is added):

```
python -B scripts/run_systems.py --jobs 12
python -B scripts/run_systems.py --timing          # after the run, nothing else running
```

## Amendment 2 (2026-11-14, after a failed first run, before the run reported)

Written and pushed, with its code and test, after the first attempt of the main run failed
and before the run whose results are reported. Nothing above this section was edited; where
this section differs, this section holds.

**What happened.** The first attempt of the main run (`--jobs 12`, home PC, 61 minutes)
finished 107 of the 110 tasks. Three printed FAILED, all with the same error from shap's
`TreeExplainer`: "Additivity check failed" (Wine seed 120; Letter seeds 120 and 127).

**The cause.** A bug in how the tree was explained, not in the tree or in the other two
systems. scikit-learn compares a float32 input with the split's float64 threshold. shap's
tree code rounds the threshold to the nearest float32 and compares with that. When the
rounding goes up to exactly the input's value, shap sends the point left where the tree
sends it right. On Wine seed 120, the tree's split on feature 12 has threshold
0.34379456937 (float64); explained point 19's value rounds to 0.3437945843 (float32), which
is above the threshold, so the tree sends it right; the threshold also rounds to
0.3437945843, so shap sent it left, and its SHAP values added up to another leaf's output.
The plan's rounding of the rows to float32 ("so it follows the same paths as the tree")
covered the inputs but not the thresholds.

**The fix** (`float32_floor` in `scripts/run_systems.py`): before explaining, each of the
explainer's thresholds is set to the largest float32 value at or below the tree's float64
threshold. For any float32 input x, x ≤ t holds exactly when x ≤ that value, so shap now
follows the tree's own paths for every explained point and every explainer row. The tree,
its predictions and every other system are unchanged. shap's additivity check stays on.

**Checked:**
- a new test, `tests/test_systems_amendment2.py`: the floor is the largest float32 at or
  below each threshold and gives the same comparisons; a one-split tree whose threshold
  rounds up to a point's value now gives TreeSHAP values that add up to the point's score
  difference. Without the fix that test fails with the same ExplainerError;
- on the three failed tasks' trees (Wine 120, Letter 120 and 127), the tree's margin
  values now add up to the margin minus its explainer-row mean to within 5e-8;
- the full test suite passes.

**What was known when this was written.** The plan and Amendment 1; the first attempt's
console output, of which only its first task's line (Shuttle seed 121: accuracy, lead and
milliseconds of the three systems) and the three FAILED lines were read; the number of rows
it saved (321). Its `runs.csv` and `summary.md` were not opened. They were not committed and
are discarded.

**Effect on the run.** shap's additivity check allows a small tolerance, so an explainer row
that took the wrong path could have shifted a task's tree values slightly without failing.
So the whole main run is repeated from the start (`--fresh`) with the fixed code, not only the
three failed tasks; every reported row comes from the same code. Everything else is as
planned: P1 and P2, the near-zero rule, Amendment 1's untied reading and timing pass, the
seeds, datasets, models and explainers.
