# Run M, Simpson's classifier trained from scratch: pre-registration

Written on 2026-11-15 on the home PC (i5-14400F) and pushed, with the code it describes, before
any study seed ran. Plan: `research_audit/PART_B_PLAN.md`, "Run M".

- **The code (new):** `pace_cfmnn/simpson_fmm.py` (the model, `SimpsonFMM`),
  `scripts/run_simpson_fmm.py` (the run), `tests/test_simpson_fmm.py` (the tests).
- **Used as it is, by import:** `pace_cfmnn/cfmnn_core.py` (`CFMNN`, `feature_terms`,
  `predict_details`), `pace_cfmnn/contraction.py` (Run F's `_candidates`, `overlap_test`,
  `contract_pair`), `pace_cfmnn/explain.py` (`PaceShap`), `pace_cfmnn/xai_baselines.py`
  (`exact_shap`), `pace_cfmnn/ties.py`, `scripts/run_generality.py` (`prepare`, `describe`),
  `scripts/run_faithfulness.py` (`setup`, `backgrounds_v3`), `scripts/accuracy_table.py`
  (`DATASETS`). No existing file was changed for this run.

## Why

Run I (`results/generality/`) showed that PACE-SHAP stays exact on "the Simpson model", but
that model was CFMNN's boxes contracted once after training (Run F's SC-all arm), not Simpson's
classifier trained from scratch, and it still broke ties by the nearest centroid. A reviewer
can fairly say that "PACE carries over to Simpson's FMM" was not tested on Simpson's FMM. This
run trains Simpson's classifier from scratch, with the contraction inside training, and
repeats Run I's Part A on it.

## Before the plan: is GFMM's membership a minimum over features?

**Yes.** Read on the home PC in hyperbox-brain 0.1.6 (the version installed in the project's
`.venv`):

- **File:** `.venv/Lib/site-packages/hbbrain/utils/membership_calc.py`.
- **Function `membership_func_gfmm(xl, xu, V, W, g=1)`** (Gabrys and Bargiela, IEEE TNN 11(3),
  2000): it computes, per box and feature, `viol_max = 1 - _ramp_func(xu - W, g)` and
  `viol_min = 1 - _ramp_func(V - xl, g)`, and returns
  `b = np.minimum(viol_max, viol_min).min(axis=1)`: the **minimum over features** (and over the
  two sides).
- **For comparison, `membership_func_fmnn(x, V, W, g=1)`** in the same file (Simpson 1992)
  returns `np.sum(viol_mat, axis=1) / (2 * n_features)`: the **mean over features** of the two
  sides' terms, the form CFMNN uses.

So, as `PART_B_PLAN.md` says, this plan (and later the paper) states that PACE covers
mean-type memberships (CFMNN, Simpson's FMM) and **not GFMM as it is**: PACE-SHAP's exact mode
relies on the membership being a sum (mean) of per-feature terms, and a minimum over features
is not.

## The model: `SimpsonFMM` (`pace_cfmnn/simpson_fmm.py`)

A subclass of `CFMNN` whose `fit` trains one row at a time, in the given order, in Simpson's
three steps. θ and γ are CFMNN's (θ = 0.3, γ = 4). It exposes `V_`, `W_`, `box_class_`,
`gamma`, `classes_` and CFMNN's `predict_details`, so `PaceShap`, `xai_baselines.exact_shap`
and `ties` run on it unchanged.

1. **Membership:** CFMNN's (`feature_terms`, averaged over the features), the same as
   hyperbox-brain's `membership_func_fmnn`.
2. **Expansion:** if the row already has membership 1 in a box of its class, nothing changes.
   Otherwise the box of its class with the highest membership **among those that can take the
   row within the size limit** is expanded; with none, a new point box is made. This is
   CFMNN's `expansion="search"`, kept so that the only difference from CFMNN is the
   contraction. (hyperbox-brain's `fmnn.py` tries only the box of highest membership and
   makes a new box if that one cannot expand; this run does not follow it there.)
3. **Size limit:** CFMNN's sum-of-sides form, `sum_i (w_i − v_i) ≤ d × θ` (also the form in
   hyperbox-brain's `fmnn.py`), not a per-feature bound.
4. **Overlap test and contraction, only after an expansion:** the expanded box is tested
   against every box of another class, in box order, with Simpson's published test (four
   cases, strict inequalities: `contraction._candidates` and `overlap_test`), and each pair
   that overlaps is contracted on the feature of smallest overlap, the lowest feature on a tie
   (`contraction.contract_pair`, Run F's code, unchanged). Each box is tested on the corners
   as they stand at its turn. **A box made as a new point is not tested**: Simpson (1992)
   tests the box expanded in the last step, and hyperbox-brain's `fmnn.py` (`fit`) runs the
   overlap test only when `adjust` is true, that is, after an expansion.
5. **Tie rule (Simpson's classifier has no centroid):** the class is that of the box with the
   highest membership; an exact tie (within `tie_tol` = 1e-9) goes to **the box made first
   (the lowest box index)**. To keep `predict_details` unchanged, every centroid is set to
   the same point (the origin), so the nearest-centroid step always ties and the lowest box
   index wins. `ties.centroid_decided` then counts the test predictions this tie rule
   decides.
6. `members_` keeps the rows each box took when they joined; a later contraction may leave a
   member outside its box, and nothing is recomputed from members. `add_points` (refinement)
   is not defined for this model.

## A deviation from `PART_B_PLAN.md`, found while building

The plan asks for a test that "no two boxes of different classes overlap by Simpson's test
after `fit`". **That property does not hold for the published algorithm**, because of step 4:
a new point box placed inside a box of another class is never tested, and a later contraction
can also change which features a pair shares. `tests/test_simpson_fmm.py` shows this with a
hand-made example (3 rows; the pair (0, 1) still overlaps after `fit`, no contraction ran).
Changing the algorithm to make the property hold (for example, testing new boxes too) would
no longer be Simpson's classifier. So, instead of that test:

- a test with a hand-made example where an expansion into another class's box **is**
  contracted and no pair overlaps afterwards;
- a test with the hand-made example above, where a new point box is **not** tested and one
  pair still overlaps;
- the run records, for every model, **`overlapping_pairs`**: the number of pairs of boxes of
  different classes that Simpson's test still finds overlapping after training
  (`SimpsonFMM.overlapping_pairs`). It is reported, whatever it is.

## What ran before this was written

- **Unit tests** on made-up data while building; then `tests.test_simpson_fmm` (8 tests, all
  pass) and the full suite: **366 tests, all pass** (`python -B -m unittest discover -s
  tests`, 512 s).
- **Smoke slices on seed 192** (not a study seed, never analysed), written outside the
  repository:
  1. Skin alone, the models part (`--datasets skin --seeds 192 --part models --out
     $HOME\scratch\run_m`): **12 s** in all; CFMNN fit 3 s, SimpsonFMM fit 4 s; peak memory
     196 MB. Looked at: the printed line, the summary, `runs.csv`.
  2. Everything on seed 192, 12 jobs (`--seeds 192 --jobs 12 --out $HOME\scratch\run_m_all`):
     **181 s**; peak memory of the Python processes 2.15 GB, at least 9.2 GB free. Looked at:
     every printed line, the summary and both CSVs.

  So these were seen, for seed 192: the exact check printed **PASS** (largest difference
  1.9e-16 over the six datasets); SimpsonFMM's test accuracy was lower than CFMNN's on
  Shuttle (88.8% against 99.8%), Letter (63.9% against 75.7%), MAGIC and Thyroid and some
  others, and higher on Skin (97.5% against 92.2%), Breast Cancer and Landsat; Shuttle had
  34,649 contractions with 15 boxes; Thyroid and Digits had none; one pair still overlapped
  on Landsat and none on the others; datasets with no training error show no tie share
  ("nan"). One seed is not evidence; only seeds 220–229 are.
- **What changed because of the smoke slices:** nothing.

## Setup

- **Seeds:** 220–229 (10). No earlier run or smoke slice used them.
- **θ = 0.3, γ = 4** for both models.

**(1) The exact check** (`exact.csv`), as Run I's Part A (1), on SimpsonFMM:

- **Datasets:** S1, S2, S3, Shuttle, MAGIC, Page Blocks (at most 12 features, where PACE-SHAP
  is exact).
- **Per seed:** the second explanation run's split, training rows and labels
  (`run_faithfulness.setup`, whose row order the script repeats to recover the labels), the
  first **5** explained test points and the 200 explainer rows (`backgrounds_v3`).
  SimpsonFMM is trained on those training rows.
- **Compared:** `PaceShap(simpson, rows).explain(points)` (the script stops if it is not in
  exact mode) and `xai_baselines.exact_shap(simpson, points, rows)` (shap's ExactExplainer).
- **Measure:** the largest absolute difference over every point, feature and seed.
- **Check, fixed now:** **PASS** if the largest difference is at most **1e-9** on every
  dataset, otherwise **FAIL**. The summary prints `Check: PASS` or `Check: FAIL`. A check,
  not a statistical test.

**(2) SimpsonFMM beside CFMNN** (`runs.csv`), on the 13 accuracy datasets (S1, S2, S3, Wine,
Breast Cancer, Digits, Thyroid, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks), with the
accuracy table's split (`run_generality.prepare`) and `run_generality.describe`: for each
model, boxes, training error, the training errors that are ties (both of `ties.py`'s forms),
test accuracy and macro-F1, test predictions decided by the tie rule and the test errors among
them, seconds to train, `overlapping_pairs`; and for SimpsonFMM the contractions and how many
fell in each of Simpson's four cases. Ties and tie-rule-decided shares are pooled over seeds.
Descriptive: no test.

## Design and what the result changes

- **The exact check is a check; the rest is descriptive.** No primary, no direction predicted.
- If the check passes, the paper may say that PACE-SHAP stays exact on Simpson's classifier
  trained from scratch (6 datasets, 10 seeds, 5 points each). If it fails, the run stops
  there, and that is reported.
- Accuracy, ties and overlapping pairs are reported whichever way they go, including
  "SimpsonFMM is less accurate than CFMNN" on any dataset.
- With contraction, Proposition 1 (every training error is a tie) is not expected to hold for
  SimpsonFMM; its tie share is reported whatever it is.
- Any change to this plan after the push is a dated amendment (before the run) or a reported
  deviation (after it). Any analysis not listed here is labelled post hoc.

## Time

From the smoke slice (181 s for one seed, 12 jobs, mostly the exact check on S1–S3, about
29 s per point with ExactExplainer), the 10 seeds with 12 jobs should take about **10–20
minutes**, with about 2–3 GB of memory. Stop and ask if it projects over 6 hours or free
memory comes within 2 GB of running out.

## Tests

`tests/test_simpson_fmm.py` checks that:

- on toy problems at four (θ, γ) settings, `SimpsonFMM` gives exactly the boxes of a literal,
  slow, row-by-row and feature-by-feature transcription of the published algorithm (with
  CFMNN's size limit and expansion), and that contractions happen;
- with one class (so no overlap test can fire) its boxes are CFMNN's;
- the two hand-made examples above (an expansion contracted; a new point box not tested);
- ties go to the lowest box index, all centroids are the origin, `ties.py` runs on it, and
  `add_points` is refused;
- PACE-SHAP on it equals shap's ExactExplainer and brute-force Shapley values
  (`tests.test_explain.exact_shapley`) to 1e-12;
- the script's seeds, datasets and constants are as above; one models task writes one row per
  model with the right columns and CFMNN's accuracy; and the summary prints PASS at 1e-16 and
  FAIL at 1e-6.

## How to run

```
python -B scripts/run_simpson_fmm.py --jobs 12
python -B scripts/run_simpson_fmm.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_simpson_fmm.py --summary-only
```

Outputs in `results/simpson_fmm/`: `runs.csv` (13 datasets × 10 seeds × 2 models =
**260 rows**), `exact.csv` (6 datasets × 10 seeds = **60 rows**), `summary.md` and the log
`run_log.txt`.
