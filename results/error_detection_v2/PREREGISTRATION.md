# Pre-registration: RQ2 v2 confirmatory run

Written and pushed to GitHub before the run started. The commit that adds this file fixes the code:
- `pace_cfmnn/signals.py` (`box_evidence`);
- `pace_cfmnn/trust_combine.py`;
- `scripts/run_error_detection_v2.py --protocol confirmatory` (run with `--jobs 4`; parallel runs give
  identical results).

Nothing is changed after the run starts. Any bug fix found later is reported here with its effect.

## Why this run

The RQ2 confirmatory run (seeds 10–19) tested the blueprint's signals. PD + U\* beat the model's own
margin, but every PACE signal lost clearly to standard methods that read the training labels (kNN
disagreement, Trust Score, an order ensemble).

A later review found the reasons:
- EI could hardly add information for CFMNN.
- No single-model, label-free signal can know what a more accurate model would predict.
- The signals look at different failure modes:
  - **overlap:** two classes' boxes contain the point (U\*, margin, tie);
  - **empty space:** a box covers regions with no training points (box evidence, box member count);
  - **order dependence:** single-pass training (order ensemble);
  - **the data:** kNN disagreement, Trust Score.

This run tests whether CFMNN's own signals **add** to the standard data-based ones when they are
combined.

## What was known before this run

- **The RQ2 confirmatory results** (seeds 10–19).
- **An exploratory check on seeds 0–2** (6 datasets): a logistic regression over all signals beat
  kNN and Trust Score on all 6. Adding the CFMNN-native signals to kNN + Trust Score helped in 16 of
  18 seed runs: a lot on Thyroid, a little elsewhere.
- **A parity check on seed 10** (Thyroid, Digits): the v2 runner reproduces the RQ2 confirmatory
  splits and baseline scores exactly.
- **The design run** (seeds 0–9, all 16 datasets; `results/error_detection_v2_design/`). Its one
  decision rule was written down before its results were seen: **PD is included as a signal only if
  adding it to LR(all) raises AUPRC by at least 0.01 on at least half of the tested units.**
  Outcome: adding PD never raised AUPRC by 0.01 on any of the 8 units (changes from −0.020 to
  +0.000), so **PD is not used**.
- **What the design run showed:**
  - **The order ensemble carries the whole native gain.** Removing it from LR(all) costs 0.281
    AUPRC on Thyroid, 0.035 on the synthetic family and 0.010 on MAGIC.
  - **The geometry signals add nothing once kNN and Trust Score are present.** That covers U\*,
    margin, tie, box evidence and box count (drop-one changes −0.043 to +0.012).
  - On Digits, with about 10–15 errors per cross-validation fold, they lower LR(all) below
    LR(kNN + TS): −0.049.
- **Changed after the design run, from the committed draft (0e2c940):**
  - A group LR(kNN + TS + order ensemble) is added.
  - The primary endpoint becomes that group vs LR(kNN + TS).
  - The draft's primary, LR(all) vs LR(kNN + TS), is kept as secondary endpoint Q1c.
  - Q2 now uses the new group.
  - Nothing else changed.

## Fixed settings

- **Seeds:** 20–29 (never used before).
- **Model:** CFMNN, fixed centroid rule + search expansion, θ = 0.3, γ = 4.
- **Datasets, splits and eligibility:** exactly as in the RQ2 confirmatory run.
  - 16 datasets.
  - Under 2,000 rows: stratified 5-fold CV, with 20% of each training part as the inner rows and
    pooled out-of-fold scores.
  - Otherwise the role split 50/20/10/10/10, scoring pool + acceptance + test rows (at most 20,000).
  - Eligible: at least 50 errors in every seed.
  - The 6 synthetic sets are one unit (per-seed mean).
- **Signals** (higher = more suspicious):
  - **U\*;**
  - **margin uncertainty:** 1 − (s′₁ − s′₂);
  - **tie:** 1 if the top two classes tie;
  - **box evidence:** distance to the nearest training row the winning box holds;
  - **box count:** −log(1 + members of the winning box);
  - **order ensemble:** the share of 5 CFMNNs, trained in other orders, that disagree;
  - **kNN disagreement:** k = 10;
  - **Trust Score:** k = 10, α = 0.1.
  - PD (prediction disagreement) is not used (see the design rule above).
- **Learned combinations:** standardised signals → logistic regression (C = 1, class-balanced),
  fitted on the **inner rows with their labels** (a labelled calibration set), scored on the
  evaluation rows.
  - LR(kNN + TS): the two data-based signals.
  - LR(kNN + TS + order ens.): plus the order ensemble.
  - LR(native): the six CFMNN-native signals.
  - LR(all): all eight.
  - Every group gets the same combiner.

## Endpoints

**Metric:** AUPRC. **Test:** paired two-sided Wilcoxon over the 10 seeds per unit, Holm across units,
α = 0.05.

- **Q1 (primary).** LR(kNN + TS + order ens.) vs LR(kNN + TS). Hypothesis: higher, i.e. CFMNN's
  training-order sensitivity adds to the data-based signals.
- **Q1b.** LR(all) vs LR(kNN + TS + order ens.): do the geometry signals add beyond that?
  (Two-sided; the design run suggests no.)
- **Q1c.** LR(all) vs LR(kNN + TS), the draft's primary. Hypothesis: higher.
- **Q2a / Q2b.** LR(kNN + TS + order ens.) vs kNN disagreement, and vs Trust Score. Hypothesis:
  higher.
- **Q3.** LR(native) vs kNN disagreement (two-sided, no direction predicted). Can the model's own
  signals alone compete?
- **Descriptive:**
  - AUROC;
  - each single signal;
  - drop-one importance (AUPRC lost when one signal is removed from LR(all)).

## How results will be reported

- Every endpoint is reported for every unit, whether it supports the hypothesis or not. Excluded
  datasets are listed.
- The earlier RQ2 runs stay as they are. This run uses labelled calibration rows, which the
  blueprint's PRS did not.
