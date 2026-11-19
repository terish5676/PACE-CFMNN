# Run P5, explaining the decision at ties: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review S6 and
B3). The code is `scripts/run_decision.py` (tests in `tests/test_decision_run.py`). It uses
`pace_cfmnn/decision.py` (commit 7843fee): `PaceDecision` and `brute_force_decision`.

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 13.

## Background (in plain words)

PACE-SHAP explains the **margin**: the winning class's rescaled support minus the best other
class's. At a **tie** the two supports are equal, so the margin is 0, and CFMNN's nearest-centroid
rule picks the class. The margin then says little about why the model chose that class (review
S6).

PACE-decision explains the **decision** itself. For a set S of features, v(S) is the share of
explainer rows z for which the mixed input (x on S, z elsewhere) is predicted as the explained
class. The prediction uses CFMNN's full rule, ties included. Its Shapley values say how much each
feature moves the chance of keeping the class.

**Tied point:** the runner-up class's rescaled support is within 2 × `tie_tol` (2e-9) of the
winner's.

## What ran before this was written

- **Tie counts in the second explanation run** (v3, seeds 30–39, 50 test points per seed):

  | Dataset | Fewest tied points in a seed |
  |---|---|
  | MAGIC | 30 |
  | Page Blocks | 29 |
  | Letter | 25 |
  | Landsat | 5 |
  | Shuttle | 3 |
  | Thyroid | 1 |
  | the other 5 datasets | 0 |

  This fixed the datasets below: those with at least 10 tied points in every seed. That gives
  enough tied points per seed for a per-seed mean.
- **Tests while building:** `tests/test_tough_explain.py`, class `TestPaceDecision`:
  - exact PACE-decision equals `brute_force_decision` to 1e-12 on toy models;
  - the sampled mode adds up;
  - tied rows get non-zero values.
- **Smoke slice on the laptop, seed 194** (10 points, not a study seed, written outside the
  repository; 7–33 s per dataset). Decision deletion AUC on the tied points (lower is better):

  | Dataset | Tied of 10 | PACE-decision | PACE-SHAP |
  |---|---|---|---|
  | Page Blocks | 8 | 0.566 | 0.779 |
  | MAGIC | 6 | 0.528 | 0.525 |
  | Letter | 6 | 0.076 | 0.081 |

  Check: PASS (difference from the brute force 0 where exact).

The design did not change after the slice.

## The question

**At tied points, is the decision explanation more faithful to the model's decision than the margin
explanation?**

## Setup

- **Datasets:** MAGIC, Letter and Page Blocks.
- **Seeds:** 320–329 (new); 50 test points per seed.
- **Models and rows:** those of the second explanation run (`run_faithfulness.setup` and
  `backgrounds_v3`):
  - CFMNN at θ = 0.3;
  - 200 evaluation rows for the deletion and insertion tests;
  - 200 explainer rows for both explainers.
- **Explainers:** PACE-decision and PACE-SHAP. Both are exact where d ≤ 12 (MAGIC and Page Blocks).
  Letter (d = 16) is sampled for both, with 32 antithetic permutations and the seed.

## Primary test

- **Measure:** the **decision deletion AUC** on tied points. Delete the most supportive feature
  first, filling it in from the evaluation rows. At each of k = 0..d deletions, record the share of
  fill-ins that keep the class. Average over k. This is `run_faithfulness.deletion_auc`'s first
  output, and lower is more faithful.
- **Per seed:** the mean over the seed's tied points, for each explainer.
- **Comparison:** PACE-decision against PACE-SHAP, a paired two-sided Wilcoxon test over the 10
  seeds, Holm across the 3 datasets.

## Secondary (reported, not used for the reading)

- Decision deletion AUC on untied points.
- Decision insertion AUC (higher is better) on tied points.
- Margin deletion and margin insertion AUCs on tied points. These measure what PACE-SHAP explains.
- Milliseconds per explanation.

## Check (pass/fail)

Where exact, PACE-decision equals `brute_force_decision` (the model's own prediction on every mixed
input) within 1e-9, on the first 2 points of each seed.

## A bias stated in advance

The primary measure counts kept decisions, which is close to the quantity PACE-decision explains.
The margin measures are close to what PACE-SHAP explains. So each explainer has a home measure.
The write-up reports both families side by side, and does not call either explainer better
overall from its home measure alone.

## How the result will be read (written now, before the run)

- **PACE-decision better on at least 2 of the 3 datasets** (Holm p < 0.05):
  - the paper offers PACE-decision as the explanation to use at ties;
  - it fixes S6's sentence to say what the margin explains;
  - it reports the margin measures next to it.
- **No significant difference on at least 2 of the 3:** the paper keeps PACE-SHAP. It says plainly
  that at a tie the margin is 0 and the decision comes from the centroid rule. It gives
  PACE-decision as an exact alternative, with this null result.
- **PACE-SHAP better on at least 2 of the 3:** as the previous case, and the paper says the
  decision game was less faithful even on its own measure.
- **Any other split:** each dataset is reported with its verdict, with no overall claim.
- **Check FAIL:** PACE-decision's exactness claim is withdrawn until it is fixed.

## Time

The smoke slice took 7–33 s per dataset for 10 points. For 50 points, about 1–3 minutes per seed on
the laptop. **About 15–30 minutes on the home PC with 6 jobs.** Stop and ask if it is past 2 hours.

## Tests

`tests/test_decision_run.py`:
- the comparison's direction and Holm step;
- the check;
- the summary's primary heading;
- one small task end to end.

## How to run (home PC, batch 2)

```
.\.venv\Scripts\python -B scripts/run_decision.py --jobs 6 *>&1 | Tee-Object results/decision/run_log.txt
.\.venv\Scripts\python -B scripts/run_decision.py --jobs 6 --resume *>&1 | Tee-Object -Append results/decision/run_log.txt   # after a stop
```

Commit `results/decision/` unedited (`points.csv`, `summary.md`, `run_log.txt`) as "Run P5:
results of the pre-registered run on the home PC".
