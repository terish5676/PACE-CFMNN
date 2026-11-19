# Run P4, exact PACE-SHAP for GFMM-family networks: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review B2).
The code is `scripts/run_gfmm_exact.py` (tests in `tests/test_gfmm_exact_run.py`). It uses
`pace_cfmnn/gfmm_explain.py` (commit d003ed1): `GFMMBoxes.from_hbbrain`, `PaceShapMin` and
`brute_force_min`. Since the smoke slice, `brute_force_min` works through the coalitions in blocks
of at most 4 million numbers. The values are unchanged; it just no longer runs out of memory on
MAGIC.

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 13.

## Background (in plain words)

CFMNN scores a point by the **mean** of one-feature membership terms. General fuzzy min-max (GFMM)
models score it by their **minimum**. PACE-SHAP is exact for CFMNN because the mean of a coalition's
terms is a sum it can build from a table. `PaceShapMin` builds every coalition's minimum from a
table of subset minima instead, so it should give the exact Shapley values of the GFMM margin
(the winning class's membership minus the best other class's).

This run checks that claim on models fitted by hyperbox-brain 0.1.6, and measures the time.

## What ran before this was written

- **Tests while building:** `tests/test_gfmm_explain.py`, on toy models:
  - exact PACE-SHAP-min equals the brute force;
  - the adapter predicts like hyperbox-brain;
  - the sampled mode works;
  - PACE-flip-min equals occlusion.
- **Laptop timing check, seed 194** (one point, not a study seed):

  | Dataset | Model | Boxes | Fit | PACE-SHAP-min | Brute force |
  |---|---|---|---|---|---|
  | Page Blocks | Online GFMM | 85 | 3 s | 0.06 s | 7.3 s |
  | Page Blocks | IOL-GFMM | 181 | 1 s | 0.13 s | 17.4 s |
  | Shuttle | Online GFMM | 31 | 17 s | 0.07 s | 1.0 s |
  | Shuttle | IOL-GFMM | 116 | 3 s | 0.06 s | 4.2 s |
  | MAGIC | Online GFMM | 319 | 45 s | 0.38 s | 25.6 s |
  | MAGIC | IOL-GFMM | 2,221 | 129 s | 5.44 s | 181.9 s |
  | S1 | Online GFMM | 2,371 | 92 s | 21.8 s | 201.3 s |
  | S1 | IOL-GFMM | 2,371 | 12 s | 18.7 s | 212.8 s |

  The first MAGIC brute force ran out of memory (4.87 GiB). That is why `brute_force_min` now
  works in blocks.
- **Smoke slice of the timing part, seed 194** (Page Blocks and Shuttle, written outside the
  repository): check PASS. Differences from the brute force were about 1e-16. ExactExplainer took
  31–139× as long as PACE-SHAP-min.

The design did not change after the slice. Because of the brute-force times above, only the first
3 points per seed are compared with the brute force, and the timing part explains 1 point per seed.

## The question

**Does PACE-SHAP-min give the exact Shapley values of a GFMM margin on real fitted models, and how
much faster is it than enumerating every coalition?**

## Setup

- **Datasets:** the explanation datasets (RQ1) with at most 12 features, where an exact method is
  practical: S1, S2, S3, Shuttle, MAGIC and Page Blocks.
- **Models:**
  - Online GFMM (θ = θ_min = 0.3, γ = 4);
  - IOL-GFMM (θ = 0.3, γ = 4).
- **Data:**
  - the paper's 80/20 split of each seed, with features scaled on the training rows;
  - explained points and explainer rows clipped to [0, 1], the range hyperbox-brain expects;
  - explainer rows: `backgrounds_v3`'s 200.
- **`check` part:**
  - seeds 310–319, 10 test points per seed;
  - the adapter must predict like hyperbox-brain on the explained points;
  - PACE-SHAP-min explains all 10 points (exact, with pruning);
  - the first 3 are compared with the brute force (`brute_force_min`: the margin on every mixed
    input).
- **`timing` part:**
  - seeds 310–312, 1 point each, run alone (`--jobs 1`);
  - PACE-SHAP-min, the brute force and shap's ExactExplainer on the same margin, each timed with
    its own set-up;
  - PACE-SHAP-min gets a warm-up call first, which is not timed.

## Check (pass/fail)

- The adapter predicts like hyperbox-brain on every explained point.
- PACE-SHAP-min equals the brute force within **1e-12** on every compared point.
- Where timed, ExactExplainer agrees with PACE-SHAP-min within **1e-9**.

## Reported (descriptive)

- Milliseconds per explanation for each arm.
- The ratios brute force ÷ PACE and ExactExplainer ÷ PACE.
- Boxes.
- The share of (row, box) pairs that pruning skipped.

## How the result will be read (written now, before the run)

- **PASS:** the paper states that PACE-SHAP-min gives exact values for GFMM-family models (B2),
  with the measured speed ratios. This is the GFMM coverage the TFS version needs. The paper says
  the cost still grows as 2^d (Proposition 5's hardness argument is for the mean; it is not
  claimed for the minimum).
- **FAIL:** the GFMM extension is withdrawn from the paper until the cause is found. A fix would be
  run as a new variant with its own pre-registration, not in place of this run.

## Time

From the laptop timing check (the home PC is about 3× faster):
- **check:** about 0.5–1 hour on the home PC with 10 jobs;
- **timing:** about 1 hour alone.

Stop and ask if either is past 3 hours.

## Tests

`tests/test_gfmm_exact_run.py`:
- ExactExplainer on the margin matches PACE-SHAP-min on a toy model;
- the check fails as it should;
- one small task end to end.

## How to run (home PC, batch 2)

```
.\.venv\Scripts\python -B scripts/run_gfmm_exact.py --part check --jobs 10 *>&1 | Tee-Object results/gfmm_exact/run_log.txt
.\.venv\Scripts\python -B scripts/run_gfmm_exact.py --part timing *>&1 | Tee-Object -Append results/gfmm_exact/run_log.txt   # alone, nothing else running
```

Add `--resume` to either after a stop. Commit `results/gfmm_exact/` unedited as "Run P4: results
of the pre-registered run on the home PC".
