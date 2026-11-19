# Run P10, RQ1 and RQ2 on held-out datasets: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before any study seed ran (tough
review B11). The code is `scripts/run_heldout.py` (tests in `tests/test_heldout_run.py`; the
loader's tests are in `tests/test_openml_data.py`). Runbook: `research_audit/HOME_PC_RUNBOOK.md`,
section 15.

## The datasets

The rule in [`RULE.md`](RULE.md) was frozen and pushed (commit 94dd0ed) before any file was
downloaded. It gives four datasets from OpenML-CC18:

| Dataset | Rows | Features | Classes | Smallest class |
|---|---|---|---|---|
| phoneme | 5,404 | 5 | 2 | 1,586 |
| wilt | 4,839 | 5 | 2 | 261 |
| mfeat-morphological | 2,000 | 6 | 10 | 200 |
| jungle_chess_2pcs_raw_endgame_complete | 44,819 | 6 | 3 | 4,335 |

- The user approved the downloads on 2026-11-16.
- Every file matched the MD5 that OpenML publishes, and every file loaded. So none is left out.
- The files' SHA-256 are pinned in `scripts/download_data.py`.
- The counts above are checked against the loaded files in `tests/test_openml_data.py`.
- Every feature is numeric, as the rule requires. Nominal attributes with numeric codes are read
  as numbers.

## What is known before the run (stated now)

**KernelSHAP is exact on these datasets.**
- With d features, there are 2^d − 2 coalitions to evaluate. With 5 or 6 features that is 30 or
  62, fewer than the 200 that KernelSHAP is given.
- In that case `xai_baselines.shapley_kernel_coalitions` enumerates every coalition, so
  KernelSHAP computes the exact Shapley values (the shap library does the same).
- So in the like-for-like comparison, PACE-SHAP and KernelSHAP compute the **same** values on
  the same rows: two exact methods.

**What that means for RQ1.**
- RQ1's primary test cannot show a difference in faithfulness here. It is kept, as the review
  asked, but it is read as a check of exactness, not as evidence for either method.
- Deletion curves can still differ slightly. Values that agree to 1e-12 can sort differently
  when two features are almost tied.
- RQ1 has two informative parts here:
  - **time:** two exact methods giving the same values, so speed is the whole comparison;
  - **KernelSHAP at its usual setting** (the second explanation run, protocol v3: 20 background
    rows against PACE-SHAP's 200). KernelSHAP is exact on its own 20 rows, so this compares
    explaining with 20 rows against explaining with 200.

**The check, run on every dataset and seed.** The script recomputes both explainers on the
like-for-like rows and records the largest |PACE-SHAP − KernelSHAP|. **PASS** if it is below
1e-9 everywhere. A FAIL would mean a bug in one of them. The script then exits 1: stop and ask,
and no RQ1 result is used until the bug is found.

## Setup

**RQ1 (seeds 360–369, 50 test points per seed):** the paper's two explanation protocols,
unchanged.
- **Like for like** (`run_faithfulness.run_cond`): every explainer on the same 200 explainer
  rows; KernelSHAP with 200 coalitions.
- **The second explanation run** (`run_faithfulness.run`, protocol v3): PACE-SHAP on 200 rows;
  KernelSHAP (the shap library) on 20 rows; also KernelSHAP-strong, LIME, PACE-flip and Random.

**RQ2 (seeds 370–379):** the repair-at-θ protocol (`run_repair_theta`).
- The paper's 80/20 stratified split; the seed's row order; min-max scaling fitted on the
  training rows.
- **B0-0.3:** CFMNN at θ = 0.3.
- **SR-0.3:** B0-0.3, then self-repair: 300 errors, 10 rounds, the smallest cut, gated on the
  training rows. No label beyond them.
- **B0-val:** CFMNN at the θ chosen by 3-fold validation on the training rows, from the paper's
  grid. It is tuned in the run, because no earlier run covers these datasets.
- Every model's test predictions are compared with B0-0.3's: changed, fixed and broken.

## Primary tests

Each is a paired, two-sided Wilcoxon signed-rank test over the 10 seeds. Holm correction runs
across the 4 datasets, separately for each primary, at α = 0.05.

1. **RQ1:** deletion AUC (interventional, margin, same rows), PACE-SHAP minus KernelSHAP.
2. **RQ2 (a):** macro-F1, SR-0.3 minus B0-0.3.
3. **RQ2 (b):** broken correct test predictions (NFR), SR-0.3 minus B0-val. Both are counted
   against B0-0.3.

A dataset where the two arms are equal on every seed is reported as "no change" (no test is run).

## Secondary

- **RQ1:**
  - insertion AUC on the same rows;
  - deletion and insertion AUC against KernelSHAP at its usual setting (v3);
  - time per explanation on the same rows.
- **RQ2:**
  - macro-F1, SR-0.3 minus B0-val;
  - training errors at θ = 0.3;
  - the θ chosen by validation;
  - boxes per model;
  - the seeds on which SR-0.3 changed any test prediction.

## How the result will be read (written now, before the run)

**RQ1**
- **Check PASS, primary "no change" or no significant difference on all 4 (expected):**
  - The paper says that on 4 datasets chosen by a rule, both explainers are exact and agree, so
    these datasets add to "exact in practice", not to "more faithful".
  - The datasets are **not** counted in the like-for-like tally of better, no difference, worse.
  - The time result says how much faster the exact computation is.
- **Check PASS, primary significant on a dataset:** the cause can only be the order of near-tied
  values. It is reported with that cause. It is not claimed as a difference in faithfulness.
- **v3 (secondary):** reported in the same form as the paper's second explanation run, as better,
  no significant difference, or worse on each of the 4 datasets.

**RQ2 (a)**
- **Worse on none:** the paper's claim that self-repair never made the model significantly worse
  extends to 4 held-out datasets. The datasets where it was better are named.
- **Worse on any:** RQ2 reports it, and the claim narrows to the datasets where it held.
- **A dataset with almost no training errors at θ = 0.3** (under 0.5%): repair has nothing to fix
  there. It is reported as outside RQ2's scope, as the review foresaw ("RQ2 narrows to datasets
  where the model makes training errors").

**RQ2 (b)**
- A repair that changed no test prediction on any seed cannot break one. Such a dataset is listed
  apart, and the tally is given without it (tough review: "NFR 10/13 includes 5 no-ops").
- **Repair breaks fewer than re-tuning** (on the datasets where it changed something): the claim
  extends.
- **More:** the claim narrows, and the paper says where.

**Both research questions:** the paper reports every result above, whichever way it falls. A
dataset is never dropped after its results are seen.

## What ran before this was written

Laptop smoke slice on seed 194 (not a study seed, never analysed), 10 test points for RQ1,
written outside the repository.

**First attempt.** RQ1 failed to start: its task was a lambda, which the job pool cannot send to
a worker. That was fixed by passing the number of points in the task. RQ2 was not affected.

**RQ2, one seed per dataset (macro-F1 and NFR):**

| Dataset | θ by validation | B0-0.3 | SR-0.3 (NFR) | B0-val (NFR) |
|---|---|---|---|---|
| wilt | 0.05 | 56.9 | 65.0 (3.82%) | 69.2 (2.58%) |
| mfeat-morphological | 0.5 | 70.8 | 72.2 (0.50%) | 69.5 (10.00%) |
| phoneme | 0.05 | 73.3 | 74.2 (10.55%) | 83.9 (6.66%) |
| jungle_chess | 0.15 | 65.6 | 66.2 (4.43%) | 73.0 (9.50%) |

The RQ2 slice took 43 minutes in all with 4 jobs. Jungle Chess was the slowest, and another run
shared the laptop.

**RQ1, rerun after the fix (1 min 48 s with 4 jobs):**
- **Check: PASS.** The largest |PACE-SHAP − KernelSHAP| on the same rows was between 5.6e-17 and
  4.2e-16 on every dataset.
- The deletion AUCs of the two were equal, as expected.
- Time per explanation on the same rows:

  | Dataset | PACE-SHAP | KernelSHAP | Ratio |
  |---|---|---|---|
  | wilt | 3.5 ms | 25 ms | 7.2× |
  | phoneme | 30 ms | 162 ms | 5.4× |
  | mfeat-morphological | 8.5 ms | 109 ms | 12.8× |
  | jungle_chess | 365 ms | 2,075 ms | 5.7× |

**Training errors at θ = 0.3** were 12.6–29.4% on this seed, so RQ2 has errors to repair on all
four datasets.

The design did not change after the slice.

## Time

- **Laptop, one seed:** RQ2 43 minutes, set by Jungle Chess (θ tuning and repair on 35,855
  training rows). RQ1 under 2 minutes at 10 points; it is about 5 times that at 50.
- **Home PC with 6 jobs, 10 seeds per part:** about 2–4 hours. Jungle Chess's 20 tasks set the
  pace, and they start first.
- **Stop and ask if it passes 8 hours.**

## How to run (home PC, batch 3)

```
.\.venv\Scripts\python -B scripts/run_heldout.py --jobs 6 *>&1 | Tee-Object results/heldout/run_log.txt
.\.venv\Scripts\python -B scripts/run_heldout.py --jobs 6 --resume *>&1 | Tee-Object -Append results/heldout/run_log.txt   # after a stop
```

The script exits 1 on **Check: FAIL**; stop and ask. Commit `results/heldout/` unedited
(`rq1_runs.csv`, `rq2_runs.csv`, `summary.md`, `run_log.txt`) as "Run P10: results of the
pre-registered run on the home PC".
