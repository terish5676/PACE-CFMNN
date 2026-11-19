# Run P1, time to a given accuracy and one clean re-time: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review S2 / D2).
The code is `scripts/run_time_to_accuracy.py` (tests in `tests/test_time_to_accuracy.py`).
- It uses `xai_baselines.kernel_shap_game` (paired coalitions; exact with every coalition) and
  `PaceShap`, unchanged.
- It uses the second explanation run's set-up (`run_faithfulness.setup`, `backgrounds_v3`,
  `SHAP_V3_R`, `SHAP_V3_P`, `COND_NSAMPLES`).

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 11.

## What ran before this was written

- **The like-for-like comparison** (`results/conditional_settings/`, and the paper): PACE-SHAP
  against KernelSHAP with 200 coalitions on the same 200 rows. Better on 3, level on 10, worse on
  none, and KernelSHAP took 3.5–10× as long. Those times come from paired games (two games per
  call) and from more than one run; the review (S2) asked for one clean timing.
- **Smoke slices on the laptop, seed 194** (not a study seed, never analysed), 1 point, with other
  jobs running, so the times are inflated:
  - part (a), S1: PACE-SHAP exact 5.4 s; KernelSHAP 12.8 s at 100 coalitions, 26.5 s at 200,
    54.2 s at 400, 90.2 s at 700 and 199 s with all 1,022;
  - part (b), Wine: PACE-SHAP 37 ms, KernelSHAP 377 ms.

  The errors at each budget were not looked at. The design did not change after the slices.
- **Tests while building:** `tests/test_time_to_accuracy.py` (6 tests) passes on the laptop.

## The question

At 200 coalitions KernelSHAP is an estimate and PACE-SHAP (d ≤ 12) is exact, so comparing their
times at that budget is not like for like. **How long does KernelSHAP need to come within 1% of
the exact values, and how does that time compare with PACE-SHAP's?** And, for the speed sentence
in the paper: **in one clean run, how much longer does KernelSHAP at 200 coalitions take than
PACE-SHAP in its default mode?**

## Part (a): time to 1% (primary)

- **Datasets:** S1, S2, S3, Shuttle, MAGIC and Page Blocks (at most 12 features, exact mode).
- **Seeds:** 300–302 (new). **Points:** 10 test points per seed, each explained in its own call.
- **Arms:** PACE-SHAP exact (the reference values); KernelSHAP at 100, 200, 400 and 700
  coalitions (those below 2^d − 2) and with every coalition ("all", which is exact).
- **Error:** relative L1, ||φ_K − φ||₁ / ||φ||₁, against PACE-SHAP's exact values, per point.
- **b\*:** the smallest budget whose median error over the dataset's 30 points is at most 1%
  ("all" if no sampled budget reaches it).
- **Primary number, per dataset:** KernelSHAP's mean time at b\* divided by PACE-SHAP's mean time,
  with a 95% bootstrap interval over the 30 points (2,000 resamples, seed 0).
- PACE-SHAP's set-up (built once per task) is shared equally among the task's points.

## Part (b): one clean re-time (replaces "3.5–10×")

- **Datasets:** the 13 like-for-like datasets (S1, S2, S3, S1c, S2c and the 8 real explanation
  datasets). **Seed:** 303 (new). **Points:** 10 per dataset, each in its own call.
- **Arms:** KernelSHAP with 200 coalitions; PACE-SHAP in its default mode (exact for d ≤ 12,
  else 32 permutations); both on the same 200 rows, one game each, each timed alone.
- **Number:** KernelSHAP ÷ PACE-SHAP per dataset (mean time over points, 95% bootstrap interval),
  and the range over datasets.

Order: one task at a time, with an untimed warm-up of both explainers on one point first.

## How the result will be read (written now, before the run)

- **PACE-SHAP faster at b\* on every dataset (ratio > 1 with the interval above 1):** the paper
  says that to match PACE-SHAP's accuracy KernelSHAP needs b\* coalitions and r× the time, with
  the per-dataset numbers.
- **KernelSHAP faster at b\* on some dataset (interval below 1):** the paper says so for those
  datasets (the review's fallback: "KernelSHAP reaches 1% faster on …") and restricts the speed
  claim to where it holds.
- **Interval contains 1:** "about the same time" for that dataset.
- **Part (b):** the paper's speed sentence uses this run's range, whatever it is, in place of
  "3.5–10×" ("better on 3, level on 10, worse on none, and KernelSHAP took a–b× as long").
- Nothing here changes the faithfulness results.

## Time

From Run P's clean home-PC times (ExactExplainer about 28 s per point on S1–S3, 6 s on MAGIC),
part (a) is about 70 s per point on S1–S3 and much less elsewhere: about 2 hours. Part (b) is
about 15–20 minutes. **About 2–2.5 hours on the home PC, alone.** Stop and ask if the pilot
projects more than 4 hours.

## Tests

`tests/test_time_to_accuracy.py`:
- relative L1 error; b\* is the smallest budget at or under 1%, and "all" when none is;
- the bootstrap ratio and its interval;
- the summary's columns and the part (b) range line;
- end to end on S1 (smoke seed 194): KernelSHAP with every coalition matches PACE-SHAP within
  1e-9, and a part (b) task on Wine.

## How to run (home PC, alone, after P7 and P3)

```
.\.venv\Scripts\python -B scripts/run_time_to_accuracy.py --pilot --out $env:TEMP\tta_pilot
.\.venv\Scripts\python -B scripts/run_time_to_accuracy.py *>&1 | Tee-Object results/time_to_accuracy/run_log.txt
.\.venv\Scripts\python -B scripts/run_time_to_accuracy.py --resume *>&1 | Tee-Object -Append results/time_to_accuracy/run_log.txt   # after a stop
```

The pilot runs 1 point on the first seed of each part (all 6 + 13 datasets). Project part (a) as
30 × its S1–S3 times plus the rest, part (b) as 10 × its times.

Commit `results/time_to_accuracy/` unedited (`runs.csv`, `summary.md`, `run_log.txt`; not the
pilot) as "Run P1: results of the pre-registered run on the home PC".
