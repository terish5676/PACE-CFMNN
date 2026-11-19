# Run P8, a faster exact mode and a fair brute-force baseline: pre-registration (descriptive, with a decision rule)

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review B9).
The code is `scripts/run_recurrence.py` (tests in `tests/test_recurrence_run.py`).
- It uses `PaceShap(exact_algo="recurrence")`, `explain.subset_sums` and `explain.brute_force_shap`,
  added in commit 7843fee with tests in `tests/test_tough_explain.py`.
- It uses the second explanation run's set-up (`run_faithfulness.setup`, `backgrounds_v3`,
  `SHAP_V3_R`), as Run P (`results/speed_source/`) did.

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 11.

## What ran before this was written

- **Run P** (`results/speed_source/`, seeds 240–242): PACE-SHAP's exact mode is 14–23× faster than
  shap's ExactExplainer on the 6 datasets with at most 12 features; the per-feature structure gives
  7.3–10.7× and pruning 1.8–2.9×.
- **Smoke slices on the laptop, seed 194** (not a study seed, never analysed), 1 point each, with
  the full unit suite running at the same time, so the times are inflated:
  - Wine (13 features, 32 boxes): matmul 0.5 s, recurrence 0.2 s, brute force 17.6 s,
    ExactExplainer 10.9 s.
  - Letter (16 features, 545 boxes): matmul 43.6 s, recurrence 30.5 s. Brute force had not
    finished one point after more than an hour and was stopped.
  - Shuttle (the end-to-end test): matmul 20 ms, recurrence 24 ms, brute force 921 ms.

  So the likely direction is known: the recurrence is somewhat faster than matmul at 13–16
  features, and **Letter is far above the 5-second rule below**. The design changed after these
  slices in one way: brute force and ExactExplainer explain one Letter point in total (the first
  point of seed 350), not one per seed.
- **Tests while building:** `tests/test_recurrence_run.py` (7 tests) passes on the laptop.

## The question

The exact mode stops at 12 features. PACE-SHAP builds every coalition's memberships with one
matrix product (K · d · 2^d multiply-adds for K kept (row, box) pairs). Building the subset sums one
feature at a time costs K · 2^d additions. **Does the recurrence give the same values faster, and
is it fast enough to raise the exact limit to 16 features?** The review also asked for a fair
brute-force baseline: shap's ExactExplainer is general code, so part of PACE-SHAP's speed over it
might come from vectorised numpy rather than from the structure.

## Arms (cost per explained point; R rows, H boxes, d features, K ≤ R · H kept pairs)

- **PACE-SHAP, matmul** (the current exact mode): O(K · d · 2^d).
- **PACE-SHAP, recurrence**: O(K · 2^d).
- **Brute force, vectorised** (`brute_force_shap`): the model on every mixed input, in large
  numpy blocks, with no Proposition 2 and no pruning: O(2^d · R · H · d).
- **shap ExactExplainer** (Wine and Letter only): the reference for the check above 12 features.

## Setup

- **Datasets:** S1, S2, S3, Shuttle, MAGIC and Page Blocks (at most 12 features), Wine (13) and
  Letter (16). The script refuses more than 16 features.
- **Seeds:** 350–352 (new). **Points:** 10 test points per seed.
- **Model and rows:** each seed's model from the second explanation run's set-up, 200 explainer rows.
- **Exact limit:** `exact_max_d = max(12, d)`, so Wine and Letter run in exact mode.
- **Letter:** brute force and ExactExplainer explain one point in total (the first point of seed
  350), because each takes about an hour there.
- **Order:** one task at a time; an untimed warm-up of PACE-SHAP on one point; then matmul,
  recurrence, brute force, ExactExplainer. Each time includes the explainer's set-up.

## Measures and check

Per dataset, seed and arm: milliseconds per explanation, the largest absolute difference from
matmul (and from ExactExplainer on Wine and Letter), the largest absolute value, d and boxes.

**Check (the script exits 1 if it fails; stop and ask):** the recurrence and brute force agree with
matmul within 1e-12 on every point; on Wine and Letter both PACE-SHAP modes agree with
ExactExplainer within 1e-9 (shap's `np.isclose` shortcut, as in Runs I and M).

Summary per dataset (means over seeds): matmul ÷ recurrence and brute force ÷ recurrence.

**No primary test.** The run is descriptive, with one decision rule.

## How the result will be read (written now, before the run)

- **Decision rule:** if the recurrence takes at most 5 s per explanation on Letter, the default
  exact limit is raised to 16 features and the paper says so. If not (as the smoke slice
  suggests), the limit stays at 12, and the paper gives Letter's time as the reason.
- **Recurrence vs matmul:** the paper's exact-mode paragraph reports the ratio per dataset,
  whatever it is. If the recurrence is faster everywhere, it becomes the default exact algorithm
  (a code change, with the full suite rerun); if it is slower at 9–10 features, matmul stays the
  default there and the paper says so.
- **Brute force:** brute force ÷ recurrence is reported as the speed-up over a fair vectorised
  baseline, next to Run P's speed-up over ExactExplainer. If brute force is within 2× of PACE-SHAP
  on some dataset, the paper says the structure gives little there.

## Time

The laptop slices, under load, suggest: Letter about 12 minutes for the two PACE-SHAP modes per
seed, plus about an hour each for one brute-force point and one ExactExplainer point; Wine about
15 minutes; S1–S3 about 5–10 minutes each for brute force over 30 points; the rest a few
minutes. **About 3–4 hours on the home PC.** Stop and ask if it is past 6 hours.

## Tests

`tests/test_recurrence_run.py`:
- on a toy model the four ways agree within the tolerances, also at 13 features with the limit
  raised;
- the check passes within the limits and fails outside them (recurrence, brute force,
  ExactExplainer);
- the summary's ratios, and the decision line (yes at 3 s, no at 9 s);
- one end-to-end task on Shuttle, smoke seed 194.

## How to run (home PC, alone, last in batch 1)

```
.\.venv\Scripts\python -B scripts/run_recurrence.py --datasets shuttle --seeds 194 --n 1 --out $env:TEMP\recurrence_smoke
.\.venv\Scripts\python -B scripts/run_recurrence.py *>&1 | Tee-Object results/recurrence/run_log.txt
.\.venv\Scripts\python -B scripts/run_recurrence.py --resume *>&1 | Tee-Object -Append results/recurrence/run_log.txt   # after a stop
```

Commit `results/recurrence/` unedited (`runs.csv`, `summary.md`, `run_log.txt`) as "Run P8: results
of the pre-registered run on the home PC".
