# Run N, Covertype in the sampled-baseline comparison: pre-registration

Written on 2026-11-15 on the home PC (i5-14400F) and pushed, with the code it describes, before
any study seed ran. Plan: `research_audit/PART_B_PLAN.md`, "Run N".

- **The code (new):** `scripts/run_sampled_baselines_covertype.py` (tests in
  `tests/test_sampled_baselines_covertype.py`).
- **Used as it is, by import:** `scripts/run_sampled_baselines.py` (Run J): its task
  `run_one`, its timing pass `time_one`, its methods (`explainers`), its budget rule
  (`budget`), `top3`, `jaccard` and its row layout; `pace_cfmnn/sampled_shap.py`
  (`permutation_shap`, `paired_kernel_shap`); `scripts/run_faithfulness.py` (`setup`,
  `backgrounds_v3`). The new script only passes Covertype and its own seeds to Run J's
  functions and writes its own summary. No existing file was changed for this run.

## Why

Run J (`results/sampled_baselines/`, seeds 160–169) compared PACE-SHAP's sampled mode with
PermutationSHAP and paired-sampling KernelSHAP at an equal budget on the explanation datasets
with 13 to 36 features. Covertype has 54, the most of any dataset in the study. This run adds
it, so the comparison covers the widest dataset too.

## What ran before this was written

- **Unit tests** on made-up Covertype-shaped data (`tests/test_covertype.fake_covertype`,
  900 rows, no download), while building; then the full suite: **366 tests, all pass**
  (`python -B -m unittest discover -s tests`, 512 s).
- **`run_faithfulness.setup` accepts the name `"covertype"`:** it only calls `data.load`,
  `data.paper_split`, `data.fit_scaler` and `CFMNN().fit`, which take any dataset name. So
  the plan's fallback (`run_covertype.prepare`) is not needed and not used.
- **One smoke slice on seed 192** (not a study seed, never analysed), alone, written outside
  the repository (`$HOME\scratch\run_n`):

  ```
  python -B scripts/run_sampled_baselines_covertype.py --seeds 192 --n 10 --out $HOME\scratch\run_n --fresh
  ```

  What was looked at: the printed line per method, the summary and `runs.csv` (4 rows), the
  elapsed time and the memory. On that seed, with 10 points: CFMNN had 114 boxes; deletion
  margin AUC PACE-SHAP −0.290, PermutationSHAP −0.288, KernelSHAP-paired −0.291, Random
  −0.168; milliseconds per explanation during the run 440, 1,781 and 5,698; rows per
  explanation 352,000, 107,164 and 351,801 (every baseline within the budget). The whole
  slice took **173 s**; the Python processes peaked at **637 MB**, with at least 10.1 GB of
  the PC's memory free. One seed on 10 points is not evidence; only seeds 230–234 are.
- **What changed because of the smoke slice:** nothing.

## Setup

- **Dataset:** Covertype (UCI 31), 581,012 rows, **54 features**, 7 classes; already on the
  PC (Run K).
- **Seeds:** 230–234 (5). No earlier run or smoke slice used them.
- **Model, split, points, rows:** those of Run J and the second explanation run
  (`run_faithfulness.setup`, `backgrounds_v3`):
  - the paper's 80/20 split (464,809 training rows), training rows in the seed's order,
    min-max scaling fitted on them, `CFMNN()` (θ = 0.3, γ = 4) on **all** the training rows
    (not Run L's subsample);
  - **50 explained test points** per seed;
  - two disjoint sets of 200 training rows: the evaluation rows (used by the deletion and
    insertion tests) and the explainer rows (the only rows an explainer sees).
- **The game, methods and budget:** Run J's, unchanged: PACE-SHAP
  (`PaceShap(model, rows, n_permutations=32)`), PermutationSHAP (shap 0.54), KernelSHAP with
  paired sampling (Covert and Lee, AISTATS 2021; `sampled_shap.paired_kernel_shap`) and a
  Random ranking. The budget per explanation is N = R × 32 × (d + 1) =
  **200 × 32 × 55 = 352,000 model rows**; PermutationSHAP gets `max_evals` = 1,760 and
  KernelSHAP-paired 1,758 sampled coalitions plus the empty and full ones. Run J's script
  stops if a method goes over N.
- **No exact reference:** 2^54 coalitions per row. So no error against exact values; the
  `n_exact` column is 0 and the error columns are empty.

## Measures

Per seed and method (`runs.csv`), as in Run J: deletion margin AUC (lower is more faithful),
insertion margin AUC (higher is more faithful), top-3 stability under a new seed (seed + 1000),
model rows per explanation, milliseconds per explanation during the run, boxes, d, R.

A **timing pass run alone** after the run (`--timing`): one pass per method on the first 10
explained points of seeds 230–232 (`timing.csv`). Its times are the ones to quote.

## Design: descriptive

One dataset with 5 seeds: no test is made, and no comparison is primary. The summary gives
means ± standard deviations over the 5 seeds, the per-seed differences (PACE-SHAP minus each
baseline, deletion margin AUC), whether every baseline kept to the budget, and the times
measured alone. No direction is predicted. Results are reported whichever way they go,
including "PACE-SHAP is no more faithful than the baselines" or "a baseline is more
faithful".

## What the result changes

- PACE-SHAP's settings (32 orders per row, 200 rows) stay as they are, whatever this run says.
- The paper may add Covertype to Run J's table as a descriptive line, labelled as 5 seeds
  without a test.
- Any change to this plan after the push is a dated amendment (before the run) or a reported
  deviation (after it). Any analysis not listed here is labelled post hoc.

## Time

From the smoke slice (173 s for 10 points, alone), one seed with 50 points takes about 13 to
15 minutes on one core. With 5 jobs (one per seed): about **15–25 minutes**; memory about
5 × 0.64 ≈ 3.2 GB. The timing pass, alone, about 5 minutes. Stop and ask if the run projects
over 6 hours or free memory comes within 2 GB of running out.

## Tests

`tests/test_sampled_baselines_covertype.py` checks that:

- the seeds, points, timing seeds and methods are as above, and Run J's budget rule gives
  352,000 rows, 1,760 `max_evals` and 1,758 coalitions at d = 54, R = 200;
- the task and the timing pass are Run J's (`run_one`, `time_one`), called with Covertype and
  50 or 10 points;
- one task on made-up 54-feature data writes one row per method, every baseline within the
  budget, no exact values, finite AUCs; the timing pass writes one row per timed method;
- the summary says it is descriptive, states the budget, lists every method, checks the
  budget, and adds the times measured alone when `timing.csv` exists.

## How to run

```
python -B scripts/run_sampled_baselines_covertype.py --jobs 5
python -B scripts/run_sampled_baselines_covertype.py --jobs 5 --resume     # continue after a stop
python -B scripts/run_sampled_baselines_covertype.py --timing              # nothing else running
python -B scripts/run_sampled_baselines_covertype.py --summary-only
```

Outputs in `results/sampled_baselines_covertype/`: `runs.csv` (5 seeds × 4 methods =
**20 rows**), `timing.csv` (3 seeds × 3 methods = **9 rows**), `summary.md` (rewritten by
`--timing` with the times measured alone) and the logs `run_log.txt` and `timing_log.txt`.
