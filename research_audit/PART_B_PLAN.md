# Part B: Runs L, M and N, built and run on the home PC

## Context

Runs F–K are done and committed on `peer-review-fixes` (origin `1c75a68`). Three gaps remain
that a reviewer is likely to raise, and the user chose to close them with three more runs:

- **L:** CoverType gave CFMNN 51.5% accuracy at θ = 0.3. Does a smaller θ, or self-repair, help?
- **M:** "PACE carries over" was tested only on CFMNN's boxes after contraction, not on
  Simpson's classifier trained from scratch.
- **N:** the sampled-baseline comparison (Run J) stops at 36 features; CoverType has 54.

This plan is for a Claude Code session on the home PC (i5-14400F, `$HOME\Documents\pace-cfmnn`),
which builds, tests, pre-registers and runs all three. The write-up of F–K and of these runs
is a separate job on the laptop and is not part of this plan. Nothing here can guarantee
acceptance; each run is reported whichever way it goes.

## Rules (as for Runs I–K)

- Branch `peer-review-fixes` only; commit as
  `git -c user.name="Srujan K" -c user.email="176512181+Srujankasturi@users.noreply.github.com" commit`,
  no `Co-Authored-By` or "Generated with" lines; push after each commit.
- Never edit a pushed `PREREGISTRATION.md`, a committed `runs.csv`, `summary.md` or log, or a
  script a pushed plan describes. New behaviour goes in **new files**.
- Each run: code, tests and `PREREGISTRATION.md` (in the format of
  `results/generality/PREREGISTRATION.md`) committed and pushed before any study seed.
- Smoke slices use seeds 192–199 only, write outside the repository, and are listed in the
  plan's "What ran before this was written" with what was looked at.
- Study seeds: L 210–214, M 220–229, N 230–234 (none used before).
- Stop and ask if: a test fails; a run projects over 6 hours or memory comes within 2 GB of
  the PC's; anything needs installing, downloading, deleting or renaming; a pushed plan is
  wrong. No subagents. Do not edit the paper, `results/FINDINGS.md` or `research_audit/CLAIMS.md`.
- Timed runs (L's explanation times, N's timing pass) run alone.

## Step 0: sync and check

`git pull origin peer-review-fixes`; `download_data.py --check` (CoverType is already there);
full suite `python -B -m unittest discover -s tests -t .` (348 tests, all pass, none skipped).

## Run L: CoverType with a smaller θ, then self-repair (descriptive)

- **Files (new):** `scripts/run_covertype_theta.py`, `tests/test_covertype_theta.py`,
  `results/covertype_theta/`.
- **Reuse:** `prepare`, `peak_memory_mb`, `timed` from `scripts/run_covertype.py`; `tune`,
  `cv_folds`, `GRIDS["CFMNN-val"]` (θ from 0.5 down to 0.05) from
  `scripts/run_readable_baselines.py`; the four-model layout of `scripts/run_repair_theta.py`
  (B0-0.3, SR-0.3, B0-val, SR-val); `run_self_repair(..., chooser="volume", budget=300,
  n_rounds=10, gate_labels_are_new=False, fast_gate=True)` from `pace_cfmnn/active.py`;
  `training_error_ties`, `centroid_decided` from `pace_cfmnn/ties.py`.
- **Recorded per seed and model:** θ, boxes, training error, tie share, centroid-decided
  share, test accuracy and macro-F1, edits accepted, seconds, peak memory. 10-NN and Random
  Forest are quoted from Run K, not rerun.
- **Check built in:** B0-0.3 on a Run K seed (smoke only: seed 190's printed 55.97%) must
  match, showing the setup is Run K's.
- **Settle in the smoke slice (seed 192, 10% then full):** small θ may give very many boxes,
  and repair sees hundreds of thousands of training errors. If tuning on the full training
  rows projects over 2 hours, fix a stratified tuning subsample in the plan before any study
  seed. If repair on the full data is too slow with the cached gate, the plan says so and
  drops to a stated gate subsample or reports repair as infeasible; it does not change the
  budget or the gate's limits.
- **Design:** descriptive, no test (one dataset). Stated in the plan.

## Run N: CoverType in the sampled-baseline comparison (descriptive)

- **Files (new):** `scripts/run_sampled_baselines_covertype.py`,
  `tests/test_sampled_baselines_covertype.py`, `results/sampled_baselines_covertype/`.
- **Reuse, by import, unchanged:** `explainers`, `budget`, `top3`, `jaccard`, the row layout
  of `run_one` and `time_one` from `scripts/run_sampled_baselines.py`; `permutation_shap`,
  `paired_kernel_shap` from `pace_cfmnn/sampled_shap.py`; `rf.setup("covertype", …)` and
  `backgrounds_v3` (check that `rf.setup` accepts the name; if not, use
  `run_covertype.prepare` and say so).
- **Setup:** 54 features, 200 explainer rows, 50 points per seed, the same budget rule:
  200 × 32 × 55 = 352,000 model rows per explanation. No exact reference at 54 features.
- **Measures:** deletion and insertion margin AUC, top-3 stability, rows used, and a timing
  pass alone on 10 points. Means over the 5 seeds; no test.

## Run M: Simpson's classifier trained from scratch

- **Before the plan is written:** read hyperbox-brain's membership function and record
  whether GFMM's membership is a minimum over features. If it is, the plan and later the
  paper state that PACE covers mean-type memberships (CFMNN, Simpson) and not GFMM as it is.
- **Files (new):** `pace_cfmnn/simpson_fmm.py`, `scripts/run_simpson_fmm.py`,
  `tests/test_simpson_fmm.py`, `results/simpson_fmm/`.
- **The model:** a subclass or sibling of `CFMNN` (`pace_cfmnn/cfmnn_core.py`) whose `fit`
  runs, after every expansion, Simpson's overlap test and contraction against every box of
  another class (`overlap_test`, `contract_pair` from `pace_cfmnn/contraction.py`), with θ
  and γ as CFMNN's. It exposes `V_`, `W_`, `box_class_`, `gamma`, `classes_`,
  `predict_details`, so `PaceShap` and `xai_baselines.exact_shap` run on it unchanged. The
  plan states the tie rule (Simpson's model has no centroid) and that the size limit is
  CFMNN's sum-of-sides form.
- **Tests:** on a toy problem the trainer matches a literal, slow transcription of the
  published algorithm (as `tests/naive_reference.py` does for CFMNN); no two boxes of
  different classes overlap by Simpson's test after `fit`; PACE-SHAP equals brute-force
  Shapley values (`tests.test_explain.exact_shapley`).
- **The run (as Run I's Part A, on the new model):**
  - exact check on S1, S2, S3, Shuttle, MAGIC, Page Blocks: 5 points per seed, largest
    difference from `exact_shap`, limit 1e-9 (PASS or FAIL);
  - on the 13 accuracy datasets: boxes, training error, tie share, centroid-decided share,
    test accuracy and macro-F1, beside CFMNN (θ = 0.3, γ = 4). Reuse `prepare` and `describe`
    from `scripts/run_generality.py`.
- **Design:** the exact check is a check; the rest is descriptive. No primary.
- **Time:** unknown until the smoke slice; Run I's exact part took most of its 21 minutes,
  and per-row contraction makes training slower than CFMNN's. Skin (196,000 training rows) is
  the one to time first.

## Order and commits

L, then N (both small and both on CoverType), then M. For each: "Run X: pre-registration,
code and tests before the run" → push → run → "Run X: results of the … run on the home PC" →
push. Finish with a hand-over section in `research_audit/HOME_PC_RUNBOOK.md` (commands,
commits, times, deviations).

## Verification

- Full suite passes on the final commit, with each run's new tests included.
- Row counts match each plan; no task printed FAILED; Run M's summary prints its check.
- `git diff 1c75a68 --stat` shows only new files, plus the runbook; no earlier
  `PREREGISTRATION.md`, `runs.csv`, `summary.md`, log or registered script changed.
- `git status` clean; no commit has a `Co-Authored-By` line.
- Final report to the user: a table of run, commits, what was measured and the few numbers
  that matter, then deviations and anything that stopped early, in plain words.
