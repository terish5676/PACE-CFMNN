# Home PC runbook (i5-14400F)

A Claude Code session on the user's home PC follows this file. It builds and runs E, A, B, C and D, as designed in `research_audit/LIMITATIONS_PLAN.md`. That session has no other context, so read both files fully before starting.

## Rules (keep all of them)

**Git:**
- Work on the `final-review` branch only. Never touch `main`, other branches or pull requests.
- Commit with this identity:
  ```
  git -c user.name="Srujan K" -c user.email="176512181+Srujankasturi@users.noreply.github.com" commit
  ```
- No `Co-Authored-By` or "Generated with" lines in commit messages.
- Never use a bare `git stash`.

**Files:**
- Never edit a pushed `PREREGISTRATION.md`, a committed `runs.csv` or `summary.md`, or the code a pushed pre-registration describes.
- If something is wrong after a pre-registration is pushed, stop and tell the user.
- Never change a method, default, hyperparameter or dataset to improve a score.

**Running:**
- Never report a test or run as passed unless you ran it and saw the output.
- Never fabricate results, tests or references.

**Stop and ask the user if:**
- a test fails;
- a check prints FAIL;
- a smoke slice projects more than twice the time in the plan;
- anything needs deleting or renaming.

**Things the user does, not you:**
- Installs: ask them first.
- GitHub sign-in: never type a password or token.
- Sleep: they set "Sleep: Never" in Settings > System > Power. Don't change sleep or power settings.

**Style:** explain things to the user plainly; they are a research student.

## 1. Setup (about 20 minutes)

The user has copied `pace-cfmnn-data.zip` (2.3 MB, the UCI datasets) to the PC, for example to `Downloads`.

```powershell
winget install -e --id Python.Python.3.14     # if missing; ask the user first
winget install -e --id Git.Git                # if missing; ask the user first

cd $HOME\Documents
git clone https://github.com/Srujankasturi/pace-cfmnn.git
cd pace-cfmnn
git checkout final-review

# The data: the zip holds data\<dataset>\... paths
Expand-Archive "$HOME\Downloads\pace-cfmnn-data.zip" -DestinationPath . -Force

py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -B -m unittest discover -s tests -t .
```

All 198 tests must pass before anything else.

Record the processor and Python version for the paper:

```powershell
.\.venv\Scripts\python -c "import platform; print(platform.processor(), platform.python_version())"
```

## 2. For each run, in order E, A, B, C, D

1. **Build:** write the code and tests the plan gives for that run. Follow the plan's "Rules for every run".
2. **Test:** run the full suite. Everything must pass.
3. **Smoke slice:**
   - Seed 120 on 1–2 small datasets, written to a scratch folder outside the repo, for example `$HOME\scratch\<run>`.
   - E has no seed 120: its smoke slice is Thyroid and Page Blocks, seeds 50–51.
   - From the time per task, project the full run. If the projection is over twice the plan's estimate, stop and ask.
4. **Pre-register:** write `results/<folder>/PREREGISTRATION.md` in the format of `results/repair_theta/PREREGISTRATION.md`. It covers:
   - the question;
   - the design;
   - the primaries and the pass rule;
   - that defaults stay whatever the results;
   - everything that ran before it was written (the smoke slice and timing, and for E the reverted laptop prototype).
5. **Commit and push before running:** the code, tests and pre-registration in one commit, for example "<Run>: pre-registration, code and tests before the run". Then:
   ```powershell
   git push origin final-review
   ```
6. **Run:**
   ```powershell
   .\.venv\Scripts\python -B scripts/<script>.py --jobs 12 *>&1 | Tee-Object results/<folder>/run_log.txt
   ```
   - Use the Bash/PowerShell tool's background mode.
   - After a stop, rerun with `--resume` and append to the log.
7. **Check:**
   - `runs.csv` has the expected number of rows;
   - no task failed;
   - `summary.md` was written.
8. **Commit the outputs unedited and push:** `runs.csv`, `summary.md`, `run_log.txt` and any other files the script wrote. Use a message like "<Run>: results of the pre-registered run on the home PC".
9. **Tell the user** the verdicts and times in plain words, then go on to the next run.

**Special rule for E:** A uses `fast_gate=True` only if E's summary says **Verdict: PASS**. If E fails, A uses the standard gate, and A's pre-registration says so.

## 3. When all five are done

Report to the user with:
- each run's verdicts;
- the total time;
- the commit hashes.

The write-up (paper, FINDINGS, CLAIMS) is done later in another session. Don't edit the paper.

## 4. Runs T and H (branch `peer-review-fixes`)

These were built, tested and pushed from the laptop. Run H's pre-registration is pushed
(`results/repair_order/PREREGISTRATION.md`); Run T is a descriptive check
(`results/repair_timing/README.md`). **Build nothing and change no code.** The rules above
still hold, except that this section works on `peer-review-fixes`, not `final-review`.

### Setup

```powershell
cd $HOME\Documents\pace-cfmnn
git fetch origin
git checkout peer-review-fixes
git pull origin peer-review-fixes
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -B -m unittest discover -s tests -t .
```

All 307 tests must pass (302 ran on the laptop in about 8 minutes; Run H's amendment 1 added 5,
written on the laptop but not run there). Then close other programs:
nothing else may run during Run T.

### Run T first, alone (about 15–30 minutes, at most about 40)

```powershell
.\.venv\Scripts\python -B scripts/time_repair_arms.py *>&1 | Tee-Object results/repair_timing/run_log.txt
```

- It runs one task at a time. Don't pass `--jobs`, and don't run anything else until it ends.
- **Check:** `results/repair_timing/summary.md` says **Verdict: PASS**, and `runs.csv` has 30
  rows. If it says FAIL, stop and tell the user.
- Commit `runs.csv`, `summary.md` and `run_log.txt` unedited, with the message "Run T: timing
  check on the home PC", and push to `peer-review-fixes`.

### Then Run H (about 1 hour with 12 jobs; 45 minutes to 1.5 hours)

Run T is done (`8789b2e`, PASS); if you are starting again, skip it. Run H now includes
**amendment 1** (end of its `PREREGISTRATION.md`): one more arm, random-300-real, and three
more comparisons.

1. **Smoke slice** (seed 140, written outside the repository):
   ```powershell
   .\.venv\Scripts\python -B scripts/run_repair_order.py --datasets landsat --seeds 140 --out $HOME\scratch\run_h --fresh
   ```
   It should take about 1 minute (57 s on the laptop). If it takes more than 2 minutes, stop
   and ask the user before the full run.
2. **Run:**
   ```powershell
   .\.venv\Scripts\python -B scripts/run_repair_order.py --jobs 12 *>&1 | Tee-Object results/repair_order/run_log.txt
   ```
   After a stop, rerun with `--resume` and append to the log.
3. **Check:** `runs.csv` has 540 rows (6 datasets × 10 seeds × 9 rows), no task failed, and
   `summary.md` was written.
4. Commit `runs.csv`, `summary.md` and `run_log.txt` unedited, with the message "Run H: results
   of the pre-registered run on the home PC", and push to `peer-review-fixes`.
5. **Tell the user**, in plain words: Run T's verdict and the time of each setup, and Run H's
   primary verdict (greedy-300 vs random-300) per dataset, the amendment-1 reading table (A1,
   against random-300-real) and the two efficiency verdicts (E1 seconds, E2 edits), with the
   commit hashes.

The write-up is done later on the laptop. Don't edit the paper, FINDINGS or CLAIMS.

## 5. Runs I, J and K (branch `peer-review-fixes`)

Built, tested and pre-registered on the laptop on 2026-11-14. **Change no code**, except the
two small additions Run K's plan lists (the `download_data.py` entry and the `data/README.md`
row). The rules above still hold. Each plan's "How to run" has the commands; this is the
order.

### Setup

```powershell
cd $HOME\Documents\pace-cfmnn
git fetch origin
git checkout peer-review-fixes
git pull origin peer-review-fixes
.\.venv\Scripts\python -B scripts/download_data.py --check
.\.venv\Scripts\python -B -m unittest discover -s tests -t .
```

Every test must pass (one Covertype test is skipped until the file is there). Run nothing
else while a run is going: Run J's timing pass and Run K measure time.

### Run I (`results/generality/PREREGISTRATION.md`; about 30 minutes)

1. Smoke slice, seed 190 (73 s on the laptop; stop and ask if over 3 minutes):
   ```powershell
   .\.venv\Scripts\python -B scripts/run_generality.py --datasets magic page_blocks --seeds 190 --out $HOME\scratch\run_i --fresh
   ```
2. Run:
   ```powershell
   .\.venv\Scripts\python -B scripts/run_generality.py --jobs 12 *>&1 | Tee-Object results/generality/run_log.txt
   ```
3. **Check:** `runs.csv` has 520 rows, `exact.csv` has 60, no task failed, and `summary.md`
   says **Check: PASS**. If it says FAIL, or the summary's count of training errors that are
   not ties is not 0, stop and tell the user.
4. Commit `runs.csv`, `exact.csv`, `summary.md` and `run_log.txt` unedited ("Run I: results of
   the pre-registered run on the home PC") and push.
5. **Tell the user:** P1 and P2 (accuracy at γ = 2 and γ = 8 against γ = 4) per dataset; the
   tie count at every γ; the Simpson model's tie share and accuracy next to CFMNN's; the
   exact check's largest difference.

### Run J (`results/sampled_baselines/PREREGISTRATION.md`; about 2 hours, then 20 minutes)

1. Smoke slice, seed 190 (6 min 16 s on the laptop; stop and ask if over 13 minutes):
   ```powershell
   .\.venv\Scripts\python -B scripts/run_sampled_baselines.py --datasets wine breast_cancer --seeds 190 --out $HOME\scratch\run_j --fresh
   ```
2. Run (stop and ask if it projects to more than 6 hours):
   ```powershell
   .\.venv\Scripts\python -B scripts/run_sampled_baselines.py --jobs 12 *>&1 | Tee-Object results/sampled_baselines/run_log.txt
   ```
3. Then the timing pass, alone:
   ```powershell
   .\.venv\Scripts\python -B scripts/run_sampled_baselines.py --timing *>&1 | Tee-Object results/sampled_baselines/timing_log.txt
   ```
4. **Check:** `runs.csv` has 200 rows, `timing.csv` has 45, no task failed.
5. Commit `runs.csv`, `timing.csv`, `summary.md` and both logs unedited ("Run J: results of
   the pre-registered run on the home PC") and push.
6. **Tell the user:** P1 and P2 (PACE-SHAP against PermutationSHAP and against
   KernelSHAP-paired, deletion margin AUC) per dataset; the insertion verdicts; the error
   against exact values on Wine and Letter; stability; the times measured alone.

### Run K (`results/covertype/PREREGISTRATION.md`; time unknown until its smoke slice)

Follow the plan's "Before the run" exactly: ask the user before the download, add the hash,
run the two smoke slices on seed 190, project, append and push Amendment 1. Only then:

```powershell
.\.venv\Scripts\python -B scripts/run_covertype.py *>&1 | Tee-Object results/covertype/run_log.txt
```

**Check:** `runs.csv` has 15 rows and no task failed. Commit `runs.csv`, `summary.md` and
`run_log.txt` unedited ("Run K: results of the run on the home PC") and push. **Tell the
user:** training time, boxes, the tie share, the centroid-decided share, the three models'
accuracy and macro-F1, and the two explanation times. It is descriptive: no verdict.

The write-up is done later on the laptop. Don't edit the paper, FINDINGS or CLAIMS.

## 6. Runs F and G (branch `peer-review-fixes`), with their amendments

Both were built earlier and reviewed on the laptop on 2026-11-14. Each plan now ends with
**Amendment 1**; read it before the run. **Change no code.** Run them before or after Runs I,
J and K; nothing depends on the order.

### Run F (`results/contraction/PREREGISTRATION.md`; about 30–45 minutes)

Amendment 1 changes no code: it corrects the wording on Page Blocks' budget and fixes how the
GFMM-rule arms are read.

```powershell
.\.venv\Scripts\python -B scripts/run_contraction.py --jobs 12 *>&1 | Tee-Object results/contraction/run_log.txt
```

**Check:** `runs.csv` has 420 rows (6 datasets × 10 seeds × 7 arms) and no task failed. Commit
`runs.csv`, `summary.md` and `run_log.txt` unedited ("Run F: results of the pre-registered run
on the home PC") and push. **Tell the user:** P1 (self-repair against Simpson's contraction,
macro-F1) per dataset, the secondaries, the evicted rows of each arm, and whether Simpson's
test contracted anything on Thyroid.

### Run G (`results/systems/PREREGISTRATION.md`; about 2–3 hours, then 1–1.5 hours)

Amendment 1 adds scores on untied points and a timing pass. Close other programs.

1. Run:
   ```powershell
   .\.venv\Scripts\python -B scripts/run_systems.py --jobs 12 *>&1 | Tee-Object results/systems/run_log.txt
   ```
2. Then the timing pass, alone (stop and ask if it projects to more than 3 hours):
   ```powershell
   .\.venv\Scripts\python -B scripts/run_systems.py --timing *>&1 | Tee-Object results/systems/timing_log.txt
   ```
   After a stop, add `--resume`.
3. **Check:** `runs.csv` has 330 rows (11 datasets × 10 seeds × 3 systems), `timing.csv` has
   33, no task failed.
4. Commit `runs.csv`, `timing.csv`, `summary.md` and both logs unedited ("Run G: results of
   the pre-registered run on the home PC") and push.
5. **Tell the user:** P1 and P2 (normalised deletion lead against the EBM and the tree) per
   dataset, with the near-zero rule; Amendment 1's reading table (which verdicts hold on
   untied points); accuracy; model size; and the times from the timing pass.

## 7. What ran: Runs F, G, I, J and K (home PC, 2026-11-14)

Branch `peer-review-fixes`, from `b3eb20e`. Machine: "Intel64 Family 6 Model 191 Stepping 2,
GenuineIntel", 16 logical CPUs, 15.8 GB, Python 3.14.10. One run at a time.

**Setup.** `download_data.py --check` passed; the full suite ran 346 tests, OK (1 skipped: the
Covertype file was not there yet), in 254 s.

| Run | Commands (as in sections 5 and 6) | Time | Commits |
|---|---|---|---|
| F | `run_contraction.py --jobs 12` | 14.7 min | results `f6d8c26` |
| G | `run_systems.py --jobs 12 --fresh`, then `--timing` alone | first attempt 61 min (3 tasks FAILED, discarded); rerun 63 min; timing pass 34 min | Amendment 2 and fix `e3033c8`; results `f6486fd` |
| I | smoke slice (seed 190, 34 s), then `run_generality.py --jobs 12` | 21 min | results `35a756b` |
| J | smoke slice (seed 190, 2.9 min), then `run_sampled_baselines.py --jobs 12`, then `--timing` alone | 75 min, then 9 min | results `b2c9638` |
| K | download, hash, tests; smoke slices (seed 190: 10% 22 s, full 130 s); then `run_covertype.py` | 11 min | Amendment 1 `59f1a40`; results `00d73ce` |

**Run G's deviation (Amendment 2 in its plan).** The first attempt's main run printed FAILED
on 3 of 110 tasks (Wine 120, Letter 120 and 127): shap's TreeExplainer "Additivity check
failed". The cause: shap compares float32 inputs with thresholds rounded to the nearest
float32, while scikit-learn compares them with the float64 thresholds, so a point whose
value equals the rounded threshold went the other way in shap. With the user's approval the
explainer's thresholds are now rounded down (`float32_floor` in `scripts/run_systems.py`),
with a new test (`tests/test_systems_amendment2.py`; the suite then ran 348 tests, OK, 1
skipped). The whole main run was repeated from the start with the fixed code. The failed
attempt's files were not committed; a copy is outside the repository
(`$HOME\scratch\run_g_failed`).

**Run K's code change** (allowed by its plan): the `covertype` entry in
`scripts/download_data.py`, with SHA-256 `614360d0…5b15771`, and the Covertype row in
`data/README.md`, committed with its Amendment 1 before the first study seed.

The write-up is done later on the laptop. The paper, FINDINGS and CLAIMS were not edited.

## 8. What ran: Runs L, N and M (home PC, 2026-11-15)

Branch `peer-review-fixes`, plan `research_audit/PART_B_PLAN.md` (`22832d2`), in the plan's
order: Step 0, then L, N, M. Same machine as section 7 (i5-14400F, 16 logical CPUs, 15.8 GB).
Each run's code, tests and `PREREGISTRATION.md` were committed and pushed before its first
study seed; smoke slices used seed 192 (and 190 for Run L's check only), wrote to
`$HOME\scratch\...`, and are listed in each plan. Times below are wall-clock, from each log's
`ELAPSED_S` line or the plan.

**Before the runs.** The write-up of F–K (claims scripts, FINDINGS §13–17, CLAIMS rows,
CHANGELOG) was committed first, as `a31a490`, with the user's agreement, because this plan
forbids editing FINDINGS and CLAIMS. **Step 0:** `download_data.py --check` printed "all
files present and identical to the ones our runs used"; the full suite ran 348 tests, OK, in
280 s.

| Run | Commands | Time | Commits |
|---|---|---|---|
| L | tests (6) and suite (354, OK, 464 s); `run_covertype_theta.py --check-k 190` (PASS, 18 s); smoke slice seed 192 on 10% (40.7 min) and a full-data probe; then `run_covertype_theta.py` alone | 11,438 s (3.2 h); 35.7–40.8 min per seed; peak 4.0 GB | plan, code, tests `bd0d023`; results `f7bcb4a` |
| N | tests (4) and suite (366, OK, 512 s, with Run M's tests); smoke slice `--seeds 192 --n 10` (173 s, 0.64 GB); then `run_sampled_baselines_covertype.py --jobs 5`, then `--timing` alone | 1,108 s, then 290 s; peak 4.1 GB, at least 6.6 GB free | plan, code, tests `075cefa`; results `6434f8e` |
| M | tests (8, OK); smoke slices: Skin seed 192 models part (12 s), then all of seed 192 with 12 jobs (181 s, Check: PASS); then `run_simpson_fmm.py --jobs 12` | 1,315 s; peak 4.1 GB, at least 7.5 GB free | plan, code, tests `53c3f81`; results `3a02af2` |

**Row counts, all as planned:** L `runs.csv` 20; N `runs.csv` 20 and `timing.csv` 9; M
`runs.csv` 260 and `exact.csv` 60. No task printed FAILED and no log has a Traceback. Run M's
summary prints **Check: PASS**.

**Deviations and things to know (each is in the run's plan or summary):**

- **Run L is on a stratified 10% of Covertype**, not the full data. The smoke slice showed the
  full data projecting to about 50 hours (the cached gate alone would need about 58 GB). The
  user was told the times and chose the 10% subsample before any study seed; the method, grid,
  budget and gate are unchanged. So Run L's accuracies are not directly comparable with Run
  K's full-data ones.
- **The plan's "51.5%" for Run K** is a value rounded twice; the raw mean is 51.447% (noted in
  Run L's plan).
- **Run L's explanation times ran alone; Run N's timing pass ran alone.**
- **Run N:** `run_faithfulness.setup` accepted `"covertype"`, so the plan's fallback was not
  needed.
- **Run M, a test in the plan replaced:** "no two boxes of different classes overlap after
  `fit`" does not hold for Simpson's published algorithm (a new point box is never tested;
  hyperbox-brain's `fmnn.py` does the same). Two hand-made tests and a per-model count of the
  pairs still overlapping (`overlapping_pairs`) stand in its place; recorded in Run M's plan
  before the run.
- **Run M, GFMM:** hyperbox-brain 0.1.6, `hbbrain/utils/membership_calc.py`,
  `membership_func_gfmm` returns a minimum over features; `membership_func_fmnn` returns a mean.
  Recorded in Run M's plan.
- **Run M, exact check:** PASS, but the largest difference, 1.4e-10, is from one row (S1, seed
  222); the other 59 rows are at most 3.1e-16. Not investigated further.
- **Run M took 22 minutes**, a little over the plan's 10–20, because ExactExplainer ran slower
  with 12 jobs sharing the CPU.

The paper, `results/FINDINGS.md` and `research_audit/CLAIMS.md` were not edited by these runs.
The write-up of L, M and N is left for the laptop.

## 9. Runs O and P (branch `peer-review-fixes`)

Plans: `results/theta_churn/PREREGISTRATION.md` (Run O, re-tune θ or repair?) and
`results/speed_source/PREREGISTRATION.md` (Run P, where PACE-SHAP's speed comes from). Both were
built and smoke-tested on the laptop and pushed with their code and tests before any study seed
ran. The rules at the top of this file hold, on `peer-review-fixes`. Run P is timed, so nothing
else heavy may run while it runs.

**Step 0.**
```
git fetch origin
git checkout peer-review-fixes
git pull origin peer-review-fixes
.\.venv\Scripts\python -B scripts/download_data.py --check
.\.venv\Scripts\python -B -m unittest discover -s tests -t .
```
Every data file must match (Sepsis may be missing; no run here uses it). The full suite must
pass: about 384 tests (366 at Run M's commit, plus 12 for Run O and 6 for Run P).

**Run O** (12 jobs; about 15–45 minutes):
1. Smoke slice, all 13 datasets on seed 193 (not a study seed; θ-val is tuned there, which the
   study seeds skip):
   ```
   .\.venv\Scripts\python -B scripts/run_theta_churn.py --seeds 193 --jobs 12 --out $env:TEMP\theta_churn_smoke
   ```
   Project the run as 20 × (sum of the summary's `task_s` over the datasets) ÷ 12. Stop and ask
   if it is more than 90 minutes.
2. The run:
   ```
   .\.venv\Scripts\python -B scripts/run_theta_churn.py --jobs 12 *>&1 | Tee-Object results/theta_churn/run_log.txt
   ```
   After a stop, add `--resume` and `Tee-Object -Append`. Every task checks that B0 at θ = 0.3
   and at θ-val reproduces the readable-baselines run; a task that fails it raises an error.
   Stop and ask if any task failed.
3. Commit `results/theta_churn/runs.csv`, `summary.md` and `run_log.txt` unedited, as "Run O:
   results of the pre-registered run on the home PC", and push.

**Run P** (alone, after Run O; about 10–30 minutes):
1. Smoke slice:
   ```
   .\.venv\Scripts\python -B scripts/run_speed_source.py --datasets S1 shuttle --seeds 194 --n 1 --out $env:TEMP\speed_source_smoke
   ```
   Project the run: S1, S2 and S3 each take about 6 × the S1 smoke's seconds, and the other
   three far less. Stop and ask if it is more than 60 minutes.
2. The run, with nothing else running:
   ```
   .\.venv\Scripts\python -B scripts/run_speed_source.py *>&1 | Tee-Object results/speed_source/run_log.txt
   ```
   The script exits with code 1 if its check fails (the summary then says **Check: FAIL**);
   stop and ask if so.
3. Commit `results/speed_source/runs.csv`, `summary.md` and `run_log.txt` unedited, as "Run P:
   results of the pre-registered run on the home PC", and push.

**Hand-over.** Add a section 10, "What ran: Runs O and P", in the style of section 8:
- the commands, wall times and commits;
- the suite's test count;
- anything that did not go as planned.

Commit and push it. Do not edit the paper, `results/FINDINGS.md` or `research_audit/CLAIMS.md`;
the write-up is done on the laptop.

## 10. What ran: Runs O and P (home PC, 2026-11-15)

Branch `peer-review-fixes`, from `0c5dff4` (plans, code and tests of both runs, pushed from the
laptop before any study seed). Same machine as sections 7 and 8. Nothing was built or changed;
no script, module, test or `PREREGISTRATION.md` was edited. Times are wall-clock, from the
`ELAPSED_S` line printed after each command.

**Step 0.** `git pull origin peer-review-fixes` (fast-forward to `0c5dff4`);
`download_data.py --check` printed "all files present and identical to the ones our runs used";
the full suite ran **384 tests, OK**, in 504 s.

| Run | Commands | Time | Rows | Commit |
|---|---|---|---|---|
| O | smoke slice `run_theta_churn.py --seeds 193 --jobs 12 --out $env:TEMP\theta_churn_smoke` (162 s; sum of `task_s` 210.74 s, so 20 × 210.74 ÷ 12 ≈ 351 s projected); then `run_theta_churn.py --jobs 12` | 879 s | `runs.csv` 1,820 (260 dataset-seed pairs × 7 θ) | `d9c2602` |
| P | alone: smoke slice `run_speed_source.py --datasets S1 shuttle --seeds 194 --n 1 --out $env:TEMP\speed_source_smoke` (38 s, Check: PASS; about 12 minutes projected); then `run_speed_source.py` | 679 s | `runs.csv` 54 (6 datasets × 3 seeds × 3 arms) | `6baa19d` |

**Results as printed (`summary.md`):**

- **Run O check:** "503 of 503 B0 rows at θ = 0.3 or θ-val reproduce the readable-baselines
  accuracy and macro-F1 exactly" (520 tasks' rows less the 17 where θ-val is 0.3). No task
  raised an error.
- **Run O, P1** (negative flip rate, repair against re-tuning, for repair): **better 10, level
  0, worse 3**; worse on Shuttle, Skin and Letter.
- **Run P:** **Check: PASS**. Structure factor (ExactExplainer ÷ no pruning) 7.3×–10.7×;
  pruning factor (no pruning ÷ PACE-SHAP) 1.8×–2.9×; total 14.3×–22.7×; (row, box) pairs
  skipped 0.623–0.759.

**What did not go as planned:** nothing. Run O ran in 15 minutes, inside the plan's 15–45;
Run P ran alone in 11 minutes, inside the plan's 10–30. Both runs printed pandas'
"float_precision is deprecated" warning to stderr, which PowerShell's `Tee-Object` records in
each `run_log.txt` as a `NativeCommandError` block; it is a warning, not an error, and the
exit code was 0 both times.

The paper, `results/FINDINGS.md` and `research_audit/CLAIMS.md` were not edited. The write-up
is done on the laptop.

## 11. Tough review, batch 1: Runs P7, P3, P1 and P8 (branch `tough-review-fixes`)

Plans: `results/flip_intervals/PREREGISTRATION.md` (P7, exact flip sets),
`results/skin_dedup/PREREGISTRATION.md` (P3, Skin without duplicates),
`results/time_to_accuracy/PREREGISTRATION.md` (P1, time to 1% and a clean re-time) and
`results/recurrence/PREREGISTRATION.md` (P8, a faster exact mode). All four were built and
smoke-tested on the laptop and pushed with their code and tests before any study seed ran. The
rules at the top of this file hold, on `tough-review-fixes`. P1 and P8 are timed: nothing else
heavy may run while either runs.

**Step 0.**
```
git fetch origin
git checkout tough-review-fixes
git pull origin tough-review-fixes
.\.venv\Scripts\python -B scripts/download_data.py --check
.\.venv\Scripts\python -B -m unittest discover -s tests -t .
```
Every data file must match (Sepsis may be missing; no run here uses it). The full suite must
pass: **446 tests** (384 at Run P's commit, plus the tough-review tests). On the laptop it takes
about 25 minutes; Run P's commit took 504 s on the home PC.

**Order:** P7 and P3 together (12 + 4 jobs would oversubscribe 16 threads; run P7 with 10 jobs
and P3 with 4), then P1 alone, then P8 alone. About 6–8 hours in all.

**P7** (10 jobs; about 1–1.5 hours):
1. Smoke slice (seed 194, not a study seed):
   ```
   .\.venv\Scripts\python -B scripts/run_flip_intervals.py --datasets S1 shuttle --seeds 194 --n 2 --out $env:TEMP\flip_smoke
   ```
   It must print **Check: PASS** twice.
2. The run:
   ```
   .\.venv\Scripts\python -B scripts/run_flip_intervals.py --jobs 10 *>&1 | Tee-Object results/flip_intervals/run_log.txt
   ```
   The script exits 1 if its check fails (the summary says **Check: FAIL**); stop and ask.
3. Commit `results/flip_intervals/` (`runs.csv`, about 25 MB; `summary.md`; `run_log.txt`)
   unedited as "Run P7: results of the pre-registered run on the home PC", and push.

**P3** (4 jobs, at the same time as P7; about 30–60 minutes):
1. Smoke slice: `.\.venv\Scripts\python -B scripts/run_skin_dedup.py --parts reference accuracy --seeds 194 --out $env:TEMP\skin_dedup_smoke`
2. The run:
   ```
   .\.venv\Scripts\python -B scripts/run_skin_dedup.py --jobs 4 *>&1 | Tee-Object results/skin_dedup/run_log.txt
   ```
   Stop and ask if any task printed FAILED.
3. Commit `results/skin_dedup/` (five CSVs, `summary.md`, `run_log.txt`) unedited as "Run P3:
   results of the pre-registered run on the home PC", and push.

**P1** (alone, after P7 and P3; about 2–2.5 hours):
1. Pilot: `.\.venv\Scripts\python -B scripts/run_time_to_accuracy.py --pilot --out $env:TEMP\tta_pilot`.
   Project part (a) as 30 × the pilot's S1–S3 task times plus the other three, and part (b) as
   10 × its times. Stop and ask if the projection is more than 4 hours.
2. The run, with nothing else running:
   ```
   .\.venv\Scripts\python -B scripts/run_time_to_accuracy.py *>&1 | Tee-Object results/time_to_accuracy/run_log.txt
   ```
3. Commit `results/time_to_accuracy/` (`runs.csv`, `summary.md`, `run_log.txt`; not a pilot
   folder) unedited as "Run P1: results of the pre-registered run on the home PC", and push.

**P8** (alone, last; about 3–4 hours):
1. Smoke slice: `.\.venv\Scripts\python -B scripts/run_recurrence.py --datasets shuttle --seeds 194 --n 1 --out $env:TEMP\recurrence_smoke`
   (must print **Check: PASS**).
2. The run, with nothing else running:
   ```
   .\.venv\Scripts\python -B scripts/run_recurrence.py *>&1 | Tee-Object results/recurrence/run_log.txt
   ```
   Letter's one brute-force point and one ExactExplainer point take about an hour each; that
   is expected. Stop and ask if the run passes 6 hours, or if it exits 1 (**Check: FAIL**).
3. Commit `results/recurrence/` (`runs.csv`, `summary.md`, `run_log.txt`) unedited as "Run P8:
   results of the pre-registered run on the home PC", and push.

After a stop, any of the four continues with the same command plus `--resume` and
`Tee-Object -Append`.

**Hand-over.** Add a section 12, "What ran: batch 1", in the style of section 10: the commands,
wall times and commits; the suite's test count; anything that did not go as planned. Commit and
push it. Do not edit the paper, `results/FINDINGS.md` or `research_audit/CLAIMS.md`; the
write-up is done on the laptop.

## 12. What ran: batch 1 (home PC, 2026-11-16/17)

Branch `tough-review-fixes`, from `d39d0e5` (plans, code and tests of all four runs, pushed from
the laptop before any study seed). Same machine as sections 7–10. Apart from Run P8's Amendment 1
(below, with the user's approval), no script, module, test or `PREREGISTRATION.md` was edited.
Times are wall-clock, from the `ELAPSED_S` line printed after each command.

**Step 0.** `git pull origin tough-review-fixes`; `download_data.py --check`; then, with the
user's approval, `download_data.py --datasets phoneme wilt mfeat_morphological jungle_chess
electricity` (the five OpenML files; the script checks OpenML's MD5 and our SHA-256), and
`--check` again, which passed. The full suite ran **501 tests, OK**, in 651 s (not the 446 this
section's step 0 says: batches 2 and 3 added tests since).

| Run | Commands | Time | Rows | Commit |
|---|---|---|---|---|
| P7 | smoke slice `run_flip_intervals.py --datasets S1 shuttle --seeds 194 --n 2 --out $env:TEMP\flip_smoke` (35 s, Check: PASS); then `run_flip_intervals.py --jobs 12`, beside P3 | 2,561 s | `runs.csv` 85,680 (point, feature) pairs | `8746697` |
| P3 | smoke slice `run_skin_dedup.py --parts reference accuracy --seeds 194 --out $env:TEMP\skin_dedup_smoke` (40 s); then `run_skin_dedup.py --jobs 4`, beside P7 | 1,329 s | refinement 110, reference 20, accuracy 30, theta 70, repair 20 | `331cb45` |
| P1 | alone: pilot `run_time_to_accuracy.py --pilot --out $env:TEMP\tta_pilot` (308 s; projected 2.07 h, under the 4 h limit); then `run_time_to_accuracy.py` | 8,043 s | `runs.csv` 1,310 | `aca8e37` |
| P8, first run | alone: smoke slice `run_recurrence.py --datasets shuttle --seeds 194 --n 1 --out $env:TEMP\recurrence_smoke` (Check: PASS); then `run_recurrence.py` | 14,164 s | `runs.csv` 74; exit 1, **Check: FAIL** | `6130b66` (kept in `results/recurrence/first_run/`) |
| P8, Amendment 1 | the comparison fix and its test (below); full suite **502 tests, OK**, in 641 s | – | – | `ab27a82` |
| P8 | alone: the smoke slice again (Check: PASS); then `run_recurrence.py --fresh` | 14,254 s | `runs.csv` 74 | `1d2acfc` |

**Results as printed (`summary.md`):**

- **P7:** **Check: PASS** on 17,500 densely checked (point, feature) pairs and all 85,680 pairs
  at the 32 values (2,414 values within 1e-9 of a breakpoint left out). The 32 values miss 6,356
  of 34,832 flipping pairs (18.2%) over all datasets.
- **P3:** SR − B0 on `skin_dedup` = **+6.20** macro-F1 points (better on 10 of 10, p = 0.0020):
  "the Skin result stands". Reference, original Skin, same seeds: +7.56 (10 of 10, p = 0.0020).
- **P1, part (a):** no sampled budget reached a median error of 1% on any of the 6 datasets, so
  b\* is "all" everywhere; KernelSHAP at b\* took 16.9–26.4× PACE-SHAP's time (every interval
  above 1). **Part (b):** KernelSHAP-200 took 3.3–8.0× as long as PACE-SHAP's default mode
  (median 4.9×).
- **P8 (rerun):** **Check: PASS.** Recurrence vs matmul 1.03–1.23× faster at 9–10 features,
  2.13× on Wine, 1.40× on Letter; brute force ÷ recurrence 47–112×. Letter's recurrence takes
  26.1 s per explanation, so the exact limit stays at 12 (the decision rule says **no**).

**What did not go as planned:**

- **P8's first run failed its check because of a comparison bug, not the values.** On Letter,
  ExactExplainer explains one point and the PACE-SHAP modes ten; the script compared `phi -
  exact[:10]`, and numpy broadcast the one ExactExplainer point against all ten, so points 2–10
  were compared with point 1 ("differs from ExactExplainer by 3.9e-02"). On the one shared point
  everything agreed within 2e-16. With the user's approval: the first run was committed as a
  record in `results/recurrence/first_run/`; `run_one` now compares only the points both arms
  explained (nothing else changed); a test that fails on the old code was added; the change is
  written up in `results/recurrence/AMENDMENT_1.md` (the plan itself is unchanged); and the whole
  run was repeated alone with `--fresh`. The rerun's Letter rows agree with ExactExplainer within
  1.9e-16. Batch 1 took about 4 hours longer than planned because of the rerun.
- **P7's smoke slice** prints one check line covering both datasets, not "Check: PASS twice" as
  section 11 says; it printed PASS and exited 0.
- **P7's `runs.csv`** is 8.1 MB, not about 25 MB.
- Every run's time was inside its plan's estimate: P7 43 min (45–90 planned, beside P3), P3
  22 min (30–60), P1 2.23 h (2–2.5), P8 3.96 h (3–4), and the first P8 run 3.93 h.
- As before, pandas' "float_precision is deprecated" warning appears in each `run_log.txt` as a
  `NativeCommandError` block; it is a warning, and the exit codes were as stated.

The paper, `results/FINDINGS.md` and `research_audit/CLAIMS.md` were not edited. The write-up
is done on the laptop.

## 13. Tough review, batch 2: Runs P2, P5, P11, P6 and P4 (branch `tough-review-fixes`)

Plans:
- `results/fmm_baselines/PREREGISTRATION.md` (P2, tuned fuzzy min-max baselines);
- `results/decision/PREREGISTRATION.md` (P5, explaining the decision at ties);
- `results/cannot_hurt/PREREGISTRATION.md` (P11, edits that provably cannot hurt);
- `results/shap_certificate/PREREGISTRATION.md` (P6, a certificate for explanations after an edit);
- `results/gfmm_exact/PREREGISTRATION.md` (P4, exact PACE-SHAP for GFMM-family networks).

All five were built and smoke-tested on the laptop, and pushed with their code and tests before
any study seed ran. Batch 2 does not depend on batch 1's results. It may start once batch 1 has
finished (P1 and P8 are timed and need the machine to themselves). The rules at the top of this
file hold, on `tough-review-fixes`. The number of jobs changes only the speed, never a result.
Only P4's timing part is timed, and it runs last, alone.

**Step 0.**
```
git fetch origin
git checkout tough-review-fixes
git pull origin tough-review-fixes
.\.venv\Scripts\python -B scripts/download_data.py --check
.\.venv\Scripts\python -B -m unittest discover -s tests -t .
```
The full suite must pass: **475 tests** (446 in batch 1, 21 for batch 2, 8 for the front end in `tests/test_api.py`). On the laptop it took 24 minutes, with 1 test skipped (each skip names a missing dataset or library).

**Order** (16 threads; about 10–20 hours in all, most of it P2):
1. The P2 pilot (6 jobs), with P5 (4 jobs) and then P11 (4 jobs) beside it.
2. The P2 study (6 jobs), with P6 (4 jobs) and then P4's check part (4 jobs) beside it.
3. P4's timing part, alone, last.

**P2 pilot** (seed 194, not a study seed; it decides the Skin amendment):
1. The pilot:
   ```
   New-Item -ItemType Directory -Force results/fmm_baselines/pilot
   .\.venv\Scripts\python -B scripts/run_fmm_baselines.py --seeds 194 --jobs 6 --out results/fmm_baselines/pilot *>&1 | Tee-Object results/fmm_baselines/pilot/run_log.txt
   ```
   Skin's five tasks start first and run side by side.
2. The projection, once the pilot has finished:
   ```
   .\.venv\Scripts\python -B scripts/run_fmm_baselines.py --project --jobs 6 --out results/fmm_baselines/pilot
   ```
   The last line gives the rule's verdict: Skin in full, or `--skin-share 0.2`.
3. A pilot may also be stopped early. If Skin tasks are still running after some hours, run the
   projection with `--running <Skin tasks still running> --elapsed-hours <hours since the pilot
   started>`. That gives a lower bound. If it already says more than 24 h, stop the pilot (Ctrl+C):
   the amendment applies. Otherwise let the pilot finish.
4. Commit `results/fmm_baselines/pilot/` (`runs.csv`, `summary.md`, `run_log.txt`) and the
   projection's output, pasted into `results/fmm_baselines/pilot/projection.txt`, as "Run P2:
   pilot and the pre-registered Skin decision". Push before any study seed runs.

**P5** (4 jobs, beside the pilot; about 15–30 minutes):
```
.\.venv\Scripts\python -B scripts/run_decision.py --jobs 4 *>&1 | Tee-Object results/decision/run_log.txt
```
The script exits 1 if its check fails (**Check: FAIL**); stop and ask. Commit
`results/decision/` unedited as "Run P5: results of the pre-registered run on the home PC", and
push.

**P11** (4 jobs, after P5, beside the pilot; about 1–2 hours):
```
.\.venv\Scripts\python -B scripts/run_cannot_hurt.py --jobs 4 *>&1 | Tee-Object results/cannot_hurt/run_log.txt
```
It exits 1 on **Check: FAIL**; stop and ask. If the summary's "Second stage" line says **run
on** a dataset, stop and ask: that stage needs its own code and pre-registration first. Commit
`results/cannot_hurt/` unedited as "Run P11: results of the pre-registered run on the home PC",
and push.

**P2 study** (6 jobs, after the pilot's commit; about 8–15 hours):
```
.\.venv\Scripts\python -B scripts/run_fmm_baselines.py --jobs 6 *>&1 | Tee-Object results/fmm_baselines/run_log.txt
```
Add `--skin-share 0.2` if, and only if, the pilot's projection said so. Stop and ask if any
task printed FAILED, or if the run passes 30 hours. Commit `results/fmm_baselines/` (`runs.csv`,
`summary.md`, `run_log.txt`) unedited as "Run P2: results of the pre-registered run on the home
PC", and push.

**P6** (4 jobs, beside the P2 study; about 2–4 hours):
```
.\.venv\Scripts\python -B scripts/run_shap_certificate.py --jobs 4 *>&1 | Tee-Object results/shap_certificate/run_log.txt
```
It exits 1 on **Check: FAIL**; stop and ask. Commit `results/shap_certificate/` unedited as "Run
P6: results of the pre-registered run on the home PC", and push.

**P4 check** (4 jobs, after P6, beside the P2 study; about 1–2 hours):
```
.\.venv\Scripts\python -B scripts/run_gfmm_exact.py --part check --jobs 4 *>&1 | Tee-Object results/gfmm_exact/run_log.txt
```

**P4 timing** (alone, after everything else; about 1 hour):
```
.\.venv\Scripts\python -B scripts/run_gfmm_exact.py --part timing *>&1 | Tee-Object -Append results/gfmm_exact/run_log.txt
```
Either part exits 1 on **Check: FAIL**; stop and ask. Commit `results/gfmm_exact/` (`check.csv`,
`timing.csv`, `summary.md`, `run_log.txt`) unedited as "Run P4: results of the pre-registered run
on the home PC", and push.

After a stop, any of them continues with the same command plus `--resume` and
`Tee-Object -Append`. P2 resumes per (dataset, seed, model family), so a stop loses only the
tasks in progress.

**Hand-over.** Add a section 14, "What ran: batch 2", in the style of section 10:
- the commands, wall times and commits;
- the suite's test count;
- the pilot's projection and the Skin decision;
- anything that did not go as planned.

Commit and push it. Do not edit the paper, `results/FINDINGS.md` or `research_audit/CLAIMS.md`;
the write-up is done on the laptop. If section 12 (batch 1's hand-over) is not written yet, write
it first.

## 14. What ran: batch 2 (home PC, 2026-11-17/18)

Branch `tough-review-fixes`, from `7ff19cf` (section 12; the plans, code and tests of all five
runs were pushed from the laptop before any study seed). Same machine as sections 7–12. Apart from
Run P4's Amendments 1 and 2 (below, each with the user's approval), no script, module, test or
`PREREGISTRATION.md` was edited. Times are wall-clock, from the `ELAPSED_S` line printed after
each command.

**Step 0.** `git pull origin tough-review-fixes`; `download_data.py --check` passed. The full
suite ran **502 tests, OK**, in 653 s (not the 475 this section's step 0 says: batch 1's P8
amendment and batch 3 added tests since). After P4's Amendment 1 it ran **503 tests, OK**
(1,021 s, beside P6), and after Amendment 2 **504 tests, OK** (655 s, no skips).

| Run | Commands | Time | Rows | Commit |
|---|---|---|---|---|
| P2 pilot | `run_fmm_baselines.py --seeds 194 --jobs 6 --out results/fmm_baselines/pilot`, beside P5 and P11; then `--project --jobs 6 --out results/fmm_baselines/pilot` into `projection.txt` | 5,661 s | `runs.csv` 72 | `c7a054f` |
| P5 | `run_decision.py --jobs 4`, beside the pilot | 374 s | `points.csv` 3,000 | `a591bb3` |
| P11 | `run_cannot_hurt.py --jobs 4`, beside the pilot, after P5 | 3,987 s | `edits.csv` 8,750 | `cb83050` |
| P2 study | `run_fmm_baselines.py --jobs 6` (Skin in full, no `--skin-share`) | 52,516 s | `runs.csv` 720 | `9a3bcb6` |
| P6 | `run_shap_certificate.py --jobs 4`, beside the study, then beside P4's check | 107,133 s | `edits.csv` 6,085 | `38d6eb7` |
| P4 check, first run | `run_gfmm_exact.py --part check --jobs 4`, beside P6 | 18,795 s | `check.csv` 240; exit 1, **Check: FAIL** | `90d8ddb` (kept in `results/gfmm_exact/first_run/`) |
| P4, Amendment 1 | the adapter's tie rule for IOL-GFMM and its test (below); suite 503, OK | – | – | `1961d8a` |
| P4 check | `run_gfmm_exact.py --part check --jobs 4 --fresh`, beside P6 | 19,125 s | `check.csv` 240 | `b061ac0` |
| P4 timing, first run | alone: `run_gfmm_exact.py --part timing` | 2,495 s | `timing.csv` 24; **14 of 18 tasks FAILED** (out of memory) | `ed91d33` (kept in `results/gfmm_exact/timing_first_run/`) |
| P4, Amendment 2 | the ExactExplainer arm in blocks and its test (below); suite 504, OK | – | – | `aa86e32` |
| P4 timing | alone: `run_gfmm_exact.py --part timing --fresh` | 5,273 s | `timing.csv` 108 | `b061ac0` |

**The pilot's projection and the Skin decision** (`results/fmm_baselines/pilot/projection.txt`):
pilot hours per dataset, all families, one seed: Landsat 0.33, Letter 3.06, MAGIC 2.05, Page
Blocks 0.06, Skin 2.41, Thyroid 0.11; "Projected study: 13.4 h with 6 jobs"; "at most 24 h, so
Skin runs in full". The pilot and `projection.txt` were pushed before the study started, and the
study ran without `--skin-share`.

**Results as printed (`summary.md`):**

- **P2:** CFMNN-SR-val against each tuned baseline, test macro-F1, Holm across 24 tests:
  **better 6, no significant difference 17, worse 1** ("Mixed"). Better: Thyroid against all four
  baselines (+14.59 to +29.40 points), Landsat (+2.34) and Letter (+3.17) against IOL-GFMM-val.
  Worse: MAGIC against IOL-GFMM-val (−1.22, 0 of 10 seeds better, Holm p = 0.0469). Test
  accuracy, the secondary, has the same split (6 / 17 / 1).
- **P5:** **Check: PASS.** Decision deletion AUC on tied points: PACE-decision lower than
  PACE-SHAP (better) on all 3 datasets, 10 of 10 seeds each, Holm p = 0.0059 (MAGIC −0.0506,
  Letter −0.0161, Page Blocks −0.1085). Secondary: on the margin measures Letter is worse for
  PACE-decision (deletion +0.0268, Holm p = 0.0059; insertion −0.0246, Holm p = 0.0410); MAGIC
  and Page Blocks show no significant difference there.
- **P11:** **Check: PASS** ("No edit meeting condition S broke a correct training prediction",
  116 such edits). 116 of 6,110 accepted edits meet S (1.9%; 0% on Thyroid to 8.0% on Page
  Blocks); the gate rejected none of them. **Second stage: not needed.**
- **P6:** **Check: PASS** ("no row outside the certificate changed by more than 1e-12", over
  6,085 accepted edits and 608,500 (edit, row) pairs). Certified share 99.1–100% per edit; rows
  that really changed 31.7% (Letter), 46.2% (Skin) and 80.3–82.7% elsewhere.
- **P4:** **Check: PASS.** The adapter predicts like hyperbox-brain on every explained point;
  PACE-SHAP-min equals the brute force within 6.8e-16 (check) and ExactExplainer within 1e-9
  (timing). Timing, one point each, alone: brute force ÷ PACE-SHAP-min 10.7–10.8× on S1–S3 and
  33.5–94.2× on Shuttle, MAGIC and Page Blocks; ExactExplainer ÷ PACE-SHAP-min 10.7–10.8× and
  33.4–90.3×.

**What did not go as planned:**

- **P6 took 29.8 h against its plan's 2–4 h** (stop and ask past 6 h). At 9.8 h, with 17 of 60
  seeds done, the run was reported and the user chose to keep it running; it was not stopped.
  Letter took 4.2–5.0 h per seed, Landsat 2.7–6.5 h and MAGIC 1.4–2.4 h (the plan's laptop guess
  was "several minutes to under an hour"); Skin, Thyroid and Page Blocks took under 2 minutes
  each. It shared the machine with the study for its first 14.6 h and with P4's check after that.
- **P4's check part was started beside P6 when the study ended, not after P6** as this section's
  order says. That order kept the study and its neighbours within 16 threads; with the study done
  it would have left the machine half idle. The check part is not timed, and the number of jobs
  changes only speed.
- **P4's check part took 5.2 h and 5.3 h**, against the plan's 0.5–1 h with 10 jobs (this
  section says 1–2 h with 4 jobs; it ran with 4). At 3.7 h it was reported and the user chose to
  keep it running. Most of the time is the brute force on S1–S3 (about 30 minutes per task).
- **P4 Amendment 1 (`results/gfmm_exact/AMENDMENT_1.md`).** The first check run failed: on 10 of
  60 IOL-GFMM tasks the adapter predicted another class than hyperbox-brain on one point each, a
  point where every box has membership 0. The adapter broke every tie by Manhattan distance
  (Online GFMM's rule); IOL-GFMM's `predict` uses hyperbox-brain's sample-count rule
  (`predict_with_probability`). With the user's approval: the first run was kept as a record;
  `GFMMBoxes.from_hbbrain` now reads the model's own rule and the adapter applies it; a test that
  fails on the old code was added; and the check part was rerun with `--fresh`. The explanations
  had matched the brute force in the first run too.
- **P4 Amendment 2 (`results/gfmm_exact/AMENDMENT_2.md`).** The first timing run printed FAILED
  for 14 of 18 tasks: shap's ExactExplainer passes 204,800 rows to the model at once, which needs
  about 33–36 GiB on S1–S3 and MAGIC (this PC has 16 GB). The laptop's smoke slice timed only
  Shuttle and Page Blocks. That run still exited 0 and printed Check: PASS, because the check
  reads only finished tasks. With the user's approval: the run was kept as a record; the
  ExactExplainer arm now evaluates the model in blocks of at most 4 million numbers, as the brute
  force already did (values unchanged); a test was added; and the whole timing part was rerun
  alone with `--fresh`. `results/gfmm_exact/run_log.txt` holds the check rerun, the failed timing
  run and the timing rerun, in that order; the first check run's log is in `first_run/`.
- **P4's timing part took 1.46 h**, against about 1 hour.
- Every other time was inside its estimate: the pilot 1.57 h (2–4 h), P5 6 minutes (15–30), P11
  1.11 h (1–2 h), the study 14.6 h (8–15 h; the pilot projected 13.4 h).
- Batch 2 took about 34 h in all, from the start of the pilot (2026-11-17 12:48) to the end of
  P4's timing rerun (2026-11-18 23:18), against the 10–20 hours this section gives, mostly because
  of P6 and P4.
- As before, pandas' "float_precision is deprecated" warning appears in each `run_log.txt` as a
  `NativeCommandError` block; it is a warning, and the exit codes were as stated.

The paper, `results/FINDINGS.md` and `research_audit/CLAIMS.md` were not edited. The write-up
is done on the laptop.

## 15. Tough review, batch 3: Runs P10 and P9 (branch `tough-review-fixes`)

Plans:
- `results/heldout/PREREGISTRATION.md` (P10: RQ1 and RQ2 on 4 held-out datasets chosen by the
  rule in `results/heldout/RULE.md`);
- `results/time_ordered/PREREGISTRATION.md` (P9: a model in use on time-ordered data).

Both were built and smoke-tested on the laptop, and pushed with their code and tests before any
study seed ran. Batch 3 does not depend on batches 1 or 2. It may start once batch 2 has
finished, because P4's timing part needs the machine to itself. The rules at the top of this file
hold, on `tough-review-fixes`. Neither run is timed alone. P10's times per explanation are a
secondary result, and the write-up uses only the ratio between the two explainers.

**Step 0.** The five OpenML files are new. The user approved downloading them on 2026-11-16, and
the script checks each one against OpenML's MD5 and our SHA-256.
```
git fetch origin
git checkout tough-review-fixes
git pull origin tough-review-fixes
.\.venv\Scripts\python -B scripts/download_data.py --datasets phoneme wilt mfeat_morphological jungle_chess electricity
.\.venv\Scripts\python -B scripts/download_data.py --check
.\.venv\Scripts\python -B -m unittest discover -s tests -t .
```
- If a file differs from ours, stop and ask.
- The full suite must pass: **501 tests** (475 after batch 2, plus 26 for batch 3). On the
  laptop it took 27 minutes, with 1 test skipped.
- With the files present, nothing in the new tests is skipped.

**Order** (16 threads): P10 (6 jobs) and P9 (5 jobs) side by side, both started at once.

**P10** (6 jobs; about 2–4 hours; Jungle Chess's tasks start first):
```
.\.venv\Scripts\python -B scripts/run_heldout.py --jobs 6 *>&1 | Tee-Object results/heldout/run_log.txt
```
- It exits 1 on **Check: FAIL**; stop and ask.
- Stop and ask if any task printed FAILED, or if it passes 8 hours.
- Commit `results/heldout/` (`rq1_runs.csv`, `rq2_runs.csv`, `summary.md`, `run_log.txt`)
  unedited as "Run P10: results of the pre-registered run on the home PC", and push.

**P9** (5 jobs; about 4–5 hours, the 10 seeds in two waves):
```
.\.venv\Scripts\python -B scripts/run_time_ordered.py --jobs 5 *>&1 | Tee-Object results/time_ordered/run_log.txt
```
- It exits 1 on **Containment: FAIL**; stop and ask.
- A FAIL in the bound column is a result, not an error. Do not stop for it.
- Stop and ask if any seed printed FAILED, or if it passes 10 hours.
- Commit `results/time_ordered/` (`weeks.csv`, `summary.md`, `run_log.txt`) unedited as "Run P9:
  results of the pre-registered run on the home PC", and push.

After a stop, either run continues with the same command plus `--resume` and
`Tee-Object -Append`. P9 resumes per seed, so a stop loses the seeds in progress.

**Hand-over.** Add a section 16, "What ran: batch 3", in the style of section 10:
- the commands, wall times and commits;
- the suite's test count;
- anything that did not go as planned.

Commit and push it. Do not edit the paper, `results/FINDINGS.md` or `research_audit/CLAIMS.md`;
the write-up is done on the laptop.

## 17. Claims checks for batches 1 and 2 (branch `tough-review-fixes`)

Section 16 is batch 3's hand-over, which the home PC writes after batch 3. If it is not there yet, leave a heading for it above this section.

The laptop wrote batch 1's and batch 2's results into the paper (revs 624–657), `results/FINDINGS.md` §23 and `research_audit/CLAIMS.md` rows TR1–TR14, but ran nothing. Two read-only scripts recompute every number from the committed CSVs. Run them **after batch 3 has finished**, or in any gap when no timed run is going. Each takes well under a minute and touches nothing in `results/`.

From the repository root (`cmd /c` writes plain UTF-8, as the earlier `*_out.txt` files are):

```powershell
git pull origin tough-review-fixes
cmd /c ".venv\Scripts\python -B research_audit\scripts\claims_tough_batch1.py > research_audit\scripts\claims_tough_batch1_out.txt 2>&1"
cmd /c ".venv\Scripts\python -B research_audit\scripts\claims_tough_batch2.py > research_audit\scripts\claims_tough_batch2_out.txt 2>&1"
Get-Content research_audit\scripts\claims_tough_batch1_out.txt -Tail 1
Get-Content research_audit\scripts\claims_tough_batch2_out.txt -Tail 1
```

Each output file's last line is `ALL OK` or `N BAD: [...]`. Either way:

1. Commit both `*_out.txt` files unedited, as "Claims checks for tough-review batches 1 and 2", and push.
2. If a script crashes, commit the output with the traceback anyway.
3. Report the two last lines.

Do not change the scripts, the paper, FINDINGS or CLAIMS. A BAD line or a crash is fixed on the laptop.
