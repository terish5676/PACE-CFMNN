# Run P9, a model in use on time-ordered data: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before any study seed ran (tough
review B10 and D6). The code is `scripts/run_time_ordered.py` (tests in
`tests/test_time_ordered_run.py`; the loader's tests are in `tests/test_openml_data.py`).
Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 15.

## Why this run

§1 motivates the paper with a model in use, whose errors must be fixed without silent changes to
its other decisions. Every earlier test used random splits. This run follows one model through
time, the way a team would use it, and asks whether self-repair keeps its promise there.

## The data

- **electricity** from OpenML (data id 151, licence Public on OpenML): prices and demand in the
  New South Wales electricity market, 1996–1998. The class says whether the price went up or down
  against a 24-hour moving average.
- 45,312 rows, one every half-hour, in time order: 944 days of 48 rows each
  (`tests/test_openml_data.py` checks the days are unbroken).
- The download was approved by the user on 2026-11-16. The file matched the MD5 that OpenML
  publishes, and its SHA-256 is pinned in `scripts/download_data.py`. The dataset was named in
  `results/heldout/RULE.md` (commit 94dd0ed) before it was downloaded.
- **Features (7):**
  - the day of the week, mapped to (day − 1) / 6;
  - the period of the day and the five numeric columns (nswprice, nswdemand, vicprice,
    vicdemand, transfer), as the file gives them.
- **Dropped:** the date column. It is not in time order in the file, and a model in use cannot
  learn from a date it has not seen.
- **Scaling:** the file's authors already scaled the numeric columns to [0, 1] over the whole
  file. That is a small look-ahead (the scale uses later weeks), which every user of this file
  shares. It is stated in the write-up, not corrected.
- **Drift:** vicprice, vicdemand and transfer are constant for the first 17,424 rows (about 52
  weeks), then vary. That drift is part of the data, and the run keeps it.
- **Windows:** weeks of 336 rows (7 days × 48). The study uses the first 134 full weeks (45,024
  rows). The last 288 rows, a partial week, are not used.

## The protocol

Each seed (380–389) fixes the order of the rows **inside** each week. CFMNN depends on the order
it sees rows in. The weeks themselves are fixed by time.

**Start.** Weeks 0–3 (1,344 rows) train the first model. θ₀ is chosen by 3-fold validation on
those rows, from the paper's grid (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05), as a team would at
deployment.

**Each week w = 4, …, 132:**
1. **In use.** Every arm's deployed model predicts week w.
   - For INC and SR, a detector flags likely errors (below).
   - PACE-SHAP explains 10 of SR's decisions, on 100 background rows drawn from the last 4 weeks
     of labelled rows.
2. **The labels arrive.** Each arm updates its model with week w's labels.
3. **Scored on week w + 1**, which no model has seen:
   - changed, fixed and broken predictions (the updated model against the one before it);
   - the certificate's churn bound (below).

## The arms

| Arm | θ | Update with each week's labels |
|---|---|---|
| **INC** | θ₀ | The incremental update: `CFMNN.add_points`, one-pass training continued on the new rows (S8's comparator). |
| **SR** | θ₀ | The same update, then self-repair on the training rows. The paper's settings: 300 errors, 10 rounds, the smallest cut, the gate on the training rows (`fast_gate`, which makes the same decisions as the standard gate). No label beyond the weeks received. |
| **RETUNE** | re-chosen | Every 4th week, θ is re-chosen by 3-fold validation on all rows so far and the model is refitted on them. In the other weeks, the incremental update. |
| **INC-0.3** | 0.3 | As INC, at the paper's θ. |
| **SR-0.3** | 0.3 | As SR, at the paper's θ. |

- SR and SR-0.3 also repair their first model before week 4, so the model in use is always a
  repaired one.
- **Known before the run:** one-pass training gives the same boxes for fit(all rows) as for
  fit(first rows) followed by add_points(the rest). So RETUNE equals INC exactly whenever the
  re-chosen θ equals θ₀. RETUNE differs from INC only when the data move θ.

## Primary test

**Broken predictions in the next week (NFR), SR minus INC.**
- NFR = the share of week w + 1's rows that the model before the update got right and the model
  after it gets wrong.
- For each week, the difference is averaged over the 10 seeds. A two-sided Wilcoxon signed-rank
  test then runs over the 129 weeks, at α = 0.05.

**A deviation from the review's sketch, stated here.** The review asked for a test "over windows
and seeds". The seeds change only the order of rows inside each week, so they are not independent
copies of the stream. Pooling 1,290 (week, seed) pairs would overstate the sample. So the
primary averages over seeds, and two robustness tests follow.

## Secondary (Holm across all of them)

- **Robustness of the primary:**
  - by seed: the mean over weeks for each seed, 10 values;
  - by month: 4-week blocks of the weekly means, 32 values, which damps the dependence between
    neighbouring weeks.
- NFR at θ = 0.3: SR-0.3 minus INC-0.3.
- NFR: SR minus RETUNE.
- Accuracy in use (each week's deployed model on that week): SR minus INC, SR minus RETUNE, and
  SR-0.3 minus INC-0.3.

## Check: does the certificate bound the next week's churn?

- For INC, SR, INC-0.3 and SR-0.3, the certificate of each whole update is the union of:
  - the certificate of the incremental step (`audit.impact_set`, tight; it covers a step that
    grows or appends boxes);
  - the certificate of every accepted self-repair edit.

  That union contains every changed prediction (Proposition 4 and Corollary 1). It is computed
  from the two models and unlabelled rows alone.
- **Split halves.** Week w + 1 is split at random into two halves.
  - The certificate's share on half A gives a one-sided 95% Clopper–Pearson upper bound
    (`audit.churn_bound`).
  - The churn (the share of changed predictions) is measured on half B.

  Measured on the same rows, the bound would hold by construction; split halves make it a real
  prediction.
- **PASS** for an arm if the bound holds in at least 95% of weeks × seeds.
- **Containment** (a hard check): every changed prediction on week w + 1 lies inside the
  certificate, in every week, seed and certified arm. If it fails, Proposition 4 or its code is
  wrong. The script then exits 1: stop and ask.

## Descriptive (no test)

- Accuracy and macro-F1 in use, NFR, changed share, boxes, and update seconds, per arm.
- **Detector (INC and SR).** AUROC for the deployed model's errors in each week, and the
  precision of the top 10% of flags, for:
  - the order ensemble: 5 CFMNNs at θ₀, each updated with `add_points` in its own row order;
  - kNN disagreement: k = 10, over the labelled rows;
  - the two added.

  **Deviation:** the paper's strongest detector also uses Trust Score and a learned combination
  fitted on labelled calibration rows. A stream has no such rows at the start, so this run uses
  the unweighted sum of the two strongest signals.
- **PACE-SHAP:** milliseconds per explanation, and whether it was exact (d = 7 ≤ 12, so it should
  be in every week).
- RETUNE's θ at each refit, and θ₀ per seed.

## What ran before this was written

Laptop smoke slices on seed 194 (not a study seed, never analysed), written outside the
repository:
- **A cost check:**
  - At θ = 0.3, CFMNN fitted on all 45,024 rows makes 36.6% training errors with 204 boxes. At
    θ = 0.05: 11.2% with 5,170 boxes.
  - So θ = 0.3 is badly under-fitted here, which is why the main arms start at θ₀ chosen on the
    first weeks.
  - Self-repair took 4 s per week at week 40 and 22 s at the end (θ = 0.3).
- **16 weeks:**
  - θ₀ = 0.05.
  - The containment and bound checks held in every week.
  - In those 11 scored weeks, SR broke more predictions than INC.
  - The design did not change after it.
- **Full length, one seed:** stopped by the user at week 110 of 132, after about 83 minutes
  (another job shared the laptop for part of it). Later weeks cost more as the models grow, so
  one seed is about 2–2.5 hours on the laptop.

## Time

- Each seed is one task.
- With 5 jobs, the 10 seeds run in two waves: **about 4–5 hours on the home PC**.
- **Stop and ask if it passes 10 hours.**

## How the result will be read (written now, before the run)

- **SR breaks fewer predictions than INC** (primary significant, SR lower):
  - the paper gains a short practitioner section (about 800 words), with this protocol;
  - §1's motivation stands.
  - If a robustness test disagrees in direction, the section says the result depends on how
    weeks are grouped.
- **SR breaks more** (primary significant, SR higher):
  - the advice changes from "repair after deployment" to "use the incremental update; repair
    under a size budget";
  - §1's motivation is softened (tough review D6);
  - the section reports the result as a negative one.
- **No significant difference:** the paper says self-repair's change to the model in use was
  not detectably different from the incremental update's, and reports SR's accuracy against
  INC's beside it.
- **Certificate check FAIL** for an arm: the paper does not claim the label-free churn bound is
  usable on drifting data for that arm. It reports how often the bound held.
- **Check PASS:** the paper reports the bound's median looseness (median bound against median
  churn), so a reader can see how useful it is, not only that it holds.
- **RETUNE:** reported as found. If it never changes θ, it equals INC, and the paper says so.

## How to run (home PC, batch 3)

```
.\.venv\Scripts\python -B scripts/run_time_ordered.py --jobs 5 *>&1 | Tee-Object results/time_ordered/run_log.txt
.\.venv\Scripts\python -B scripts/run_time_ordered.py --jobs 5 --resume *>&1 | Tee-Object -Append results/time_ordered/run_log.txt   # after a stop
```

A stop loses the seeds in progress: each seed is one task, from week 0 to the end. Commit
`results/time_ordered/` unedited (`weeks.csv`, `summary.md`, `run_log.txt`) as "Run P9: results of
the pre-registered run on the home PC".
