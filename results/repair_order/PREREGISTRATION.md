# Run H, the order of training errors in self-repair: pre-registration

Written on 2026-11-14 and pushed, with the code it describes, before the run.
The code is `scripts/run_repair_order.py` (tests in `tests/test_repair_order.py`).
It adds an `order` option to `run_self_repair` in `pace_cfmnn/active.py` (the
default, `"random"`, is the order every committed run used), and `Gate.verdict`,
`Gate.peek` and `CachedGate.peek` to `pace_cfmnn/refine.py` (the gate's
decisions are unchanged). Runbook: `research_audit/HOME_PC_RUNBOOK.md`, run H.

## What ran before this was written

- **Runs A and E** (`results/repair_sensitivity/`, `results/fast_gate_check/`).
  E's summary says **Verdict: PASS**, so this run uses the cached gate. A's
  training-error counts and times per arm set the arms and the time estimate below.
- **The default order is unchanged.** With the new code, RQ3's SR arm was replayed
  for Thyroid and Page Blocks, seed 50 (both gates), and Thyroid, seed 51 (in the
  tests): every edit and the test macro-F1 equal the committed RQ3 results.
- **Tests while building:** the full suite, with the new tests (see Tests below).
  The first toy problem in the new tests gave the model only one training error,
  too few to test an order; its data was changed to 3 features (62 errors). No
  design changed.
- **A smoke slice, seed 140** (not a study seed, never analysed), on the laptop,
  written outside the repository: Thyroid, Page Blocks and Landsat, then MAGIC,
  Letter and Skin, for timing. It printed each arm's test macro-F1 and a one-seed
  summary of the comparisons. The only
  things taken from it are the times and that the code runs on every dataset. No
  arm, order, budget or comparison changed after it.
  - With 3 tasks at a time, one task (all 8 arms) took: Thyroid 5 s, Page Blocks
    27 s, Landsat 57 s, MAGIC 11 min, Letter 18 min, Skin 33 min.
  - Almost all of it is the two greedy arms: MAGIC 341 and 274 s, Letter 571 and
    362 s, Skin 1,188 and 726 s (greedy-300 and greedy-1000).
  - Run A's home-PC times per arm are about the laptop's here, so the full run
    (60 tasks, 12 jobs) projects to about 1 hour.

## The question

Self-repair tries up to 300 of the model's own training errors, in a seeded
random order. With the same 300 tries, does a smarter order make the repaired
model better? Does it reach what trying every error gives?

**Why the order can matter more on some datasets.** Mean training errors at the
start in Run A: Thyroid 71, Page Blocks 227, Landsat 291, Letter 2,080, MAGIC
2,316, Skin 4,983. On the first three a 300-try budget reaches every error
whatever the order, so only the sequence changes. That can still matter, because
each accepted edit changes the model the next one is checked against. On the
other three, 300 tries cover only 6–15% of the errors, so the order also decides
which errors are tried at all. This is an expectation, not a test: all 6 datasets
are in the test.

## Setup

- **Data and model:** RQ3's split (`run_refinement.split`) and fitted model
  (`CFMNN()`: θ = 0.3, γ = 4), on Thyroid, Landsat, Page Blocks, MAGIC, Letter
  and Skin.
- **Seeds:** 130–139. No earlier run used them.
- **Repair:** RQ3's SR arm as in Run A: the volume cut (`chooser="volume"`), 10
  rounds, gated on the training rows (`gate_labels_are_new=False`), the default
  gate (macro-F1 may fall at most 0.5 points, no class recall more than 1 point,
  each edit compared with the model before it), box cap 1.1 times the start, and
  the cached gate (`fast_gate=True`).
- **The seeded random order** (`rng.permutation` of the training errors, seed =
  the study seed) is drawn in every arm, as today; the box and greedy orders use
  it only to break ties.

## Arms

All arms start from the same fitted model, so B0 is shared. "300" is RQ3's
default budget, `min(300, len(pool) // 4)`: 273 on Page Blocks, 300 elsewhere.

| Arm | Order | Errors tried |
|---|---|---|
| B0 | none: no repair | none |
| **random-300** | seeded random (today's SR; the control) | 300 |
| random-1000 | seeded random | 1,000 |
| random-all | seeded random | every training error |
| box-300 | box | 300 |
| box-1000 | box | 1,000 |
| **greedy-300** | greedy | 300 |
| greedy-1000 | greedy | 1,000 |

**Box order** (`active.box_order`). The training errors are grouped by the box
that wins them at the start (always a box of a wrong class). The biggest group
goes first; groups of equal size, and the rows inside a group, keep the random
order. The order is fixed at the start.

**Greedy order.**

- At the start of each round, every error not yet tried is scored on the current
  model by a trial edit: the same cut the real try would make (`shrink_target`,
  the volume cut), applied to a copy.
- The score is the number of training rows the trial edit would fix minus the
  number it would break (rows right after minus rows right before), computed
  through the cached gate (`CachedGate.peek`), which leaves the gate unchanged.
- Rank: errors whose trial edit the gate would accept come first, then those it
  would reject, then rows that cannot be edited (already right, or not inside the
  winning box). Within each class the higher score goes first; ties follow the
  random order.
- Before each try, an error whose score was computed before the last accepted
  edit is re-scored on the current model and tried only if it still ranks first;
  otherwise it goes back with its new score (lazy re-scoring). So every try uses
  a score from the model it is tried on.
- A round tries as many errors as the random order's round would
  (`np.array_split` of the budget into 10 rounds).

In every arm, an error that is already right by its turn is skipped, and the try
still counts against the budget, as today. On a dataset with fewer training
errors than a budget, arms with that budget may try the same errors and can be
identical; that is reported, not corrected.

## Measures

For each dataset, seed and arm:

- test macro-F1, accuracy, lowest class recall and balanced accuracy (%);
- boxes and training error, at the start and the end; training errors at the start;
- tries; edits accepted; edits rejected by reason; tries skipped (already right;
  not inside the winning box);
- test predictions fixed and broken against B0 (`audit.prediction_churn`);
- trial edits scored (greedy only);
- seconds. Several tasks share the CPU, so times are compared within a dataset
  only.

## Primary and pass rule

**Primary:** greedy-300 vs random-300 on test macro-F1.

- Paired two-sided Wilcoxon over the 10 seeds, Holm-corrected across the 6
  datasets, α = 0.05. When all 10 differences are zero, p = 1.
- greedy-300 is **better** on a dataset if the Holm p is below 0.05 and the mean
  difference favours it, **worse** in the other direction, otherwise **level**.
- No direction is predicted.

**Secondary** (the same test, each Holm-corrected across datasets, not corrected
for each other, and reported as secondary): box-300 vs random-300; greedy-300 vs
random-all; greedy-1000 vs random-1000; box-1000 vs random-1000; greedy-300 vs
box-300. Everything else (accuracy, recall, boxes, edits, fixed and broken,
trial edits, times) is descriptive.

## What the result changes

- Results are reported whichever way they go.
- Every committed result (RQ3, RQ4, Runs A–E) used the random order and stays as
  it is. If greedy-300 is better on some datasets and worse on none, the paper
  reports the greedy order as an improvement found in this run, with its extra
  time; it does not rerun the earlier studies with it.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.

## Time

From the smoke slice and Run A's home-PC times per arm: about 1 hour with 12
jobs, between 45 minutes and 1.5 hours. Most of it is the two greedy arms, which
score every remaining error once per round; a Skin task is the longest (about
half an hour). If the home PC's smoke slice (runbook, run H) takes more than
twice its expected time, stop and ask before the full run.

## Tests

`tests/test_repair_order.py` checks that:

- the default order is the random order, edit for edit, and reproduces RQ3's
  committed SR edits on real data (Thyroid, seed 51);
- the trial copy leaves the model unchanged and edits exactly like a deep copy;
- the box order is a permutation of the errors, with whole groups, biggest first,
  and the random order inside and between groups of equal size;
- `peek` gives the trial model's predictions and leaves the cached gate as it was;
- greedy's first try is the error with the best brute-force score, every accepted
  edit's score equals its real fixes minus breaks, and a predicted pass or reject
  always matches the gate's real decision;
- greedy makes the same edits with the cached and the standard gate, tries each
  error at most once, and uses the round sizes of the random order;
- the script builds the arms as above and refuses to run unless Run E passed.

## How to run

```
python -B scripts/run_repair_order.py --jobs 12
python -B scripts/run_repair_order.py --jobs 12 --resume     # continue after a stop
python -B scripts/run_repair_order.py --summary-only
```

Outputs: `results/repair_order/runs.csv` (one row per dataset, seed and arm, the
same columns on every row; NaN where a column does not apply), `summary.md` and
`run_log.txt`. Expected: 6 datasets × 10 seeds × 8 arms = 480 rows.

## Amendment 1 (2026-11-14, before the run)

Written and pushed, with its code and tests, before any study seed (130–139) was run.
Nothing above this section was edited; where this section differs, this section holds.

**Why.** A review of this plan, before the run, found two gaps.

1. **The primary comparison favours greedy for a reason other than its order.** A try on
   a row that is already right still counts against the budget. Greedy ranks such rows
   last, so nearly all of its 300 tries reach the gate; the random order spends many of
   its tries on rows an earlier edit already fixed (in Run A's SR-default arm, out of 300
   tries: Skin 252, MAGIC 98, Landsat 90, Letter 50, Page Blocks 44). A greedy win could
   then mean only "it wastes fewer tries".
2. **The plan does not test the reason to change the order: less work.** Time and edits
   were descriptive only.

**What was known when this was written.** The plan above, the laptop smoke slice on
seed 140 (it printed each arm's test macro-F1 and a one-seed summary; seed 140 is not a
study seed and is not analysed), Run T's results (`results/repair_timing/`, seed 50),
and Runs A and E. No study seed had been run.

**What is added.**

- **A new arm, random-300-real:** the seeded random order with RQ3's default budget,
  but a try on a row that is already right, or not inside the winning box, does not count
  (`run_self_repair(..., count_skips=False)`). The order runs on until 300 edits (273 on
  Page Blocks) have reached the gate, or the errors run out. Each round gates as many
  edits as a round of random-300 tries. All other settings are those of random-300.
- **A new column, `edits_gated`:** the tries that reached the gate (an edit was made and
  checked). `edits_logged` = `edits_gated` + both skip counts.
- **A new comparison, A1: greedy-300 vs random-300-real** on test macro-F1, the same test
  as the primary (paired two-sided Wilcoxon over the 10 seeds, Holm across the 6
  datasets).
- **Two efficiency tests,** each the same test, Holm across datasets:
  - **E1: greedy-300 vs random-all, seconds** (lower is less work);
  - **E2: greedy-300 vs random-all, edits that reached the gate** (lower is less work).

  The arms of one task run one after another in one process, so seconds are paired
  within a dataset and seed; they are still measured while other tasks share the CPU.

**What stays the same.** The primary (greedy-300 vs random-300), every arm above, the
seeds, the datasets, the gate and the secondary comparisons. A1, E1 and E2 are reported
as amendment-1 comparisons, each corrected across datasets and not for each other.

**How the results will be read** (written now, before the run):

- On a dataset where the primary says **better**, greedy is called *a better order* only
  if A1 also says **better**. If A1 says level or worse, the gain is reported as coming
  from *not wasting tries*, not from choosing better errors.
- Greedy is said to *reach what trying every error gives, with less work* on a dataset
  only if greedy-300 vs random-all (a secondary comparison) is not **worse** on macro-F1
  **and** E1 says greedy-300 is **faster**. If it is only E2 that favours greedy (fewer
  edits, but more seconds), it is reported as *fewer edits, more time*.
- The summary prints this reading per dataset (`reading` in the script).

**Effect on the run.** 6 datasets × 10 seeds × 9 rows = **540 rows** (was 480). The new
arm costs about as much as random-300 with the cached gate (seconds per task, from Run
A and Run T), so the time estimate (about 1 hour with 12 jobs) does not change.

**Tests** (in `tests/test_repair_order.py`, written on the laptop, to be run on the home
PC before the run, as in the runbook):

- the default still counts skips, edit for edit;
- with real tries, the edits are a prefix of the random order, exactly `budget` of them
  reach the gate unless the errors run out, each round gates a default round's number,
  and the edits equal the first edits of trying every error;
- greedy refuses `count_skips=False`;
- the script builds 8 repair arms (random-300-real with `count_skips=False`), every row's
  tries add up, and the summary prints A1, the reading table and E1–E2 with the right
  direction words.
