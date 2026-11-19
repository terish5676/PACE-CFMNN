# Run F: self-repair against contraction: pre-registration

Written on 2026-11-13 and pushed, with the code it describes, before the run.

- **The code:**
  - `scripts/run_contraction.py` runs it;
  - `pace_cfmnn/contraction.py` (new) holds the overlap tests and contractions;
  - the tests are in `tests/test_contraction.py`.
- **Self-repair:** `run_self_repair` from `pace_cfmnn/active.py`, with the edits in
  `pace_cfmnn/refine.py`. These files last changed in commit a4c08f1 (Run E, which added the
  cached gate; it is off by default and is not used here).

## Why

The peer review (`reviews/fable_peer_review.md`, weakness B3 and question 2) asks how self-repair
compares with the standard fix for class overlap in fuzzy min-max networks. CFMNN trains by
expansion only, so boxes of different classes overlap, and every training error is a tie inside
such an overlap (Proposition 1). Two fixes exist for that:

- **Simpson's overlap test and contraction** (Simpson 1992), which classic fuzzy min-max runs
  after every expansion;
- **GFMM's online learning**, which contracts while it trains.

Self-repair instead shrinks the wrong-class box only at training errors, and keeps an edit only
if a gate passes. The paper has never compared the two. This run does, on RQ3's datasets and
split. It also closes audit item N1: the first repair run's "B-Simpson" arm had no code.

## What ran before this was written

All of it was on seed 99, which is not a study seed. All of it was thrown away.

- **Overlap counts on thyroid, page_blocks and landsat.** CFMNN at θ = 0.3, on the training
  rows of RQ3's split. For each pair of boxes of different classes, it counted:
  - pairs whose closed boxes meet: 130, 44 and 1,113;
  - pairs that also meet strictly (with open intervals): 0, 43 and 857;
  - pairs Simpson's published test finds: 0, 15 and 64.

  Simpson's test uses strict inequalities, so it misses boxes that share a face, and boxes
  that are equal on a feature. Many CFMNN boxes are flat on many features: on thyroid, half of
  all box sides have zero width. On thyroid, Simpson's test finds nothing to contract.
- **Both contraction rules on those three datasets.** Test macro-F1, contractions, rows
  evicted and seconds were printed.
- **Timing of the contraction arms** on letter, skin and magic. It printed seconds only, no
  accuracies.
- **Smoke runs of the script:**
  - thyroid, seed 99, with an earlier version that had no GC arms. Its third contraction arm
    (ungated contraction at every training error) was dropped later; see below;
  - page_blocks and landsat, seed 99, with the final version.

  Each printed every arm's test macro-F1 for seed 99. Output went outside the repository.
- **The unit tests,** on synthetic data only.

**What changed because of these:**

- **GC-all and GC-err were added** after the overlap counts showed that Simpson's published
  test finds nothing on thyroid. Alone, it would be a weak comparator there. They use GFMM's
  published overlap test and contraction, which counts shared faces.
- **The "at every training error" arm was dropped,** to keep the number of arms down. With a
  300-error budget, SC-err and SR try the same rows, which is the like-for-like comparison.
- **The primary comparison did not change:** self-repair against Simpson's contraction on every
  overlap. Neither did any setting of self-repair, GFMM or the split.

The seed-99 accuracies favoured self-repair on page_blocks and landsat. These were seen before
this was written, so they are not evidence; only seeds 120–129 are.

## Setup

- **Datasets:** RQ3's six: thyroid, landsat, page_blocks, magic, letter, skin.
- **Seeds:** 120–129 (10), new.
- **Data:** RQ3's split (`split` in `scripts/run_refinement.py`):
  - `data.role_split`, a stratified 50/20/10/10/10 split;
  - the training rows in the seed's order;
  - min-max scaling fitted on the training rows.
- **Rows used:** only the training rows (50%) and the test rows (10%). Every arm uses the same
  ones. No arm buys labels, and none uses the pool, inner or acceptance rows.
- **Starting model:** CFMNN with θ = 0.3, γ = 4, and the fixed centroid rule.

## Arms

| Name | What it is |
|---|---|
| **B0** | The starting model |
| **SR** | B0, then self-repair: RQ3's SR arm |
| **SC-all** | B0, then Simpson's overlap test and contraction on every pair of boxes of different classes |
| **SC-err** | B0, then Simpson's test and contraction only at training errors: the same rows, in the same order, as SR. No gate |
| **GC-all** | As SC-all, with GFMM's overlap test and contraction rule |
| **GC-err** | As SC-err, with GFMM's overlap test and contraction rule |
| **GFMM** | hyperbox-brain's `OnlineGFMM`, trained on the same rows in the same order. It runs its own overlap test and contraction while it trains |

### Self-repair (SR)

Called as in RQ3:
`run_self_repair(B0, X_train, y_train, X_train, y_train, chooser="volume", budget=300, n_rounds=10, seed=seed, gate_labels_are_new=False)`.

- It tries up to 300 of the model's training errors, in a seeded random order, in 10 rounds.
- For each one, it cuts the winning wrong-class box on the feature where the cut removes the
  smallest share of the box's width.
- The gate reads the training rows. It keeps an edit only if:
  - macro-F1 falls by at most 0.5 points;
  - no class's recall falls by more than 1 point;
  - the boxes stay at most 1.1 times the start.

### Simpson's contraction (SC-all, SC-err)

The four cases of Simpson's (1992) overlap test, with his strict and non-strict inequalities,
as written in `pace_cfmnn/contraction.py`:

- the boxes overlap only if one case holds on every feature;
- the feature with the smallest overlap is contracted;
- the lowest feature wins a tie.

The contraction:

- in cases 1 and 2, both boxes move to the middle of the overlap;
- in cases 3 and 4, the outer box gives up the shorter side;
- the boxes still touch on the new face, as in Simpson's paper.

**SC-all** tests every pair of boxes of different classes, in box order, in sweeps until a sweep
contracts nothing.

**SC-err** takes the training errors in the order SR uses (the same seed, the first 300). For
each row, on the model as it is by then, it is skipped if:

- the model now gets the row right;
- the row is not inside the winning box;
- the row is not inside any box of its own class;
- the test finds no overlap between the winning wrong-class box and the row's best own-class box.

Otherwise that pair is contracted and kept, with no gate.

### GFMM's rule (GC-all, GC-err)

The same procedure as SC-all and SC-err, with a different test and contraction, both from
hyperbox-brain 0.1.6 (Khuat and Gabrys 2021):

- the test is `is_two_hyperboxes_overlap_num_data_general`, which uses closed intervals, so it
  counts shared faces;
- the contraction is `overlap_resolving_num_data`: Simpson's four cases with non-strict
  inequalities, four shared-face cases, and a gap of 1e-5 after each contraction.

This is GFMM's rule applied to the trained CFMNN. It is not the same as training GFMM.

### After every contraction

As after a self-repair shrink (`refine.shrink_box`):

- each changed box keeps only the training rows still inside it;
- its Sum, Count and centroid are recomputed from them;
- a box left with no row is removed.

The rows dropped this way are **evicted**: they belong to no box. `validate_structure` is checked
in the tests.

### GFMM

`OnlineGFMM(theta=0.3, theta_min=0.3, gamma=4)`, as in the readable-baselines run: one pass at
θ = 0.3, with no shrinking loop.

- GFMM's θ caps each side of a box. CFMNN's caps the sum of the sides.
- So GFMM is a reference point, not a like-for-like arm.

## Recorded

For each dataset, seed and arm (`runs.csv`, one row per arm):

- test macro-F1, accuracy and lowest class recall, in %;
- boxes at the start and at the end;
- training rows evicted from their boxes (count, and % of the training rows);
- training rows inside a box of another class at the end;
- training error at the end;
- test predictions changed, fixed (wrong → right) and broken (right → wrong), against B0;
- edits or contractions tried and kept, boxes removed, Simpson's case of each contraction,
  sweeps, and the training errors skipped (and why);
- seconds: fitting for B0 and GFMM, the repair or contraction step for the others. Twelve
  processes share the CPU, so times are compared within a dataset only.

For GFMM, evicted rows and rows inside a box of another class are not recorded: its library
keeps no members.

## Comparisons

Each comparison is paired over the 10 seeds:

- a two-sided Wilcoxon signed-rank test (`test_endpoint` in `scripts/run_refinement_v2.py`);
- Holm-corrected across the 6 datasets of that comparison (`holm` from the same script);
- p = 1 when all 10 differences are zero.

The first arm's value is **higher** on a dataset if the Holm p is below 0.05 and the mean
difference is positive, and **lower** in the other direction. Otherwise there is **no
significant difference**. That is not the same as equal: no equivalence test is made.

| Code | First − second | Measure | Question |
|---|---|---|---|
| **P1 (primary)** | SR − SC-all | macro-F1 | Self-repair or Simpson's contraction on every overlap: which is more accurate? |
| S1 | SR − SC-err | macro-F1 | Self-repair or ungated contraction at the same training errors? |
| S2a | SR − GC-all | macro-F1 | Self-repair or GFMM's contraction rule on every overlap? |
| S2b | SR − GC-err | macro-F1 | Self-repair or GFMM's contraction rule at the same training errors? |
| S3 | SR − GFMM | macro-F1 | Self-repair or GFMM, which contracts while it trains? |
| S4a–f | each of SR, SC-all, SC-err, GC-all, GC-err, GFMM − B0 | macro-F1 | Does each one help? |
| S5 | SR − SC-all | lowest class recall | The primary pair, on the worst class |
| S6 | SR − SC-all | boxes at the end | Which leaves fewer boxes? |
| S7 | SR − SC-all | test predictions broken (% of test rows) | Which breaks fewer correct predictions? |
| S8 | SR − SC-all | training rows evicted (%) | Which evicts fewer training rows? |
| S9 | SR − SC-all | accuracy | The primary pair, by accuracy |

- The primary comparison is the one the review asks for.
- The secondaries are not corrected for each other or for the primary. They are read as
  support, not as separate findings.
- No direction is predicted.

**Descriptive:** the means over seeds of every recorded value, by arm and dataset, and the count
of each of Simpson's cases.

## What the result will mean

- **If SR is higher than SC-all on macro-F1 on most datasets,** the paper may say that gated
  repair at training errors is more accurate than Simpson's contraction on this model.
- **If there is no significant difference, or SC-all is higher,** the paper's case for
  self-repair rests on what contraction does not give:
  - the gate;
  - the edit-level certificates;
  - the record of which errors were repaired.

  The paper then says so, and does not claim an accuracy gain over contraction.
- **Either way,** the evicted rows (S8) and broken predictions (S7) are reported beside the
  accuracies. This also answers review weakness B7, on evicted rows.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.
- Every number the paper takes from this run is recomputed from `runs.csv` by a read-only
  script.

## How to run

On the home PC (i5-14400F), from the repository root, with the data folder in place:

```
python -B scripts/run_contraction.py --jobs 12
```

- **Tasks:** 60 (6 datasets × 10 seeds). Letter starts first.
- **Expected time:** about 30–45 minutes. Letter's self-repair (about 17 minutes per seed in
  RQ3, on the laptop) and GFMM's fit (a few minutes per seed on Letter) set the length. Every
  contraction arm takes seconds.
- **Output:**
  - `runs.csv` gets each dataset-seed pair's rows as it finishes. A stopped run continues with
    `--resume`. `--fresh` deletes the results and starts over;
  - `summary.md`. `--summary-only` rebuilds it from `runs.csv`.

## Amendment 1 (2026-11-14, before the run)

Written and pushed before any study seed (120–129) was run. Nothing above this section was
edited; where this section differs, this section holds. **No code changed.**

**Why.** A review of the plan and code, before the run, found no bug, and three statements
that need correcting or disclosing.

**What was known when this was written.** The plan above; its seed-99 runs; and a smoke
slice on **seed 191** (not a study seed, never analysed), on the laptop, written outside the
repository: Thyroid and Page Blocks, all seven arms (20 s). It printed each arm's test
macro-F1 for seed 191. One seed is not evidence. No study seed had been run.

**1. The self-repair budget is 300 on every dataset, which is not RQ3's budget on Page
Blocks.**

- The plan calls SR "RQ3's SR arm" and "called as in RQ3", and gives the call with
  `budget=300`. The code does what the call says: 300 on all six datasets.
- RQ3 used `min(300, pool // 4)`, which is **273 on Page Blocks** and 300 elsewhere.
- This matters on Page Blocks: in Run H (seeds 130–139) 5 of the 10 seeds had more than 273
  training errors there, so 300 tries reach errors that 273 would not.
- **Nothing changes in the run.** SR and SC-err both use 300, on the same rows in the same
  order, so the comparisons between arms are like for like. What changes is the wording:
  **on Page Blocks, SR here is RQ3's self-repair with a budget of 300, not RQ3's SR arm
  exactly,** and its numbers are not to be set beside RQ3's or Run H's Page Blocks numbers as
  if they were the same arm. On the other five datasets it is RQ3's call.

**2. Seeds 120–129 are also Run G's seeds.** Both plans say no earlier study used them. The
two runs use different splits (this run: RQ3's 50/20/10/10/10 split; Run G: the paper's
80/20 split) and neither had run when the other was written, so no result of one could shape
the other. Disclosed; no change.

**3. How the GFMM-rule arms (GC-all, GC-err) will be read.** Written now, before the run:

- GFMM's overlap test counts shared faces, and CFMNN's boxes share many. Applied once to a
  trained CFMNN, the rule can cut away most of a box. On seed 191 GC-all evicted 2,352 of
  2,736 training rows on Page Blocks and 618 of 3,600 on Thyroid, and its macro-F1 was far
  below the starting model's.
- So S2a and S2b (SR against GC-all and GC-err) are reported with the evicted rows beside
  them, and **a win for SR there is not presented as a win over GFMM,** nor as evidence about
  contraction in general. S3 (SR against GFMM trained from scratch) is the comparison with
  GFMM.
- On Thyroid, Simpson's test found no overlap on seed 191 (as on seed 99), so SC-all was the
  starting model there and SC-err contracted nothing. If that holds on the study seeds, P1
  and S1 on Thyroid are the same comparison as S4a (SR against B0), and the paper says so.

**What stays the same.** Every arm, setting, seed, dataset, measure and comparison, and the
primary (P1: SR against SC-all on macro-F1).
