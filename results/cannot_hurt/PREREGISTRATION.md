# Run P11, edits that provably cannot hurt: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review B6,
part 2). The code is `scripts/run_cannot_hurt.py` (tests in `tests/test_cannot_hurt.py`). It uses:
- `audit.training_safe_shrink` (condition S, commit 79c5584, with clause (iii) added in this
  commit; see below);
- a new `on_try` hook in `active.run_self_repair`, which is called after every gated edit,
  accepted or not, before a rejected edit is rolled back. A test shows the hook changes nothing:
  the repaired model's boxes and predictions are identical with and without it.

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 13.

## Background (in plain words)

Self-repair shrinks a box to fix one of its training errors. After each shrink, a *gate* re-scores
the whole training set and keeps the edit only if macro-F1 and the lowest class recall did not
drop too far. **Condition S** is a cheap test of whether a shrink cannot break any correct training
prediction. It holds when the edit changes only that box and keeps its centroid, and all three of
these are true:

- (i) no training row of the box's own class lies in the slab the shrink cut away;
- (ii) no own-class row outside the old box, whose membership in the box fell, was won by that box
  before the edit;
- (iii) no training row predicted correctly and won by another box has a higher membership in the
  shrunk box than in its winner.

Proposition B6 says that a shrink meeting S breaks no correct training prediction. If many
accepted edits met S, the gate could be skipped for them (a *fast path*).

**Clause (iii) was added while writing the proof, before this run.**
- The first version had only (i) and (ii). The proof's step "a row the box did not win keeps its
  winner" fails for near-ties. CFMNN treats memberships within `tie_tol` (1e-9) of the top as
  tied, and picks the nearest centroid among them.
- Here is how it fails. Suppose the shrunk box is strictly on top, and another box wins within the
  tolerance. Lowering the top can let a third box into the tied group, and that box can win.
- `tests/test_tough_audit.py` now has such a case: one feature, `tie_tol` = 0.1. There, (i) and
  (ii) hold and a correct prediction still breaks. Clause (iii) catches it.
- With `tie_tol` = 0, clause (iii) always holds.

## What ran before this was written

- **The reanalysis of saved edit logs** (tough review B6, part 3) suggested that S rarely holds. A
  box's faces sit on its own members, so moving a face inward usually cuts one away.
- **Smoke slice on the laptop, seed 194** (not a study seed, never analysed). Written outside the
  repository; Thyroid, Page Blocks and Skin; 3–190 s per dataset:

  | Dataset | Accepted edits | Meet S | Gated edits meeting S that the gate rejected |
  |---|---|---|---|
  | Thyroid | 14 | 0 | 0 |
  | Page Blocks | 18 | 3 (16.7%) | 0 |
  | Skin | 43 | 1 (2.3%) | 0 |

  Check: PASS. No edit meeting S broke a correct training prediction. Of the accepted edits, 83–100%
  cut away own-class rows, which fails part (i).

  The slice was run again after clause (iii) was added (13 s, 39 s and 381 s, with other runs
  sharing the laptop). The counts are the same. One accepted Skin edit fails clause (iii), and it
  already failed another part of S.

The design did not change after the slice.

## The question

**How many accepted self-repair edits meet condition S, and does every one of them leave every
correct training prediction intact?**

## Setup

- **Datasets:** RQ2's six: Thyroid, Landsat, Page Blocks, MAGIC, Letter and Skin.
- **Seeds:** 390–399 (new).
- **Split, model, budget and rounds:** the refinement run's, unchanged (`run_refinement.split`;
  CFMNN at θ = 0.3; budget min(300, pool/4); 10 rounds).
- **Repair:** gated self-repair on the training rows, exactly as `run_refinement_v2.run_one`'s SR
  arm.
- **What the hook records, per gated edit:**
  - whether S holds, and which part fails;
  - whether the gate accepted the edit, and the gate's time;
  - how many correct training predictions the edit broke, and how many errors it fixed.
- **Output:** `results/cannot_hurt/edits.csv`, one row per gated edit.

## Primary (descriptive)

The share of accepted edits that meet S, per dataset, pooled over the 10 seeds. It comes with a
95% Clopper–Pearson interval on the pooled count (the summary gives the counts; the write-up adds
the interval).

## Check (pass/fail)

No edit meeting S, accepted or not, breaks a correct training prediction.

## Second stage (the fast path)

If the gate never rejects an edit meeting S, then skipping the gate for those edits changes no
decision, so the fast path gives the same model. Its only saving is the gate time on those edits,
which the summary reports.

The fast path is therefore run as its own model only on a dataset where both hold:

- S holds for at least 10% of accepted edits; and
- the gate rejected at least one edit meeting S.

That stage's code and its own short pre-registration would be pushed first. Its primary test
would be the review's: macro-F1 of the fast path against gated self-repair, one-sided
non-inferiority with a margin of 0.5 points, Wilcoxon over the 10 seeds, Holm across the datasets
where it runs.

**A deviation from the review's sketch, stated here.** The review made that non-inferiority test
the primary. Where the gate never rejects an edit meeting S, the fast path and gated self-repair
make every decision the same way. The test would then compare two identical models and pass by
construction. So this run makes the share of edits meeting S the primary, and keeps the
non-inferiority test for the datasets where the two paths can differ.

## How the result will be read (written now, before the run)

- **Check FAIL:** Proposition B6 (part 2) is wrong as stated. It is withdrawn from the paper until
  the proof is fixed, and the failing edit is reported.
- **Check PASS, and S holds for under 10% of accepted edits on every dataset:** the paper keeps the
  proposition as a safety guarantee. It says plainly that S rarely holds in practice, because a
  box's faces sit on its own members, so the gate cannot usually be skipped. The fast path is
  given as a "next step" line only.
- **Check PASS, and S holds for at least 10% on some dataset:** the paper reports the share and the
  gate time saved there. The second stage runs only if the gate also rejected an S edit there.

## Time

The laptop smoke slice took 3 s (Thyroid), 11 s (Page Blocks) and 190 s (Skin) for one seed.
MAGIC and Letter are the slowest refinement datasets: self-repair takes several minutes per seed.
**About 1–2 hours on the home PC with 6 jobs.** Stop and ask if it is past 4 hours.

## Tests

`tests/test_cannot_hurt.py`:
- the hook changes nothing (same boxes and predictions) and sees every gated edit;
- the check fails on a fake edit that meets S and breaks a prediction;
- the second-stage rule.

## How to run (home PC, batch 2)

```
.\.venv\Scripts\python -B scripts/run_cannot_hurt.py --jobs 6 *>&1 | Tee-Object results/cannot_hurt/run_log.txt
.\.venv\Scripts\python -B scripts/run_cannot_hurt.py --jobs 6 --resume *>&1 | Tee-Object -Append results/cannot_hurt/run_log.txt   # after a stop
```

Commit `results/cannot_hurt/` unedited (`edits.csv`, `summary.md`, `run_log.txt`) as "Run P11:
results of the pre-registered run on the home PC".
