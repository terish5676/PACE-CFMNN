# Pre-registration: refinement v2 confirmatory run (reframed RQ3 and RQ4)

Written, committed and pushed to GitHub (branch `refinement-v2`) before the run started. The commit that adds this file fixes the code:
- `pace_cfmnn/active.py` (`run_refinement`, `run_self_repair`);
- `pace_cfmnn/refine.py`;
- `pace_cfmnn/audit.py`;
- `scripts/run_refinement_v2.py`.

Nothing is changed after the run starts. Any bug found later is reported in a "Deviations" section of the summary, with its effect on the results.

The team started the run without waiting for the guide's review. The team's v2 proposal explains the reframing.

## Why this run

v1 (seeds 10–19; `results/refinement_confirmatory/`, recomputed in `summary_v1.md`) answered the questions it registered:
- **The explanation's cut was no better.** It did not beat the smallest cut (R1) or a random cut (R3).
- **Shrinking helped.** Shrinking behind the gate beat adding points on all 6 datasets (R2).

But v1 cannot say where the gains come from:
- **Every training error is a tie.** Each one has membership 1 in a box of its own class and in a box of the wrong class, and the centroid rule picks the wrong box. This holds for 100% of training errors on all 6 datasets, at seeds 0 and 10. So the same shrink can be applied to training errors with no labels bought.
- **Not every label was counted.** The gate reads the labels of the acceptance rows (548–24,506 per dataset), and v1 did not count them.
- **No other learner got the same labels.**

v2 asks:
- **RQ3.** With every label counted, does buying labels improve CFMNN beyond repairing its own training errors? And does repair keep the lowest class recall?
- **RQ4.** Before an edit is applied, can we compute a set of inputs (the *certificate*) that contains every prediction the edit changes? Of the predictions repair changes, are more of them corrections than with retraining on the same labels?

## What was known before this run

**Exploratory checks E1–E4.** These used seeds 0–2 and the same 6 datasets, and are descriptive only (`results/refinement_v2_explore/summary.md`).

Mean change in test macro-F1 against B0, in points:

| Arm (exploratory name) | Labels beyond the training rows | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|---|
| SR-300-train (SR here) | 0 | +6.27 | +1.16 | +0.80 | +2.55 | +2.50 | +8.51 |
| OE \| A+B+C′-volume (AL here) | budget + acceptance rows | +6.11 | +0.57 | +1.24 | +4.99 | +2.57 | +8.90 |
| SR-all, then AL | budget + acceptance rows | +6.75 | +0.30 | +0.91 | +4.48 | +4.53 | +9.38 |
| CFMNN refit, same labels | budget + acceptance rows | −0.62 | −0.28 | −4.33 | +0.57 | +0.66 | −1.13 |
| RF, training rows only | 0 | +12.50 | +6.20 | +7.68 | +12.05 | +20.68 | +9.85 |

**E1, self-repair.**
- Gating on the training rows uses no labels at all. That version was the best of the three self-repair versions on 5 of 6 datasets. The other two, SR-300 and SR-all, gate on the acceptance rows.
- So SR here is that version, with AL's budget: SR and AL try the same number of points (300; 273 on Page Blocks).

**E2, which feature to cut.**
- The oracle cut tries every feature and keeps the one the gate scores best. Against the smallest cut, it gained between −0.57 and +1.13 points.
- PACE-SHAP's cut fell short of the smallest cut on 5 of 6 datasets.
- So the smallest cut (C′-volume) is kept. The fewest-members cut was best on Page Blocks (+4.66 against B0), but it was never tried with self-repair, so it is not used.

**E3, which points to buy.**
- kNN disagreement and the order ensemble were within 1.4 points of each other on every dataset.
- The order ensemble is kept. It is the RQ2 detector and v1's selector.

**E4, certificates.**
- **No misses.** 69,488 accepted edits in 252 edit-based runs changed no prediction outside the tight or the box-level certificate.
- **Loose.** The tight certificate held 1–36% of the audit rows per edit. Of the rows it held, 0.0–9.4% actually changed.
- **Explanations not covered.** Outside the tight certificate, the PACE-flip top-3 set still changed on 0–63% of the explained rows.

**E4, why P4 is the fix share.**
- **Repair changes more, because it fixes more.** On most datasets, repair changed more test predictions than refitting on the same labels (1.4–24% against 0.6–7.9%; Page Blocks was the exception).
- **Broken predictions alone favour doing little.** Refitting gained nothing in macro-F1. Yet SR-all then AL broke more predictions than refitting did on Landsat, MAGIC and Letter.
- **The fix share separates them.** The fix share is fixed / (fixed + broken). It was higher for SR-all then AL than for refitting on all 6 datasets (53–98% against 12–60%).
- **The cost is still reported.** Broken predictions are kept as a secondary endpoint (S4a).

**Smoke tests on seed 100.** These ran on Thyroid and Page Blocks and are excluded from all analysis. They checked the code and the run time. Their summaries printed test results, but every design choice in this document comes from E1–E4.

## Fixed settings

- **Seeds:** 50–59, never used for refinement before:
  - v1 used 10–19;
  - the pilot and E1–E4 used 0–2;
  - the smoke tests used 100.
- **Model, split, budget, gate, shrink and datasets:** v1's, unchanged (`results/refinement_confirmatory/PREREGISTRATION.md`), through `run_refinement.split`, `active.run_refinement` and `refine.py`:
  - CFMNN with θ = 0.3 and γ = 4;
  - role split 50/20/10/10/10;
  - budget min(300, a quarter of the pool) in 10 rounds: 300 everywhere except Page Blocks (273);
  - gate on each edit:
    - macro-F1 may fall by at most 0.5 points;
    - no class's recall may fall by more than 1 point;
    - boxes ≤ 1.1 × the count at the start of that arm;
    - a failed edit is rolled back;
  - the shrink moves the chosen face 0.001 past the point, on the feature where the cut removes the smallest share of the box's width (C′-volume);
  - datasets: Thyroid, Landsat, Page Blocks, MAGIC, Letter and Skin (in full).
- **Audit rows:** a fixed sample of min(2,000, test size) test rows per seed (generator seed + 3).
  - They are used for the certificates.
  - PACE-flip explains the first 100 of them, with the first 200 training rows as background.
- **Execution:** `python scripts/run_refinement_v2.py --jobs 4`.
  - After an interruption, add `--resume`. It reruns only the unfinished dataset × seed jobs, and each job is deterministic.

## Arms

| Arm | What it does | Labels beyond the training rows |
|---|---|---|
| B0 | No refinement | 0 |
| B9 | CFMNN refit on the training rows and the whole labelled pool (v1's reference) | Pool size |
| **SR** | Self-repair: the smallest-cut shrink, tried on `budget` of the model's own training errors in a seeded random order, gated on the training rows. A row already right by its turn is skipped; nothing is added. | **0** |
| **AL** | Order-ensemble selection, add, and smallest-cut shrink (v1's A+B+C′-volume), gated on the acceptance rows | Budget + acceptance rows |
| **SR+AL** | SR, then AL on the repaired model | Budget + acceptance rows |
| **Retrain, shuffled** | CFMNN fit on the training rows, SR+AL's bought labels and the acceptance rows, in a fresh random order | Budget + acceptance rows |
| Retrain, appended | The same rows, with the new ones appended after the training rows | Budget + acceptance rows |
| kNN / RF, training rows | kNN (k = 10) and a Random Forest (100 trees) on the training rows | 0 |
| kNN / RF, every label | The same learners on every row SR+AL read | Budget + acceptance rows |

## Endpoints

**How they are measured:** on the whole test set, after the last round.

**Test:**
- a paired Wilcoxon signed-rank test over the 10 seeds of each dataset (scipy defaults; zero differences are dropped);
- Holm correction across the datasets of each endpoint;
- significance level 0.05.

**Primary endpoints:**
- **P3 (RQ3).** Test macro-F1, SR+AL − SR, two-sided.
  - The question: do bought labels add anything after self-repair?
- **P4 (RQ4).** Fix share, SR+AL − Retrain shuffled, two-sided.
  - **Fix share:** of the test rows whose correctness changed from B0, the share that went from wrong to right, i.e. fixed / (fixed + broken).
  - If an arm changes no row's correctness in a seed, that seed is dropped from P4 for that dataset. The number of seeds tested is reported.

**Secondary endpoints:**

| Endpoint | Value | Comparison | Test |
|---|---|---|---|
| S3a | Test macro-F1 | SR − B0 | Two-sided |
| S3b | Test macro-F1 | SR − AL | Non-inferiority: one-sided test that SR − AL > −1 point |
| S3c | Lowest class recall | SR+AL − B0 | Non-inferiority, margin 2 points |
| S3d | Lowest class recall | SR − B0 | Non-inferiority, margin 2 points |
| S4a | Broken share: test rows B0 got right and the arm gets wrong (% of test rows) | Retrain shuffled − SR+AL | Two-sided |
| S4b | PACE-flip top-3 change share, 100 test rows | Retrain shuffled − SR+AL | Two-sided; Skin not tested (3 features, so its top-3 set is every feature) |
| S4c | Test macro-F1 | SR+AL − Retrain shuffled | Two-sided |

**Margins:**
- **1 macro-F1 point** is the largest loss we would accept in exchange for buying no labels at all.
- **2 points for the lowest class recall**, because one class's recall moves more between seeds than macro-F1 does.

**Check C4 (must hold):**
- Every change to an audit row's prediction made by an accepted edit of AL, SR or SR+AL must lie inside that edit's tight certificate and its box-level certificate.
- A single miss means a bug. The certificate claim is then withdrawn until the bug is found, and the bug is reported.

**Descriptive only:**
- test accuracy and balanced accuracy;
- labels used;
- the share of the gap to RF (with every label) closed;
- changed, fixed and broken shares;
- the PACE-flip top feature changed;
- certificate size per edit, precision and union;
- PACE-flip change outside the certificates;
- training error at the end;
- accepted edits;
- run time;
- B9, Retrain appended and kNN.

## How results will be read

**RQ3, on each dataset:**
- Bought labels add value after self-repair if P3 is significantly positive.
- Self-repair alone is enough there if S3b holds and P3 is not significantly positive.

**RQ4:**
- Repair's changes are more targeted than retraining's if P4 is significantly positive.
- S4a and S4b are reported as the cost side, whatever they show.
- C4 decides whether the certificate claim stands.

**Reporting:**
- Every endpoint is reported for every dataset, whether it supports the hypothesis or not.
- E1–E4 stay as they are and are reported as exploratory.

## Expected run time

About 3.5–4 hours with 4 workers on an i7-8650U, estimated from E1–E4's timings. Letter and Skin take the longest.
