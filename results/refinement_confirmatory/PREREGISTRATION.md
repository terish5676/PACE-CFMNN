# Pre-registration: refinement confirmatory run (Phase 6, RQ3–RQ4)

Written and pushed to GitHub before the run started. The commit that adds this file fixes the code:
- `pace_cfmnn/cfmnn_core.py` (`add_points`, `_remove_box`);
- `refine.py`, `active.py` and `signals.py`;
- `scripts/run_refinement.py --protocol confirmatory`.

Nothing is changed after the run starts. Any bug fix found later is reported here with its effect on the
results.

## Why this run

The RQ2 confirmatory run showed that PACE's trust signals are weak error detectors next to methods that
read the training labels. So the refinement study asks what PACE's explanations can do that those
methods cannot: **decide where to fix the model.**

When a labelled point is misclassified inside a box of the wrong class, one face of that box is moved
just past the point. The question is whether choosing that face with the explanation works better than
choosing it by geometry or at random.

## What was known before this run

- **The pilot** (seeds 0–2; S2, Thyroid, Landsat, Page Blocks and MAGIC; `results/refinement_pilot/`).
  It is exploratory. Its main results:
  - **Shrinking helps.** The smallest-cut shrink beat add-only in 12 of the 12 real dataset × seed
    runs; the explanation-guided shrink did so in 9 of 12. Refitting on the whole labelled pool (B9)
    barely helped.
  - **The explanation did not choose better.** The explanation-guided shrink was below the
    smallest-cut control in all 3 seeds on Thyroid and MAGIC, and mixed on Landsat and Page Blocks.
    So the pilot does not favour the primary hypothesis. The endpoint is kept as the blueprint
    planned it.
  - **Selection barely mattered with add-only.** Random, margin and PD + U\* all landed within about
    ±0.5 macro-F1 points.
    - So the selection test now compares the order ensemble with random selection, both with the
      explanation-guided shrink (the team's decision after the pilot).
    - Margin and PD + U\* are dropped; their pilot result is reported as exploratory.
  - **Synthetic data:** no misclassified pool point lies inside a wrong-class box, so the shrink never
    applies (see the dataset rule below).
- **Code changed after the pilot:**
  - summary formatting;
  - `--jobs` (parallel runs; identical results, checked on 4 pilot dataset × seed runs);
  - the `eligibility` and `confirmatory` protocols.
  - The loop, the edits and the gate are unchanged.
- **A smoke test** of Letter and Skin on seed 100 (excluded from all analysis), run to check the code
  and the run time. Its log printed test results. No definition was changed after it.

## Fixed settings

- **Seeds:** 10–19 (never used for refinement before).
- **Model:** CFMNN, fixed centroid rule + search expansion, θ = 0.3, γ = 4 (the main model).
- **Split:** the role split 50/20/10/10/10 (`data.role_split`).
  - Training rows are shuffled with the seed. The scaler is fitted on them, and all rows are clipped
    to [0, 1].
  - The model trains on the training rows.
  - The pool's labels are revealed only for selected points.
  - The gate uses the acceptance rows.
  - Test rows are only measured.
- **Budget:** min(300, a quarter of the pool) labels, in 10 rounds of equal size. That is 300 on every
  dataset except Page Blocks (273).
- **Gate**, compared with the model before each edit, on the acceptance rows:
  - macro-F1 may fall by at most 0.5 points;
  - no class's recall may fall by more than 1 point;
  - boxes ≤ 1.1 × the starting count.
  - A failed edit is rolled back. A failed shrink falls back to the plain add, gated the same way.
- **Shrink:** applies when the selected point is misclassified and lies inside the winning wrong-class
  box.
  - The face of that box nearer to the point, on the chosen feature, moves 0.001 past the point.
  - The box keeps only its training members still inside it. A box left empty is removed.
  - Then the point is added with its true label.
- **Datasets:** the datasets of at least 2,000 rows (so the role split applies) where every pilot
  seed's pool holds at least 25 points that the starting model gets wrong while they lie inside the
  winning wrong-class box (`results/refinement_pilot/eligibility.md`; no refinement result is used).
  - **Included:** Thyroid, Landsat, Page Blocks, MAGIC Gamma Telescope, Letter Recognition and Skin
    Segmentation.
  - **Excluded:**
    - the synthetic sets: 0 such points;
    - Shuttle: 18–20.
  - **Not considered:** datasets under 2,000 rows (Wine, Breast Cancer, Digits). The blueprint asks
    for nested cross-validation there, and it is not implemented.
  - **Skin is used in full** (245,057 rows), although the blueprint suggests a subsample for its
    refinement loop (the team's decision; it makes the run about 7 hours instead of 3.5).
- **Execution:** `python scripts/run_refinement.py --protocol confirmatory --jobs 4`. Parallel runs
  give results identical to a sequential run; only the timing columns differ.

## Arms

| Test | Selection | Operations |
| --- | --- | --- |
| Repair (RQ4) | order ensemble | A+B · A+B+C · A+B+C′-random · A+B+C′-volume |
| Selection (RQ3) | order ensemble · random | A+B+C |
| References | B0: no refinement · B9: refit on training + the whole labelled pool | — |

- **A+B:** add the point with CFMNN's own training rule.
- **C:** the shrink feature is the largest term of the contrastive explanation "why the wrong class,
  not the true one".
- **C′-random:** a random feature.
- **C′-volume:** the feature where the cut removes the smallest share of the box's width.

## Endpoints

**Metric:** test macro-F1 (%) after the last round.

**Test:** a paired two-sided Wilcoxon signed-rank test over the 10 seeds per dataset, with Holm
correction across the datasets of each comparison. Significance level 0.05.

**Primary endpoint (RQ4):**
- **R1.** A+B+C vs A+B+C′-volume, on each repair-test dataset. Blueprint hypothesis: A+B+C is higher.

**Secondary endpoints:**
- **R2.** A+B+C vs A+B: does shrinking help at all?
- **R3.** A+B+C vs A+B+C′-random.
- **S1.** Order-ensemble selection vs random selection, both with A+B+C.
- **Descriptive:**
  - test accuracy and the lowest class recall;
  - box counts;
  - accepted and rejected edits by type;
  - learning curves over the rounds;
  - run time.

## How results will be reported

- Every endpoint is reported for every dataset, whether it supports the hypothesis or not.
- Excluded datasets are listed with the reason.
- The pilot stays as it is and is reported as exploratory.
