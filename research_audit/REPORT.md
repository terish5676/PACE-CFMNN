# Final review: report

Branch `final-review` (pushed), commits `0614e66` (the review) and the one after it (this report and the
changelog's commit ids). Paper: Claude Doc rev 262 → rev 282 (review), then rev 293 (reframe, at the
user's request), then rev 302 (readable baselines, at the user's request), then rev 311 (repair at the
θ chosen by validation, at the user's request). Date: 2026-11-06, updated 2026-11-08.

## What was done

- **Every claim in the paper checked.**
  - [CLAIMS.md](CLAIMS.md) has 134 rows covering well over 500 numbers. Each was recomputed from the committed `runs.csv` files, not the rounded summaries.
  - 122 rows were right as written.
  - 4 numbers were slightly wrong (P1–P4): a table cell off in the last digit, a range end off by one point, and two lists each missing one dataset.
  - 10 sentences said a little more than the evidence (W1–W10).
  - One claim, "every training error is a tie", had no committed source. A new post hoc check now backs it: 100% in all 18 cases (P5).
  - All of these are corrected in the paper (revs 263–280).
- **Every pre-registered endpoint is reported** in the paper, including the ones that went against our methods.
- **Equations against the code.** No mismatch was found. 13 new hand-checked tests cover:
  - the paper's worked example;
  - U and U\*;
  - Holm correction;
  - non-inferiority p-values;
  - fix share;
  - the gate's limits;
  - four edge cases.
- **Code.** No bug that changes any result was found. One log field means two things (C2, unused, Won't fix). Five scripts run in full when given `--help` (C3, Won't fix here).
- **Docs.**
  - FINDINGS and README now cover all three repair runs and match the paper.
  - README's 38 commands all exist, with valid flags.
  - `requirements.txt` covers every third-party import.
- **Related work.**
  - Each contribution names its closest prior work.
  - One citation was added: Quteishat and Lim, rule extraction from fuzzy min-max networks.
  - The newest references (three 2026 arXiv papers) and one claim about Khuat, Chen and Gabrys were checked on the web.

No conclusion changed. No pre-registered study was rerun. No committed result or pre-registration was edited.

## Issues and status

Full list: [ISSUES.md](ISSUES.md). Changes and revisions: [CHANGELOG.md](CHANGELOG.md).

| Status | Count | Items |
|---|---|---|
| Fixed | 22 | P1–P5, W1–W10, R1, R2, C1, D1–D4 |
| Won't fix | 3 | W11 (no change needed), C2 (unused log field), C3 (scripts without a parser; out of scope) |
| Needs you | 2 | N1, N2 (below) |

**What is left (Needs you):**
1. **N1: the protocol amendment's two arms in the first repair run.** "B-Simpson" and "C-minimal" have no code in the repository: `pace_cfmnn/contraction.py` is missing. The teammate should commit the code, or the team should mark those rows as not reproducible. About 30 minutes. The paper does not use them.
2. **N2: the paper is longer than 9 pages.** Choose a venue, then cut. Likely cuts: RQ1 extension details, RQ2 v2 bullets, the deviations paragraph. About 3–5 hours once the page limit is known.

Also open, outside this review's scope:
- the venue choice;
- the teammate's PR #17 and zip;
- the anonymised repository.

## Tests run

| What | Kind | Result |
|---|---|---|
| Full unit suite, `python -B -m unittest discover -s tests -t .` | full | 178 tests, all pass (155 s) |
| `tests/test_final_review.py` (part of the 178) | full | 13 new tests pass |
| Claims recomputation, `research_audit/scripts/claims_*.py` | re-analysis of committed CSVs | all numbers as in CLAIMS.md |
| Tie check, `check_tie_errors.py` (6 datasets × seeds 0, 10, 50) | new post hoc check | 100% of training errors are cross-class ties |
| README commands, `check_readme_commands.py` | small check | 38 of 38 exist, flags valid |
| Imports against `requirements.txt` | small check | 7 of 7 third-party packages listed |
| Smoke runs, `smoke_all.py` (1 dataset, 1 seed, few points; outputs outside the repository) | small check, numbers meaningless | 36 of 36 exit 0 ([SMOKE.md](SMOKE.md)) |
| The 6 scripts with no `--out`, run in full with writes redirected | full re-run of quick post hoc scripts | 8 of 8 outputs byte-identical to the committed files |

## Paper changes

19 small edits, revs 263–282, listed with their revisions in [CHANGELOG.md](CHANGELOG.md). The
title, RQs, contributions and section structure are unchanged. All citations use author names, with no
bracket citations and no TBD markers.

**Reframe, at the user's request, revs 283–293.**
- The abstract, intro, contributions and conclusion now lead with what is strong: exact values, never less faithful, the tie finding and certificates. They state the accuracy cost openly.
- The like-for-like result now reads "better on 3 of 13, level on the rest, never worse, 3.5–10× faster".
- One contribution was reworded and moved to second place.
- No result changed. The title, RQs and headings are kept. Details are in [CHANGELOG.md](CHANGELOG.md).

## Readable baselines (2026-11-06 to 2026-11-07)

A test planned in advance, at the user's request:
- rules in [PREREGISTRATION.md](../results/readable_baselines/PREREGISTRATION.md), pushed in `42a71d4` before the run;
- results in `ba21cd3`;
- write-up in [FINDINGS §9](../results/FINDINGS.md) and the paper at rev 302.

The run took 11 h 15 min, with no failures and no deviations.

**Result.** CFMNN is not the most accurate readable model, nor the smallest. Each count below gives the datasets where CFMNN was better / level / worse.

| CFMNN against | θ tuned | θ = 0.3 |
|---|---|---|
| EBM (main effects only) | 2 / 2 / 9 | 0 / 1 / 12 |
| Decision tree | 5 / 0 / 8 | 3 / 0 / 10 |
| Logistic regression | 7 / 2 / 4 | 4 / 2 / 7 |
| GFMM at the same θ = 0.3 | – | 4 / 3 / 6 |

- θ chosen by validation beat θ = 0.3 on 7 datasets, by up to 20 points (Letter). There it mostly chose θ = 0.05, with 6–23× as many boxes.
- The tuned tree had fewer leaves than CFMNN at θ = 0.3 had boxes on 10 of 13 datasets.

**Tests run.**

| What | Kind | Result |
|---|---|---|
| The run | full, pre-registered | 2,080 of 2,080 rows, exit code 0 |
| `claims_readable.py` | re-analysis of the committed CSV | every verdict count matches the run's summary |
| Full unit suite | full | 190 tests, all pass (93 s) |
| `check_readme_commands.py` | small check | 39 of 39 commands exist, flags valid |

**How it changes the review below.**
- **Fuzzy systems:**
  - "No comparison with other hyperbox models" is now answered for accuracy: against GFMM, CFMNN is better on 4 datasets and worse on 6.
  - "How do the results change with θ?" is answered for accuracy only. RQ1–RQ4 still use θ = 0.3.
- **The accuracy weakness is sharper.** Reviewers can no longer be told CFMNN trades accuracy for readability against black boxes only, because an EBM and a tree beat it on most datasets. The paper now says so and rests the case on CFMNN's structure.
- **A new question (post hoc observation, not tested).** On the six repair datasets, a smaller θ raised accuracy by up to 20 points (Letter). Self-repair raised macro-F1 by 1.5–6.0 points. These are different seeds and measures. But reviewers may ask why one should repair at θ = 0.3 instead of choosing a smaller θ. The paper's answer would be the box count (6–23× more) and the certificates. A run of self-repair at the chosen θ would test it. **Now tested:** see the next section.
- **Grid ends.** Several tuned models often chose the end of their grid, so wider grids might change some counts.

## Repair at the θ chosen by validation (2026-11-07 to 2026-11-08)

A test planned in advance, at the user's request, of the question above:
- rules in [PREREGISTRATION.md](../results/repair_theta/PREREGISTRATION.md), pushed in `22e687d` before the run;
- results in `4dd2e59`;
- write-up in [FINDINGS §10](../results/FINDINGS.md) and the paper at rev 311 (the abstract now gives the result too).

The run took 15 h 4 min (the plan said about 11 h), with no failures and no deviations. Every model
without repair reproduced the readable-baselines run exactly.

**Result.** Choosing a smaller θ by validation is more accurate than repairing at θ = 0.3. Each count
gives the datasets where the first model was better / level / worse.

| Comparison | Accuracy | Macro-F1 |
|---|---|---|
| Repair at θ = 0.3 vs the chosen θ without repair (P1) | 1 / 5 / 7 | 1 / 6 / 6 |
| Repair at the chosen θ vs no repair there (P2) | 3 / 10 / 0 | 3 / 10 / 0 |
| Repair at θ = 0.3 vs no repair there (S1) | 7 / 6 / 0 | 7 / 6 / 0 |
| The chosen θ vs θ = 0.3, both repaired (S2) | 8 / 5 / 0 | 7 / 6 / 0 |

- Repair at θ = 0.3 falls short by up to 17.5 points (Letter). It wins only on Thyroid, where
  validation chose larger boxes.
- At the chosen θ, repair adds at most 1.6 points. On Letter, MAGIC and Landsat that θ leaves only
  2–11 training errors.
- Repair never added a box. The chosen θ needs 6–23× as many on six datasets.
- Every training error of all 520 models without repair is a cross-class tie.

**Tests run.**

| What | Kind | Result |
|---|---|---|
| The run | full, pre-registered | 1,040 of 1,040 rows, exit code 0 |
| `claims_repair_theta.py` | re-analysis of the committed CSV | every verdict count matches the run's summary |
| Full unit suite | full | 198 tests, all pass (104 s) |
| `check_readme_commands.py` | small check | 40 of 40 commands exist, flags valid |

**How it changes the review below.**
- **The open question is answered, against repair on accuracy.** Reviewers who ask "why not a smaller
  θ?" are right about accuracy: the paper now says so in §7. Its answer is size. Repair keeps the
  model small, while the smaller θ multiplies the boxes.
- **RQ3's result holds on a new split.** By macro-F1, repair at θ = 0.3 helped on all 6 of RQ3's
  datasets (+1.7 to +5.6 points, against RQ3's +1.5 to +6.0).
- **The edit budget is a limit.** At θ = 0.3 the 300 edits reached only a small share of the errors
  (300 of about 4,000 on MAGIC). Repairing more was not tried. The paper now says so in Limitations.

## Review from three angles

An honest reading of how reviewers would likely see the paper. It is not a prediction of the outcome.

### Fuzzy systems

**Strengths.**
- The paper uses the structure of the fuzzy min-max model itself: membership is an average of one-feature terms, so Shapley values can be computed exactly, and single-feature changes can be computed cheaply.
- The finding that every training error is a tie between boxes of two classes is a real insight into how these networks learn. It explains why a gated contraction ("self-repair") works without new labels.
- Repair is linked to Simpson's contraction step and to the general fuzzy min-max network, and its certificates follow from the locality of a box edit.
- Two fixes to the base CFMNN are found and reported, with the original rules always shown alongside.

**Weaknesses.**
- Only one model variant (CFMNN) is studied, at one setting (θ = 0.3, γ = 4).
- CFMNN is 0.2–21.5 points less accurate than the better of 10-NN and a Random Forest on all 13 datasets, and repair narrows but does not close this gap.
- There is no comparison with other hyperbox models, such as the general fuzzy min-max network, or with rule extraction as an explanation.
- The Simpson-contraction arm of the first repair run cannot be reproduced (N1).

**What reviewers would ask.**
- Does PACE carry over to other hyperbox models?
- How do the results change with θ?
- Why explain a model that is less accurate than simple alternatives? The paper's answer is transparency and auditable repair; reviewers will want that argued up front.
- How does self-repair compare with a standard contraction step run on the same errors?

### Explainable AI

**Strengths.**
- Exactness is shown, not assumed: PACE-SHAP matches general exact Shapley methods to rounding error, 16–30× faster. PACE-flip equals brute-force occlusion.
- The benchmark is fair and pre-registered. An earlier scoring error (ranking signed values by size) was found and withdrawn in the open.
- The paper reports the like-for-like result honestly: with the same rows and features, PACE-SHAP is more faithful on 3 of 13 datasets, level on the rest and never worse, and KernelSHAP takes 3.5–10× as long. It also reports that LIME ranks the truly relevant features best.
- The counterfactuals are checked on the model, land closer to real data, and are far faster than DiCE.

**Weaknesses.**
- Much of PACE-SHAP's lead comes from its setup (no estimation error, more background rows), not from a better quantity.
- Exact Shapley values for a model with per-feature additive structure is close to known results for other model classes, so the novelty is in the engineering and the checks.
- PACE-SHAP with 200 rows is slower than KernelSHAP's 20-row default on 9 of 11 datasets. Given the same 200 rows, KernelSHAP is the slower one, by 3.5–10×.
- PACE-flip is less faithful than KernelSHAP on 6 of 11 datasets.
- The conditional version did not help.
- Deletion and insertion tests have known pitfalls.
- There is no user study.

**What reviewers would ask.**
- What is new beyond "this model allows exact Shapley values"?
- Would people make better decisions with PACE than with KernelSHAP?
- Why does the faithfulness test use this baseline?

### Experimental machine learning

**Strengths.**
- Seven studies were planned in advance (seven pre-registrations) and run on fresh seeds each time.
- The tests are paired Wilcoxon tests over 10 seeds with Holm correction, and every endpoint is reported, including the negative ones:
  - the explained cut never beat the thinnest cut;
  - kNN and Trust Score beat the PACE trust signals;
  - EI adds nothing.
- The label cost counts the gate's acceptance rows.
- Repair is compared with retraining, 10-NN and a Random Forest given the same labels.
- Every paper number traces to a committed file and a script; every script smoke-runs.

**Weaknesses.**
- With 10 seeds the smallest possible Wilcoxon p is about 0.002, and two of the three "labels add" results sit at Holm p = 0.049.
- Error is controlled within each test family, not across the many studies and post hoc checks in one paper.
- The datasets are small to medium UCI sets, and all timing comes from one laptop.
- The RQ3 and RQ4 framing was written after the first repair run failed its primary test. This is disclosed, and the second run was planned in advance, but reviewers will notice it.
- The certificates are exact but loose (an edit's tight certificate holds 1.2–35% of test rows, of which 0.2–3.7% change), and they do not bound changes to explanations.
- Repair broke more predictions than appended retraining on Landsat, MAGIC and Letter.

**What reviewers would ask.**
- How sensitive are the repair results to the gate's limits?
- Would a simpler method, such as refitting with a smaller θ, give the same gain as self-repair?
- Can the certificates be made tighter?
- Why 10 seeds?

### Verdict

The work is careful, honest and reproducible. Its claims now match its results, and its negative results are reported as plainly as its positive ones. Its clearest contributions are:
- exact, fast and stable explanations for this model;
- the tie finding with label-free self-repair;
- certificates that are never wrong, although loose.

Its weak points for reviewers are:
- the accuracy gap of the explained model;
- the small like-for-like faithfulness gain;
- trust signals that lose to standard baselines;
- the length.

It fits best at a venue that values careful empirical work on fuzzy or interpretable models, after it is cut to length. A top general machine-learning venue would likely ask for more models, a user study, or a stronger reason to use CFMNN. Acceptance cannot be promised anywhere.
