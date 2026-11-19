# Final review: issues

Branch `final-review`, 2026-11-06. Paper: the Claude Doc at
https://claude.ai/artifact/9z3UBSnVXvHC3zKSEJywSS (rev 262 at the start, rev 282 after the fixes).
Every claim in the paper is checked in [CLAIMS.md](CLAIMS.md). Every change is listed in
[CHANGELOG.md](CHANGELOG.md).

**Severity.**
- **High:** a conclusion changes.
- **Medium:** a number, a list or a claim in the paper or the docs is wrong.
- **Low:** the wording says more, or something different, than the evidence shows; or a docs or tooling gap.

**Status.**
- **Fixed:** with a test or evidence.
- **Won't fix:** with a reason.
- **Needs you:** with what is needed and how long it would take.

**Summary.**
- No High issues: no bug was found that changes a result, and no conclusion changed.
- 29 items (P6 and P7 added on 2026-11-13 by the check of the peer-review revision):
  - 24 Fixed;
  - 3 Won't fix (no change needed, or out of scope);
  - 2 Needs you.

## Paper numbers (Mismatch rows in CLAIMS.md)

### P1. RQ4 table: Letter's tight certificate per edit
- **Where:** paper §7 "Certificates and targeted change", RQ4 table, row "Tight certificate per edit, SR then AL", Letter.
- **Problem:** the paper says 1.6. `refinement_v2_confirmatory/runs.csv` gives 1.546, which rounds to 1.5 (`research_audit/scripts/claims_repair_out.txt`). FINDINGS already had 1.5.
- **Severity:** Medium (one table cell, off by one in the last digit).
- **Fix:** cell changed to 1.5 (doc rev 263).
- **Verified:** read back from the doc at rev 281.
- **Status:** Fixed.

### P2. Figure 2 caption: intervals that exclude zero without significance
- **Where:** paper §7 "Faithfulness", Figure 2 caption.
- **Problem:** the caption lists S3, Wine and Letter in grey. MAGIC's grey interval also excludes zero (upper end −3.2e-5) while its Holm-corrected test is not significant (`claims_figures_out.txt`).
- **Severity:** Medium (a list missing one item).
- **Fix:** "S3, Wine, MAGIC and Letter in grey" (doc rev 264).
- **Verified:** read back at rev 281.
- **Status:** Fixed.

### P3. PACE-sparse: share of points with a single changed feature
- **Where:** paper §7 "Sparse counterfactuals", first bullet; `results/FINDINGS.md` §4.2.
- **Problem:** the paper says 86–97%. `extensions_v1/counterfactuals/runs.csv` gives 85.5–97.3% (Letter 85.5), so 85–97% (`claims_rq1_out.txt`).
- **Severity:** Medium (the low end off by one point).
- **Fix:** "85–97%" in the paper (doc rev 265) and in FINDINGS.
- **Verified:** read back at rev 281; FINDINGS diff.
- **Status:** Fixed.

### P4. The planned U against random ranking
- **Where:** paper §7 "Trust signals", bullet "U\* is confirmed"; `results/FINDINGS.md` §3.2.
- **Problem:** the paper names MAGIC and Letter as datasets where the planned U is no better than random. On Page Blocks it is not significantly better either (Holm p = 0.17; `claims_base_rq2_out.txt`, "U vs Random").
- **Severity:** Medium (a list missing one item).
- **Fix:** "…on MAGIC and Letter, and not significantly better on Page Blocks" (doc rev 266); the same in FINDINGS, with its numbers.
- **Verified:** read back at rev 281; FINDINGS diff.
- **Status:** Fixed.

### P5. "Every training error is a tie" had no committed evidence
- **Where:** paper abstract and §6; FINDINGS §5.3.
- **Problem:** the claim rested on the pre-registration text and a proposal. No committed result showed it.
- **Severity:** Medium (a headline claim without a traceable source).
- **Fix:** a post hoc check, `research_audit/scripts/check_tie_errors.py`, writes [`results/final_review/tie_errors/`](../results/final_review/tie_errors/summary.md). For the 6 repair datasets at seeds 0, 10 and 50, it fits CFMNN on the training split and counts the training errors where both the true class and the predicted class have membership 1.
- **Verified:** the result is 100% of training errors in all 18 cases (66 to 10,980 errors per case). No paper change was needed. FINDINGS §5.3 now cites the check.
- **Status:** Fixed (evidence added).

### P6. Three numbers of the peer-review revision were rounded twice
- **Where:** paper §7 "Equivalence, like for like" (Landsat's interval), §9 interval sentence (Letter), §9 bullet "Retraining on the same labels helps little" (appended range). Found by `claims_peer_review.py` (2026-11-13), rows PR1, PR6 and PR7.
- **Problem:** each was rounded from the 4- or 2-decimal value in `results/peer_review/reanalysis.md` instead of from the raw value:
  - Landsat's lower end is −0.05347, printed −0.054;
  - self-repair's gain on Letter is 2.2497, printed 2.3;
  - appended retraining's largest change is +0.847, printed +0.9.
- **Severity:** Medium (three numbers off by one in the last digit; no verdict changes).
- **Fix:** −0.053, 2.2 and +0.8 in the paper (doc revs 440–442).
- **Verified:** the edits were read back from the doc, and the script now prints OK for all three.
- **Status:** Fixed.

### P7. Appendix C left out one primary comparison
- **Where:** paper Appendix C, table "Primary comparisons", row "Extensions" (seeds 40–49).
- **Problem:** `results/extensions_v1/PREREGISTRATION.md` names two primaries for the conditional study: faithfulness like for like, and realism (the share of inputs off the data, conditional game against the usual one). The table listed only the first. The realism result itself is reported in Appendix B.
- **Severity:** Low (a list incomplete; the result was already in the paper).
- **Fix:** the row now also lists "share of inputs off the data, conditional vs usual game" (doc rev 443).
- **Verified:** the edit was read back. The other 16 rows match their plans (row PR18).
- **Status:** Fixed.

## Paper wording (Wording rows in CLAIMS.md)

All wording fixes are small find-and-replace edits made through the Docs connector, and each was read back at rev 281.

### W1. "on 2,000 test rows"
- **Where:** §7, the RQ4 paragraph.
- **Problem:** `run_refinement_v2.py` uses `min(2000, n_test)` rows, so 548–2,000 per dataset.
- **Severity:** Low.
- **Fix:** "on up to 2,000 test rows (548–2,000 per dataset)" (rev 267).
- **Status:** Fixed.

### W2. "Retraining with the new rows appended was not part of the plan"
- **Where:** §7, the RQ4 paragraph.
- **Problem:** `refinement_v2_confirmatory/PREREGISTRATION.md` lists "Retrain, appended" as an arm (line 101) and as a descriptive measure (line 152). It was planned, just not tested.
- **Severity:** Low.
- **Fix:** "was planned as a comparison to describe, with no test" (rev 268).
- **Status:** Fixed.

### W3. Abstract: "Buying up to 300 labels"
- **Where:** abstract, last paragraph.
- **Problem:** the methods that buy labels also read the gate's labelled acceptance rows (821–24,806 labels in all; PREREGISTRATION label-cost column). The sentence read as if 300 labels were the whole cost.
- **Severity:** Low.
- **Fix:** "…chosen by the order ensemble, plus the gate's labelled acceptance rows, added…" (rev 269).
- **Status:** Fixed.

### W4. "Retraining CFMNN on the same labels did not help"
- **Where:** abstract; §7 repair bullet "Retraining on the same labels does not help."
- **Problem:** against the starting model, shuffled retraining changed macro-F1 by −3.4 to +1.8 points, never significantly. Appended retraining was significant on Letter (+0.85). Both are post hoc (`claims_repair_out.txt`).
- **Severity:** Low.
- **Fix:**
  - abstract: "helped significantly only once (Letter, by 0.85 points, in a post hoc test)" (rev 270);
  - §7: bullet title "helps little", plus both ranges and their tests (revs 271–272).
- **Status:** Fixed.

### W5. "samples 32 orders … each paired with its reverse"
- **Where:** §4 "How PACE-SHAP works", item 2.
- **Problem:** `pace_cfmnn/explain.py` `_sampled` draws 16 random orders per row and adds their 16 reverses, 32 in all. The text read as 32 random orders plus 32 reverses.
- **Severity:** Low.
- **Fix:** "samples 16 orders of the features per background row and adds each one's reverse (32 orders in all)" (rev 273).
- **Status:** Fixed.

### W6. "gives the same answer every time"
- **Where:** §4, first paragraph.
- **Problem:** one random generator serves the whole batch, so the same point in a different batch can get other orders. The same call on the same points always gives the same answer.
- **Severity:** Low.
- **Fix:** "so the same call on the same points always gives the same answer" (rev 274).
- **Status:** Fixed.

### W7. Definition of a tie in §5
- **Where:** §5 "A correction to U".
- **Problem:** the text defined a tie as "a point lies inside boxes of two classes". In `pace_cfmnn/signals.py` a tie is when the top two class supports are within 2·tie_tol. Lying inside boxes of both classes is the usual cause, not the definition.
- **Severity:** Low.
- **Fix:** "When boxes of two classes give a point the same highest membership (a tie), as when it lies inside boxes of both" (rev 275).
- **Status:** Fixed.

### W8. "Gaussian weights as in Aas, Jullum and Løland"
- **Where:** §7 "Conditional replacement".
- **Problem:** Aas, Jullum and Løland use a Mahalanobis distance with the full covariance. `explain.py` (`cond_scale`) divides each feature by its own standard deviation: a diagonal covariance.
- **Severity:** Low.
- **Fix:** "…, but on distances scaled by each feature's spread instead of the full covariance" (rev 276).
- **Status:** Fixed.

### W9. "the same", "no difference" for non-significant results
- **Where:**
  - paper §7, RQ2 v2 bullet: "made no difference on Skin or Page Blocks";
  - paper §7 counterfactuals: "the same on the other 6" and "(the same on S3)";
  - README and FINDINGS §6, §7.
- **Problem:** these are non-significant differences, not equal values.
- **Severity:** Low.
- **Fix:**
  - "no significant difference" in all three places in the paper (revs 277–279);
  - the same change in README and in FINDINGS' counterfactual lines;
  - FINDINGS uses "no difference" and "level" for non-significant results in many places, so its header
    now says that these words mean "not significant after Holm".
- **Status:** Fixed.

### W10. "Outside the tight certificate … 51% of the explained rows"
- **Where:** §7, RQ4 bullet "They do not bound changes to explanations".
- **Problem:** the 51% (MAGIC, `expl_tight_out_top3`) is a share of the explained rows that lie outside every edit's tight certificate, not of all explained rows.
- **Severity:** Low.
- **Fix:** "Among the explained rows outside every edit's tight certificate, … up to 51% (MAGIC)" (rev 280).
- **Status:** Fixed.

### W11. Occlusion equals PACE-flip "up to rounding"
- **Where:** README and paper §7 "Speed".
- **Observation:** the two methods' scores differ by at most 5.4e-15. In 23 of 110 dataset-seeds the deletion AUCs differ by up to 1.2e-3, because points with equal scores are put in a different order.
- **Severity:** none; the claim is about the scores and is correct.
- **Status:** Won't fix (no change needed; recorded so a reviewer's question has an answer).

## Related work and references

### R1. Rule extraction from fuzzy min-max networks not cited
- **Where:** §8 "Boxes, prototypes and clusters".
- **Problem:** the closest prior work on explaining this model family, rule extraction from a trained fuzzy min-max network, was covered only through the survey.
- **Severity:** Low.
- **Fix:** added "; Quteishat and Lim extract such rules from a trained network" (rev 281). Added the reference A. Quteishat and C. P. Lim, *Applied Soft Computing* 8(2):985–995, 2008 (rev 282).
- **Verified:** web check (dblp, researchr and the Deakin repository record).
- **Status:** Fixed.

### R2. Web check of the references most at risk
- **What:** the three 2026 arXiv papers (Khouna, Ferry and Vidal 2602.09128; Enaieh, Fercoq and García 2605.00889; Iakovets and colleagues 2607.01306) and the claim that Khuat, Chen and Gabrys use an ensemble against training-order sensitivity.
- **Result:** all four exist and say what the paper says they say. Khuat, Chen and Gabrys (IJCNN 2020, arXiv 2001.02391) add a simple ensemble to reduce sensitivity to the presentation order.
- **Status:** Fixed (verified; no change needed).

## Code, statistics and tests

### C1. Equations and statistics against the code
- **Where:**
  - `cfmnn_core.py` (membership, training, tie-break);
  - `explain.py` (PACE-SHAP, PACE-flip, the conditional game);
  - `signals.py` (U, U\*);
  - `refine.py` and `audit.py` (gate, certificates);
  - the Holm, Wilcoxon and non-inferiority code in `run_refinement_v2.py`, `run_error_detection.py` and `summarise_refinement_v1.py`.
- **Result:** no mismatch with the paper's equations.
- **Fix:** 13 new tests in `tests/test_final_review.py`, each checked by hand:
  - the paper's §4 worked example (boxes, memberships, PACE-SHAP);
  - U and U\* for a tied and an untied point;
  - Holm on fixed p-values, against all three implementations;
  - non-inferiority p-values (1/1024 and > 0.5);
  - fix-share counts;
  - gate limits at exactly 0.5 and 1.0 points (inclusive), and the box-cap floor;
  - edge cases: a constant feature, a class with one row, d = 1, and an edit that removes a class's only box.
- **Verified:** the full suite passes, 178 tests (165 before plus 13 new), 155 s.
- **Status:** Fixed (tests added).

### C2. `x_correct_after` in the repair edit log means two different things
- **Where:** `pace_cfmnn/active.py`, lines 249 and 330.
- **Problem:** in the oracle branch, a rejected cut leaves `model` unchanged, so the field describes the model before the edit. In the other branches it describes the trial edit, even when the edit was rejected.
- **Severity:** Low.
- **Status:** Won't fix. No paper or FINDINGS number uses this field, and changing the log format would make old and new `runs.csv` files disagree.

### C3. Five scripts have no argument parser
- **Where:** `pace_shap_settings.py`, `knn_reference.py`, `counterfactual_table_seeds.py`, `summarise_refinement_v1.py` and `posthoc_shuttle_sparse.py`.
- **Problem:** `python scripts/X.py --help` runs the whole script, and it overwrites its committed outputs in `results/`.
- **Evidence:** this happened in this review. `knn_reference.py` and `counterfactual_table_seeds.py` were run by the first version of `check_readme_commands.py`. Their three outputs came back byte-identical to the committed files (`git hash-object` against `HEAD`). `pace_shap_settings.py` was stopped after 5 minutes, before it writes anything, and nothing in `results/` changed.
- **Severity:** Low.
- **Fix:**
  - `check_readme_commands.py` no longer calls `--help` on scripts without a parser;
  - `smoke_redirect.py` runs such scripts with their writes sent outside the repository.
- **Status:** Won't fix in the scripts themselves: adding a parser to five finished scripts is out of scope for this review. Recorded as a follow-up.

## Docs and reproducibility

### D1. FINDINGS lacked refinement v2
- **Where:** `results/FINDINGS.md` §5–§7.
- **Problem:** §5 stopped at the paused pilot.
- **Fix:** §5 rewritten as "Repair (RQ3, RQ4)":
  - 5.1, the pilot;
  - 5.2, the first pre-registered run with R1–R3, S1 and the amendment arms;
  - 5.3, the exploratory checks;
  - 5.4, the second pre-registered run with the RQ3 and RQ4 tables.
  - Also: a §6 claim bullet and five §7 must-not-claim bullets.
- **Verified:** every number traced to CLAIMS.md rows Q1–Q18.
- **Status:** Fixed.

### D2. README lacked refinement v2, and its status and test count were out of date
- **Where:** `README.md`.
- **Problem:**
  - Phases 6–7 were shown as "Paused";
  - "139 tests" (now 178);
  - no commands for `run_refinement_explore.py`, `run_refinement_v2.py` or `summarise_refinement_v1.py`;
  - the key results described only the pilot;
  - the layout table missed `audit.py` and `resume.py`.
- **Fix:**
  - status table updated;
  - test count and run time updated;
  - three commands added, with run times from the `time_s` sums in the committed `runs.csv` files;
  - key results replaced by the three repair runs;
  - layout rows added;
  - intro mentions trust signals and repair.
- **Verified:** `research_audit/scripts/check_readme_commands.py`: all 38 README commands exist, with valid flags and choices (`check_readme_commands_out.txt`).
- **Status:** Fixed.

### D3. requirements.txt comment named too few scripts
- **Where:** `requirements.txt`.
- **Problem:** the comment said shap, lime and dice-ml are needed only for `xai_baselines.py`, `run_faithfulness.py` and their tests. The counterfactual, global-importance and review scripts need them too.
- **Fix:** comment rewritten. All seven third-party imports in the repository (numpy, scipy, scikit-learn, pandas, shap, lime, dice-ml) are listed; checked by parsing every `.py` file's imports.
- **Status:** Fixed.

### D4. Smoke runs of every script
- **What:** `research_audit/scripts/smoke_all.py` runs each of the 36 experiment configurations once, with outputs in a scratch folder outside the repository:
  - 1 dataset, 1 seed and a few points per run;
  - the 7 scripts without `--out` run through `smoke_redirect.py`.
- **Result:** 36 of 36 runs exited 0, and none wrote into the repository (`git status` unchanged).
  - The 6 scripts run in full through the wrapper gave 8 output files.
  - All 8 are byte-identical to the committed files (line endings aside), so those results reproduce exactly.
  - Details: [SMOKE.md](SMOKE.md).
- **Status:** Fixed (every script runs end to end).

## Needs you

### N1. The protocol amendment's two arms and the missing `contraction.py`
- **Where:** `results/refinement_confirmatory/PROTOCOL_AMENDMENT_1.md` (teammate's commit b2352f1); `summary_v1.md` appendix.
- **Problem:** the amendment adds two arms, "B-Simpson non-active contraction" and "order_ensemble | A+B+C-minimal", to the first repair run.
  - It points to `pace_cfmnn/contraction.py`, which is not in the repository, so these rows cannot be reproduced.
  - They were not pre-registered, and the paper does not use them.
  - B-Simpson's macro-F1 is far below B0 (for example MAGIC 61.8 against 71.1).
- **What is needed:** the teammate commits `contraction.py` (or the code that made those rows), or the team agrees to mark the rows as not reproducible in `summary_v1.md`. About 30 minutes for the teammate. This review did not touch it: the teammate's work is out of scope.
- **Status:** Needs you.

### N2. The paper is longer than 9 pages
- **Where:** the whole paper.
- **Problem:** this review's fixes added a few words, not pages. The paper was already over a 9-page limit, and no venue has been chosen.
- **What is needed:** choose a venue, then cut to its limit. Likely cuts:
  - the RQ1 extension details in §7;
  - the RQ2 v2 bullets;
  - the "Deviations" paragraph.
  - About 3–5 hours of editing once the limit is known. The title, RQs and contributions should not need to change.
- **Status:** Needs you.
