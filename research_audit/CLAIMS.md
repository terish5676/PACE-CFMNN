# Claims ledger: the paper against the committed results (revs 262–657)

Every number, comparison and claim in the paper (Claude Doc `9z3UBSnVXvHC3zKSEJywSS`), traced to a committed file and recomputed from the raw `runs.csv` files, not from the rounded summaries. The first sections check rev 262. Each later section checks the text added in the revisions its heading names, up to rev 575 (2026-11-15). Revs 494–537 moved text into two research questions and Appendices B–E without new numbers. On 2026-11-16 every claims script was re-run on branch tough-review-fixes, after the code added for the tough review; all 20 exited 0 and every `*_out.txt` was unchanged byte for byte.

- **Scripts:** `research_audit/scripts/`. These are `claims_repair.py`, `claims_rq1.py`, `claims_base_rq2.py`, `claims_figures.py`, `check_tie_errors.py` and `check_matched_speed.py`, plus the helpers in `common.py`. Later sections add `claims_readable.py`, `claims_repair_theta.py`, `claims_followup.py`, `claims_peer_review.py`, `claims_order_timing.py`, `claims_sr_method.py`, `claims_contraction.py`, `claims_systems.py`, `claims_generality.py`, `claims_sampled.py` and `claims_covertype.py`, and the post hoc `check_exact_simpson.py`.
- **Outputs:** each script writes a `*_out.txt` file next to it. `check_tie_errors.py` writes to `results/final_review/tie_errors/`.
- **Tests used:** the paper's own conventions. Each comparison is a paired two-sided Wilcoxon test over seeds, with Holm correction across datasets. A result counts as significant when the Holm-adjusted p is below 0.05.

**Status**

| Status | Meaning |
|---|---|
| Verified | The recomputed value matches the paper at the paper's rounding |
| Mismatch | The number or list is wrong |
| Wording | The number is right, but the sentence says more, or something different, than the evidence shows |
| Untraceable | No committed file supports it |

Each Mismatch and Wording row points to an item in `ISSUES.md`.

**Summary:**

- 134 claim rows checked. Many rows cover a whole table or list, so well over 500 single numbers were compared.
- 122 rows Verified. Four of these also carry a wording note.
- 4 rows Mismatch, all small: P1, P2, P3 and P4.
- 8 rows Wording only. With the four notes on Verified rows, these give items W1–W11 in ISSUES.md.
- 0 Untraceable. The one claim that had no backing file, "100% of training errors are ties", now has one: a new post hoc check, `results/final_review/tie_errors/`.
- Every pre-registered endpoint is reported, including the ones that went against the method. See "Pre-registered endpoints" at the end.
- **After the fixes:** all four Mismatch rows and all Wording rows except W11 (a note, no change needed) were corrected in the paper at doc revs 263–280. The rows below still show what rev 262 said; [ISSUES.md](ISSUES.md) gives the new wording and its revision.
- **Reframe (revs 283–293, user request):** the abstract, intro, contributions and conclusion were reworded. Only one number in the new text was not already in the paper: matched KernelSHAP took 3.5–10× as long as PACE-SHAP. It has its own row, F10b, and script, `check_matched_speed.py`. Every other number in the new text is in rows A1–A16, F10 or Q6.
- **Readable baselines (revs 294–302):** a new pre-registered run (seeds 70–89) added the §7 paragraph "Against other readable models", its table and 5 bullets, a sentence pair in the abstract, and a sentence in §1, Settings, Deviations, Limitations and the Conclusion, plus 3 references. Every number is in rows RB1–RB15 and is recomputed by `claims_readable.py`, whose verdict counts match the run's own `summary.md`. All 15 rows are Verified.
- **Repair at the chosen θ (revs 303–311):** a second pre-registered run on the same seeds (70–89) extended the §7 bullet "θ = 0.3 was not the most accurate choice", and added a clause to Settings, Deviations and two Limitations bullets. At the user's request (revs 309–311), the abstract now gives the result and states the tie finding for all 13 datasets, and §6 gives the new evidence for it; rows A10 and R1 keep their old wording, as the rows above do. Every number is in rows RT1–RT14 and is recomputed by `claims_repair_theta.py`, whose verdict counts match the run's own `summary.md`. All 14 rows are Verified.
- **Follow-up tests (revs 312–326):** five runs on seeds 110–119 answered the limitations. Every number is in rows FU1–FU8, recomputed by `claims_followup.py`. All 8 rows are Verified; one error it caught (PACE-sparse's r1 ratio) was fixed at rev 326.
- **Peer-review revision (revs 327–443):** the paper was rewritten to answer `reviews/fable_peer_review.md`, with no new runs. It gained a new title, three contributions, a new section order with appendices, equivalence tests and intervals, a table of tied and untied points, Figure 1, the S1–S3 generators, the power floor and a table of primary comparisons. Every new number is in rows PR1–PR19 and is recomputed by `claims_peer_review.py` (78 checks).
  - The check found 3 numbers rounded twice (P6) and one primary comparison missing from Appendix C (P7). All four were fixed at revs 440–443, and all 19 rows are now Verified.
  - Runs F and G are not done: the paper marks each place for their results "[Result pending: Run F]" or "[Result pending: Run G]", and the ledger has no rows for them yet.
- **Runs H and T (revs 444–449):** the order of the training errors in self-repair (pre-registered, seeds 130–139) and the time of each repair setup (descriptive, seed 50), both run on the home PC. Every number is in rows OT1–OT10, recomputed by `claims_order_timing.py` (30 checks). All 10 rows are Verified.
- **Self-repair as the method (revs 450–453, no new run):** a §9 paragraph and table (gains over the starting model in +x form, labels and seconds), a post hoc §10 paragraph on RQ4 for self-repair alone, two closing sentences in §9's Run H paragraph (why self-repair keeps the random order), and a clause in the Conclusion. Every number is in rows SM1–SM9, recomputed by `claims_sr_method.py` (51 checks). All 9 rows are Verified.
- **Runs F, G, I, J and K (revs 455–478):** self-repair against contraction, whole systems against an EBM and a tree, Proposition 1 at three γ and PACE-SHAP on contracted boxes, PACE-SHAP against modern samplers above 12 features, and one large dataset (CoverType), all run on the home PC. They filled the seven "[Result pending]" places and added a §6 table and paragraph, a §7 paragraph, sentences in the abstract, §3, §5, Limitations, Settings, Deviations and Appendix C. Every number is in rows CF1–CF8, SY1–SY7, GE1–GE5, SB1–SB5 and CT1–CT4, recomputed by five scripts (159 checks). All 29 rows are Verified; four values first taken from rounded summaries were corrected before they reached the paper (see that section).
- **Check of that write-up on the laptop (rev 479, 2026-11-15):** the five scripts were run again (all 159 checks OK; three p-values print differently because of pandas' float parser, with no verdict changed; see that section). `check_exact_isclose.py` (post hoc) found the cause of Run I's 1e-10 exact-check differences: shap's ExactExplainer skips a model call when a background value is within `np.isclose`'s tolerance of the explained point's. §3's "three worst cases" was wrong (the three checked are the 1st, 2nd and 5th largest) and was reworded at rev 479 with the cause. New row GE6; GE3 keeps its old wording. At the user's request, rev 480 adds a sentence on Runs F and G to the Conclusion (rows CF9 and SY8).
- **Runs L, M and N (revs 481–493, 2026-11-15):** CoverType with θ chosen by validation and self-repair (descriptive, on 10% of the rows), Simpson's classifier trained from scratch (an exact check, PASS, and a descriptive comparison with CFMNN), and CoverType among the modern samplers (descriptive), all run on the home PC. They added sentences to §3, §6, §7, the Limitations, Settings, Deviations and "Which tests are primary", and three rows to Appendix C's table. Every number is in rows CL1–CL5, SF1–SF7 and SN1–SN4, recomputed by three scripts (92 checks), plus a post hoc check of Run M's one large exact-check difference (5 checks: the same shap shortcut as Run I's). All 16 rows are Verified. Rows GE5 and CT4 keep the wording they checked; revs 486 and 488 changed it.
- **Tough-review batches 1 and 2 (revs 624–657, 2026-11-19):** nine pre-registered runs on seeds 250–399 (time to accuracy and a clean re-time, Skin without duplicates, exact flip sets, the recurrence, tuned fuzzy min-max baselines, GFMM exactness, the decision explanation, the PACE-SHAP certificate, edits that cannot hurt). They changed "3.5–10×" to "3.3–8.0×" everywhere, and added sentences to the abstract, contributions, §3, §4, §6, §7, §8, Limitations, the conclusion, Next steps, Appendix A (Deviations), Appendix E (nine table rows) and Appendix F (Propositions 6–10 and the recurrence remark). Rows TR1–TR14: 12 Pending until `claims_tough_batch1.py` and `claims_tough_batch2.py` run on the home PC, 2 Verified from the hand-overs and plans.
- **Rows above at rev 443:** the rows keep the wording of the version they checked. The corrected text of the four Mismatch rows (P1–P4, rows Q14, F7, K6, E6) and of rows A4 and A13 is still in the paper at rev 443. This was read from an export of rev 439; revs 440–443 changed only the four numbers of P6 and P7.

## Abstract

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| A1 | PACE-SHAP equals shap's exact explainer to rounding error, 16–30× faster, for d ≤ 12 | review_v1/exact | max abs diff 3.3e-16; speed-up 16.5–30.4× | Verified |
| A2 | Above 12 features, a 1–2% error against exact values | review_v1/exact (32 orders) | Wine 0.96%, Letter 1.70% | Verified |
| A3 | More faithful than KernelSHAP on 6 datasets, never significantly less; against the stronger KernelSHAP, better on 3 | faithfulness_v3 | 6/11 deletion, 0 worse; strong: 3/11 | Verified |
| A4 | Same rows and features: more faithful on only 3 of 13 | extensions_v1/conditional (usual tests) | 3/13 deletion (S3, BC, Landsat) | Verified |
| A5 | More stable than KernelSHAP even with new background rows (10 of 11) | review_v1/faithfulness `stability_new` | 10 better, 1 level (Page Blocks) | Verified |
| A6 | PACE-flip: 0.05–1.4 ms; 25–134× faster than batched occlusion; about 90–300× faster than LIME; 300–1,200× faster than KernelSHAP | faithfulness/timing, review_v1/occlusion_batched | 0.047–1.367 ms; 25–134×; 89–316×; 292–1,194× | Verified |
| A7 | Own signals beat the plain margin but not the standard methods | error_detection_confirmatory | PD+U\* > margin on 6/8; TS and kNN > PD+U\* on 7/8; order ensemble on 8/8 | Verified |
| A8 | The order ensemble improves the combination on most datasets; bootstrap copies do as well | error_detection_v2, review_v1/ensembles | 5/8 higher; bagged vs order level on 8/8 | Verified |
| A9 | CFMNN is 0.2–21.5 points below the better of 10-NN and RF on all 13 | review_v1/accuracy | 13/13, gap 0.2–21.5 | Verified |
| A10 | Every training error is a cross-class tie on the 6 datasets | results/final_review/tie_errors (new, post hoc) | 100% at seeds 0, 10, 50 (66–10,980 errors per case) | Verified (see P5) |
| A11 | Self-repair raised macro-F1 on all 6, by 1.5–6.0 | refinement_v2_confirmatory S3a | +1.48 to +6.02, Holm p < 0.05 on 6/6 | Verified |
| A12 | "Buying up to 300 labels … added a further 0.7–2.0 points on 3" | v2 P3 | +0.73 (Letter), +1.08 (Landsat), +2.03 (MAGIC) | Verified, Wording W3 |
| A13 | "retraining CFMNN on the same labels did not help" | v2, post hoc | shuffled − start −3.4 to +1.8, never significant; appended +0.85 on Letter, significant post hoc | Wording W4 |
| A14 | 0 prediction changes outside the certificate in 46,153 edits | v2 C4 | 0 / 46,153 / 180 runs | Verified |
| A15 | Of the changed predictions, 64–94% were corrections, against 41–52% for retraining | v2 fix share | SR+AL 64.2–94.1; shuffled 40.5–52.4 | Verified |
| A16 | The explanation-chosen cut did no better than the thinnest slice | refinement_confirmatory R1 | no significant difference on any dataset | Verified |

## Sections 3 and 7: the base model

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| B1 | The centroid fix raises Shuttle from 93.3% to 99.8% | g0 | 93.33 → 99.80 | Verified |
| B2 | Expansion search: Digits 81.7 → 95.3; Skin 97.0 → 94.7 | g0 | 81.72 → 95.33; 97.01 → 94.74 | Verified |
| B3 | Sepsis: ours 48.65% (5 seeds), paper's ensemble 68.76%, always "survives" 92.65% | g0 runs; base paper value recorded in g0/summary.md | 48.65; 68.76 (quoted); majority 92.65 | Verified |
| B4 | Accuracy table: 13 rows × (training rows, 3 models with ± sd, gap) | review_v1/accuracy | all 78 cells match | Verified |
| B5 | Gap under 3 points on Shuttle, Wine, BC, Thyroid, Digits; 5.2 and 7.4 on Skin and Landsat; 9.5–21.5 on S1–S3, PB, MAGIC, Letter | same | same lists and values | Verified |
| B6 | RF has 200 trees in this table | scripts/accuracy_table.py | n_estimators=200 | Verified |
| B7 | Parity table: ours with the paper's rules 95.0 / 93.3 / 96.9; both fixes 97.2 / 99.8 / 94.7; paper 96.9 / 94.0 / 97.6 | g0 (both CFMNN columns, from runs.csv); paper column from the base paper (g0/summary.md) | all match | Verified |
| B8 | Within 2 points of the paper's ensemble on Thyroid, Shuttle, Skin | same | 1.9, 0.7, 0.7 | Verified |

## Section 4: PACE

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| M1 | Toy example: memberships 0.97 and 0.93; the 4-row table; margin drops | hand check (summary of earlier session) and `tests/` | match | Verified |
| M2 | PACE-flip skips 71–98% of boxes on all 11 datasets | speed/runs.csv + speed_uci/runs.csv | 71.1–98.2% | Verified |
| M3 | PACE-SHAP sampled mode: "samples 32 orders of the features per background row, each paired with its reverse" | pace_cfmnn/explain.py `_sampled` | 32 orders per row = 16 random orders and their 16 reverses | Wording W5 |
| M4 | "uses a fixed seed, so it gives the same answer every time" | explain.py: one generator for the whole batch | the same call gives the same answer; the same point in a different batch can get other orders | Wording W6 |
| M5 | 200 background rows (71 on Wine); at most 4,096 coalitions | code and settings | match | Verified |
| M6 | The margin "is 0 for a tied point" | cfmnn_core, explain | true for cross-class ties (the code's tie) | Verified |

## Section 5: trust signals

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| T1 | 50 nudges, 5% of each feature's standard deviation, clipped to [0, 1]; redraw with probability 0.1 if ≤ 10 distinct values | signals.py | match | Verified |
| T2 | PRS = mean percentile rank | pace_score.py | match | Verified |
| T3 | "When a point lies inside boxes of two classes (a tie)" | signals.py: a tie = top two class supports within 2·tol | the code's tie is equal top scores; inside two boxes is the common case, not the definition | Wording W7 |
| T4 | Planned U ranks ties as less uncertain than most points | error_detection/summary.md; U formula | match | Verified |

## Section 6: repair

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| R1 | Every training error is a tie inside an own-class and a wrong-class box; 100% on 6 datasets at seeds 0 and 10 | before: only PREREGISTRATION and a proposal. Now: results/final_review/tie_errors | 100% at seeds 0, 10 and 50 | Verified (P5: evidence added) |
| R2 | Shrink moves the face 0.001 past the point, on the feature with the smallest share of width cut; empty box removed | refine.py | MARGIN = 1e-3; C′-volume chooser | Verified |
| R3 | Gate: macro-F1 −0.5, class recall −1, box cap 1.1× | refine.Gate | match | Verified |
| R4 | Points tried: 300, or a quarter of the pool if smaller | run_refinement_v2 `budget = min(300, len(Xp)//4)` | 273 on Page Blocks, 300 elsewhere | Verified |
| R5 | Certificate argument (loose; tight = drop single-class groups) | audit.certify | match | Verified |
| R6 | Fix share = fixed ÷ (fixed + broken) | run_refinement_v2 | match | Verified |

## Section 7: faithfulness (RQ1)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| F1 | 50 points (36 Wine), 10 seeds; first run 0–9; second run 30–39 | runs.csv | match | Verified |
| F2 | PACE-SHAP never significantly less faithful than any method in the second run | faithfulness_v3 | 0 worse, all methods, both tests | Verified |
| F3 | More faithful than KernelSHAP by deletion on 6 of 11 (S3, Wine, BC, Thyroid, Letter, Landsat), by insertion on 7 | same | same lists | Verified |
| F4 | Against the stronger KernelSHAP: 3 by deletion (BC, Thyroid, Landsat), 4 by insertion | same | same | Verified |
| F5 | Relative lead: 2–4% on S1–S3, Wine, Letter; 10–28% on MAGIC, Thyroid, BC; 67% on Landsat; d_z 0.6–8.6; Shuttle and PB: KernelSHAP slightly ahead, not significantly | same | 1.6–4.1%; 9.9–28.2%; 67.2%; d_z 0.61–8.63; Holm p 0.55 and 0.63 | Verified |
| F6 | Figure 2: 22 rows of mean, interval and Holm p | widget data vs faithfulness_v3 | all 22 rows match to 4 decimals | Verified |
| F7 | Figure 2 caption: the intervals that exclude zero without significance are S2 and MAGIC (blue) and S3, Wine and Letter (grey) | same | grey also includes MAGIC (upper end −0.00003) | Mismatch P2 |
| F8 | Mean deletion AUC table (66 cells) and its bold marks | faithfulness_v3 | all 66 match; bold matches | Verified |
| F9 | PACE-SHAP keeps every feature; shap keeps at most 10 by default; BC and Landsat have 30 and 36 features | code, data | match | Verified |
| F10 | Matched KernelSHAP (200 rows, all features, 200 coalitions): 3/13 deletion (S3, BC, Landsat), 2/13 insertion (BC, Landsat), never worse | extensions_v1/conditional | same | Verified |
| F10b | (Added at rev 284, also in the abstract, contribution 1 and the conclusion.) Matched KernelSHAP took 3.5–10× as long as PACE-SHAP on every dataset; timed while other runs shared the laptop | extensions_v1/conditional `ms_per_explanation`; `check_matched_speed.py` | ratio of mean times 3.46–10.01; slower in 130 of 130 dataset-seed pairs (1.16–12.14) | Verified |
| F11 | PACE-SHAP-20 never significantly less faithful than KernelSHAP | faithfulness_v3 | 0 worse | Verified |
| F12 | PACE-SHAP beats PACE-flip on 8/11 by both tests; level on Thyroid, Shuttle, PB | same | same | Verified |
| F13 | First run: KernelSHAP beat PACE-flip on 6 (S1–S3, Wine, MAGIC, Letter); PACE-flip won on 3 (BC, Thyroid; Landsat by deletion) | faithfulness | same | Verified |
| F14 | Occlusion gives PACE-flip's result up to rounding | review_v1/occlusion_batched; faithfulness | scores differ by ≤ 5.4e-15; deletion AUCs differ by ≤ 1.2e-3 in 23 of 110 dataset-seeds, where equal scores are put in a different order | Verified (note in ISSUES W11) |
| F15 | Stability: fixed seed identical; new seed: PACE-SHAP sampled 0.94–0.99, LIME 0.52–0.90, KernelSHAP 0.15–0.73 | faithfulness_v3 | same | Verified |
| F16 | Post hoc: PACE-SHAP-20 with new rows more stable than KernelSHAP on only 5/11 | same | 5/11 | Verified |
| F17 | Planned test: more stable on all 11 with fixed rows, 10/11 with new rows (level PB) | review_v1/faithfulness | same | Verified |
| F18 | Ground truth top-3: LIME 0.89–0.92, PACE-SHAP 0.79–0.85, KernelSHAP 0.72–0.81, PACE-flip 0.49–0.55 | faithfulness_v3 | same | Verified |
| F19 | LIME less faithful than PACE-SHAP on 9 of 11 | same (deletion) | 9/11 | Verified |
| F20 | 42–47% of PACE-flip's single changes involve the relevant features | faithfulness_v3/flip_mass | 41.7–47.1% | Verified |
| F21 | Ties in the second run: PB 79%, MAGIC 75%, Letter 55%, Landsat 25%, Shuttle 11% | review_v1/ties_v3.csv | 78.6, 75.0, 55.4, 24.8, 11.2 | Verified |
| F22 | Raw curves: 10/11 by deletion (level PB), 11/11 by insertion | review_v1/faithfulness | same | Verified |
| F23 | Untied points, 9 datasets with ≥ 10 in every seed: 7 by deletion, 6 by insertion; never worse | same | same | Verified |
| F24 | LIME with 200 rows less faithful on 9/11 deletion, 11/11 insertion | same | same | Verified |
| S1 | Speed table numbers: 0.05–1.4 ms; 29–141×, 89–316×, 292–1,194×; batching saves at most 2.2×; still 25–134× | faithfulness/timing; occlusion_batched | 0.047–1.367; 29–141; 89–316; 292–1,194; 1.01–2.23; 25–134 | Verified |
| S2 | Figure 3: 55 time values | widget vs faithfulness/timing | all match to 4 decimals | Verified |
| S3 | PACE-SHAP 1.2–3.0× as long as KernelSHAP on 9 of 11, less on Wine and Shuttle | faithfulness_v3/timing | 1.22–3.04 on 9; 0.43 and 0.73 | Verified |
| S4 | PACE-SHAP-20 4.3–14× faster than KernelSHAP | same | 4.35–14.47 | Verified |
| S5 | Exact: within 3.3e-16 of ExactExplainer and every-coalition KernelSHAP; 16–30× faster; S1 2.5 s vs 53.7 s; Shuttle 19 vs 478 ms | review_v1/exact | 3.33e-16; 16.5–30.4×; 2,499 vs 53,708 ms; 19.2 vs 477.7 ms | Verified |
| S6 | Sampled 32 orders: 1.0% (Wine), 1.7% (Letter); top-3 0.995 and 1.0; Letter exact 41.7 s vs 0.9 s | same | 0.96%, 1.70%; 0.9954, 1.0; 41,655 vs 909 ms | Verified |
| C1 | Conditional game: off the data at most 10% outside Shuttle, up to 67% for the usual game; Shuttle 57% and 71% | extensions_v1/conditional/realism.csv | 10.1, 67.1; 56.7, 70.6 | Verified |
| C2 | Conditional PACE-SHAP beats conditional KernelSHAP on 6/13 by both tests, never worse | conditional/runs.csv | same | Verified |
| C3 | PACE-SHAP beat its conditional version on the conditional test on 10/13 (deletion), 9/13 (insertion); Shuttle the exception (0.452 vs 0.482, 9 of 10 seeds) | same | same | Verified |
| C4 | At σ = 0.1 the game rests on about one row once half the features are kept, on 7 of 13 | extensions_v1/posthoc_review.md §2 (seed 40) | effective rows 1.1–1.2 at d/2 on S1–S3, S1c, S2c, Wine, Letter | Verified |
| C5 | σ checks: near PACE-SHAP on synthetic at σ = 1; slightly more faithful on Shuttle and MAGIC at 0.5 and 1; the noisy copy gets slightly more credit; MAGIC off 5% / 12% / 2% / 20% | conditional/posthoc_sigma.csv and .md | match | Verified |
| C6 | Four times the rows halved the loss on S1, changed little elsewhere | conditional/posthoc_rows.md | S1 loss 0.061 → 0.028 | Verified |
| C7 | "Gaussian weights as in Aas, Jullum and Løland" | explain.py `cond_scale`: per-feature standard deviation | diagonal scale, not Aas et al.'s full covariance | Wording W8 |
| L1 | On Shuttle, Letter and Landsat, 63–70% of evaluated inputs lie off the data | realism.csv `test_int` | 62.6–70.4 | Verified |

## Section 7: trust signals (RQ2)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| E1 | Confirmatory table: 8 units × 7 columns (56 cells), bold | error_detection_confirmatory | all 56 match; bold matches | Verified |
| E2 | Excluded: Wine, BC, Shuttle, with as few as 3, 20, 39 errors | same | 3, 20, 39 | Verified |
| E3 | PD + U\* beats the margin on 6/8, all 10 seeds each; not Digits (small, ns) or Thyroid (slightly worse) | same | same | Verified |
| E4 | EI makes the combination worse on 6/8, never better | same (PRS\* vs PD+U\*) | 6 lower, 2 level | Verified |
| E5 | U\* beats U on all 6 real datasets with ≥ 1% ties, all 10 seeds | same | 6/6, 10/10 each | Verified |
| E6 | "The planned U is no better than random on MAGIC and Letter" | same | also not significantly better on Page Blocks (Holm p 0.17) | Mismatch P4 |
| E7 | TS and kNN beat PD + U\* on 7/8; order ensemble on 8/8 | same | same | Verified |
| E8 | 10-NN more accurate than CFMNN on 12 of 13 (92% vs 74% on Letter); Thyroid the exception | knn_reference.csv | 12/13; 92.2 vs 74.1 | Verified |
| E9 | Combination table: 8 × 5 (40 cells), bold | error_detection_v2 | all 40 match | Verified |
| E10 | Page Blocks 0.10 vs 0.14; two confirmatory seeds at 0.20 and 0.24 | both runs | 0.103 vs 0.141; 0.198, 0.237 | Verified |
| E11 | Order ensemble added on 5/8 in all 10 seeds; Thyroid 0.12 → 0.46; no difference on Skin and PB; lower on Digits | error_detection_v2 | 5 higher (10/10 seeds each), 2 level, Digits lower | Verified, Wording W9 ("no difference") |
| E12 | Planned ensembles test: repeated on 5 units, lower on Digits; bagged as good on 8/8; order adds to bagged on 4 (synthetic, Thyroid, MAGIC, Letter); SGD order ensemble helps only synthetic, worse on Skin | review_v1/ensembles | same | Verified |
| E13 | Strongest detector beats kNN on 8/8 and TS on 7/8 | error_detection_v2 | same | Verified |
| E14 | Geometry signals: significant only on Letter, +0.0015 | same | +0.0015, Holm p 0.016 | Verified |

## Section 7: counterfactuals, global importance and interactions (RQ1)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| K1 | Counterfactual table, 88 cells, bold = Holm p < 0.05; both valid on 27–30 of 30 | extensions_v1/counterfactuals | all 88 match (time = median over the both-valid points); bold matches | Verified |
| K2 | Model confirms 99–100% of PACE's, 97–100% of DiCE's | same | 99.3–100; 97.3–100 | Verified |
| K3 | Closer to real data on all 11; smaller change on 5 of 8 real datasets "and the same on the other 6"; 70–1,000× faster; DiCE fewer features on 10/11 "(the same on S3)" | same | 11/11; 5 sig, 6 not significant; 70–1,043×; 10/11 | Verified, Wording W9 ("the same") |
| K4 | Nearest unlike neighbour changes 6–30 features, more than twice either method; keeps Thyroid's locks in 9% | same | 5.9–30.3; ratio ≥ 2.33; 9.3% | Verified |
| K5 | PACE-sparse: 0.1–1.1 fewer features than DiCE on 10/11; smaller change on all 11 | same | 0.13–1.14 on 10; 11/11 | Verified |
| K6 | Single feature for 86–97% of points on S1–S3, MAGIC, Letter, PB; DiCE 56–68% | same | 85.5–97.3%; DiCE 56.1–68.3% | Mismatch P3 |
| K7 | Shuttle: 0.9 more features than DiCE | same | +0.94 | Verified |
| K8 | Closer to data than DiCE on the 8 real datasets; 5–55% farther than PACE on all 11; 13 ms to 4.4 s; faster than DiCE on 10/11, 3.5× slower on BC | same | match (4.7–54.8%; 12.9 ms–4.36 s; 3.51×) | Verified |
| G1 | All rankings find the 3 relevant synthetic features | global_importance | gt_top3 = 1.0 for every method | Verified |
| G2 | PACE-flip agrees with permutation importance more than KernelSHAP on 10/11, significant on 3 (Shuttle, Letter, Landsat); MAGIC KernelSHAP ahead, ns | global_importance (seeds 0–9) | same | Verified |
| G3 | At most 11 s per dataset on average (14.3 s slowest run) vs about 3 minutes for KernelSHAP | same | 11.1 s; 14.3 s; 175.6 s | Verified |
| G4 | Limiting PACE-flip to 50 points removes about a quarter of its lead (PB) | extensions_v1/global_importance/posthoc_points_page_blocks.md | lead 0.330 → 0.244 (26%) | Verified |
| G5 | Second run: PACE-SHAP vs KernelSHAP, no significant difference on any of 11; PACE-flip lead replicated on Letter, Landsat, appeared on PB, not on Shuttle | extensions_v1/global_importance | same | Verified |
| I1 | (x0, x1) strongest pair on S3 in all 10 seeds, and on S1 and S2 too | extensions_v1/interactions | 10/10 on each | Verified |
| I2 | Figure 4: 30 per-seed leads; means 2.06, 1.62, 1.35; S3's range overlaps S1's but not S2's | widget vs runs.csv | all match (max diff 5e-5) | Verified |

## Section 7: repair (RQ3, RQ4)

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| Q1 | First run: explained cut level with the thinnest cut on all 6, worse than random on Letter; shrink beat add-only on all 6 by 0.7–8.6; order ensemble beat random choice on 3 | refinement_confirmatory | R1 none; R3 Letter; R2 0.74–8.61 on 6; S1 on 3 | Verified |
| Q2 | Exploratory: SR within half a point of 300 bought labels on 5/6; acceptance rows 548–24,506; retraining gained nothing; best cut −0.6 to +1.1 over the thinnest | refinement_v2_explore | MAGIC −2.44, others ≥ −0.5; 548–24,506; −0.57 to +1.13 | Verified |
| Q3 | Split 50/20/10/10/10; 10 seeds 50–59; margins 1 and 2 points | code, PREREGISTRATION | match | Verified |
| Q4 | Labels beyond training rows: 821–24,806 | v2 design | 273 + 548 to 300 + 24,506 | Verified |
| Q5 | RQ3 table (42 cells) | refinement_v2_confirmatory | all match | Verified |
| Q6 | SR +1.5 to +6.0 on all 6; closes 10–83% of the gap to RF | v2 S3a | +1.48 to +6.02; 10.5–82.8% | Verified |
| Q7 | SR then AL beat SR on Landsat (+1.1), MAGIC (+2.0), Letter (+0.7); Holm p = 0.049 for the first and last | v2 P3 | +1.08, +2.03, +0.73; 0.0488, 0.0117, 0.0488 | Verified |
| Q8 | SR no worse than AL (1 point) only on Letter; within half a point on Thyroid, Landsat, Skin; trails by 2.3 and 1.8 on PB and MAGIC | v2 S3b | same | Verified |
| Q9 | SR then AL beat shuffled retraining by 1.7–8.2 on all 6; appended retraining changed macro-F1 by −2.6 to +0.9 | v2 S4c; descriptive | same | Verified, Wording W4 |
| Q10 | Lowest class recall: kept within 2 points on 4/6, Thyroid +20, Skin +18; not on PB or MAGIC; SR lowered MAGIC's by 2.7 in 9 of 10 seeds | v2 S3c, S3d | +20.48, +17.87; −2.73, 9 losses | Verified |
| Q11 | RF ahead by 5–19 on 5 of 6; kNN ahead on Landsat, MAGIC, Letter, Skin, behind on PB, far behind on Thyroid | v2 | 5.01–18.53 (Skin 0.62); match | Verified |
| Q12 | "computed each accepted edit's certificates on 2,000 test rows" | run_refinement_v2 `min(2000, n_test)` | 548–2,000 rows | Wording W1 |
| Q13 | "Retraining with the new rows appended was not part of the plan" | PREREGISTRATION lines 101, 152 | it was planned, as a descriptive arm with no test | Wording W2 |
| Q14 | RQ4 table, 42 cells | v2 | 41 match; Letter tight certificate 1.546 is printed 1.6 | Mismatch P1 |
| Q15 | 0 misses in 46,153 edits (both certificates); 0 in 69,488 exploratory edits | v2 and explore | 0 / 46,153; 0 / 69,488 | Verified |
| Q16 | Tight certificate 1.2–35% of test rows; 0.2–3.7% of those changed; union 5–74% | v2 | 1.19–34.55; 0.15–3.74; 5.36–74.01 | Verified |
| Q17 | "Outside the tight certificate, PACE-flip's top 3 … changed on up to 51% of the explained rows (MAGIC)" | v2 `expl_tight_out_top3` = share of the explained rows outside the union of tight certificates | 0.51 of those rows (MAGIC) | Wording W10 |
| Q18 | Fix share beat shuffled retraining on all 6 by 12–54 points; fewer broken; fewer top-3 changes on all 5; 1.7–8.2 more accurate | v2 P4, S4a–c | +11.86 to +53.60; all significant | Verified |
| Q19 | Shuffled retraining changed 2.4–30% of predictions, appended 0.7–8% | v2 | 2.36–30.00; 0.73–8.00 | Verified |
| Q20 | Against appended retraining: higher fix share on all 6, 2.1–7.5 points more accurate, but broke more on Landsat, MAGIC, Letter | v2 | 6/6; 2.14–7.46; those 3 | Verified |
| Q21 | SR then AL changed PACE-flip's top 3 on 26–78% of explained rows | v2 | 0.26–0.78 | Verified |

## Deviations, settings, limitations, conclusion

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| D1 | Three deviations (LIME statistics in v3; conditional run restarted after 6 of 130 jobs; matched comparison missing from earlier drafts) | run logs, earlier review | match | Verified |
| D2 | Seeds per run, as listed in Settings | runs.csv seed columns | all match | Verified |
| D3 | Settings: θ, γ, KernelSHAP 200 samples, LIME 1,000, DiCE 1,000 and 5, ensemble 5, RF 100 and kNN 10 in repair, PACE-flip 100 rows | code | match | Verified |
| D4 | Software versions (Python 3.14, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, shap 0.54.1, lime 0.2.0.1, dice-ml 0.12) | installed environment | all match | Verified |
| L2 | Limitations repeat the numbers above (1–2%, 0.94–0.99, 63–70%, 55–79%, 0.2–21.5, 2.7, …) | as above | match | Verified |
| X1 | Conclusion repeats A1–A16 | as above | match | Verified |

## Section 7: against other readable models (revs 294–302)

Source: `results/readable_baselines/runs.csv` (pre-registered, seeds 70–89), recomputed by
`research_audit/scripts/claims_readable.py` (output in `claims_readable_out.txt`). Codes P1–P4 and
S1–S3 are the run's pre-registered comparisons.

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| RB1 | Abstract: an EBM was more accurate on 12 of 13 datasets and a tree on 10 (the explained model, θ = 0.3) | S1c, S1a | 0 / 1 / 12; 3 / 0 / 10 | Verified |
| RB2 | Abstract: choosing θ by validation helped on 7, with up to 23× as many boxes | S2; mean boxes | 7 / 6 / 0; ratios 6.24–23.21 (Skin 23.21) | Verified |
| RB3 | §1: in the base run CFMNN is less accurate than the better of NN and RF on every dataset; in a later test an EBM was more accurate on 12 of 13 | B4 (review_v1/accuracy); S1c | 13 of 13; 12 of 13 | Verified |
| RB4 | §7 table, 7 rows × 3 counts | P1, P2, P3, S1a, S1b, S1c, P4 | 5/0/8, 7/2/4, 2/2/9, 3/0/10, 4/2/7, 0/1/12, 4/3/6: all 21 match | Verified |
| RB5 | EBM: at θ = 0.3 level only on Wine, up to 14.2 points less elsewhere; tuned CFMNN beat it only on Skin and Letter | S1c, P3 | Wine level; MAGIC −14.20; Skin +2.54, Letter +7.37 | Verified |
| RB6 | θ by validation more accurate on 7, by 0.2–20.0 (Letter 20.0, Page Blocks 10.3, MAGIC 10.1), never less | S2 | +0.16 to +19.97; 19.97, 10.33, 10.12; 0 worse | Verified |
| RB7 | On 6 of these it chose θ = 0.05, the smallest tried, with 6–23× as many boxes (Letter 5,325 against 565) | settings and rules | θ = 0.05 in 13–20 of 20 seeds on Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks; 6.24–23.21×; 5,324.9 vs 564.8 | Verified |
| RB8 | The tuned tree had fewer leaves than CFMNN-0.3 had boxes on 10 of 13; a leaf tests 2.3–9.3 features; a box limits 79–100% of them | sizes | 10 (not Shuttle, Skin, Letter); 2.25–9.29; 0.79–1.00 | Verified |
| RB9 | Tuned CFMNN trailed the better of tuned kNN and RF on every dataset, by up to 11.9; at θ = 0.3 it tied the RF on Wine (98.3%) and trailed elsewhere by 0.2–21.3 | gap, per-dataset means | −11.92 to −0.05; Wine 0.000 (98.33 both); −0.21 to −21.29 | Verified |
| RB10 | Macro-F1: the same counts for the tree and LogReg; EBM (2, 3, 8), GFMM (4, 5, 4) | S3 | 5/0/8, 7/2/4, 2/3/8, 4/5/4 | Verified |
| RB11 | Several tuned models often chose the end of their grid (θ = 0.05, C = 100, k = 1) | settings | in more than 10 of 20 seeds: θ = 0.05 on 6 datasets, C = 100 on 6, k = 1 on 4 (and 1,024 leaves on Letter) | Verified |
| RB12 | Setup: 3-fold CV on the training rows; θ 0.5–0.05, 4–1,024 leaves, C 0.01–100; EBM main effects only and GFMM at θ = 0.3, both untuned, fixed before the run | PREREGISTRATION.md (pushed in 42a71d4 before run_log's start); scripts/run_readable_baselines.py | match | Verified |
| RB13 | Settings: seeds 70–89; interpret-core 0.7.8 (EBM), hyperbox-brain 0.1.6 (GFMM) | runs.csv seeds; installed versions | 20 seeds 70–89; 0.7.8, 0.1.6 | Verified |
| RB14 | Limitations: other readable models more accurate on most datasets; EBM main effects only, GFMM untuned, grid ends | RB1, RB11, RB12 | match | Verified |
| RB15 | Conclusion: "on every dataset in our base run" (was "on every dataset we tested", no longer true once the new run tied on Wine) | B4; RB9 | match | Verified |

## Section 7, abstract and §6: repair at the θ chosen by validation (revs 303–311)

Source: `results/repair_theta/runs.csv` (pre-registered, seeds 70–89), recomputed by
`research_audit/scripts/claims_repair_theta.py` (output in `claims_repair_theta_out.txt`). Codes P1,
P2 and S1–S3 are the run's pre-registered comparisons. All but the last are by accuracy.

| ID | Claim | Source | Recomputed | Status |
|---|---|---|---|---|
| RT1 | θ chosen by validation without repair was more accurate than repair at θ = 0.3 on 7 datasets, by up to 17.5 points (Letter); repair at θ = 0.3 was more accurate only on Thyroid | P1 | 1 / 5 / 7; worse by 0.09–17.53 (Letter −17.53); better: Thyroid +1.07 | Verified |
| RT2 | With repair at both θ, the chosen θ was more accurate on 8 datasets and never less | S2 | 8 / 5 / 0 | Verified |
| RT3 | At the chosen θ, repair was more accurate on 3 datasets, by at most 1.6 points, and never less | P2 | 3 / 10 / 0; Thyroid +1.55, Page Blocks +0.26, Skin +0.14 | Verified |
| RT4 | On Letter, MAGIC and Landsat the chosen θ left only 2–11 training errors on average | B0-val train_errors | Letter 5.70, MAGIC 10.95, Landsat 1.85 | Verified |
| RT5 | Macro-F1 gave similar counts | S3 | P1 1/6/6, P2 3/10/0, S1 7/6/0, S2 7/6/0 (accuracy: 1/5/7, 3/10/0, 7/6/0, 8/5/0) | Verified |
| RT6 | Repair at θ = 0.3 was still more accurate than no repair on 7 datasets and added no box | S1; boxes | 7 / 6 / 0; repair added a box in 0 of 260 pairs at θ = 0.3 (and 0 of 260 at the chosen θ) | Verified |
| RT7 | "A second test planned in advance, on the same seeds and split" | PREREGISTRATION.md, pushed in 22e687d (2026-11-07 10:40) before run_log's start (2026-11-08 01:06); B0 check | the 520 models without repair reproduce the readable-baselines run exactly (max difference 0.0) | Verified |
| RT8 | Settings: seeds 70–89 for this test | runs.csv seeds | 20 seeds, 70–89 | Verified |
| RT9 | Deviations: this test was planned in advance, with no change after the push | 22e687d; `git log` for scripts/run_repair_theta.py, pace_cfmnn/active.py, pace_cfmnn/refine.py | no commit to these files between 22e687d and the run | Verified |
| RT10 | Limitations: repair tried only 300 edits, a small share of the training errors on the large datasets (about 4,000 on MAGIC) | edits_logged; B0-0.3 train_errors | 300 used on all 20 seeds of Skin, MAGIC, Letter, Landsat and Page Blocks at θ = 0.3; MAGIC 4,048.35 on average | Verified |
| RT11 | Limitations: repair was tested with one budget and one gate, on 6 datasets with 10 seeds for RQ3 and RQ4, and on 13 with 20 seeds here | refinement_v2_confirmatory runs.csv; this run | 6 datasets, seeds 50–59; 13 datasets, seeds 70–89; budget 300 (RQ3: 273 on Page Blocks, where its pool was smaller), RQ3's gate | Verified |
| RT12 | Abstract: "On all 13 datasets, every training error of CFMNN we checked is a tie between a box of the right class and a box of the wrong class" (was "On the 6 datasets we checked") | this run's B0 rows; A10 | all 520 models without repair (13 datasets × 20 seeds × 2 θ): train_errors_in_winning_box = train_errors_own_box_full = train_errors | Verified |
| RT13 | Abstract: "In a second planned test, it did not match choosing θ by validation, which stayed more accurate on 7 of 13 datasets, but it added no boxes" | P1; boxes | 1 / 5 / 7; repair added a box in 0 of 520 models | Verified |
| RT14 | §6: "it held for all 347,339 training errors on all 13 datasets, at both θ, over 20 seeds" | this run's B0 rows | 347,339 errors over the 353 of 520 models that had any; 520 of 520 all ties | Verified |

## Pre-registered endpoints: all reported?

| Run | Endpoints in the plan | In the paper |
|---|---|---|
| Explanations v3 (seeds 30–39) | PACE-SHAP vs KernelSHAP (deletion, main); insertion; strong KernelSHAP; PACE-SHAP-20; stability; ground truth | All reported, including Shuttle and Page Blocks, where KernelSHAP is ahead (not significant) |
| Review tests (seeds 50–69) | exact values and speed; raw and untied curves; stability with new rows; LIME with 200 rows; ensembles; accuracy | All reported, including the bagged ensemble doing as well (against the order-ensemble story) |
| Conditional (B3) | conditional vs usual; matched KernelSHAP; realism; noisy copy | All reported, including "rarely more faithful" and "less credit to the noisy copy, not more" |
| Counterfactuals (B1) | PACE-sparse vs DiCE | Reported, including Shuttle's loss and BC's slower time |
| Global importance (B2), interactions (B4) | as planned | Reported, including the weak B4 test |
| Trust, confirmatory | PD+U\* vs margin; EI; U\* vs U; vs baselines | All reported, including EI failing and the baselines winning |
| Trust, combination | order ensemble added; geometry signals | Reported, including Digits being lower |
| Repair v1 (seeds 10–19) | R1 (explained cut, main), R2, R3, S1 | Reported; R1 failed and is stated |
| Repair v2 (seeds 50–59) | P3, S3a–d, P4, S4a–c, C4 | All 9 endpoints reported, including S3b holding only on Letter and S3d (MAGIC's lowest recall falls) |
| Readable baselines (seeds 70–89) | P1–P4, S1–S3; descriptive gap, settings, sizes, times | All reported in §7, including CFMNN losing to EBM on most datasets, to the tree at θ = 0.3, and θ = 0.3 being a poor choice on 7 datasets. Times are in FINDINGS §9 only |
| Repair at the chosen θ (seeds 70–89) | P1, P2, S1, S2, S3; descriptive boxes, errors, edits, fixed and broken, gap, times | P1, P2, S1, S2 and S3 reported in §7, including repair at θ = 0.3 losing to the chosen θ on 7 datasets. The descriptive items are in FINDINGS §10, and the edit budget in Limitations |


## Follow-up tests of the limitations (seeds 110–119; paper rev 312–326)

Checked by `research_audit/scripts/claims_followup.py`; its output is in `claims_followup_out.txt`.

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| FU1 | §7: "PACE-SHAP was more faithful than KernelSHAP at its usual setting on 7 of 11 datasets by deletion and 8 by insertion, and never less … same 200 rows … better on 4 and 5" | C | 7/4/0, 8/3/0; same rows 4/7/0, 5/6/0 | Verified |
| FU2 | §7: "The counts were the same on untied points only, and the raw margin curves favoured PACE-SHAP on all 11" | C | untied as FU1; raw 11/0/0 both | Verified |
| FU3 | §7: "on 3–6 of the 6 datasets at every setting, and never less"; "1–3 effective rows at σ = 0.05 and 27–120 at σ = 0.4"; "Letter 10% … 34%"; Limitations "Shuttle 67%" | D | 3–6 better, 0 worse; 1.0–2.9 and 27.4–119.8; 10% → 34%; Shuttle 49% → 67% | Verified |
| FU4 | §7: "PACE's answers moved 2.5–13 times less … PACE-sparse's 1.5–10 times less, on all 11"; r3 counts 3 / 1 and 7 / 1 | B | 2.53–12.81 and 1.54–9.70; r1 11/0/0 for both; r3 1/7/3 and 1/3/7 | Verified |
| FU5 | §7: budgets "only on Letter (−1.1, +1.4 and +2.1)"; no gate worse on all 6 "by 0.7–10.3"; strict worse on 3, loose on 1, cumulative on 2; MAGIC recall 70.8 / 69.6 / 68.0 | A | as stated | Verified |
| FU6 | §7: "6,272 accepted edits … 0.8% … 6.3% … 38% … 6% … missed no change … 5.0% … 99% … 55% … 23%" | A | 0.79, 6.33, 38.29, 6.22; 4.98, 99.44; 55.40, 23.33; 0 misses | Verified |
| FU7 | §7: "all 14,827 edit decisions the same … all 60 runs … 2–22 times faster (12 times in total)" | E | 14,827 with 0 mismatches; 60/60; 2.1–21.9; 11.9 | Verified |
| FU8 | Deviations: DiCE kdtree left out of r1, "about 25 seconds" per call | B pre-registration | 25.3 s per call on Wine (smoke slice) | Verified |

| Run | Endpoints in the plan | In the paper |
|---|---|---|
| E (seeds 50–59) | Pass rule | Reported (PASS) |
| A (seeds 110–119) | Each arm vs SR-default; certificates | All 7 arms and 5 certificates reported, including the gates that did worse |
| B (seeds 110–119) | PACE and PACE-sparse vs DiCE random, on r1 and r3 | All 4 reported, including the r3 losses, which are also in Limitations |
| C (seeds 110–119) | PACE-SHAP vs KernelSHAP (usual setting and same rows), by deletion and insertion | All 4 reported, plus the untied and raw versions |
| D (seeds 110–119) | The -cond pair at each setting | All 8 reported as a range: better on 3–6, never worse |

## Peer-review revision (paper revs 327–443)

Checked by `research_audit/scripts/claims_peer_review.py`; its output is in `claims_peer_review_out.txt` (78 checks, all OK after the fixes). Sources are under `results/`. Section numbers are those of rev 443. Seeds: second explanation run 30–39 (`faithfulness_v3`), like for like 40–49 (`extensions_v1/conditional`), tie test 50–59 (`review_v1/faithfulness`), second repair run 50–59 (`refinement_v2_confirmatory`), readable models and repair at the chosen θ 70–89. The script refits nothing except Figure 1's 2-D case. The base-model tie counts (PR5) come from `peer_review/ties_all.csv`, which `scripts/peer_review_reanalysis.py` refitted.

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| PR1 | Abstract, §1, §7: like for like, PACE-SHAP and KernelSHAP "equivalent on 12 of 13 datasets, by deletion and by insertion"; Landsat the exception, "−0.045, 95% interval −0.053 to −0.036"; at KernelSHAP's usual setting "equivalent on 6 of 11 by deletion" | extensions_v1/conditional; faithfulness_v3; margin a tenth of KernelSHAP's lead over random | 12/13 both, Landsat not; −0.0446 (−0.0535 to −0.0357); 6/11 | Verified after P6 (rev 439 said −0.054) |
| PR2 | §7: S1–S3 as one unit, more faithful "on 6 of 9 units by deletion and by insertion, and like for like on 3 of 9 by deletion"; across units "p = 0.02 … by both tests", like for like "p = 0.20 by deletion and p = 0.004 by insertion, where all 9 … favour PACE-SHAP (median 0.0015)" | faithfulness_v3; conditional | 6/3/0 both; 3/6/0; p 0.0195, 0.0195, 0.2031, 0.0039; 9 of 9, +0.0015 | Verified |
| PR3 | §7: the contrastive explanation, score breakdown and PACE-multi rank worse than KernelSHAP "on 9, 11 and 6 of 11"; all beat random "except the score breakdown on 2" | faithfulness_v3 | 9, 11, 6 worse; breakdown 9/2/0 against random (level on Thyroid, Page Blocks) | Verified |
| PR4 | §7 table of tied and untied points: share tied, raw deletion AUC and untied deletion AUC with verdicts, 11 datasets | review_v1/faithfulness | all 33 cells; 9 datasets with at least 10 untied points in every seed | Verified |
| PR5 | §7 Ties: the nearest-centroid rule decides "0% of test rows on S1–S3 and Wine, under 1% on Breast Cancer and Digits, 5% … 76% on MAGIC"; tied rows hold "90–100% of test errors on Skin, MAGIC, Letter and Page Blocks, 83% on Shuttle and 53–60% on Landsat and Thyroid" | peer_review/ties_all.csv (base model, seeds 0–4) | 0; 0.35, 0.67; 5.3, 12.6, 20.2, 24.7, 56.5, 72.8, 76.3; 90.3–99.9, 83.1, 53.4–59.8 | Verified |
| PR6 | §9: self-repair's macro-F1 gain with 95% intervals on the 6 datasets; SR and AL "not shown to be equivalent within 1 point on any dataset" | refinement_v2_confirmatory | as printed; Letter 2.25 (1.78 to 2.72); 0 of 6 equivalent | Verified after P6 (rev 439 said 2.3 on Letter) |
| PR7 | §9: appended retraining "changed macro-F1 by −2.6 to +0.8 points, significantly only on Letter (+0.85)", shuffled "by −3.4 to +1.8 points, never significantly" | refinement_v2_confirmatory | −2.59 to +0.85, Letter only; −3.4 to +1.8, none | Verified after P6 (rev 439 said +0.9) |
| PR8 | §10 table: broken predictions for SR then AL, shuffled and appended retraining (18 cells), beside fix share | refinement_v2_confirmatory | all 18 cells; fix shares as Q14 | Verified |
| PR9 | §8: the signals "took longer than the three standard methods together on 14 of 16 datasets (Letter: 87 s against 9 s per seed)" | error_detection_confirmatory | 14/16; 87.0 and 9.1 s | Verified |
| PR10 | §6: training error at θ = 0.3 per dataset group; "2.6 training rows per box, against 10.7–3,336 on the real datasets other than Wine and Digits" | repair_theta, B0-0.3 | 0; 0.2–1.6; 4.0; 10.4–14.4; 21.3–26.6; 2.6; 10.7–3,335.5 | Verified |
| PR11 | §1: "14–1,533 boxes, each with conditions on 79–100% of the features" | repair_theta and readable_baselines (CFMNN-0.3) | 14.0 (Shuttle) to 1,533.3 (S2) in both runs; 79–100% | Verified |
| PR12 | §5: accepted shrinks cut off "2% of the training rows on Skin, 6% on Page Blocks, 10% on Letter, 22% on Landsat, 32% on MAGIC and 48% on Thyroid" | refinement_v2_confirmatory/edits.csv.gz (SR, accepted), over n_train | 2, 6, 10, 22, 32, 48 | Verified |
| PR13 | §7: after a refit a PACE counterfactual still gives the target class "in 17–55% of refits on 10 of the 11 datasets (98% on Shuttle)", PACE-sparse "in 24–42% (98% on Shuttle)" | cf_robustness, r3 (seeds 110–119) | as printed | Verified |
| PR14 | Figure 1 caption: "2 training rows are cut off", certificate "17% of the square", prediction changed on "0.2%", "inside the certificate" | scripts/figure_boxes.py (refit) | 2; 17%; 0.2%; 0 grid points changed outside | Verified |
| PR15 | §6: "the smallest two-sided Wilcoxon p is 0.002"; across 11 datasets "at least 9 of 10 seeds", across 13 "all 10" | exact Wilcoxon null, n = 10 | 0.00195; 9 agreeing 0.0039 < 0.05/11, 8 agreeing 0.0098; only 0.00195 < 0.05/13 | Verified |
| PR16 | §6 Setup: S1–S3 generators (5,000 rows, 10 uniform features, 2x₀ + 1.5x₁ − x₂ above 1.25; S1 margin 0.1; S2 noise 0.25; S3 rule and margin 0.02) | pace_cfmnn/data.py | as printed | Verified |
| PR17 | Appendix C: software versions | requirements-lock.txt (new) | all 9 match | Verified |
| PR18 | Appendix C: primary comparisons, 17 runs | each results/*/PREREGISTRATION.md | 16 rows matched; the extensions run's realism primary was missing | Verified after P7 (rev 443) |
| PR19 | §6: the base paper's CoverType (581,012 rows, 54 attributes) and Crop mapping (325,834 rows, 175 attributes) left out | UCI dataset pages (not a result of ours) | 581,012 × 54; 325,834 × 175 | Verified (web, 2026-11-13) |

## Repair order and repair time (Runs H and T; paper revs 444–449)

Checked by `research_audit/scripts/claims_order_timing.py`; its output is in `claims_order_timing_out.txt` (30 checks, all OK). H: `results/repair_order/runs.csv` (seeds 130–139). T: `results/repair_timing/runs.csv` (seed 50).

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| OT1 | §9: greedy "more accurate than the random order on MAGIC (+4.9 macro-F1 points) and Letter (+3.0), less accurate on Skin (−1.1), and not significantly different on Thyroid, Landsat and Page Blocks" | H, primary | +4.87, +2.98, −1.08; 3 level | Verified |
| OT2 | §9: the random order spent "108 on MAGIC and 260 on Skin" tries on rows already right; greedy "spent none on MAGIC and Letter" | H, skip_already_right | 108, 260; 0, 0 | Verified |
| OT3 | §9: against real tries, greedy "still more accurate on MAGIC (+4.7) and Letter (+2.7), and not significantly different elsewhere, Skin included (−0.2)" | H, amendment A1 | +4.65, +2.67; 4 level; −0.16 | Verified |
| OT4 | §9: on Skin random-300 "reached the gate only 39 times", and continuing to 300 checked edits "lowered macro-F1 from 98.9 to 98.0 (not tested)" | H, edits_gated, macro-F1 | 39; 98.88 and 97.96 | Verified |
| OT5 | §9: greedy-300 "not significantly different from trying every training error on 5 datasets and more accurate on MAGIC (+8.0)" | H, secondary | 5 level; +7.97 | Verified |
| OT6 | §9: "took longer than trying them all, on every dataset (Skin 1,160 s against 27 s per seed …)"; "MAGIC and Letter … 241 s and 420 s, against 5 s and 12 s" | H, E1 and seconds | slower on 6/6; 1,159.9 and 26.6; 241.2, 419.7, 5.2, 11.8 | Verified |
| OT7 | §9: the box order "less accurate than the random order on Letter and Skin, and more accurate nowhere" | H, secondary | worse on Letter, Skin; box-1000 6 level | Verified |
| OT8 | §9 Time: CFMNN "0.04–1.9 s", self-repair "0.2–4.7 s (2–45 times faster than with the gate RQ3 used)", active learning "3.9–142 s, 5–56 times as long", SR then AL "4.6–145 s"; every setup equal to RQ3 | T | 0.04–1.87; 0.20–4.68; 2.4–45.0; 3.92–142.0, 4.8–56.3; 4.57–145.0; 30/30 match | Verified |
| OT9 | §9 Time: leaving out AL "saves 82–98% of the time" and gives up SR then AL's gains on Landsat (+1.1), MAGIC (+2.0), Letter (+0.7) | T; RQ3 P3 (row Q7) | 82.1–98.3%; as Q7 | Verified |
| OT10 | Limitations, Next steps, Appendix C (seeds 130–139 and seed 50; desktop; primary comparisons row) | H and T plans | match | Verified |

## Self-repair as the method (paper revs 450–453)

Checked by `research_audit/scripts/claims_sr_method.py`; its output is in `claims_sr_method_out.txt` (51 checks, all OK). RQ3: `results/refinement_v2_confirmatory/runs.csv` (seeds 50–59). T: `results/repair_timing/runs.csv` (seed 50). H: `results/repair_order/runs.csv` (seeds 130–139). Tests redone with `common.paired`.

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| SM1 | §9 table, macro-F1: starting model 84.0 … 93.1; self-repair +6.0, +1.5, +2.6, +3.3, +2.2, +5.7; active learning +6.4 … +6.1; SR then AL +6.6 … +6.2; SR then AL over self-repair +0.6, +1.1, +2.3, +2.0, +0.7, +0.6 | RQ3 | 83.96 … 93.10; 6.02, 1.48, 2.58, 3.27, 2.25, 5.66; …; 0.57, 1.08, 2.28, 2.03, 0.73, 0.55 | Verified |
| SM2 | §9 table, bold cells: self-repair over the starting model on all 6; SR then AL over self-repair on Landsat, MAGIC, Letter | RQ3, S3a and P3 (pre-registered) | 6 of 6 better; Landsat, MAGIC, Letter better, none worse | Verified |
| SM3 | §9 table, lowest class recall: starting model 59.2 … 80.9; self-repair +19.4, +6.4, +0.1, −2.7, +2.9, +16.7; active learning +19.2, +8.0, −1.9, +0.4, +1.9, +17.4; SR then AL +20.5, +9.1, −1.3, −1.2, +3.8, +17.9 | RQ3 | 59.19 … 80.92; 19.37, 6.41, 0.08, −2.73, 2.93, 16.65; 19.24, 8.02, −1.87, 0.35, 1.93, 17.39; 20.48, 9.14, −1.26, −1.22, 3.76, 17.87 | Verified |
| SM4 | §9 table, labels 1,020, 944, 821, 2,202, 2,300, 24,806; seconds, self-repair 0.2, 1.4, 0.8, 2.1, 4.7, 2.5; SR then AL 6.8, 17.1, 4.6, 33.3, 117.5, 145.0 | RQ3 total_labels; T | same each seed; 0.20, 1.44, 0.82, 2.15, 4.68, 2.52; 6.79, 17.07, 4.57, 33.29, 117.47, 145.04; T 30/30 match | Verified |
| SM5 | §9 "Self-repair as the method": labels on top "added +0.6 to +2.3 macro-F1 points, significantly on 3 of 6 datasets, at a cost of 821–24,806 labels and 5.5–57 times the time"; active learning alone "ahead … by 2.3 points on Page Blocks and 1.8 on MAGIC"; self-repair "lowered MAGIC's lowest class recall by 2.7 points" | RQ3; T | 0.55–2.28, 3 of 6; 821–24,806; 5.55–57.48; 2.30, 1.80; −2.73 | Verified |
| SM6 | §10 "Self-repair alone (post hoc)": fix share "59.0–87.5%, against 40.5–52.4%", "higher on every dataset, by 6.6–47.0 points, and significantly on 4 (not Landsat or Page Blocks)" | RQ3, churn_fix_share, post hoc | 58.96–87.50; 40.54–52.36; +6.61 to +46.97, Holm p 0.012 on 4 and 0.074 on Landsat, Page Blocks | Verified |
| SM7 | §10: broke fewer predictions "on all 6 datasets, by 0.8–8.4 points of the test rows"; changed PACE-flip's top 3 on fewer rows "on all 5 datasets tested, by 22.9–64.5 points, all significantly"; "None of its 6,233 accepted edits changed a prediction outside its certificate, tight or plain" | RQ3, churn_broken_share, expl_topk_change_share (a share, ×100), cert_* | 0.83–8.35, 6 of 6; 22.9–64.5, 5 of 5; 6,233 edits, 0 and 0 misses | Verified |
| SM8 | §9 Run H paragraph, last two sentences: a smarter order "can help, on MAGIC and Letter, but there it took 36–46 times as long as the random order's 300 tries, and 4.5–132 times as long across the 6 datasets, with no gain on the other 4" | H, seconds, primary | 35.7 and 46.1; 4.48–132.3; better only on MAGIC, Letter | Verified |
| SM9 | Conclusion: bought labels added less, on 3, "at 5.5–57 times the time" | T; RQ3 P3 | as SM5 | Verified |

## Runs F, G, I, J and K (paper revs 455–478)

Checked by five scripts in `research_audit/scripts/`, each with its `*_out.txt`: `claims_contraction.py` (Run F, 42 checks), `claims_systems.py` (Run G, 57), `claims_generality.py` (Run I, 22), `claims_sampled.py` (Run J, 18) and `claims_covertype.py` (Run K, 20); all 159 OK. The post hoc checks of Run I's exact check are `check_exact_simpson.py` (which side is in error) and `check_exact_isclose.py` (the cause, 5 checks OK). Every value is rounded from the raw CSV (`runs.csv`, `exact.csv`, `timing.csv`), never from a summary. Tests redone with `common.paired`. Sources: F `results/contraction/` (seeds 120–129), G `results/systems/` (120–129; timing pass seed 120), I `results/generality/` (150–159), J `results/sampled_baselines/` (160–169; timing pass 160–162), K `results/covertype/` (170–174). All run on the home PC.

Four values in the first drafts of the write-up came from the rounded summaries and were wrong at the paper's rounding; the raw values were used instead: Run F's S1 on Skin (9.0, not 9.1) and S4a on Landsat (2.0, not 1.9), self-repair's evicted rows on Letter (9%, not 10%), and CFMNN's accuracy on CoverType (51.4%, not 51.5%).

**Rerun on the laptop (2026-11-15).** All five scripts ran again, with every check OK. Three of the scripts reproduce their `*_out.txt` byte for byte. In the other two, three p-values printed differently, all on comparisons whose verdict is level either way:

| Script | Comparison | Laptop | Saved output and `summary.md` |
|---|---|---|---|
| `claims_contraction.py` | S9 (accuracy, SR vs SC-all), Page Blocks | 0.0840 | 0.0879 |
| `claims_contraction.py` | S7 (broken %, SR vs SC-all), Landsat | 0.8652 | 0.9219 |
| `claims_generality.py` | P1 (γ = 2 vs 4), Thyroid | 0.8867 | 0.8047 |

The cause is pandas' default CSV float parser, which is not exact in the last bit. On the laptop it reads one value about 1e-14 differently. In the Page Blocks case, that breaks an exact tie between two seeds' differences (+0.365 and −0.365 points, one test row each way), and the Wilcoxon p changes. With `float_precision="round_trip"`, both scripts reproduce their saved output byte for byte. So the saved values are the exact ones, and no verdict or quoted number changes.

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| CF1 | Abstract: self-repair "more accurate than Simpson's contraction … on all 6"; §9 "Against contraction": "on all 6 datasets, by 2.4–22.5 macro-F1 points" | F, P1 | 6 of 6 higher; +2.44 to +22.46 | Verified |
| CF2 | §9: "than the same contraction at the same training errors, without a gate, on all 6, by 1.7–9.0" | F, S1 | 6 of 6; +1.7 to +9.0 (raw) | Verified |
| CF3 | §9: contraction on every overlap "lowered macro-F1 on MAGIC (−13.0) and Skin (−14.9) and changed it significantly nowhere else" | F, S4b | −13.01, −14.85; 4 level | Verified |
| CF4 | §9: on Thyroid "Simpson's test found no overlap on any seed", so the primary equals self-repair against the starting model | F, edits_kept and macro-F1 | 0 contractions on all 10 seeds; SC-all = B0 on every seed | Verified |
| CF5 | §9: against GFMM from scratch "more accurate on Thyroid, MAGIC and Skin, not significantly different on Landsat and Page Blocks, and less accurate on Letter (76.9 against 82.3, −5.4)" | F, S3 | as printed; 76.9, 82.3, −5.39 | Verified |
| CF6 | §9: GFMM's rule "on every overlap … left 21–90% of the training rows in no box"; not presented as wins over GFMM (Amendment 1) | F, GC-all rows_evicted_pct; S2a, S2b | 21.1–89.6; S2a, S2b 6 of 6 | Verified |
| CF7 | §9: "300 tries on every dataset … on Page Blocks … more than RQ3's 273" | F, edits_tried ≤ 300; plan Amendment 1 | max 300 | Verified |
| CF8 | §5 "Rows a shrink cuts off": self-repair left more rows in no box on Thyroid (59% against 0%), Landsat (24% against 12%) and Letter (9% against 0.8%), fewer on MAGIC (33% against 89%) and Skin (3% against 18%) | F, rows_evicted_pct; S8 | 59.46/0, 23.54/12.47, 9.496/0.847, 32.65/89.28, 2.78/17.80; S8 worse on 3, better on 2 | Verified |
| CF9 | Conclusion (rev 480): "self-repair was more accurate than Simpson's contraction on all 6 repair datasets" | F, P1 (`claims_contraction.py`) | as CF1: 6 of 6 higher, +2.44 to +22.46 macro-F1 | Verified |
| SY1 | Abstract, §6, Limitations: CFMNN with PACE-SHAP "less accurate than an explainable boosting machine on 10 of 11 … and than a decision tree with TreeSHAP on 9", "slower to explain on all 11"; §6 "level on Wine", "more accurate on Wine and Breast Cancer" | G, S2 accuracy and S3 ms | EBM 0/1/10 (level Wine); tree 2/0/9 (better Wine, Breast Cancer); ms worse on 11 against both | Verified |
| SY2 | §6: by the lead, against EBM "more faithful on 2 (S1, Page Blocks), level on 6 and less faithful on 3 (Breast Cancer, Shuttle, Letter)", against the tree "3 (S1, S2, Page Blocks), level on 4 and less faithful on 4 (Wine, Breast Cancer, Shuttle, Letter)" | G, P1 and P2 | as printed | Verified |
| SY3 | §6: Landsat read from the gain under the near-zero rule, "less faithful against both"; by the gain alone "less faithful on 10 of 11 … against EBM and on 9 against the tree" | G, near-zero rule; S4 gain | flagged: Landsat only; worse both; gain 0/1/10 and 0/2/9 | Verified |
| SY4 | §6: on untied points "Letter's verdict turns from less to more faithful against both, Landsat's holds, and MAGIC and Page Blocks (77% and 75% of points tied) have too few untied points" | G, amendment 1 | Letter worse → better (both); Landsat holds; MAGIC 77.0%, Page Blocks 74.6% not eligible | Verified |
| SY5 | §6 table: accuracy, rules, ms per explanation timed alone, deletion lead and gain for the three systems on 11 datasets (165 cells) | G runs.csv (means over seeds); timing.csv (seed 120) | all cells as printed (tree leaves to 1 dp) | Verified |
| SY6 | Limitations: "much faster to explain"; the EBM and the tree "more accurate … on 10 and 9 of 11" | G | as SY1, SY5 | Verified |
| SY7 | Appendix C Deviations: the first attempt "failed on 3 of 110 tasks", thresholds rounded down, "the whole run was repeated (its Amendment 2)" | G plan, Amendment 2; `run_log.txt` of the reported run (0 FAILED) | as printed | Verified |
| SY8 | Conclusion (rev 480): "As a whole system, CFMNN with PACE-SHAP was also less accurate than an explainable boosting machine on 10 of 11 explanation datasets and than a decision tree with TreeSHAP on 9, and slower to explain on all 11" | G, S2 accuracy and S3 ms (`claims_systems.py`) | as SY1: EBM 0/1/10, tree 2/0/9, ms worse on 11 against both | Verified |
| GE1 | §3: "A planned test at γ = 2, 4 and 8 (seeds 150–159, 13 datasets) found no training error, of 569,239, that was not such a tie" | I, train_errors and train_errors_tied_inside_both | 569,239; 0 | Verified |
| GE2 | §3: after Simpson's contraction "on S1–S3, Breast Cancer, Shuttle, Skin and MAGIC only 0–8% of the training errors were still ties" | I, Simpson, pooled | 0.0, 0.0, 0.0, 3.6, 0.1, 7.7, 0.2 | Verified |
| GE3 | §3: on the contracted boxes "PACE-SHAP matched … ExactExplainer to within 1.6e-10, inside the limit of 1e-9"; post hoc "PACE-SHAP matched a brute-force Shapley sum to 1.3e-16 in the three worst cases, so the small differences come from ExactExplainer" | I, exact.csv; `check_exact_simpson_out.txt` (post hoc) | 1.60e-10; S3 150, S1 152, MAGIC 158: PACE vs brute force ≤ 1.30e-16, ExactExplainer 7.1e-11–1.6e-10 | Verified, but "three worst cases" was wrong: the three checked are the 1st, 2nd and 5th largest of five (S2 159 is 3rd); reworded at rev 479, see GE6 |
| GE4 | Appendix C Settings: γ = 2 "more accurate … on S1 (+2.1 points) and Digits (+1.0)", γ = 8 "less accurate on S1, S2, S3 and Digits (−0.9 to −4.3)", "no significant difference elsewhere" | I, P1 and P2 | +2.09, +0.97; −4.26, −2.64, −3.85, −0.94; 11 and 9 level | Verified |
| GE5 | Limitations: PACE run "on CFMNN and on CFMNN's boxes after Simpson's contraction, not on Simpson's classifier trained from scratch" | I plan ("What Simpson's FMM means here") | as printed | Verified |
| GE6 | §3 (rev 479): brute force matched PACE-SHAP "in three of the five tasks that differed by more than 1e-14 (the two largest and the smallest)"; ExactExplainer "treats a background value within numpy's isclose tolerance (relative 1e-5) of the explained point's as unchanged and does not call the model again for it"; "Each of the five tasks has one such value"; with exact equality "ExactExplainer matched PACE-SHAP to 1.3e-16" | I, exact.csv; `check_exact_isclose_out.txt` (post hoc, shap 0.54.1) | 5 tasks above 1e-14 (S3 150 1.60e-10, S1 152 1.27e-10, S2 159 1.08e-10, S3 159 9.86e-11, MAGIC 158 7.14e-11), the other 55 ≤ 2.7e-16; one near-equal value each, gaps 3.3e-6–5.7e-6 (MAGIC 152 also has one, gap 2.9e-7, difference 1.3e-16); as shipped each point reproduces its committed difference; exact equality ≤ 1.336e-16 | Verified (post hoc) |
| SB1 | §7 "Above 12 features": by deletion "better on 1 dataset (Landsat), level on 4 and worse on none against each"; "at most 0.0006 elsewhere" | J, P1 and P2 | Landsat −0.0045 and −0.0057; 4 level each; 0.0006 | Verified |
| SB2 | §7: by insertion "better on 1 (Wine) against PermutationSHAP and on 2 (Thyroid, Landsat) against paired KernelSHAP, and never worse" | J, secondary | as printed | Verified |
| SB3 | §7: error "28–40% of theirs (relative L1 error: Wine 0.010 against 0.029 and 0.025; Letter 0.014 against 0.046 and 0.049)" | J, rel_l1_error | 0.28–0.40; as printed | Verified |
| SB4 | §7: top 3 "most stable under a new seed on all 5 datasets (Landsat 0.95 against 0.76 and 0.71)"; "the two samplers took 3.8–9.6× as long" | J, stability_top3; timing.csv | 5 of 5; 0.95, 0.76, 0.71; 3.8–9.6 | Verified |
| SB5 | §7: equal budget, "the same 200 rows (71 on Wine)" | J, rows_per_explanation ≤ budget_rows; background_rows | every task within budget | Verified |
| CT1 | §6 "One large dataset": "about 119 boxes", "51.4% test accuracy, against 92.7% for 10-nearest neighbours and 95.6% for a Random Forest"; Limitations the same three figures | K | 119.0; 51.45, 92.67, 95.63 | Verified (51.4 from the raw mean, 51.447) |
| CT2 | §6, §3: "All 1,128,010 of its training errors were ties"; "the nearest-centroid rule decided 98.9% of its test predictions" | K, pooled | 1,128,010 of 1,128,010; 98.89% | Verified |
| CT3 | §6: "trained in 10 s", "PACE-flip and PACE-SHAP took 1.1 ms and 454 ms per explanation" | K | 10.26 s; 1.12 ms; 453.7 ms | Verified |
| CT4 | §6 Datasets: CoverType "ran … once on its own (Section 6)"; Appendix C seeds 120–174 and the desktop | the five plans; HOME_PC_RUNBOOK.md §7 | match | Verified |

## Runs L, M and N (paper revs 481–493)

Checked by three scripts in `research_audit/scripts/`, each with its `*_out.txt`: `claims_covertype_theta.py` (Run L, 36 checks), `claims_simpson.py` (Run M, 30) and `claims_sampled_covertype.py` (Run N, 26); all 92 OK. The post hoc check of Run M's exact check is `check_exact_isclose_simpson.py` (5 checks OK). Every value is rounded from the raw CSV, read with `float_precision="round_trip"`, never from a summary. No test is made: Runs L and N are descriptive, and Run M is a pass-or-fail check plus a descriptive comparison. Sources: L `results/covertype_theta/` (seeds 210–214, a stratified 10% of CoverType), M `results/simpson_fmm/` (220–229), N `results/sampled_baselines_covertype/` (230–234; timing pass 230–232). All run on the home PC (`research_audit/HOME_PC_RUNBOOK.md` §8).

Three values first guessed while writing the scripts were wrong and were corrected before they reached the paper or FINDINGS: the mean macro-F1 gain of self-repair at θ = 0.05 (+0.5, not +0.6; the raw mean is 0.548), CV accuracy at θ = 0.05 (72.4–73.4%), and one insertion difference against paired KernelSHAP that is −3.5e-18 (counted as level, not behind).

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| CL1 | §6 "One large dataset" (rev 483): "On a stratified 10% of CoverType (seeds 210–214, descriptive), the θ chosen by validation was 0.05, the smallest we tried, on every seed" | L, theta and cv_scores | θ = 0.05 on 5 of 5; CV accuracy highest at 0.05 on every seed (72.4–73.4%) | Verified |
| CL2 | §6: "It raised test accuracy from 50.7% to 72.9%, with 34–42 times as many boxes (about 3,010) and 9.3 s per PACE-SHAP explanation instead of 0.3 s" | L | 50.73 and 72.93; box ratio 34.1–41.7 per seed; 3,009.6; 9.25 s and 0.30 s | Verified |
| CL3 | §6: "Self-repair then added +3.1 points at θ = 0.3 and +0.4 at θ = 0.05, gaining on every seed and adding no box; before it, every training error was a tie" | L | +3.13 (+0.52 to +5.31) and +0.44 (+0.06 to +0.61); every accuracy and macro-F1 gain > 0; boxes never rose; 113,545 of 113,545 and 45,542 of 45,542 ties | Verified |
| CL4 | §6: "The 10% share was fixed before the run because the full data would have taken about 50 hours"; Appendix C Deviations (rev 491): "a trial projected about 50 hours and about 58 GB for the gate, so that run's pre-registration fixed a stratified 10% of the rows" | L plan (`bd0d023`), "What changed because of these"; subsample = 0.1 in runs.csv | as printed | Verified |
| CL5 | Limitations (rev 487): "On a 10% sample of it, θ chosen by validation reached 72.9%, with 34–42 times as many boxes" | L | as CL2 | Verified |
| SF1 | §3 (revs 481–482): PACE run "on Simpson's classifier trained from scratch", built with "CFMNN's size limit and expansion, with Simpson's overlap test and contraction after every expansion and ties to the box made first" | M plan ("The model") | as printed | Verified |
| SF2 | §3: "PACE-SHAP matched ExactExplainer to within 1.4e-10, and on 59 of the 60 tasks to 3.1e-16" | M, exact.csv | 1.35e-10 (S1, seed 222); the other 59 ≤ 3.09e-16 | Verified |
| SF3 | §3: "the other task holds one value of the same kind, and with exact equality the two matched to 1.0e-16 (a check made after the run)" | `check_exact_isclose_simpson_out.txt` (post hoc, shap 0.54.1) | one near-equal pair, gap 4.1e-6; as shipped 1.350e-10 = committed; exact equality 1.041e-16 | Verified (post hoc) |
| SF4 | §3: "less accurate on Shuttle (85.8% against 99.8%), Letter (67.0% against 75.3%), MAGIC and Thyroid, more accurate on Skin, Breast Cancer and Page Blocks, and within 0.3 points elsewhere" | M, runs.csv (means over 10 seeds) | −14.0, −8.3, −1.9, −1.0; +2.1, +1.1, +0.7; the other 6 within 0.27 | Verified (descriptive; grouped at 0.5 points) |
| SF5 | §3: tie share "ranged from 0–2.1% (S2, Breast Cancer, Shuttle, MAGIC) to 98.7% (Letter) and 100% (Thyroid, where no contraction happened)" | M, runs.csv, pooled | 0.0, 0.0, 0.7, 2.1; 98.7; 100.0 with 0 contractions on every seed | Verified |
| SF6 | Limitations (revs 485–486): "tested it in full only on CFMNN: on CFMNN's boxes after Simpson's contraction and on Simpson's classifier trained from scratch (built with CFMNN's size limit and expansion), we checked only that PACE-SHAP's values are exact" | I and M plans | as printed | Verified |
| SF7 | Appendix C Deviations (rev 491): the plan's overlap test "was replaced … the run counted the pairs that still overlap (none on 11 datasets, at most 4 per model on Landsat and Letter)" | M plan ("A deviation from PART_B_PLAN.md"); runs.csv overlapping_pairs | 0 on 11 datasets; largest 4 (Landsat) | Verified |
| SN1 | §7 "Above 12 features" (rev 484): "the three were close by deletion (margin AUC −0.215, −0.215 and −0.213 for PACE-SHAP, PermutationSHAP and paired KernelSHAP)" | N, runs.csv | −0.2151, −0.2153, −0.2128 | Verified |
| SN2 | §7: "per seed, PACE-SHAP was ahead of PermutationSHAP on 1 and behind on 4, and ahead of paired KernelSHAP on 3 and behind on 2, by at most 0.0067" | N | 1 and 4; 3 and 2; 0.0067 | Verified |
| SN3 | §7: "PermutationSHAP used about a third of the budget, and, timed alone, the two samplers took 4.6× and 13.9× as long as PACE-SHAP (434 ms)" | N, runs.csv and timing.csv | 32.6% of 352,000; 4.59 and 13.90 (of the means); 433.6 ms | Verified |
| SN4 | §6 Datasets (rev 488) "Sections 6 and 7"; Settings (revs 489–490) seeds 210–214, 230–234 and 220–229 and the desktop for seeds 120–234; "Which tests are primary" (rev 493) seeds 120–234; Appendix C (rev 492) three new rows, "the exact checks on contracted boxes and on Simpson's classifier (pass or fail); the three CoverType runs are descriptive" | the three plans; HOME_PC_RUNBOOK.md §8 | match | Verified |

## Runs O and P, the abstract and the trim (paper revs 538–575)

Checked by two scripts in `research_audit/scripts/`, each with its `*_out.txt`: `claims_theta_churn.py` (Run O, 40 checks) and `claims_speed_source.py` (Run P, 18); all 58 OK. Every value is rounded from the raw CSVs, read with `float_precision="round_trip"`; Run O's primary is recomputed with `common.paired`, not read from the run's summary. Sources: O `results/theta_churn/runs.csv` with `results/repair_theta/runs.csv` for repair's flips (seeds 70–89), P `results/speed_source/runs.csv` (240–242); both ran on the home PC (`research_audit/HOME_PC_RUNBOOK.md` §10). Two values first guessed while writing the Run O script were wrong and were corrected before they reached the paper or FINDINGS: the accuracy tally of re-tuning against repair is 7 / 5 / 1 (repair more accurate on Thyroid), as the repair-at-θ run's committed P1 already says, not 7 / 6 / 0; and the 7 datasets are S1, Shuttle, Skin, MAGIC, Letter, Landsat and Page Blocks (not S2).

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| TC1 | §8 "Repair or re-tune θ?": "Repair broke fewer on 10 of 13 datasets and more on 3 (Shuttle, Skin and Letter, by 0.07–0.93 points)" | O, P1 (NFR, paired Wilcoxon, Holm, lower is better) | 10 better / 0 level / 3 worse; +0.065, +0.185, +0.929; Holm p ≤ 0.0143 | Verified |
| TC2 | §8: "Re-tuning broke 0.02–10.5% of the test predictions, repair 0–7.8%; on S1–S3, Wine and Digits repair changed none, while re-tuning broke 0.8–10.5%" | O | 0.020–10.455; 0.000–7.760; 0 changed on those 5; 0.833–10.455 | Verified |
| TC3 | §8: "Re-tuning is the bigger update: where it is more accurate it also fixes more, and on 6 of those 7 datasets it uses 6.3–23.2 times as many boxes as repair" | O; accuracy tally recomputed (repair-at-θ P1) | re-tune more accurate on 7 (S1, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks), PFR higher on all 7; boxes ratio 6.27–23.24 on the 6 other than S1 (S1 0.52) | Verified |
| TC4 | §8: "Without repair, CFMNN needed 1.8–23.2 times its boxes at θ = 0.3 to match repair's accuracy on 7 datasets, and no θ we tried matched it on Thyroid" | O, "boxes needed to match repair" (descriptive) | 1.84 (Page Blocks) – 23.21 (Skin) on Breast Cancer, Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks; Thyroid not reached; S1–S3 reached with fewer boxes (0.48–0.74), Wine and Digits at θ = 0.3 | Verified |
| TC5 | §8: "repair at the chosen θ was never less accurate than that θ alone (more accurate on 3), and broke at most 0.41% of its predictions" | repair-at-θ S2 (`claims_repair_theta_out.txt`); O, SR-val against B0-val | 3 / 10 / 0; 0.406 (Page Blocks), next 0.285 (Thyroid) | Verified |
| TC6 | §6 θ bullet: "it was also more accurate than repair at θ = 0.3 on 7 datasets, by up to 17.5 points (Letter); repair was more accurate only on Thyroid. At the chosen θ, repair was more accurate than that θ alone on 3 datasets, by at most 1.6 points, and never less" | repair-at-θ P1 and S2 (`claims_repair_theta_out.txt`; P1 also recomputed in `claims_theta_churn_out.txt`) | 1 / 5 / 7 for repair; Letter −17.53; 3 / 10 / 0, largest +1.55 (Thyroid) | Verified |
| TC7 | Abstract, contribution 2, Limitations, conclusion: repair "broke fewer of the model's correct predictions" than re-tuning θ "on 10 of 13"; "tune θ before a model is used, repair it after"; abstract "6–23 times as many boxes" | as TC1; the 6–23 from the readable-baselines run (existing rows) | as TC1 | Verified |
| TC8 | Limitations: "did not measure re-tuning followed by repair against the original model" | O plan ("Not measured") | as printed | Verified |
| TC9 | Appendix E table, Settings, "Which tests are primary", Deviations, Timings bullet: Run O on seeds 70–89 reused, on the desktop; Run P on 240–242, descriptive; their checks | the two plans; HOME_PC_RUNBOOK.md §10 | match | Verified |
| SS1 | §4 "Why exact is not free here": "shap's ExactExplainer took 14–23 times as long as PACE-SHAP. The structure alone gave 7.3–10.7× of this … and pruning, which skipped 62–76% of the pairs, a further 1.8–2.9×; the three agreed to within 2.5e-16" | P | total 14.3–22.7; structure 7.3–10.7; pruning 1.8–2.9; skipped 0.623–0.759; largest difference 2.5e-16 (Shuttle, ExactExplainer) | Verified |
| SS2 | Abstract: "mostly because of this structure and partly by pruning"; conclusion: "mostly because of the per-feature structure (7.3–10.7×) and partly by pruning (1.8–2.9×)" | P | structure > pruning on all 6 | Verified |
| SS3 | §4: "a point costs on the order of K · d · 2^d operations, against R · H · d · 2^d" | P plan ("What the code does"), from `PaceShap._exact_values` and ExactExplainer's calls | as printed | Verified (from the code, not timed) |
| AB1 | Abstract: "more faithful than KernelSHAP … on 3 of 13 datasets and level on 10, in 3.5–10× less time"; "16–30× faster"; "1.5–6.0 points"; "more accurate than Simpson's contraction on all 6" | existing rows (like-for-like, exact check, RQ2 second run, Run F) | as before | Verified (re-worded, no new number) |

**Moved, not changed.** These blocks left the main text with their numbers unchanged, so their CLAIMS rows still hold at the new place: the exact checks on Simpson's contraction and on Simpson's classifier trained from scratch (GE1–GE6, SF1–SF5) and the accuracy and ties of that classifier → Appendix D "Other fuzzy min-max models"; "Above 12 features, against modern samplers" (SB1–SB5, SN1–SN3) → end of Appendix C, with a short summary left in §7; "The synthetic sets as one unit" → Appendix B; "What the tests can detect, in detail" and Sepsis → Appendix E, with a short summary left in §6; "The name." → Appendix A.

## Tough-review batches 1 and 2 (paper revs 624–657)

Nine pre-registered runs from the tough review (seeds 250–399, home PC, `research_audit/HOME_PC_RUNBOOK.md` §12 and §14). Write-up: `results/FINDINGS.md` §23.

Two scripts check the numbers, each with its `*_out.txt`:
- `claims_tough_batch1.py`: P1, P3, P7 and P8;
- `claims_tough_batch2.py`: P2, P4, P5, P6 and P11.

Both read the raw CSVs with `float_precision="round_trip"` and recompute each test as its run's script does.

**Status: Pending.** The scripts are written but not yet run, because nothing runs on the laptop. The home PC runs them (runbook §17), and each row then becomes Verified or Mismatch.

The rows were written from each run's own `summary.md`. One value was derived by hand and checked on the laptop by reading the CSV with `awk`: Skin's NFR without duplicates (1.27% against 0.34%). A first draft of FINDINGS had 0.99%, which is the refinement split's broken share, not the paper split's; it was corrected before the commit.

| ID | Paper text | Source | Value | Status |
|---|---|---|---|---|
| TR1 | Abstract, contribution 1, §7 ×3, conclusion: KernelSHAP "took 3.3–8.0× as long" (median 4.9×), re-timed alone, seed 303 | P1 part (b) | 3.3–8.0 over 13 datasets, median 4.9, every interval above 1 | Pending |
| TR2 | Contribution 1, §7: "to come within 1% of the exact values it needed every coalition and 17–26× the time"; "median error … still 4–7% at 700 coalitions (1.3% on Shuttle at 400 of its 510)" | P1 part (a) | b* = all on 6 of 6; 16.9–26.4; 700: 4.03–7.25%; Shuttle 400: 1.33% | Pending |
| TR3 | §6, §8, Limitations: Skin without duplicates, "51,444 of 245,057 rows"; gain "6.2 points, on all 10 seeds, against 7.6 on the full data" | P3 | +6.20 (10/10, p = 0.0020); reference +7.56 (10/10) | Pending |
| TR4 | Limitations: without duplicates "repair broke more correct predictions than re-tuning θ by 0.93 points (0.19 with them), and CFMNN trailed the Random Forest by 6.9 points (5.2 with them)" | P3 secondary; 0.19 from TC1, 5.2 from the base-model rows | NFR 1.27 vs 0.34 (10/10 more); RF 99.80 − CFMNN 92.92 | Pending |
| TR5 | §4, Limitations, App F Prop 8: exact sets agree with a 10,001-value grid on 17,500 pairs; the 32 values miss "18%", "under 10% on 7 datasets, but 14% on MAGIC and 45–56% on Thyroid, Page Blocks and Shuttle" | P7 | 6,356 / 34,832 = 18.2%; MAGIC 13.9, Thyroid 45.1, Page Blocks 51.0, Shuttle 56.2; next largest 8.3 | Pending |
| TR6 | §4, Limitations, App F remark: recurrence "1.03–2.13× faster", "1.03–1.23× at 9–10 features, 2.13× on Wine and 1.40× on Letter", "47–112×" over brute force, "26.1 s" on Letter, limit stays at 12 | P8 (rerun) | as printed | Pending |
| TR7 | Contribution 2, §8, Limitations, conclusion: against tuned EFMNN, KNEFMNN, RFMNN and IOL-GFMM "more accurate in 6 of 24, not significantly different in 17, less accurate in 1 (IOL-GFMM on MAGIC)"; Thyroid "14.6–29.4"; Landsat +2.3, Letter +3.2; MAGIC "1.2 points on all 10 seeds"; "accuracy gave the same counts" | P2 primary and accuracy secondary (Holm across 24) | 6 / 17 / 1; +14.59 to +29.40; +2.34, +3.17, −1.22 (0 of 10) | Pending |
| TR8 | §8: "fewer boxes than EFMNN, KNEFMNN and RFMNN on all 6 datasets (5,343 against 15,075 on Letter), and fewer than IOL-GFMM on 3"; "at θ = 0.3 the counts were 7, 11 and 6" | P2 | as printed | Pending |
| TR9 | §3, Limitations, App F Prop 9: PACE-SHAP-min "within 6.8e-16"; "10.7–10.8 times" on S1–S3, "33–94 times" elsewhere; first check failed on IOL-GFMM's tie rule "on one point in 10 of 60 tasks" | P4 (rerun after Amendments 1 and 2; first runs kept) | 6.8e-16; 10.7–10.8; brute 33.5–94.2, ExactExplainer 33.4–90.3; first run 10 IOL-GFMM tasks | Pending |
| TR10 | Abstract, §7, conclusion, App F Prop 10: PACE-decision more faithful to the decision on tied points "on all 3 datasets and in 10 of 10 seeds (deletion AUC lower by 0.016–0.109)", also untied and by insertion; on the margin "less faithful on Letter and not significantly different on MAGIC and Page Blocks"; exact within 1e-9 | P5 | −0.0506, −0.0161, −0.1085, Holm p = 0.0059; margin deletion Letter +0.0268, insertion −0.0246 | Pending |
| TR11 | §8, Limitations, App F Prop 7: PACE-SHAP certificate sound (no change outside above 1e-12), held "99–100%" / "99.1–100%" of rows; values changed on "32–83%", median relative change "2e-5 to 2e-3" | P6 | 6,085 edits, 608,500 pairs; 99.1–100.0; 31.7–82.7; 2.0e-5 to 2.0e-3 | Pending |
| TR12 | Limitations, Next steps, App F Prop 6: S held for "116 of 6,110 accepted edits (1.9%; at most 8.0%)", "90–100% failed (i)", none broke a correct training prediction, the gate accepted every S edit | P11 | as printed | Pending |
| TR13 | App A Deviations: three fixes after a failed first run; P6 "30 hours against 2–4", P4's check "5 against 1–2" | HOME_PC_RUNBOOK.md §12, §14; the three amendment files | 29.8 h; 5.2 h and 5.3 h | Verified (read from the hand-overs, not computed) |
| TR14 | App E table: nine rows for the runs, seeds and primaries; the last column "not re-analysed" for P2, P3 and P5 | the nine plans | match | Verified (read from the plans) |
