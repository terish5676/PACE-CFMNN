# Final review: changelog

Every change made in the final review, with where it landed. Issue IDs refer to [ISSUES.md](ISSUES.md).

## Repository (branch `final-review`)

Base: `b508703` (merge of `rq1-extensions` and `refinement-v2` into `final-review`).

| Commit | Files | Change | Issues |
|---|---|---|---|
| `0614e66` | `research_audit/CLAIMS.md` | New: claims ledger for paper rev 262 (134 rows, endpoint table) | P1–P5, W1–W11 |
| `0614e66` | `research_audit/ISSUES.md`, `research_audit/CHANGELOG.md`, `research_audit/SMOKE.md` | New: issues log, this changelog, smoke-run results | all |
| `0614e66` | `research_audit/scripts/claims_*.py`, `common.py` and their `*_out.txt` | New: read-only scripts that recompute every paper number from the committed `runs.csv` files | P1–P4 |
| `0614e66` | `research_audit/scripts/check_tie_errors.py`, `results/final_review/tie_errors/` | New post hoc check: share of training errors that are cross-class ties (100% in 18 of 18 cases) | P5 |
| `0614e66` | `research_audit/scripts/check_readme_commands.py` and its output | New: checks that every README command exists, with valid flags | D2, C3 |
| `0614e66` | `research_audit/scripts/smoke_all.py`, `smoke_redirect.py` | New: one-dataset, one-seed smoke run of every script, with outputs outside the repository | D4, C3 |
| `0614e66` | `tests/test_final_review.py` | New: 13 hand-checked tests (worked example, U and U\*, Holm, non-inferiority, fix share, gate limits, edge cases) | C1 |
| `0614e66` | `results/FINDINGS.md` | §5 rewritten for the three repair runs; §6 and §7 bullets; header note on "no difference"; P3 and P4 corrected; counterfactual wording | D1, P3, P4, W9 |
| `0614e66` | `README.md` | Status, test count (178), three repair commands, repair key results, layout rows, intro, counterfactual wording | D2, W9 |
| `0614e66` | `requirements.txt` | Comment on which scripts need shap, lime and dice-ml | D3 |

No committed result, `runs.csv`, `summary.md` or pre-registration was edited. No method, default,
hyperparameter or dataset choice was changed. The commit after `0614e66` only fills these commit ids into this file and REPORT.md.

**One unplanned write, with no change.** The first version of `check_readme_commands.py` called
`--help` on every script. Three scripts have no argument parser, so they ran in full:
- `knn_reference.py` rewrote `results/error_detection_confirmatory/knn_reference.csv` and `.md`;
- `counterfactual_table_seeds.py` rewrote `results/review_v1/counterfactual_table.md`;
- `pace_shap_settings.py` was stopped after 5 minutes, before it writes anything.

All three rewritten files are byte-identical to the committed versions (`git hash-object` against
`git rev-parse HEAD:<file>`), so nothing changed, and both outputs reproduce exactly. The checker
was then fixed (C3).

## Paper (Claude Doc `9z3UBSnVXvHC3zKSEJywSS`)

Rev 262 at the start. All edits are small find-and-replace edits through the Docs connector. The
title, RQs, contributions and section structure are unchanged. Each edit was read back at rev 281.

| Doc rev | Where | Change | Issue |
|---|---|---|---|
| 263 | §7 RQ4 table | Letter, tight certificate per edit: 1.6 → 1.5 | P1 |
| 264 | Figure 2 caption | Grey intervals excluding zero without significance: add MAGIC | P2 |
| 265 | §7 sparse counterfactuals | 86–97% → 85–97% | P3 |
| 266 | §7 trust signals, "U\* is confirmed" | Add "and not significantly better on Page Blocks" | P4 |
| 267 | §7 RQ4 paragraph | "on up to 2,000 test rows (548–2,000 per dataset)" | W1 |
| 268 | §7 RQ4 paragraph | Appended retraining "was planned as a comparison to describe, with no test" | W2 |
| 269 | Abstract | Bought labels "plus the gate's labelled acceptance rows" | W3 |
| 270 | Abstract | Retraining "helped significantly only once (Letter, by 0.85 points, in a post hoc test)" | W4 |
| 271–272 | §7 repair bullet | "Retraining on the same labels helps little", with both ranges and tests | W4 |
| 273 | §4 PACE-SHAP item 2 | "16 orders … and adds each one's reverse (32 orders in all)" | W5 |
| 274 | §4 first paragraph | "the same call on the same points always gives the same answer" | W6 |
| 275 | §5 "A correction to U" | Tie defined by equal highest membership | W7 |
| 276 | §7 conditional replacement | Diagonal scale instead of Aas, Jullum and Løland's full covariance | W8 |
| 277–279 | §7 RQ2 v2 and counterfactual bullets | "no significant difference" (three places) | W9 |
| 280 | §7 RQ4 bullet | 51% is of the explained rows outside every tight certificate | W10 |
| 281 | §8 "Boxes, prototypes and clusters" | Add Quteishat and Lim (rule extraction) | R1 |
| 282 | References | Add A. Quteishat and C. P. Lim, *Applied Soft Computing* 8(2):985–995, 2008 | R1 |

## Reframe (user request, 2026-11-06)

After the review, the user asked to reframe the paper around what is strong and true. They also pointed out that "more faithful on only 3 of 13" undersold PACE-SHAP, which is level on the rest, never worse, and faster.

The reframe:
- changes no result, method or setting;
- keeps the title, RQs and section headings;
- keeps all 6 contributions; one is reworded and moved to second place.

One number is new to the paper: matched KernelSHAP took 3.5–10× as long as PACE-SHAP. It was already in FINDINGS §4.2. It is now checked by `research_audit/scripts/check_matched_speed.py` (CLAIMS row F10b).

**Paper (doc rev 282 → 293)**

| Doc rev | Where | Change |
|---|---|---|
| 283 | Abstract ¶2 | Rewritten. PACE-SHAP computes the values KernelSHAP estimates. With the same rows its values never change between runs, and it was more stable on 10 of 11 datasets. It was never significantly less faithful. Its lead over KernelSHAP's usual setting (6 of 11) comes mostly from the setup. Like for like it was better on 3 of 13, level on 10, worse on none, and KernelSHAP took 3.5–10× as long. Dropped for length: the stronger-KernelSHAP result and the LIME and KernelSHAP speed ratios (still in §7). |
| 283 | Abstract ¶3 | Rewritten. It opens with the accuracy cost (0.2–21.5 points). Then the tie finding, self-repair (+1.5–6.0 points on all 6, winning back 10–83% of the gap to a Random Forest), bought labels, certificates and fix share. |
| 284 | §7 like-for-like bullet | "only" dropped. Adds "level on the rest" and "KernelSHAP took 3.5–10× as long on every dataset (timed while other runs shared the laptop)". |
| 285 | §1 "Our idea" | Two sentences added. We do not claim CFMNN is more accurate. Our aim is to explain, and safely fix, a model whose rules people can read. |
| 286 | Contribution 1 (PACE-SHAP) | Adds: values do not change between runs; never significantly less faithful; given the same rows, better on 3 of 13, level on the rest, 3.5–10× faster. |
| 287–288 | Contribution "Self-repair" | Label becomes "Why CFMNN errs, and self-repair with certificates". The tie finding is added to the body. |
| 289 | Contribution "Trust signals" | States the result: the model's own signals lose to standard methods that use the training labels. Order-ensemble copies add to those methods, as bootstrap copies do. |
| 290 | Conclusion | PACE-SHAP sentence: more stable, never significantly less faithful, lead mostly from the setup, and given the same rows level on most datasets and 3.5–10× faster. |
| 291 | Conclusion | Self-repair wins back 10–83% of the gap to a Random Forest. |
| 292 | Contributions | The self-repair item moves to second place. No text refers to contributions by number. |
| 293 | Abstract ¶2 | "more faithful by deletion on 3 of 13". By insertion it is 2 of 13. |

**Wording changed from the approved plan, for accuracy.** The plan said PACE-SHAP's values "do not drift between runs". They do vary when new background rows are drawn, so the text now says "with the same rows its values never change between runs" (abstract) and "do not change between runs" (contribution). The trust contribution also keeps the bootstrap caveat.

**Checks.**
- Every changed block was read back at rev 292.
- The outline shows the title, RQs and all 11 headings unchanged, and 6 contributions.
- There are no `TBD` or bracket citations.
- Every number in the new text is in CLAIMS rows A1–A16, F10, F10b or Q6.

**Repository (this commit)**

| Files | Change |
|---|---|
| `research_audit/scripts/check_matched_speed.py` and its `_out.txt` | New, read-only. KernelSHAP time over PACE-SHAP time from `extensions_v1/conditional/runs.csv`: 3.46–10.01 (ratio of means), 130 of 130 slower |
| `research_audit/CLAIMS.md` | Row F10b, and a reframe note in the summary |
| `README.md`, `results/FINDINGS.md` §8.5 | Like-for-like result: "only" dropped; level on the rest, never worse, 3.5–10× as long |
| `research_audit/CHANGELOG.md`, `research_audit/REPORT.md` | This section, and the report's paper-changes line and review notes |

No committed result, `runs.csv`, `summary.md` or pre-registration was edited, and no study was rerun.

## Readable baselines (user request, 2026-11-06 to 2026-11-07)

The user asked for a test, planned in advance, of CFMNN against other readable models, with θ chosen
by validation. The user chose to add EBM and GFMM, to use 20 seeds (70–89), to run EBM with main
effects only and GFMM untuned at θ = 0.3 (both for time), and to use six processes overnight.

**Repository**

| Commit | Files | Change |
|---|---|---|
| `42a71d4` | `scripts/run_readable_baselines.py`, `tests/test_readable_baselines.py`, `results/readable_baselines/PREREGISTRATION.md`, `requirements.txt` | New: the script, 12 tests and the pre-registration, pushed at 17:12 before the run (started 17:13:05). `interpret-core` and `hyperbox-brain` added as optional packages |
| `ba21cd3` | `results/readable_baselines/runs.csv`, `summary.md`, `run_log.txt` | The run's own output, unedited: 2,080 rows, finished 2026-11-07 04:28, exit code 0, no failures |
| this commit | `research_audit/scripts/claims_readable.py` and its `_out.txt` | New, read-only. Redoes every test with `common.paired`. Its verdict counts match the run's `summary.md` exactly |
| this commit | `research_audit/CLAIMS.md` | Rows RB1–RB15, an endpoints row and a summary note |
| this commit | `results/FINDINGS.md` | New §9; bullets in §6 and §7; a header link. In §7, "better on only 3 of 13" now reads "better on 3 of 13 … level on the rest and never worse", as the user asked after the reframe |
| this commit | `README.md` | The run's command, the key results, the optional packages, and the test count (178 → 190) |
| this commit | `research_audit/scripts/check_readme_commands_out.txt` | Rerun after the new README command: 39 of 39 exist, with valid flags |
| this commit | `research_audit/CHANGELOG.md`, `research_audit/REPORT.md` | This section and the report's section |

No pre-registration, earlier `runs.csv`, `summary.md` or paper-number file was edited. No method,
default, hyperparameter or dataset choice was changed after the push, and no study was rerun.

**Paper (doc rev 293 → 302)**

| Doc rev | Where | Change |
|---|---|---|
| 294 | §7, after the base-model tables | New paragraph "Against other readable models": setup, a 7-row table of counts, and 5 bullets (EBM strongest; θ = 0.3 not the most accurate choice; trees smaller; reference models stay ahead; macro-F1 and grid-end checks) |
| 295 | Abstract ¶3 | After the accuracy cost: an EBM was more accurate on 12 of 13 datasets and a tree on 10; θ by validation helped on 7, with up to 23× as many boxes |
| 296 | §1 "Our idea" | The base run is named as such. A later test found an EBM more accurate on 12 of 13 |
| 297–298 | Settings | Seeds 70–89 for the readable-model test; interpret-core 0.7.8 (EBM) and hyperbox-brain 0.1.6 (GFMM) |
| 299 | Deviations | "readable models" added to the list of tests planned in advance |
| 300 | Limitations, accuracy bullet | Other readable models were more accurate on most datasets; EBM had main effects only, GFMM was untuned, and several tuned models chose the end of their grid |
| 301 | Conclusion | "on every dataset we tested" → "on every dataset in our base run". At θ = 0.3, CFMNN tied the Random Forest on Wine in the new run |
| 302 | References | Added Breiman, Friedman, Olshen and Stone (CART, 1984); Lou, Caruana and Gehrke (KDD 2012); Nori, Jenkins, Koch and Caruana (InterpretML, arXiv:1909.09223, 2019). The last two were checked on the web |

**Checks.**
- Every changed block was read back at rev 302.
- The title, RQs and section headings are unchanged.
- There are no `TBD` markers or bracket citations.
- Every number in the new text is in CLAIMS rows RB1–RB15.

## Repair at the θ chosen by validation (user request, 2026-11-07 to 2026-11-08)

The readable-baselines run left a question: why repair at θ = 0.3 instead of choosing a smaller θ?
The user asked for a test planned in advance, chose 20 seeds (70–89) run overnight, and started it
on 2026-11-08 at 01:06.

**Repository**

| Commit | Files | Change |
|---|---|---|
| `22e687d` | `scripts/run_repair_theta.py`, `tests/test_repair_theta.py`, `results/repair_theta/PREREGISTRATION.md` | New: the script, 8 tests and the pre-registration, pushed on 2026-11-07 at 10:40, before the run |
| `4dd2e59` | `results/repair_theta/runs.csv`, `summary.md`, `run_log.txt` | The run's own output, unedited: 1,040 rows, finished 2026-11-08 16:10, exit code 0, no failures |
| this commit | `research_audit/scripts/claims_repair_theta.py` and its `_out.txt` | New, read-only. Redoes every test with `common.paired`. Its verdict counts match the run's `summary.md` exactly |
| this commit | `research_audit/CLAIMS.md` | Rows RT1–RT14, an endpoints row and a summary note |
| this commit | `results/FINDINGS.md` | New §10; a bullet in §6 and two in §7; a header link |
| this commit | `README.md` | The run's command, the key results, the script list, and the test count (190 → 198) |
| this commit | `research_audit/scripts/check_readme_commands_out.txt` | Rerun after the new README command: 40 of 40 exist, with valid flags |
| this commit | `research_audit/CHANGELOG.md`, `research_audit/REPORT.md` | This section, and the report's section. The report's open question is marked as now tested |

No pre-registration, earlier `runs.csv`, `summary.md` or paper-number file was edited. No method,
default, hyperparameter or dataset choice was changed after the push, and no study was rerun.

**Paper (doc rev 302 → 311)**

| Doc rev | Where | Change |
|---|---|---|
| 303, 307 | §7, bullet "θ = 0.3 was not the most accurate choice" | The new result: θ chosen by validation without repair beat repair at θ = 0.3 on 7 datasets (by up to 17.5 points), and lost only on Thyroid. With repair at both, the chosen θ was better on 8. At the chosen θ, repair helped on 3 (at most 1.6 points). Macro-F1 gave similar counts. Repair at θ = 0.3 still helped on 7 and added no box, so its case is the smaller model |
| 304 | Settings | Seeds 70–89 also for the test of repair at the chosen θ |
| 305 | Deviations | "repair at the chosen θ" added to the list of tests planned in advance |
| 306 | Limitations, accuracy bullet | Repair at θ = 0.3 did not make up for choosing θ by validation, and tried only 300 edits, a small share of the training errors on the large datasets (about 4,000 on MAGIC) |
| 308 | Limitations, repair bullet | "Repair was tested on 6 datasets with 10 seeds, one budget and one gate" → "with one budget and one gate: on 6 datasets with 10 seeds for RQ3 and RQ4, and on 13 with 20 seeds in the test of repair at the chosen θ" |
| 309 | Abstract | "On the 6 datasets we checked, every training error of CFMNN is a tie" → "On all 13 datasets, every training error of CFMNN we checked is a tie" (user request) |
| 310 | Abstract | After the self-repair sentence: "In a second planned test, it did not match choosing θ by validation, which stayed more accurate on 7 of 13 datasets, but it added no boxes." (user request) |
| 311 | §6, tie paragraph | Added: in the test of repair at the chosen θ, the tie held for all 347,339 training errors on all 13 datasets, at both θ, over 20 seeds |

**Checks.**
- Every changed block was read back at rev 311.
- The title, RQs and section headings are unchanged.
- There are no `TBD` markers or bracket citations.
- Every number in the new text is in CLAIMS rows RT1–RT14.

## RQ3 explainer (user request, 2026-11-11)

A plain explainer of RQ3 (self-repair) for the team and reviewers, with diagrams drawn by our own code. It answers the reviewer's questions on epochs, retraining, time and how errors are ordered, and corrects claims in a teammate's write-up (a theorem that inputs outside the edited box cannot change; SR and SR then AL mixed in one table; the gate called a validation set; repair called faster than retraining).

| Commit | Files | Change |
|---|---|---|
| first commit | `docs/rq3_timing.py`, `results/rq3_explainer_timing/` | New descriptive timing check, made after the RQ3 run and not pre-registered: SR as RQ3 calls it, without the certificate audit, seed 50, one dataset at a time. SR's test macro-F1 equals the committed value on all 6 datasets |
| second commit | `docs/RQ3_self_repair.md`, `docs/rq3_figures.py`, `docs/figures/` | New explainer, its 8 figures and `rq3_numbers.txt`. Toy figures come from running `CFMNN`, `run_self_repair` and `audit.certify` on made-up data; every fact stated about them is an `assert`. Result numbers are recomputed from `results/refinement_v2_confirmatory/` |
| second commit | `README.md`, `research_audit/CHANGELOG.md` | A layout row for `docs/`, a link from the repair key results, and this section |

No library code, test, pre-registration, earlier `runs.csv`, `summary.md` or paper text was changed. The only rerun is the timing check's SR on seed 50, written to its own new folder.


## Follow-up tests of the limitations (2026-11-11 to 2026-11-13)

**The runs.**
- Five runs (E, A, B, C and D) were built and run on the home PC (i5-14400F), from `research_audit/LIMITATIONS_PLAN.md` and `research_audit/HOME_PC_RUNBOOK.md`.
- Each run's code, tests and pre-registration were pushed before it ran (a4c08f1 to f43b11b).
- A laptop prototype of the cached gate (ab5a166) was reverted first (fa2f890), and its results were not used.

**The write-up, done on the laptop.**
- `results/FINDINGS.md` §11.
- CLAIMS rows FU1–FU8, and the endpoint rows.
- `research_audit/scripts/claims_followup.py` and its output.
- This section.

**The paper (doc rev 311 → 326).**

| Doc rev | Where | Change |
|---|---|---|
| 312 | §7 | New paragraph "Follow-up tests of the limitations", with 6 bullets: more points and ties; conditional settings; robust counterfactuals; repair budget and gate; tighter and explanation certificates; the faster gate |
| 313–318 | Limitations | Bullets updated with the results: 50 points; one DiCE setting (now the r3 weakness); single σ (now the trade between rows and realism); timings (now also a desktop); one budget and one gate; certificates |
| 319 | §7, certificate bullet | Now points to the follow-up explanation certificates |
| 320–321 | Next steps | Counterfactuals: find answers that still hold after refitting. Repair: use the tighter certificates in every repair, and find one for PACE-SHAP |
| 322 | Deviations | DiCE kdtree left out of r1, decided before the counterfactual run |
| 323–325 | Settings | Seeds 110–119, and the desktop's hardware |
| 326 | §7 | PACE-sparse's r1 ratio corrected to 1.5–10, an error the claims check caught |

**Checks.**
- Every changed block was read back.
- The title, RQs and section headings are unchanged.
- There are no `TBD` markers or bracket citations.

## Reproducibility package and claims check of the peer-review revision (2026-11-13)

**Repository.**
- `requirements-lock.txt`: the exact versions of the 46 packages the results were made with (Python 3.14.7, Windows 11), written by the new `scripts/make_lock.py` from the installed packages. `requirements.txt` keeps its lower bounds.
- `scripts/download_data.py`: downloads the 8 UCI datasets from the UCI repository, keeps only the files the loaders read, and compares each with the SHA-256 of the copy our runs used. `--check` compares without downloading. Tested with `--check` only; the download path has not been run.
- `README.md` (Install) and `data/README.md` point to both.
- `research_audit/scripts/claims_peer_review.py` and its output: 78 checks of the numbers added at paper revs 327–439.
- `CLAIMS.md`: header and summary brought up to rev 443, and new rows PR1–PR19. `ISSUES.md`: P6 and P7.

**Paper.**

| Rev | Where | Change |
|---|---|---|
| 440–442 | §7, §9 | Three numbers rounded twice, corrected (P6): Landsat's interval −0.053 to −0.036; self-repair's gain on Letter 2.2; appended retraining −2.6 to +0.8 |
| 443 | Appendix C | The extensions run's realism primary added to the table of primary comparisons (P7) |

## Self-repair as the method, a time table and a smarter error order (2026-11-14)

**Repository (built on the laptop; Runs T and H run on the home PC).**
- `pace_cfmnn/active.py`: `run_self_repair(..., order=...)`, with "random" (the default, which every committed run used, unchanged), "box" and "greedy"; new helpers `box_order` and `_trial_copy`. With the new code, the default reproduces RQ3's committed SR edits exactly (Thyroid and Page Blocks seed 50 with both gates, Thyroid seed 51).
- `pace_cfmnn/refine.py`: `Gate.verdict` (the check's thresholds, moved into their own method; the same decisions), and `Gate.peek` and `CachedGate.peek`, which score a trial edit without changing the gate.
- Run T (descriptive, not pre-registered): `scripts/time_repair_arms.py`, `results/repair_timing/README.md`, `tests/test_repair_timing.py`. It times train CFMNN, SR with the old and the cached gate, AL and SR+AL on one machine.
- Run H (pre-registered): `scripts/run_repair_order.py`, `results/repair_order/PREREGISTRATION.md`, `tests/test_repair_order.py`.
- `scripts/sr_method_tables.py` and `results/sr_method/tables.md`: RQ3's gains over B0 in +x format (markers only from the pre-registered endpoints, recomputed and checked against RQ3's summary.md), the labels each arm used, and RQ4 for self-repair alone (post hoc). The time table fills in once Run T has run.
- `research_audit/HOME_PC_RUNBOOK.md`: section 4, Runs T and H.

No committed result, `runs.csv`, `summary.md` or pre-registration was edited.

**Run H, amendment 1 (2026-11-14, before the run).** After a review of the plan, before any study seed ran:
- `pace_cfmnn/active.py`: `run_self_repair(..., count_skips=True)`. The default, used by every committed run, is unchanged; `False` ("real tries") counts only the tries that reach the gate (random and box orders only).
- `scripts/run_repair_order.py`: a new arm, random-300-real; a new column, `edits_gated`; a new comparison, A1 (greedy-300 vs random-300-real); two efficiency tests, E1 and E2 (greedy-300 vs random-all on seconds and on edits that reached the gate); and a per-dataset reading of the primary. 540 rows instead of 480.
- `tests/test_repair_order.py`: 5 new tests, written on the laptop and not run there; they run on the home PC before the run.
- `results/repair_order/PREREGISTRATION.md`: a dated "Amendment 1" section added at the end. The text above it was not edited.
- `research_audit/HOME_PC_RUNBOOK.md`: Run H's test count, row count and report.

## Runs H and T written up (2026-11-14)

Runs H and T were built on the laptop and run on the home PC (87aa227 to 498784c). This write-up adds:

- `research_audit/scripts/claims_order_timing.py` and its output (30 checks, all OK);
- CLAIMS rows OT1–OT10;
- `results/FINDINGS.md` §12;
- `results/sr_method/tables.md`, regenerated with Run T's time table.

**Paper.**

| Rev | Where | Change |
|---|---|---|
| 444–449 | §9 | Two new paragraphs: "Order of the training errors" (Run H) and "Time of each setup" (Run T) |
| 444–449 | Limitations | The repair bullet names the random order and the greedy result; the timing bullet covers seeds 110–139 and the timing check |
| 444–449 | Next steps | Repair: make the greedy order cheaper |
| 444–449 | Appendix C | Seeds 130–139 and seed 50, the desktop, and a primary-comparisons row for Run H |

## Self-repair as the method written up (2026-11-14)

No new run. Plan parts 2 and 3 (tables, self-repair as the method, RQ4 for self-repair alone) and the framing of part 4 (Run H answers "why not a smarter order?"). This adds:

- `scripts/sr_method_tables.py`: the explanation-change row of RQ4 for self-repair alone is now in points (the column is a share in [0, 1]; it was printed unscaled, as +0.36 for +36.3); `results/sr_method/tables.md` regenerated;
- `research_audit/scripts/claims_sr_method.py` and its output (51 checks, all OK);
- CLAIMS rows SM1–SM9;
- `results/FINDINGS.md` §5.5 "Self-repair as the method", and a note in §12;
- `README.md`: key results for self-repair as the method and for Runs H and T, and command rows for `time_repair_arms.py`, `run_repair_order.py` and `sr_method_tables.py` (`check_readme_commands.py`: 43 commands, 0 with problems), and the test count (307 tests pass on the laptop, 641 s).

**Paper.**

| Rev | Where | Change |
|---|---|---|
| 450 | §9 | New paragraph "Self-repair as the method" and a table of gains over the starting model (+x), labels and seconds |
| 451 | §10 | New paragraph "Self-repair alone (post hoc)": RQ4's tests on self-repair alone |
| 452 | §9 | Run H paragraph: opens with "Would a smarter order fix more with the same tries?" and closes with why self-repair keeps the random order (36–46 times the time on MAGIC and Letter) |
| 453 | Conclusion | "and bought labels added less, on 3, at 5.5–57 times the time" |

## Runs F, G, I, J and K written up (2026-11-15)

All five ran on the home PC (`research_audit/HOME_PC_RUNBOOK.md` §7). No new experiment; one post hoc check.

**Files.**

- `research_audit/scripts/claims_contraction.py`, `claims_systems.py`, `claims_generality.py`, `claims_sampled.py`, `claims_covertype.py` and their outputs (42, 57, 22, 18 and 20 checks, all OK), rounding from the raw CSVs;
- `research_audit/scripts/check_exact_simpson.py` and its output (post hoc): in Run I's three largest exact-check differences, PACE-SHAP agrees with a brute-force Shapley sum to 1.3e-16, and the 7e-11 to 1.6e-10 differences come from shap's ExactExplainer;
- CLAIMS rows CF1–CF8, SY1–SY7, GE1–GE5, SB1–SB5, CT1–CT4 (29 rows, all Verified);
- `results/FINDINGS.md` §13–17, one per run.

**Paper.**

| Rev | Where | Change |
|---|---|---|
| 455–457 | Abstract | Run F (more accurate than Simpson's contraction on all 6) and Run G (less accurate than EBM on 10 of 11 and a tree on 9, slower to explain on all 11) |
| 458–459 | §3 | Proposition 1 at γ = 2, 4, 8 (0 of 569,239) and on CoverType; PACE-SHAP on contracted boxes (1.6e-10; post hoc: the error is ExactExplainer's) |
| 460–461 | §5 | Rows a shrink cuts off, against contraction (Run F) |
| 461 | §6 | "Whole systems" result and table (Run G); new paragraph "One large dataset" (Run K) |
| 462 | §9 | "Against contraction" result with Amendment 1's caveats (Run F) |
| 463 | §7 | New paragraph "Above 12 features, against modern samplers" (Run J) |
| 464–466 | Limitations | PACE run on contracted boxes; accuracy cost as whole systems and CoverType; timing seeds 110–174 |
| 467–473 | §6, Next steps, Appendix C | CoverType run once on its own; Next steps repair item; γ result, seeds and desktop in Settings; Run G's Amendment 2 in Deviations |
| 474–478 | Appendix C | Primary-comparison table: F and G entries completed, rows for γ and Simpson's contraction, modern samplers and CoverType; notes on the exact check and the descriptive run |

## Check of the F–K write-up on the laptop (2026-11-15)

No new experiment. The five claims scripts were run again (159 checks OK). No plan, CSV, `summary.md`, log or registered script has changed since 22832d2.

**Files.**

- `research_audit/scripts/check_exact_isclose.py` and its output (post hoc, 5 checks OK): the cause of Run I's 1e-10 exact-check differences. shap 0.54.1's tabular masker treats a background value within `np.isclose`'s tolerance (relative 1e-5) of the explained point's as unchanged, so ExactExplainer does not call the model again for it. Each of the five tasks has one such value. As shipped, each point reproduces its committed difference, and with exact equality ExactExplainer matches PACE-SHAP to 1.3e-16.
- `research_audit/CLAIMS.md`: a note on the laptop rerun (three p-values print differently because pandas' default float parser is not exact in the last bit; `float_precision="round_trip"` reproduces the saved outputs, and no verdict changes); new row GE6; GE3's status notes the wording fix.
- `results/FINDINGS.md`: §6 and §7 bullets for Runs F–K (they were missing); §15's post hoc bullet now names the three checked cases correctly and gives the cause.

**Paper.**

| Rev | Where | Change |
|---|---|---|
| 479 | §3 "Which models this covers" | "in the three worst cases" (wrong: the three checked were the 1st, 2nd and 5th largest) becomes "in three of the five tasks that differed by more than 1e-14 (the two largest and the smallest)"; two sentences on the cause (ExactExplainer's isclose shortcut; 1.3e-16 with exact equality) |
| 480 | Conclusion | At the user's request, after "repair narrows that gap but does not close it": the whole-system result of Run G (less accurate than an EBM on 10 of 11 and a tree with TreeSHAP on 9, slower to explain on all 11) and Run F's primary result (self-repair more accurate than Simpson's contraction on all 6); CLAIMS rows CF9 and SY8 |

## Runs L, M and N written up (2026-11-15)

All three ran on the home PC (`research_audit/HOME_PC_RUNBOOK.md` §8; plans in `results/covertype_theta/`, `results/simpson_fmm/` and `results/sampled_baselines_covertype/`). No new experiment; one post hoc check. No plan, CSV, `summary.md`, log or registered script was changed.

**Files.**

- `research_audit/scripts/claims_covertype_theta.py`, `claims_simpson.py`, `claims_sampled_covertype.py` and their outputs (36, 30 and 26 checks, all OK), rounding from the raw CSVs read with `float_precision="round_trip"`;
- `research_audit/scripts/check_exact_isclose_simpson.py` and its output (post hoc, 5 checks OK): Run M's one exact-check difference above 3.1e-16 (S1, seed 222, 1.35e-10) is the same shap shortcut as Run I's; with exact equality, ExactExplainer matches PACE-SHAP to 1.0e-16;
- CLAIMS rows CL1–CL5, SF1–SF7 and SN1–SN4 (16 rows, all Verified);
- `results/FINDINGS.md` §18–20, one per run, and bullets in §6 and §7 (the §7 bullet on Simpson's classifier now says only exactness was checked on it).

**Paper.**

| Rev | Where | Change |
|---|---|---|
| 481–482 | §3 "Which models this covers" | PACE run on Simpson's classifier trained from scratch; its exact check (1.4e-10; 59 of 60 tasks to 3.1e-16; post hoc, the other is the shap shortcut, 1.0e-16 with exact equality); its accuracy and tie share beside CFMNN (descriptive) |
| 483 | §6 "One large dataset" | Run L: θ by validation 0.05 on every seed, 50.7% to 72.9% with 34–42 times the boxes and 9.3 s per PACE-SHAP explanation; self-repair +3.1 and +0.4; the 10% share and why |
| 484 | §7 "Above 12 features" | Run N: CoverType close by deletion, per-seed counts both ways, PermutationSHAP used a third of the budget, samplers 4.6× and 13.9× as long |
| 485–486 | Limitations | "tested it in full only on CFMNN: on CFMNN's boxes after Simpson's contraction and on Simpson's classifier trained from scratch … we checked only that PACE-SHAP's values are exact" (was: "not on Simpson's classifier trained from scratch") |
| 487 | Limitations | CoverType: on a 10% sample, θ by validation reached 72.9% with 34–42 times the boxes |
| 488 | §6 Datasets | CoverType "on its own (Sections 6 and 7)" |
| 489–490 | Appendix C Settings | seeds 210–214, 230–234 and 220–229; desktop for every run on seeds 120–234 |
| 491 | Appendix C Deviations | the two changes made before any study seed: Run L's 10% share, Run M's replaced overlap test |
| 492 | Appendix C primary comparisons | the exact check on Simpson's classifier; the three CoverType runs descriptive; three new table rows |
| 493 | "Which tests are primary" | later tests on seeds 120–234 |

## The four weak points: restructure, Runs O and P, abstract and trim (2026-11-15)

The approved plan "fix the four weak points" (write for fuzzy min-max users; move the trust signals to an appendix; two research questions; Runs O and P). Runs O and P were pre-registered in `0c5dff4` and run on the home PC (`d9c2602`, `6baa19d`, runbook §10 in `f18a2cd`). No plan, CSV, `summary.md`, log or registered script was changed.

**Files.**

- `research_audit/scripts/claims_theta_churn.py` and `claims_speed_source.py` and their outputs (40 and 18 checks, all OK);
- CLAIMS rows TC1–TC9, SS1–SS3 and AB1, and a list of the blocks moved out of the main text;
- `results/FINDINGS.md` §21 (Run O) and §22 (Run P).

**Section map (old → new), for CLAIMS rows written before rev 494.** Old §8 (RQ2, trust signals) → Appendix B; old §9 and §10 (RQ3, RQ4) → §8 "RQ2: repair, labels and certificates"; old §11–13 → §9–11; old Appendix B → Appendix C; old Appendix C → Appendix E; Appendix D is new (follow-up tests in detail). The pre-registrations keep the names RQ1–RQ4; Appendix A "Question names" says that RQ3 and RQ4 are merged as RQ2.

**Paper, revs 494–537 (restructure, before the runs).** Introduction re-aimed at fuzzy min-max users (¶1–3, "Our idea", "The gap"); two research questions; trust signals moved to Appendix B with one paragraph left in the main text; secondary explanation results moved to Appendix C and follow-up runs to Appendix D; new §4 paragraph "Why exact is not free here" (non-additive margin, cost per point, tractability results of Arenas and colleagues, 2023, and Van den Broeck and colleagues, 2021, checked on the web); new §7 lead "What exactness buys"; conclusion rewritten; cross-references renumbered.

**Paper, revs 538–575 (after the runs).**

| Where | Change |
|---|---|
| §4 "Why exact is not free here" | Run P: ExactExplainer 14–23× as long; structure 7.3–10.7×, pruning 1.8–2.9× (62–76% of pairs skipped); agreement 2.5e-16 |
| §8, new paragraph "Repair or re-tune θ?" | Run O: NFR 10 / 0 / 3 for repair; ranges; boxes; repair at θ-val never worse and gentle; the guidance (tune θ before use, repair after, both where possible) |
| §6 θ bullet | Shortened; points to §8 instead of "the reason to repair at θ = 0.3 is the smaller model" |
| Abstract | Rewritten in the plan's order, 250 words |
| Contributions, Limitations, conclusion | Run O and Run P results; Limitations notes "re-tune, then repair" was not measured; conclusion shortened |
| Appendix A Deviations, Appendix E Settings, primary table, "Which tests are primary", Timings | Runs O and P: seeds, machine, two table rows, descriptive status |
| Trim (moves, numbers unchanged) | Exact checks on other fuzzy min-max models → Appendix D; "Above 12 features" details → Appendix C (summary kept in §7); "The synthetic sets as one unit" → Appendix B; "What the tests can detect" details and Sepsis → Appendix E (summary kept in §6); "The name." → Appendix A; one repeated intro sentence and two follow-up sentences (timing on the desktop and the tighter check, both in Appendix D) cut |
| References fixed | Deviations: the conditional run's KernelSHAP comparison "in Appendix C" (was Section 7); Datasets: CoverType "(Appendices C and D)" |

**Length (counted by script from an export at rev 575).** Abstract 250 words. Main text (§1–11) 11,082 words without tables, 12,083 with them; at rev 537 it was 11,792 without tables (12,793 with), and the plan put it at about 17,600 before the restructure. The working target of 11,000 is 82 words short of being met; reset it to the journal's limit once one is chosen.

## Tough review: fixes (2026-11-16 onwards)

The plan: `research_audit/TOUGH_REVIEW_PLAN.md`, one row per review item (the review itself is
`reviews/tough_review_2026-11-16.md`, outside the repository). Branch `tough-review-fixes`, cut
from `peer-review-fixes` at `4d16c24`.

**Rollback point for the paper: rev 575** (the doc as the review read it). Every paper edit below
is a direct edit (the user's choice), made section by section, with its rev range recorded here.

**Code, theory and pre-registrations.**

| Commit | What |
|---|---|
| `79c5584` | Checklist; Proposition 5 construction (`hardness.py`); PACE-SHAP certificate, churn bound, condition S (`audit.py`) |
| `7843fee` | PACE-decision, exact flip sets, recurrence exact mode and brute-force baseline |
| `d003ed1` | GFMM explanations, tuned fuzzy min-max baselines, deduplicated Skin |
| `c6006a4` | Reanalysis: corrected tests, cross-dataset tests, margins, RQ2 figure (`results/tough_review/`) |
| `4043ddf`, `7b6737a`, `816d863` | Batch 1 (Runs P1, P3, P7, P8): scripts, tests, pre-registrations, runbook §11 |
| `94dd0ed` | Batch 2 (Runs P2, P4, P5, P6, P11): scripts, tests, pre-registrations, runbook §13; condition S clause (iii); `research_audit/THEORY.md`; the held-out dataset rule (`results/heldout/RULE.md`) |
| `7071a19` | Release kit, part 1 (B13, R1, R3): `pyproject.toml` (`pip install -e .`, extras `xai`, `baselines`, `all`); the front end `pace_cfmnn/api.py` (`PaceClassifier`, `explain`, `repair`, `certify`) with `tests/test_api.py`; `Dockerfile` from the lock file (not yet built); `LICENSE-PENDING.md`; README install, quick start and licence text |
| `4ede92b` | Batch 3 (Runs P9, P10): OpenML downloads with MD5 and SHA-256 pins (`scripts/download_data.py`); the ARFF reader and `data.load_openml` with `tests/test_openml_data.py`; `scripts/run_heldout.py` and `scripts/run_time_ordered.py` with tests; pre-registrations in `results/heldout/` and `results/time_ordered/`; runbook §15 |
| `05e0072` | Release kit, part 2 (R2, B13): `release/osf/` (MANIFEST.md and sha256sums.txt for the 38 plans at a89a179; steps for the OSF upload); `release/demo/pace_cfmnn_demo.html` (interactive 2-D demo, published as a private artifact); README demo section and test count (501) |

**Paper edits (direct, from rev 575).** Wording fixes from Section A of the review, before any
batch result. Each rev range is one or more edit calls, in section order.

| Revs | What |
|---|---|
| 576 | Abstract (249 words): scope of RQ1, timing from one run, exact on 6 of 11 (3 real), "without new labels", ties and the margin, #P-hardness, sound but loose certificates |
| 577–583 | §1 (idea, research questions, contributions; interactions dropped from them), §2 (related fuzzy min-max work, the narrower gap), §3 (four properties), §4 (PACE-flip wording; Proposition 5 with a proof sketch and remark) |
| 584–587 | §5 certificates (sound but loose; churn bound), §6 setup (Skin duplicates; what "no significant difference" means; post hoc margins) |
| 588–596 | §7 RQ1 (what exactness buys, ties first, faithfulness with cross-dataset tests, equivalence marked post hoc); "level" replaced by "no significant difference" in §6–§7; §8 "Repair or re-tune θ?" (5 better, 3 worse of the 8 where repair changed anything) |
| 597–608 | §8 rest, §9 related work, §10 limitations (bullets on ties and on the margin), §11 next steps (B14 lines), new references |
| 609–617 | Appendices B–E: wording ("level"), App D's PACE-SHAP certificate line, GFMM-0.3 in the readable-models row, a duplicated block removed |
| 618 | App E: a column of verdicts that survive both corrections (post hoc), and a paragraph on the corrected tests |
| 619 | Appendix F: Proposition 5 in full; Corollary 1; Propositions 6–10; the recurrence remark |
| 620–623 | §8: Figure 5 (accuracy, boxes and negative flip rate across θ): caption, reference and image; the interactions figure renumbered 6 |

**Batches 1 and 2 written up (2026-11-19).** The home PC ran batch 1 (P1, P3, P7, P8; runbook §12) and batch 2 (P2, P4, P5, P6, P11; §14) on 2026-11-16/18. The laptop wrote them up with file edits only and ran nothing.

| Revs | What |
|---|---|
| 624–627 | Abstract ("3.3–8.0×" re-timed alone; the decision explanation at ties); contributions 1 and 2 (time to 1%, GFMM, exact flip sets; tuned fuzzy min-max baselines 6 / 17 / 1); RQ1 scope (GFMM) |
| 628–632 | §3 (GFMM through Proposition 9); §4 (what the 32 values miss; the recurrence and the 12-feature limit); §6 (Skin duplicates did not drive repair's gain; seeds of the later tests) |
| 633–636 | §7 ("3.3–8.0×" in three places; time to 1%; PACE-decision at ties) |
| 637–638 | §8 (Skin without duplicates; a new paragraph "Against tuned fuzzy min-max networks"; the PACE-SHAP certificate) |
| 639–649 | Limitations (GFMM, flip sets, the 12-feature limit, tuned baselines, certificates and condition S, Skin, timings); conclusion; Next steps (gate-free repair) |
| 650 | App A Deviations (three fixes after a failed first run; two runs far over time); App E table (nine rows) |
| 651–657 | App F (Propositions 6–10 and the recurrence remark, with each run's result); rev 657 fixed a code mark that rev 651 had spread over a whole sentence |

Files: `results/FINDINGS.md` §23; `research_audit/CLAIMS.md` rows TR1–TR14; `research_audit/scripts/claims_tough_batch1.py` and `claims_tough_batch2.py` (written, not yet run: runbook §17); `research_audit/TOUGH_REVIEW_PLAN.md` statuses. Not changed: the default exact algorithm. The recurrence becomes the default after batch 3, as P8's plan says, so batch 3 runs the code its plans describe.
