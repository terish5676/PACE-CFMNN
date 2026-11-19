# Tough review (2026-11-16): checklist

Source: `reviews/tough_review_2026-11-16.md` (outside the repository), paper rev 575, code 4d16c24.
Branch: `tough-review-fixes`. Paper rollback point: **rev 575**.
Targets: a TMLR version (~7,500 words of main text) and an IEEE TFS version (~6,500 words, 10 pages). The same paper cannot be under review at both at once, so the team picks the order.

Status: `todo`, `code` (code and tests done), `pre-reg` (pushed before its run), `ran` (results pulled), `done` (in the paper), `neg` (done, opposite result reported).

## Section A

| ID | What | Cost | Status | Commit / rev |
|---|---|---|---|---|
| N1 | #P-hardness proposition (B1) | A | done | 79c5584 (construction + test); paper rev 583 (§4), 619 (App F); the guide checks the proof |
| N2 | Cite MLF and RFMM; Prop 1 as motivation | A | done | paper revs 577–583 (§1, §2) |
| N3 | Cite Fu & Li, Sinitsin et al., Santurkar et al.; "sound but loose" | A | done | paper revs 576 (abstract), 584 (§5), 597–608 (§9, references) |
| N4 | Narrow the gap; cite hyperbox-brain | A | done | paper rev 578 (§2 'The gap') |
| N5 | Frame RQ1 for mean-membership fuzzy min-max networks | A | done | paper revs 576–577 (abstract, RQ1) |
| S1 | NFR: 5 better / 3 worse on the 8 datasets where repair changed anything | A | done | paper rev 596 (§8 'Repair or re-tune θ?') |
| S2 | Speed numbers from one timed-alone run; re-time like for like (P1) | A + C | done | paper revs 576, 582; P1 ran (aca8e37): 3.3–8.0× re-timed alone, 1% needs every coalition (17–26×); paper revs 624–649 |
| S3 | Exact on 6 of 11 (3 real); 8 of 13 in the like-for-like run | A | done | paper revs 576, 588 |
| S4 | PACE-flip: 32 values, or exact flip sets (B8, P7) | A + C | done | paper rev 581; P7 ran (8746697): Check PASS, the 32 values miss 18.2% of flips (45–56% on Thyroid, Page Blocks, Shuttle); paper revs 632–656 |
| S5 | "without new labels" | A | done | paper revs 576–577 and wherever repair is described |
| S6 | Margin, not decision, at ties (B3, P5) | A + C | done | paper revs 576, 589 (§7 'Ties come first'), 597–608 (§10); P5 ran (a591bb3): PACE-decision better on 3 of 3 tied, worse on Letter's margin; paper revs 624, 636, 649 |
| S7 | Disclose floor and clip; lead with raw and untied | A + B | done | reanalysis §4; paper revs 587 (floor and clip, §6), 589 |
| S8 | Appended retraining as the main comparator | A | done | paper rev 595 (§8, appended retraining first) |
| S9 | RQ2 scope; accuracy–size–churn figure (B5) | A + B | done | paper revs 576, 596, 620–623 (Figure 5) |
| S10 | Interactions out of the contributions | A | done | paper rev 577 (contributions) |
| S11 | Deletion test favours interventional explainers | A | done | paper rev 588 (§7 'What exactness buys') |
| D1 | Nadeau–Bengio corrected tests; Holm across a run's primaries | B | done | reanalysis §1, d1_primaries.csv; paper rev 618 (App E column and paragraph) |
| D2 | Across-dataset tests beside each headline | A + B | done | reanalysis §2; paper revs 588–596 |
| D3 | Equivalence "(post hoc)"; margin sensitivity | A + B | done | reanalysis §3; paper revs 587–590 '(post hoc)' |
| D4 | Tuned fuzzy min-max baselines (P2) | C | done (mixed) | d003ed1 (fmm_baselines.py); P2 ran (c7a054f pilot: Skin in full; 9a3bcb6): better 6, no significant difference 17, worse 1 (IOL-GFMM on MAGIC); paper revs 624, 638, 649 |
| D5 | Skin duplicates: disclose; deduplicated variant (P3) | A + C | done | d003ed1 (loader); paper rev 587 (§6); P3 ran (331cb45): +6.20, the Skin result stands; paper revs 632, 638, 649 |
| D6 | Time-ordered case study (B10, P9) and held-out datasets (B11, P10) | C | pre-registered | 5 OpenML files downloaded with the user's approval (2026-11-16), SHA-256 pinned; P9 and P10 pushed with batch 3 |
| R1 | Licence (team decides); anonymised copy | A | in progress | LICENSE-PENDING.md (the team chooses; hyperbox-brain is GPL-3.0); anonymised copy: the user's account |
| R2 | OSF deposit of the plans with SHA-256 | A | bundle ready | `release/osf/` (MANIFEST.md, sha256sums.txt: 38 plans at a89a179; six amended plans, every amendment before its results); the upload and DOI need the user's OSF account |
| R3 | README test count; CLAIMS to the current rev; container | A + B | in progress | README test count (501, 27 min); Dockerfile; CLAIMS refresh to the final rev waits for the write-ups (its scripts run on the home PC) |
| P1 | Main text to ~7,500 (TMLR) / ~6,500 (TFS) words | A | todo | |
| P2 | Abstract: at most 3 claims and 1 caveat | A | todo | |
| P3 | Real figures; RQ1 and RQ2 key figures | A + B | todo | |
| P4 | Terms and placement | A | done | paper revs 588–617 ('level' → 'no significant difference') |

## Section B

| ID | What | Cost | Run | Status | Commit / rev |
|---|---|---|---|---|---|
| B1 | Proposition 5: exact Shapley values of the margin are #P-hard | A | test only | done | 79c5584; paper revs 583, 619; the guide checks the proof |
| B2 | PACE-SHAP and PACE-flip for min-membership networks (GFMM) | C | P4 | done (PACE-SHAP only) | d003ed1; P4 ran: first check FAIL (adapter's tie rule), Amendments 1 and 2 (1961d8a, aa86e32), rerun PASS (b061ac0): within 6.8e-16, 10.7–94× faster; paper revs 627, 632, 649, 656. PACE-flip-min is implemented and unit-tested against occlusion on toy models (`tests/test_gfmm_explain.py`), but P4 did not run it, so the paper claims only PACE-SHAP for GFMM |
| B3 | Tie-aware explanation of the decision | C | P5 | done | 7843fee; P5 ran (a591bb3); paper revs 624, 636, 649, 656 |
| B4 | RQ1 key figure: time against error | C | P1 | ran; figure todo | P1 ran (aca8e37); the figure is drawn with the restructure (P3) |
| B5 | RQ2 key figure: accuracy, boxes and broken predictions across θ | B | none | done | scripts/rq2_figure.py; paper revs 620–623 (Figure 5, §8) |
| B6 | Churn bound; when a shrink cannot hurt; which datasets gain | A + B + C | P11 | done | P11 ran (cb83050): Check PASS, S holds for 1.9% of accepted edits, second stage not needed; paper revs 649, 656 (Prop 6 kept as a guarantee, fast path in Next steps). Before: | 79c5584 (audit functions); P11 pushed with batch 2 (on_try hook in active.py); parts 1 and 3: reanalysis: results/tough_review/ §5. Finding: a first shrink always cuts a member, so condition S fails for first shrinks |
| B7 | Certificate for PACE-SHAP explanations | C | P6 | done (sound but loose) | 79c5584; P6 ran (38d6eb7, 29.8 h): Check PASS, certifies 99.1–100% of rows; paper revs 638, 649, 656 |
| B8 | Exact flip sets | C | P7 | done | 7843fee; 4043ddf; P7 ran (8746697); paper revs 627, 632, 649, 656 |
| B9 | Faster exact mode by recurrence; vectorised brute force | C | P8 | done (limit not raised) | 7843fee; P8 first run FAIL (comparison bug), Amendment 1 (ab27a82), rerun PASS (1d2acfc): recurrence 1.03–2.13× faster, Letter 26.1 s, so the limit stays at 12; paper revs 632, 649, 656. **To do after batch 3:** make the recurrence the default exact algorithm (the plan's rule) and rerun the suite on the home PC |
| B10 | Time-ordered case study | C | P9 | pre-registered | electricity (OpenML 151); `scripts/run_time_ordered.py`; batch 3 |
| B11 | Held-out datasets chosen by a rule | C | P10 | pre-registered | rule frozen in 94dd0ed before download; 4 datasets; `scripts/run_heldout.py`; batch 3 |
| B12 | Running example; two-question structure | A | none | todo | |
| B13 | Release: package, kit, demo | A | none | in progress | part 1: pyproject.toml, pace_cfmnn/api.py + tests, Dockerfile; part 2: OSF bundle (`release/osf/`), 2-D demo (`release/demo/`, artifact BBLuALTy1eHGeT1JeKLNUG; not yet opened in a browser); pending: Zenodo DOI at camera-ready |
| B14 | Second-paper ideas as "Next steps" lines | A | none | done | paper revs 597–608 (§11 'Next steps') |

## Runs (home PC)

New seeds start at 250. Appendix E's seeds run to 242, and none of the ranges below overlaps them.

| Run | Batch | Seeds | Datasets | Est. time |
|---|---|---|---|---|
| P1 time to accuracy + re-time | 1 | 300–303 | 6 (≤12 features) + 13 like-for-like | 2–2.5 h, alone |
| P3 Skin deduplicated | 1 | 260–269 | Skin (+ original as reference) | 0.5–1 h |
| P7 exact flip sets | 1 | 340–349 | 11 explanation datasets | 1–1.5 h |
| P8 recurrence exact mode | 1 | 350–352 | 6 + Wine, Letter | 3–4 h, alone |
| P2 tuned fuzzy min-max baselines | 2 | 250–259 | RQ2's 6 | 8–20 h (pilot first) |
| P4 GFMM exactness | 2 | 310–319 | 6 (≤12 features) | 1.5–3 h |
| P5 decision explanation | 2 | 320–329 | MAGIC, Letter, Page Blocks | 15–30 min |
| P6 PACE-SHAP certificate | 2 | 330–339 | RQ2's 6 | 2–4 h |
| P11 cannot-hurt fast path | 2 | 390–399 | RQ2's 6 | 1–2 h |
| P10 held-out datasets | 3 | 360–379 | 4 OpenML datasets (rule in RULE.md) | 2–4 h (smoke projection; first guess 6–12 h) |
| P9 time-ordered case study | 3 | 380–389 | electricity (OpenML 151) | 4–5 h (smoke projection; first guess 1–2 h) |

Batches 1 and 2 ran on 2026-11-16/18 (runbook §12, §14). Batch 1 took about 4 h longer than planned, because of P8's rerun. Batch 2 took about 34 h against 10–20 h, because of P6 (29.8 h) and P4's check (5.3 h). The write-up is in FINDINGS §23 and CLAIMS rows TR1–TR14.
