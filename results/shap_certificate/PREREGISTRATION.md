# Run P6, a certificate for PACE-SHAP explanations after an edit: pre-registration

Written on 2026-11-16 and pushed, with the code it describes, before the run (tough review B7).
The code is `scripts/run_shap_certificate.py` (tests in `tests/test_shap_certificate_run.py`). It
uses `audit.pace_shap_certificate` (commit 79c5584) and the existing `on_accept` hook of
`active.run_self_repair`.

Runbook: `research_audit/HOME_PC_RUNBOOK.md`, section 13.

## Background (in plain words)

After self-repair edits a box, which explanations can have changed? The **certificate** marks the
rows whose PACE-SHAP values the edit may have changed. A row is left out when both hold:

- its predicted class is the same before and after the edit; and
- no changed box survives pruning for any explainer row, before or after the edit.

For a row left out, every coalition's margin is a maximum over the same unchanged boxes, so its
values cannot change. In exact mode they are identical. In sampled mode they are identical given
the same seed, points and order.

A user who re-explains only the certified rows after an edit would then be sure the others are
unchanged.

## What ran before this was written

- **Tests while building:** random edits on toy models; no value changed outside the
  certificate.
- **Smoke slice on the laptop, seed 194** (not a study seed, written outside the repository; 61–128
  s per dataset):

  | Dataset | d | Mode | Accepted edits | Certified share | Changed share | Median / largest relative change |
  |---|---|---|---|---|---|---|
  | Thyroid | 21 | sampled | 14 | 100.0% | 90.3% | – |
  | Page Blocks | 10 | exact | 18 | 100.0% | 83.4% | 1.7e-2 / 10 |
  | Skin | 3 | exact | 43 | 100.0% | 46.1% | – |

  Check: PASS. As review B7 expected, the certificate is sound but marks nearly every row. On
  these datasets, nearly every row has a changed box among its surviving boxes.

  After the slice, the two relative-change columns were added: they say how large the real changes
  are. The test logic did not change.

## The question

**Is the certificate sound on every accepted self-repair edit, and how much of the explained data
does it cover?**

## Setup

- **Datasets:** RQ2's six: Thyroid, Landsat, Page Blocks, MAGIC, Letter and Skin.
- **Seeds:** 330–339 (new).
- **Split, model, budget and rounds:** the refinement run's (`run_refinement.split`; CFMNN at
  θ = 0.3; budget min(300, pool/4); 10 rounds).
- **Repair:** self-repair gated on the training rows, as `run_refinement_v2.run_one`'s SR arm.
- **Explained rows:** 100 test rows drawn with the seed.
- **Explainer rows:** 100 training rows. That is fewer than the paper's 200, for time. The
  certificate's logic does not depend on how many.
- **PACE-SHAP:** exact where d ≤ 12 (Page Blocks, MAGIC, Skin). Elsewhere (Thyroid, Landsat,
  Letter) it is sampled, with 8 permutations and a fixed seed.
- **After every accepted edit**, an `on_accept` hook computes:
  - the certificate;
  - the new values, compared with the values before the edit (cached for each model, so each
    model is explained once).
- **Output:** `results/shap_certificate/edits.csv`, one row per accepted edit.

## Check (pass/fail)

No explained row outside the certificate has any value change larger than **1e-12**, over every
accepted edit.

## Reported (descriptive)

Per dataset:
- the certified share (mean per edit);
- the changed share;
- the share of certified (edit, row) pairs that changed (how tight the certificate is);
- the median and largest relative change (a row's largest value change ÷ its largest value before
  the edit).

## How the result will be read (written now, before the run)

- **FAIL:** Proposition B7 is wrong as stated, or the code does not implement it. The certificate is
  withdrawn from the paper until the cause is found.
- **PASS, and the mean certified share is at least 90% on every dataset:** the paper states B7 as a
  sound guarantee that is loose in practice. After an edit almost every row may change, so
  re-explaining everything costs about the same. The relative changes are reported, so readers see
  whether those changes are small.
- **PASS, and the certified share is under 90% on some dataset:** as above, plus the paper reports
  the share of explanations the certificate spares on those datasets.

## Time

The smoke slice took 61–128 s per seed for the three datasets above. Landsat (36 features) and
Letter (16) are sampled, and MAGIC is exact with many boxes, so they are slower: several minutes to
under an hour per seed on the laptop. **About 2–4 hours on the home PC with 6 jobs.** Stop and ask
if it is past 6 hours.

## Tests

`tests/test_shap_certificate_run.py`:
- the hook sees every accepted edit, and finds no change outside the certificate (Page Blocks
  exact, Thyroid sampled; seed 194, 10 rows);
- values are cached per model;
- the check fails as it should;
- the summary row.

## How to run (home PC, batch 2)

```
.\.venv\Scripts\python -B scripts/run_shap_certificate.py --jobs 6 *>&1 | Tee-Object results/shap_certificate/run_log.txt
.\.venv\Scripts\python -B scripts/run_shap_certificate.py --jobs 6 --resume *>&1 | Tee-Object -Append results/shap_certificate/run_log.txt   # after a stop
```

Commit `results/shap_certificate/` unedited (`edits.csv`, `summary.md`, `run_log.txt`) as "Run P6:
results of the pre-registered run on the home PC".
