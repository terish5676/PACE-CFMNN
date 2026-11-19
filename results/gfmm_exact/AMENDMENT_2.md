# Run P4, Amendment 2: the ExactExplainer arm works in blocks

Written on 2026-11-18, after the first `timing` run and before its rerun. The pre-registration is
unchanged; this note records one fix to the code it describes.

## What happened

The first `timing` run (home PC, alone, `--jobs 1`, 2,495 s) printed **FAILED** for 14 of its 18
tasks: every S1, S2, S3 and MAGIC seed, and Shuttle seeds 311 and 312. Each failed with
`MemoryError` in shap's ExactExplainer arm, for example "Unable to allocate 36.0 GiB for an array
with shape (204800, 2358, 10)". The script still exited 0 and printed Check: PASS, because the
check reads only the tasks that finished. The four that finished (Shuttle 310, Page Blocks
310–312) agreed everywhere within 6e-16. Their `timing.csv` and the `summary.md` it wrote are kept
unedited in `timing_first_run/`; `run_log.txt` keeps the whole output, and the rerun is appended
to it.

## The cause

`exact_shap_min` (`scripts/run_gfmm_exact.py`) gives shap's ExactExplainer the margin of the
predicted class as its model. ExactExplainer evaluates all 2^d coalitions against the 200
explainer rows in one call (204,800 rows at d = 10), and `GFMMBoxes.margin` builds a (rows,
boxes, features) array: about 33–36 GiB with the 2,100–2,450 boxes of S1–S3 and MAGIC's IOL-GFMM,
on a PC with 16 GB. The laptop's timing smoke slice used only Shuttle and Page Blocks (few boxes),
so it did not meet this. The brute force had met the same problem before the plan was pushed and
already works in blocks of at most 4 million numbers.

## The fix

- `scripts/run_gfmm_exact.py`: new `blocked_margin`, which evaluates the margin in blocks of at
  most `CHUNK_ELEMS` = 4,000,000 (row, box, feature) numbers, `brute_force_min`'s default, and
  joins the results. ExactExplainer's model is now `blocked_margin` in place of one `margin` call.
  The values are the same; only memory use changes. PACE-SHAP-min, the brute force, the models,
  settings, datasets, seeds, points, tolerances and the check are unchanged.
- `tests/test_gfmm_exact_run.py`: `test_exact_explainer_arm_works_in_blocks` checks that
  `blocked_margin` equals one `margin` call exactly, that the arm never asks the model for a block
  larger than `CHUNK_ELEMS`, and that it still matches PACE-SHAP-min within 1e-9.

## Approval and rerun

The user approved the fix and a rerun on 2026-11-18. The whole `timing` part is rerun alone with
`--fresh`, so all 18 tasks are timed by the same code.
