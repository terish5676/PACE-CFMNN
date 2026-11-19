# Run P4, Amendment 1: the adapter's tie rule for IOL-GFMM

Written on 2026-11-18, after the first `check` run and before the rerun. The pre-registration is
unchanged; this note records one fix to the code it describes.

## What happened

The first `check` run (home PC, 4 jobs, 18,795 s; kept unedited in `first_run/`) printed
**Check: FAIL**. PACE-SHAP-min equalled the brute force within 7e-16 on every compared point, but
on 10 of the 60 IOL-GFMM tasks the adapter (`GFMMBoxes`) predicted a different class from
hyperbox-brain on one of the 10 explained points:

- S2 seeds 310, 312, 314, 316, 317 and 318;
- S3 seeds 311, 316 and 318;
- Page Blocks seed 311.

Online GFMM matched on every task.

## The cause

On each of those points every box has membership 0, so every class ties and the tie rule alone
decides. `GFMMBoxes` broke every tie by the Manhattan distance to the box centres
(hyperbox-brain's `predict_with_manhattan`), which is what Online GFMM's `predict` uses.
IOL-GFMM's `predict` defaults to `type_boundary_handling=PROBABILITY_MEASURE`
(`predict_with_probability`): the class with the largest share of membership times box sample
count among the tied boxes, with ties in that going to the class whose tied boxes hold more
samples, and, at membership 1, a tied box holding one sample chosen with `random.seed(0)`.

The class matters for the explanation: PACE-SHAP-min explains the margin of the predicted class.

## The fix

- `pace_cfmnn/gfmm_explain.py`: `GFMMBoxes` takes `n_samples` and `tie_rule`
  (`"manhattan"`, the default, or `"probability"`). `GFMMBoxes.from_hbbrain` reads the default of
  the model's `predict(type_boundary_handling=...)` and uses the probability rule, with the
  model's `N_samples`, when it is `PROBABILITY_MEASURE`. Online GFMM keeps the Manhattan rule.
  Nothing else changes: no model, setting, dataset, seed, tolerance or timing.
- `tests/test_gfmm_explain.py`: `test_iol_gfmm_ties_follow_its_probability_rule` builds an
  IOL-GFMM model where the two rules disagree at full ties; it fails before the fix and passes
  after it.
- On the 10 failed tasks, the fixed adapter predicts like hyperbox-brain on every explained point
  (checked from a script outside the repository before the rerun).

## Approval and rerun

The user approved the fix and a full rerun of the `check` part on 2026-11-18. The first run is
kept unedited in `first_run/`; the rerun uses the same command with `--fresh`, then the `timing`
part runs alone as planned.
