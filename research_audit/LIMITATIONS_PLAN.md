# Plan: new runs for the reviewer's limitations 4, 5 and 7, plus a faster exact gate

Approved by the user on 2026-11-11. All of it is built and run on the home PC. Follow it with `research_audit/HOME_PC_RUNBOOK.md`.

**What the runs are for:** the paper (rev 311) states these limitations, and each run tests whether they can be reduced. Item 6 (CFMNN is less accurate than kNN and Random Forest) stays a stated limitation, because only a different model would fix it.

| # | Limitation | What the run does |
|---|---|---|
| 4a | Deletion/insertion use 50 points per dataset (36 on Wine), 10 seeds | Run C: 200 points per seed, fresh seeds |
| 4b | Replacement inputs leave the data; the conditional test was run at one setting (σ = 0.1, 200 rows) | Run D: more σ and row settings. Removal tests always leave the data somewhat |
| 4c | Tied points get margin 0, which ignores the centroid tie-break (55–79% are tied on Letter, MAGIC and Page Blocks) | Run C: pre-registered tie-aware measures |
| 5 | One DiCE setting; robustness not measured | Run B: more counterfactual methods, and 4 robustness measures |
| 7a | One budget, one gate | Run A: budgets and gates |
| 7b | Certificates are exact but loose | Run A: one cheap tightening, kept only if it never misses |
| 7c | No bound on explanation changes | Run A: exact certificates for PACE-flip and PACE-decomposition. None for PACE-SHAP |
| new | Repair is slow: the gate is 92–99% of self-repair's time (`results/rq3_explainer_timing/`) | Run E: a cached gate that must give the same decisions |

## Rules for every run

- **Before a run:** each run has its own `results/<folder>/PREREGISTRATION.md`, committed and pushed before it starts.
  - No direction is predicted.
  - It lists everything that ran before it was written: the smoke slice and any timing.
- **Defaults never move, whatever the results:** 300 edits, gate 0.5/1.0, σ = 0.1, θ = 0.3, γ = 4. Sensitivity results are reported, never used to pick settings.
- **Never edit** existing scripts, pre-registrations, `runs.csv`/`summary.md` files, or any file a paper number comes from. New outputs go in new folders.
- **Library changes** only add functions or keyword arguments, and their defaults keep today's output exactly. Each one gets a test showing that.
- **Seeds:**
  - 110–119 are the study seeds for A–D (no earlier run used them).
  - E replays RQ3's seeds 50–59.
  - Seed 120 is for smoke and timing slices only and is never analysed.
- **Statistics (A–D):** paired two-sided Wilcoxon over the 10 seeds, Holm across datasets, α = 0.05. Each verdict is better, level or worse for the first arm.
- **Script form:** every script has `--datasets`, `--seeds`, `--jobs`, `--resume`, `--fresh`, `--summary-only` and `--out`. Copy the pattern of `scripts/run_repair_theta.py`, which uses `pace_cfmnn/resume.py` (`pending`, `run_all`).
- **Imports:** import from existing scripts rather than copying them.
- **Tests:** one test file per run in `tests/`, and the full suite must pass before each pre-registration.

## E. The cached gate (build and run first)

**The problem.** `Gate.measure` (`pace_cfmnn/refine.py`, class `Gate`) calls `model.predict` on every gate row after every edit. That costs n × H × d, even though a self-repair edit changes only one box. It either moves one face (`refine.shrink_box`) or removes the box (`CFMNN._remove_box`). The certificate `audit.certify` doesn't help as it stands, because it needs both models' top groups on every row, which is the same two full passes.

**The exact shortcut.** Add `class CachedGate(Gate)` to `pace_cfmnn/refine.py`:

1. **What it keeps for the gate rows:**
   - the membership table M (n × H);
   - the squared centroid distances D (n × H);
   - each row's top membership and predicted class;
   - copies of V, W, the centroids and the box classes.

   Compute M and D with exactly the expressions in `CFMNN.predict_details` (`pace_cfmnn/cfmnn_core.py`, the lines that build `M` and `D`). The winner is the top membership, ties within `tie_tol` going to the nearest centroid.
2. **Finding the change.** On each `measure`, compare the model with the cached copies to find the change:
   - nothing changed;
   - exactly one box moved;
   - exactly one box was removed.

   Anything else (including a change of box class) falls back to a full recompute.
3. **A moved box k:**
   - recompute column k of M and D (n × d work);
   - re-score only the rows where k was in the tied top group before (M[:,k] ≥ old top − tol), or where its new membership reaches the old top − tol.

   **Why the other rows can't change:** their top membership, tied set and distances are unchanged, so their prediction is too.
4. **A removed box k:** delete column k, and re-score the rows where k was tied at the top.
5. **A rejected edit** (the gate says no) restores the cache exactly. In `check`, keep an undo record of the changed column and rows, and apply it when the check fails. This matches the model's rollback to its snapshot.

**Wiring:**
- Add keyword arguments to `active.run_self_repair`:
  - `fast_gate=False`: when `True`, use `CachedGate` instead of `Gate`;
  - `gate_reference="last"`: `"start"` compares every edit with the model before any edit (the cumulative gate for run A), and any other value raises.
- Leave `run_refinement` (active learning) on the standard gate.

**Tests:**
- After random shrinks and box removals on synthetic models, the cached predictions equal `model.predict` element for element (`np.array_equal`).
- A rejected edit restores the cache.
- Box-cap rejection works as in `Gate`.
- `run_self_repair` with `fast_gate=True` gives the same model and the same per-edit log as with `fast_gate=False`. Check this under the default, strict (0/0) and loose gates.
- The defaults are unchanged.
- The `"start"` reference bounds the total loss.

**Script:** `scripts/check_fast_gate.py`, writing to `results/fast_gate_check/`.

- **Replay:** RQ3's SR arm exactly as `scripts/run_refinement_v2.py` runs it:
  - the split from `run_refinement.split`;
  - `CFMNN().fit`;
  - `budget = min(MAX_BUDGET, len(pool) // 4)`;
  - `n_rounds = N_ROUNDS`;
  - `chooser="volume"`;
  - gated on the training rows, with `gate_labels_are_new=False`.
- **Scope:** 6 datasets (Thyroid, Landsat, Page Blocks, MAGIC, Letter, Skin), seeds 50–59, with `fast_gate=True` and no certificate hook.
- **On seed 50 only,** also run the standard gate on the same machine.
- **What to compare:** each edit's `train_index`, `accepted`, `reason` and `x_correct_after`, against `results/refinement_v2_confirmatory/edits.csv.gz` (arm SR); and the final test macro-F1 against that folder's `runs.csv` (arm SR), to 1e-12.
- **Output:** `runs.csv` must have the same columns on every row, filled with NaN where a column doesn't apply.
- **Pass rule (the primary):**
  - all 60 replayed runs equal the committed log and macro-F1;
  - and the seed-50 standard-gate runs equal the cached-gate runs.

  Otherwise FAIL, and the cached gate is not used anywhere.
- **Speed:** report the ratio of standard to cached time on seed 50. No speed is required.

**Pre-registration disclosure:** a prototype of the cached gate was built and tried on the laptop on 2026-11-11 (commit ab5a166). It was then reverted (fa2f890) so that everything is rebuilt on the PC. Its results are not used.

## A. Repair budgets, gates and certificates

**Script:** `scripts/run_repair_sensitivity.py`, writing to `results/repair_sensitivity/`.

**Setup:** RQ3's split and SR arm, as in E, on the same 6 datasets, θ = 0.3, the volume cut, gated on the training rows. Use `fast_gate=True` only if E passed.

**Arms:**
- B0: no repair;
- SR-default: 300 edits (`min(300, len(pool)//4)`, which is 273 on Page Blocks), gate 0.5/1.0;
- budgets of 100, 1,000, and all training errors;
- at the default budget, gates:
  - none (both limits infinite);
  - strict (0/0);
  - loose (1.0/2.0);
  - cumulative (0.5/1.0 with `gate_reference="start"`).

**Measures:**
- test macro-F1, accuracy and lowest-class recall;
- boxes, and training error at the start and the end;
- edits accepted, and rejected by reason;
- test predictions fixed and broken against B0 (`audit.prediction_churn`);
- seconds.

**Certificates, on SR-default only.** Use an `on_accept` hook like `Certificates` in `scripts/run_refinement_explore.py` (the class starts at about line 90), on 100 test rows. Each certificate is scored for misses, precision and size, with brute-force recomputation before and after each accepted edit:
- the existing loose and tight certificates (`audit.certify`);
- **new `audit.certify_centroid`:** inside the tied set, a row can switch only if the changed box's centroid distance or tied membership changes who is nearest. It is dropped if it ever misses, in the tests or in the run;
- **new `audit.explanation_certificate`**, for PACE-flip and PACE-decomposition: a row is certified unchanged if no feature replacement can bring the edited box into the top or tied group. Use PACE-flip's pruning bound from `explain.py` (the docstring around lines 180–186: ceiling and floor per box when one feature is replaced). PACE-SHAP has no certificate.

**Primary:** each arm vs SR-default on test macro-F1.

**Tests:**
- the certificates never miss on random edits to small synthetic models, checked by brute force;
- the arms are built as specified;
- the summary verdicts are right on a made-up table.

## B. Counterfactual baselines and robustness

**Script:** `scripts/run_cf_robustness.py`, writing to `results/cf_robustness/`.

**Setup:** B1's design (`scripts/run_counterfactuals.py`, `run_one`, with `extended=True`):
- 11 datasets;
- 30 points per seed;
- the target is the runner-up class;
- Thyroid's features stay locked, as there.

**Methods:**
- PACE and PACE-sparse;
- DiCE random, as in the paper's v2 setting;
- DiCE genetic and DiCE kdtree. Add a `method=` keyword to `xai_baselines.dice_counterfactual`, default `"random"`, so today's output stays exact;
- the nearest unlike neighbour (`xai_baselines.nearest_unlike_neighbour`);
- **new:** a brute-force search over 1–2 features, using the replacement values from the training data. It returns the smallest-L1 valid answer, or none.

**Measures:**
- the existing ones: validity, features changed, L1, plausibility, ms;
- **r1, input sensitivity:** recompute the counterfactual for 5 perturbed copies of x (`neighborhood.Perturber`, κ = 0.05, as RQ2 uses it). Report the median of L1(cf′, cf) ÷ L1(x′, x). DiCE genetic is left out of r1 because of its cost;
- **r2:** the share of 20 noisy copies of the answer still predicted as the target;
- **r3:** validity under 4 CFMNNs fitted on the training rows in other orders;
- **r4:** the target margin at the answer (`xai_baselines.margin_score`).

**Primaries:** PACE and PACE-sparse vs DiCE random, on r1 and r3.

## C. RQ1 with 200 points, tie-aware

**Script:** `scripts/run_more_points.py`, writing to `results/rq1_more_points/`.

**Setup:** protocol v3 of `scripts/run_faithfulness.py` (`run(name, seed, n_explain, protocol="v3")`), with all its methods and settings.
- 200 points per seed; Wine keeps all 36.
- 11 datasets.
- An added arm: KernelSHAP on PACE-SHAP's same 200 background rows (`run_faithfulness._same_rows`), the like-for-like arm.

**Tie-aware outputs**, following `scripts/run_review_rq1.py` (around lines 108–133: `_margin`, ties at margin ≤ 1e-12, near-ties):
- tie and near-tie counts;
- AUCs on untied points only;
- raw (unnormalised) AUCs (`with_raw=True`);
- the class-kept AUC.

**Primaries:** PACE-SHAP vs KernelSHAP, at the usual setting and on the same rows, by deletion and by insertion.

## D. The conditional test at more settings

**Script:** `scripts/run_conditional_settings.py`, writing to `results/conditional_settings/`.

**Setup:** B3's conditional test (`scripts/run_faithfulness.py`, `run_cond`; see `scripts/posthoc_sigma.py` and `scripts/posthoc_rows.py` for how σ and the number of rows are varied).

**Settings:**
- σ of 0.05, 0.2 and 0.4 with 200 rows;
- σ = 0.2 with 800 rows.

The existing B3 run (`results/extensions_v1/conditional`) already covers σ = 0.1 with 200 rows.

**Datasets:** S1, Shuttle, Letter, Landsat, MAGIC and Page Blocks, with 50 points each.

**Explainers:** PACE-SHAP-cond and KernelSHAP-cond.

**Measures:** effective rows, and the off-data share (`pace_cfmnn/realism.py`).

**Primary:** the -cond pair at each setting.

## Expected run times on the i5-14400F (`--jobs 12`, ±50%)

| Run | Hours |
|---|---|
| E | ~0.2 |
| A | ~2 with the cached gate (~5 without) |
| B | ~5 |
| C | ~9 |
| D | ~6 |

If a seed-120 smoke slice projects more than twice these times, stop and ask the user.

## After the runs (not on the PC)

The write-up is done later in a laptop session:
- claims scripts;
- `results/FINDINGS.md`, `research_audit/CLAIMS.md`, `research_audit/CHANGELOG.md`;
- the paper's Limitations, §7 and Settings.
