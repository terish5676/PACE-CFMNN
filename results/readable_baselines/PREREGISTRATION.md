# Readable baselines and θ chosen by validation: pre-registration

Written on 2026-11-06 and pushed (with the code it describes) before any run on the
study seeds, 70–89.

## Smoke runs

Smoke runs used seed 99 only, which is not a study seed:

- CFMNN at θ = 0.05, 0.1, 0.3 and 0.5 on Letter, MAGIC, Shuttle, Landsat and Skin:
  one fit each, timed, with its box count.
- The script without EBM, on Skin, S1, Letter and MAGIC, with GFMM tuned like
  CFMNN-val. Skin and S1 finished, and their log printed their seed-99 accuracies.
  Letter and MAGIC were stopped after 1 h 45 min, still tuning GFMM.
- EBM, one fit each, timed, with its term count:
  - at its defaults on Wine, Breast Cancer and MAGIC. Skin was stopped after
    47 min and S1 after about 17 min;
  - with main effects only on every dataset except S2 and S3.
- GFMM at θ = 0.3: one fit on each of the 13 datasets, timed, with its box count.

Their output was thrown away. Two settings changed because of their **times**, never
their accuracies. The plan said: "If GFMM or EBM would take more than 8 h in total,
stop and ask." Both would have, so the user chose these on 2026-11-06:

- **EBM runs with main effects only** (`interactions=0`). At its default, with pair
  terms, one fit took 26 min on Wine (142 training rows) and 12.5 min on MAGIC.
- **GFMM is not tuned.** It runs at θ = 0.3, the explained model's value. Tuned
  like CFMNN-val, it took 36 min per seed on Skin and 15 min on S1.

Even so, EBM with main effects only takes 20–29 min per fit on Letter, Shuttle
and Skin, so the whole run takes about 44 h of single-core time. The user chose
to run it with six processes instead of four. This changes only the speed.

Two bugs were fixed during the smoke runs, neither of which changes a result:

- the summary step crashed, from a Python scoping error;
- EBM's own worker processes crash on this machine (joblib/loky), so EBM always
  runs in one process.

## Why

On all 13 datasets, CFMNN is 0.2–21.5 points less accurate than the better of
10-NN and a Random Forest (review v1, Test 4). The paper says it accepts that
cost for a model whose rules people can read. Reviewers will ask two things:

1. Is CFMNN as accurate as other models people can read, when every model is
   tuned the same way?
2. Was θ = 0.3 a poor choice?

θ = 0.3 stays the setting for every RQ1–RQ4 result. This run only adds a check
on the base model. Earlier results stay untouched. Everything new goes to
`results/readable_baselines/`.

## Setup

- **Datasets:** all 13: S1, S2, S3, Wine, Breast Cancer, Digits, Thyroid,
  Shuttle, Skin, MAGIC, Letter, Landsat, Page Blocks.
- **Seeds:** 70–89 (20). No earlier study used them.
- **Data:** the paper's 80/20 split (`data.paper_split`). The training rows come
  in the seed's order, as in `scripts/accuracy_table.py`. Min-max scaling is
  fitted on the training rows.
- **Fairness:** every model is trained on the same rows in the same order.

## Tuning

A tuned model:

1. scores each of its settings by stratified 3-fold cross-validation on the
   training rows. The folds come from the seed, and each fold keeps the row
   order, because CFMNN depends on it;
2. takes the setting with the highest mean CV accuracy. On an exact tie (to 12
   decimal places), it takes the simplest setting, which is listed first below;
3. is refitted with that setting on all the training rows, and scored once on
   the test rows.

## Models

| Model | Package | Settings, simplest first |
|---|---|---|
| **CFMNN-0.3** (the explained model) | this repository | θ = 0.3, γ = 4. Not tuned. |
| **CFMNN-val** | this repository | θ ∈ {0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05}, γ = 4 |
| **Tree** (CART) | scikit-learn 1.9.1 | max leaves ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024} |
| **LogReg** (logistic regression) | scikit-learn 1.9.1 | C ∈ {0.01, 0.1, 1, 10, 100}; up to 5,000 iterations |
| **EBM** (explainable boosting machine) | interpret-core 0.7.8 | main effects only (`interactions=0`), library defaults otherwise. Not tuned. |
| **GFMM-0.3** (online general fuzzy min-max) | hyperbox-brain 0.1.6, `OnlineGFMM` | θ = θ_min = 0.3, γ = 4. Not tuned. |
| **kNN** (reference) | scikit-learn 1.9.1 | k ∈ {20, 10, 5, 3, 1} |
| **Random Forest** (reference) | scikit-learn 1.9.1 | 200 trees, library defaults otherwise |

- The tree, EBM and Random Forest get `random_state` = the seed.
- GFMM's θ caps each side of a box. CFMNN's θ caps the sum of the sides at
  d × θ. So at the same θ, GFMM's boxes are at most as large as CFMNN's.
- EBM with main effects only is an additive model: one curve per feature.
  EBM's default also adds pair terms. These took too long here, and the paper
  will say they were off.

## Recorded

For each dataset, seed and model:

- test accuracy and macro-F1, in %;
- the setting chosen, and the mean CV accuracy of every setting tried;
- seconds to tune, fit and predict. Six runs share the CPU, so times are
  compared within a dataset only;
- size, as rules and conditions per rule:
  - **CFMNN and GFMM:** rules = boxes. A box's conditions = the features whose
    interval is narrower than [0, 1].
  - **Tree:** rules = leaves. A leaf's conditions = the distinct features
    tested on its path.
  - **EBM:** rules = terms, one per feature here, with 1 condition each. Each
    term is a curve, not a single test, so this does not compare directly with
    boxes or leaves.
  - **LogReg:** rules = non-zero weights.

## Comparisons

Each comparison is paired over the 20 seeds. It uses a two-sided Wilcoxon
signed-rank test, Holm-corrected across the 13 datasets of that comparison.
When all 20 differences are zero, p = 1.

- A model is **better on a dataset** if the Holm p is below 0.05 and the mean
  difference favours it.
- **Worse** is the same, in the other direction.
- Otherwise the two are **level**.

| Code | Comparison | Measure | Question |
|---|---|---|---|
| P1 (primary) | CFMNN-val vs Tree | accuracy | Is CFMNN as accurate as a readable model of the same kind (rules), both tuned? |
| P2 (primary) | CFMNN-val vs LogReg | accuracy | … as a linear model, both tuned? |
| P3 (primary) | CFMNN-val vs EBM | accuracy | … as an additive model? |
| P4 (primary) | CFMNN-0.3 vs GFMM-0.3 | accuracy | Is CFMNN as accurate as the standard fuzzy min-max learner at the same θ? |
| S1 | CFMNN-0.3 vs Tree, LogReg and EBM | accuracy | The same, for the model the paper explains |
| S2 | CFMNN-val vs CFMNN-0.3 | accuracy | Was θ = 0.3 a poor choice? |
| S3 | P1–P4 | macro-F1 | Do the primary results hold for macro-F1? |

The four primary comparisons answer separate questions, so they are not
corrected for each other. No direction is predicted.

**Descriptive:**

- the gap from CFMNN-0.3 and from CFMNN-val to the better of the tuned kNN and
  the Random Forest;
- the setting each tuned model chose on each dataset, counted over seeds;
- mean sizes and times.

## Reporting

- Results are reported whichever way they go.
- Any change to this plan after the push is reported as a deviation.
- Any analysis not listed here is labelled post hoc.
- Every number the paper takes from this run is recomputed from `runs.csv` by
  a read-only script.

## How to run

From the repository root, with the laptop kept awake:

```
python -B scripts/run_readable_baselines.py --jobs 6
```

- **Processes:** six, on a laptop with 4 cores and 8 threads. Each model inside
  runs in one process. The slowest datasets (Letter, Skin, Shuttle) start
  first.
- **Expected time:** about 8–9 h.
- **Output:** the rows are saved after each dataset-seed pair, so a stopped run
  continues with `--resume`. To rebuild `summary.md` from the saved rows, use
  `--summary-only`.
