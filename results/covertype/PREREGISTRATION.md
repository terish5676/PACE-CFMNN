# Run K, one large dataset (Covertype): pre-registration

Written on 2026-11-14 on the laptop and pushed, with the code it describes, before the run.
The run itself is on the home PC (i5-14400F).

**This plan is not complete until Amendment 1 is appended on the home PC** (the file's
SHA-256 and the smoke times; see "Before the run"). No study seed may run before that
amendment is pushed.

- **The code:** `scripts/run_covertype.py` (tests in `tests/test_covertype.py`).
- **New loader:** `data.load("covertype")` (`load_covertype` in `pace_cfmnn/data.py`). It is a
  new name; no other loader and no dataset list changed (tested: `data.UCI` and
  `data.BUILTIN` do not contain it).
- **Used as it is:** `pace_cfmnn/ties.py` (Run I's tie counts), `pace_cfmnn/explain.py`
  (PACE-flip and PACE-SHAP), and `run_faithfulness.backgrounds_v3` (the explainer rows).

## Why

The peer review (item 15) asks for "one large run (CoverType) for accuracy, tie rate and
explanation time". The paper's largest dataset is Skin (245,057 rows, 3 features). Covertype
has more rows and many more features, so it shows whether training, the tie finding and the
explanation times hold up at that size.

## The dataset

UCI dataset 31, "Covertype": **581,012 rows, 54 features** (10 measured, 4 binary wilderness
areas, 40 binary soil types), **7 classes** (labels 1–7).

- File: `covtype.data.gz` (10.7 MB on the UCI page), inside
  `https://archive.ics.uci.edu/static/public/31/covertype.zip`. The loader reads it as
  shipped, from `data/covertype/covtype.data.gz`.
- **It was not downloaded when this was written.** The user chose to download it on the home
  PC only. The loader and the script were tested on made-up rows of the same shape.

## What ran before this was written

- **Unit tests** on made-up data shaped like Covertype (54 columns, the same one-hot blocks).
  A first version of that made-up data had 44 independent random binary columns; it gave
  almost one box per row and PACE-SHAP took 9 s per point, so the test data was changed to
  one-hot blocks. This says nothing about the real data.
- **Nothing on Covertype itself:** no fit, no accuracy, no timing. So the run's time is
  unknown, and the smoke slice on the home PC decides whether the run goes ahead.

## Design: descriptive

**No comparison is tested and there is no primary.** One dataset cannot support a paired
test across datasets, and the review asks for numbers, not a verdict. Every number is
reported as mean ± standard deviation over the seeds, or pooled over them.

- **Seeds:** 170–174 (5). No earlier run or smoke slice used them.
- **Split:** the paper's 80/20 split (`data.paper_split`), the training rows in the seed's
  order, min-max scaling fitted on the training rows (the accuracy table's setup): 464,809
  training rows and 116,203 test rows.
- **CFMNN:** `CFMNN(theta=0.3, gamma=4)`, the fixed centroid rule and search expansion: the
  paper's model, with no tuning for this dataset.

**Recorded for CFMNN:**

- seconds to train; boxes; training error;
- the share of training errors that are ties (Run I's count: both memberships are 1, the
  form Proposition 1 states; the looser count is recorded too). **Proposition 1 predicts
  100%.** Any other share would contradict it and would be reported as such;
- the share of test predictions decided by the nearest centroid, and the share of the test
  errors that are among them;
- test accuracy and macro-F1.

**Two references,** on the same rows in the same order: 10-nearest neighbours
(`KNeighborsClassifier(n_neighbors=10)`) and a Random Forest
(`RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)`), as in the
accuracy table. Test accuracy and macro-F1.

**Explanation time,** on 50 test points per seed (chosen as the explanation runs choose
them), with the explainer rows of the second explanation run:

- **PACE-flip:** `PaceFlip(model, background=explainer_rows)`;
- **PACE-SHAP,** sampled mode (54 features): `PaceShap(model, first 200 explainer rows,
  n_permutations=32)`.

Milliseconds per explanation, one batch of 50 points. The run does one seed at a time with
nothing else running, so the times are measures. (The Random Forest uses every core, but
only after CFMNN and the explainers of that seed are done.)

Also recorded: the peak memory of the process and the minutes per seed.

## What the result changes

- Results are reported whichever way they go, including a low accuracy, a long training
  time, or a tie share that is not 100%.
- Nothing in the paper's method or settings changes because of this run.
- The paper may say only what one dataset supports: "on one large dataset". It may not say
  that CFMNN or PACE scales in general.
- Any change to this plan after the push is a dated amendment (before the run) or a reported
  deviation (after it). Any analysis not listed here is labelled post hoc.

## Before the run (home PC)

1. **Ask the user before downloading,** giving the URL and size above. After their yes:
   - add this entry to `DATASETS` in `scripts/download_data.py`, download with
     `python -B scripts/download_data.py --datasets covertype`, and put the SHA-256 of the
     downloaded `covtype.data.gz` in the entry (the first attempt, with a placeholder hash,
     downloads the file and then reports it as different; compute the hash with the script's
     `sha256` function and put it in):

     ```python
     "covertype": (31, "covertype", {
         "covtype.data.gz": "<SHA-256 of the file you downloaded>",
     }),
     ```

   - add a Covertype row to `data/README.md`;
   - run `python -B -m unittest tests.test_covertype`: with the file in place, one more test
     runs. It checks the shape (581,012 × 54), the labels 1–7, that columns 11–54 are binary,
     and that the file matches the recorded hash. Until the entry is added, that test fails
     on purpose.
2. **Smoke slice, seed 190** (not a study seed, never analysed), written outside the
   repository, with nothing else running:

   ```
   python -B scripts/run_covertype.py --seeds 190 --subsample 0.1 --out <scratch folder>/k10 --fresh
   python -B scripts/run_covertype.py --seeds 190 --out <scratch folder>/k100 --fresh
   ```

   The first uses a stratified 10% of the rows; the second the full data for one seed.
3. **Project the full run:** 5 × the minutes of the full-data seed.
   **Stop and ask the user if** it projects to more than 6 hours, or the peak memory comes
   within about 2 GB of the PC's memory, or the 10% slice alone takes more than 30 minutes
   (then do not start the full-data seed).
4. **Append "Amendment 1 (date, before the run)"** to this file, commit it with the
   `download_data.py` entry and the `data/README.md` row, and push. It records: the SHA-256;
   the smoke slice's minutes, peak memory, boxes and time per explanation (10% and full);
   what was looked at (the script prints seed 190's accuracies); the projected time; and any
   change it forces. Nothing above the amendment is edited.

## Tests

`tests/test_covertype.py` checks that:

- the loader reads a gzip file laid out like UCI's, refuses a file with the wrong number of
  columns, names a missing file, and leaves the other dataset lists unchanged;
- the split is the accuracy table's, and the explained points are chosen as in the
  explanation runs;
- one task writes three rows with the same columns; CFMNN's row matches a model fitted
  directly; every training error is a tie on the made-up data; PACE-SHAP runs in sampled
  mode;
- the summary says the run is descriptive and prints every model.

## How to run

```
python -B scripts/run_covertype.py
python -B scripts/run_covertype.py --resume            # continue after a stop
python -B scripts/run_covertype.py --summary-only
```

No `--jobs`: one seed at a time, nothing else running. Outputs in `results/covertype/`:
`runs.csv` (one row per seed and model: CFMNN, 10-NN, Random Forest; 5 × 3 = **15 rows**)
and `summary.md`. Save the printed output as `run_log.txt`.

## Amendment 1 (2026-11-14, before the run)

Written and pushed on the home PC, with the `download_data.py` entry and the `data/README.md`
row, before any study seed (170–174) was run. Nothing above this section was edited.

**The file.** Downloaded on the home PC on 2026-11-14 with
`python -B scripts/download_data.py --datasets covertype`, from
`https://archive.ics.uci.edu/static/public/31/covertype.zip`. `covtype.data.gz` is 11,240,707
bytes, SHA-256 `614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`. That hash is
now in the script's entry; `--check` passes for every dataset, and all 7 tests in
`tests/test_covertype.py` run and pass (the one that needs the file included).

**The machine.** i5-14400F ("Intel64 Family 6 Model 191 Stepping 2, GenuineIntel"), 16 logical
CPUs, 15.8 GB of memory, Python 3.14.10.

**Smoke slices, seed 190** (not a study seed, never analysed), written outside the repository,
one at a time with nothing else running:

| Slice | Wall time | Minutes per seed (script) | Peak memory | Boxes | PACE-flip | PACE-SHAP |
|---|---|---|---|---|---|---|
| 10% of the rows (`--subsample 0.1`) | 22 s | 0.33 | 851 MB | 80 | 0.8 ms | 299 ms per explanation |
| Full data | 130 s | 2.13 | 3,049 MB | 111 | 1.1 ms | 419 ms per explanation |

On the full data, CFMNN trained in 10 s; the tie count took 18 s; 10-NN and the Random Forest
took 34 s each.

**What was looked at.** Everything the two slices printed and wrote to their `summary.md`, for
seed 190: each model's test accuracy and macro-F1 (full data: CFMNN 55.97%, 10-NN 92.56%,
Random Forest 95.58% accuracy), CFMNN's training error (43.96%), the tie counts (204,350 of
204,350 training errors are ties inside a box of both classes), the share of test predictions
decided by the nearest centroid (99.08%), the explanation times and PACE-SHAP's share of
skipped (row, box) pairs. One seed is not evidence; only seeds 170–174 are.

**Projection.** 5 × 2.13 minutes ≈ **11 minutes** for the run. Peak memory 3.0 GB is far from
the PC's 15.8 GB. The 10% slice took 22 s. None of the plan's stop conditions applies.

**Changes forced.** None. Every setting, seed, model, measure and the descriptive design stay
as planned.
