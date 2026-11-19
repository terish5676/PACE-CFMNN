# Run P3: Skin without its duplicate rows

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Written by `scripts/run_skin_dedup.py`.

`skin_dedup` keeps the first copy of each row: 51,444 of 245,057 rows.

## Primary: does self-repair still help?

SR − B0, macro-F1 on `skin_dedup`, seeds 260–269 (10 seeds): **+6.20 points** (SR better on 10, worse on 0), Wilcoxon p = 0.0020. The paper's gain on Skin with duplicates: +5.66; half of it: 2.83.

**Reading:** Self-repair still helps without the duplicates, by at least half the paper's gain (+6.20 ≥ 2.83): the Skin result stands, reported with this number.

Reference, original Skin on the same seeds: SR − B0 = +7.56 (better on 10, worse on 0, p = 0.0020).

### Every arm (mean over seeds)

| Arm | Macro-F1 | Accuracy | Lowest class recall | Labels | Broken predictions (%) |
|---|---|---|---|---|---|
| B0 | 91.68 | 93.69 | 80.48 | 0 | 0.000 |
| B9 | 91.99 | 93.89 | 81.26 | 10288 | 0.812 |
| AL | 98.33 | 98.64 | 97.27 | 5445 | 0.466 |
| SR | 97.88 | 98.28 | 96.40 | 0 | 0.993 |
| SR+AL | 98.50 | 98.78 | 97.81 | 5445 | 0.478 |
| Retrain, appended | 91.91 | 93.87 | 80.89 | 5445 | 0.486 |
| Retrain, shuffled | 90.95 | 93.14 | 78.39 | 5445 | 5.032 |
| kNN, training rows | 99.56 | 99.64 | 99.49 | 0 | 0.196 |
| kNN, every label | 99.59 | 99.66 | 99.54 | 5445 | 0.173 |
| RF, training rows | 99.68 | 99.74 | 99.70 | 0 | 0.105 |
| RF, every label | 99.72 | 99.77 | 99.73 | 5445 | 0.093 |

SR+AL − SR (do bought labels add anything?): +0.62, better on 10, worse on 0, p = 0.0020 (secondary).

## Accuracy table (secondary)

| Model | Accuracy, `skin_dedup` |
|---|---|
| CFMNN | 92.92 |
| 10-NN | 99.73 |
| Random Forest | 99.80 |

## CFMNN across theta (secondary)

| theta | Accuracy | Boxes | NFR (%) |
|---|---|---|---|
| 0.05 | 99.43 | 2011 | 0.344 |
| 0.1 | 98.32 | 708 | 1.033 |
| 0.15 | 96.85 | 360 | 2.151 |
| 0.2 | 95.90 | 209 | 2.612 |
| 0.3 | 92.92 | 89 | 0.000 |
| 0.4 | 91.95 | 43 | 6.221 |
| 0.5 | 91.79 | 26 | 6.107 |

theta by validation: 0.05.

Broken correct predictions (NFR), repair (SR-0.3) against re-tuning (B0 at theta by validation): repair breaks -0.927 points fewer on average (fewer on 0 seeds, more on 10, p = 0.0020).

Repair on the paper's split (SR-0.3 − B0-0.3, accuracy): +5.09, better on 10, worse on 0, p = 0.0020.

