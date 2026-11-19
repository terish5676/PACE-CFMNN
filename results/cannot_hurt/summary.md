# Run P11: how many self-repair edits provably cannot hurt

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The refinement run's split and self-repair (gated on the training rows), seeds 390–399. Written by `scripts/run_cannot_hurt.py`.

**Check: PASS.** No edit meeting condition S broke a correct training prediction (116 such edits, accepted or not).

| Dataset | Accepted edits | Meet S | Share | Gated edits meeting S that the gate rejected | Own-class rows cut | Won and dropped | Above the winner (near-tie) | Centroid moved | Other boxes changed | Accepted edits that broke a correct training prediction | Gate time on accepted S edits (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 122 | 0 | 0.0% | 0 | 122 (100%) | 34 (28%) | 0 (0%) | 120 (98%) | 0 (0%) | 46 (38%) | 0.0 |
| landsat | 1,654 | 74 | 4.5% | 0 | 1,520 (92%) | 371 (22%) | 13 (1%) | 1,423 (86%) | 0 (0%) | 830 (50%) | 16.6 |
| page_blocks | 113 | 9 | 8.0% | 0 | 102 (90%) | 8 (7%) | 0 (0%) | 101 (89%) | 0 (0%) | 62 (55%) | 0.2 |
| magic | 1,899 | 13 | 0.7% | 0 | 1,882 (99%) | 390 (21%) | 0 (0%) | 1,758 (93%) | 0 (0%) | 1,680 (88%) | 4.7 |
| letter | 1,972 | 17 | 0.9% | 0 | 1,944 (99%) | 87 (4%) | 0 (0%) | 1,928 (98%) | 0 (0%) | 1,325 (67%) | 24.3 |
| skin | 350 | 3 | 0.9% | 0 | 346 (99%) | 69 (20%) | 3 (1%) | 327 (93%) | 0 (0%) | 248 (71%) | 3.1 |

Overall: 116 of 6,110 accepted edits meet S (1.9%); the gate rejected 0 gated edits that meet S. The five columns after it count the accepted edits failing each part of S (an edit can fail several). The gate bounds the drop in macro-F1 on the gate rows, not each row, so an accepted edit can still break some correct training predictions.

**Second stage (the fast path):** not needed. It runs only where at least 10% of accepted edits meet S and the gate rejected an edit meeting S; elsewhere skipping the gate for those edits changes no decision, so it gives the same model, and the saving is the gate time in the last column.
