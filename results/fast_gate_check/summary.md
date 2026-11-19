# Run E: the cached gate

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). RQ3's SR arm replayed with `fast_gate=True`, seeds 50–59, and compared with `results/refinement_v2_confirmatory/`. Written by `scripts/check_fast_gate.py`.

**Verdict: PASS**

- Replayed runs equal to the committed edit log and macro-F1 (to 1e-12): 60 of 60 expected (60 present).
- Seed 50, standard gate equal to the cached gate (edit log, model, test predictions): 6 of 6 datasets (6 run).
- Largest macro-F1 difference: 0.00e+00; edits that differ: 0.

## Per dataset

| Dataset | Seeds matching | Edits (total) | Mismatched edits | Max macro-F1 diff |
|---|---|---|---|---|
| thyroid | 10 of 10 | 596 | 0 | 0.00e+00 |
| landsat | 10 of 10 | 2883 | 0 | 0.00e+00 |
| page_blocks | 10 of 10 | 2348 | 0 | 0.00e+00 |
| magic | 10 of 10 | 3000 | 0 | 0.00e+00 |
| letter | 10 of 10 | 3000 | 0 | 0.00e+00 |
| skin | 10 of 10 | 3000 | 0 | 0.00e+00 |

## Speed on seed 50 (not part of the pass rule)

Both gates ran in the same process on the same machine; several tasks shared the CPU, so read the ratio, not the seconds.

| Dataset | Standard gate (s) | Cached gate (s) | Standard ÷ cached |
|---|---|---|---|
| landsat | 31.2 | 3.8 | 8.13 |
| letter | 223.7 | 10.2 | 21.92 |
| magic | 58.6 | 6.1 | 9.55 |
| page_blocks | 4.2 | 2.0 | 2.10 |
| skin | 28.1 | 6.6 | 4.23 |
| thyroid | 1.6 | 0.4 | 3.69 |

Total over these datasets: standard 347 s, cached 29 s, ratio 11.87.
