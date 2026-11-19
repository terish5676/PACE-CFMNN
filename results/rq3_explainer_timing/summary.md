# Self-repair timing, seed 50 (descriptive check, made after the RQ3 run)

Written by `docs/rq3_timing.py`. Not part of any pre-registered study; nothing is tested.
SR is called exactly as in `scripts/run_refinement_v2.py`, without the certificate audit,
one dataset at a time. SR's test macro-F1 equals the committed value on every dataset.

Machine: Intel64 Family 6 Model 142 Stepping 10, GenuineIntel, Python 3.14.7, Windows 11. Total 787 s.

| Dataset | Training rows | Features | Boxes | Shrinks tried | Accepted | Fit (s) | One pass over training rows (s) | SR (s) | Gate share of SR (%) | SR ÷ fit | Committed SR time, with audit, 4 jobs (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 3,600 | 21 | 42 | 23 | 14 | 0.15 | 0.051 | 1.5 | 92 | 10 | 4.2 |
| page_blocks | 2,736 | 10 | 22 | 179 | 14 | 0.08 | 0.016 | 4.1 | 96 | 50 | 8.2 |
| landsat | 3,217 | 36 | 143 | 198 | 176 | 0.17 | 0.237 | 60.5 | 99 | 363 | 154.7 |
| skin | 122,528 | 3 | 85 | 19 | 16 | 5.73 | 1.183 | 34.6 | 95 | 6 | 52.0 |
| magic | 9,510 | 10 | 241 | 208 | 207 | 0.46 | 0.483 | 138.4 | 99 | 302 | 226.7 |
| letter | 10,000 | 16 | 451 | 258 | 202 | 0.43 | 1.896 | 535.2 | 99 | 1234 | 910.3 |
