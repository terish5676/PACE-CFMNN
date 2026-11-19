# Post-hoc: CFMNN vs a plain 10-NN classifier on the confirmatory splits

Not pre-registered; descriptive only. Seeds 10-19, the same splits as `runs.csv` (the CFMNN accuracy matches it for every seed). Mean over seeds.

| Dataset | CFMNN accuracy (%) | 10-NN accuracy (%) | CFMNN errors where 10-NN predicts another class |
|---|---|---|---|
| S1 | 88.3 | 95.1 | 0.82 |
| S2 | 79.4 | 86.0 | 0.63 |
| S3 | 86.2 | 87.6 | 0.68 |
| S2-imbalanced | 90.0 | 91.7 | 0.51 |
| S2-label-noise | 74.3 | 82.0 | 0.58 |
| S2-shifted | 79.4 | 85.5 | 0.62 |
| digits | 95.6 | 97.7 | 0.70 |
| thyroid | 96.7 | 93.8 | 0.21 |
| skin | 95.4 | 99.9 | 0.98 |
| magic | 72.4 | 83.0 | 0.68 |
| letter | 74.1 | 92.2 | 0.92 |
| landsat | 84.6 | 89.0 | 0.63 |
| page_blocks | 85.9 | 94.7 | 0.83 |
