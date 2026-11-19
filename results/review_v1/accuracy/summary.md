# Review Test 4 (descriptive): accuracy of the explained model and two references

Rules: [`../PREREGISTRATION.md`](../PREREGISTRATION.md) (Test 4). The paper's 80/20 split, seeds [0, 1, 2, 3, 4]; accuracy (%) on the test rows, mean ± standard deviation over seeds.

| Dataset | Training rows | CFMNN | 10-NN | Random Forest | CFMNN minus the better reference |
|---|---|---|---|---|---|
| S1 | 4000 | 89.6 ± 1.1 | 95.1 ± 0.8 | 99.6 ± 0.1 | -10.0 |
| S2 | 4000 | 80.6 ± 1.6 | 85.8 ± 1.8 | 90.1 ± 1.3 | -9.5 |
| S3 | 4000 | 87.0 ± 0.8 | 88.7 ± 0.7 | 100.0 ± 0.0 | -13.0 |
| wine | 142 | 98.3 ± 2.5 | 95.6 ± 3.2 | 100.0 ± 0.0 | -1.7 |
| breast_cancer | 455 | 95.4 ± 1.1 | 97.4 ± 1.5 | 97.0 ± 1.0 | -1.9 |
| digits | 1437 | 95.3 ± 0.5 | 98.0 ± 0.7 | 97.6 ± 0.2 | -2.7 |
| thyroid | 5760 | 97.2 ± 0.6 | 94.2 ± 0.2 | 99.6 ± 0.1 | -2.5 |
| shuttle | 46400 | 99.8 ± 0.0 | 99.8 ± 0.0 | 100.0 ± 0.0 | -0.2 |
| skin | 196045 | 94.7 ± 2.1 | 99.9 ± 0.0 | 100.0 ± 0.0 | -5.2 |
| magic | 15216 | 72.5 ± 0.7 | 83.6 ± 0.4 | 88.3 ± 0.8 | -15.8 |
| letter | 16000 | 74.6 ± 0.7 | 94.2 ± 0.4 | 96.2 ± 0.2 | -21.5 |
| landsat | 5148 | 84.3 ± 1.1 | 89.8 ± 0.4 | 91.7 ± 0.9 | -7.4 |
| page_blocks | 4378 | 86.3 ± 5.6 | 94.8 ± 0.4 | 97.3 ± 0.6 | -11.0 |
