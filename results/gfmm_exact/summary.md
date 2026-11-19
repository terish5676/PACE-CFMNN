# Run P4: exact PACE-SHAP for GFMM-family networks

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Written by `scripts/run_gfmm_exact.py`.

**Check: PASS.** The adapter predicts like hyperbox-brain on every explained point; PACE-SHAP-min equals the brute force within 1e-12 on every point; where timed, shap's ExactExplainer agrees within 1e-09.

## Check (seeds 310–319)

| Dataset | Model | Points explained | Points compared | Boxes (mean) | Largest difference from brute force | (Row, box) pairs skipped |
|---|---|---|---|---|---|---|
| S1 | Online GFMM | 100 | 30 | 2363 | 2.2e-16 | 0.009 |
| S1 | IOL-GFMM | 100 | 30 | 2363 | 2.2e-16 | 0.009 |
| S2 | Online GFMM | 100 | 30 | 2426 | 2.1e-16 | 0.009 |
| S2 | IOL-GFMM | 100 | 30 | 2427 | 2.1e-16 | 0.009 |
| S3 | Online GFMM | 100 | 30 | 2414 | 2.7e-16 | 0.010 |
| S3 | IOL-GFMM | 100 | 30 | 2414 | 2.7e-16 | 0.010 |
| shuttle | Online GFMM | 100 | 30 | 33 | 5.0e-16 | 0.816 |
| shuttle | IOL-GFMM | 100 | 30 | 1129 | 6.7e-16 | 0.739 |
| magic | Online GFMM | 100 | 30 | 323 | 3.6e-16 | 0.703 |
| magic | IOL-GFMM | 100 | 30 | 2255 | 6.7e-16 | 0.673 |
| page_blocks | Online GFMM | 100 | 30 | 84 | 6.8e-16 | 0.712 |
| page_blocks | IOL-GFMM | 100 | 30 | 186 | 3.9e-16 | 0.691 |

## Timing (each alone, with its set-up; mean over seeds)

| Dataset | Model | ms PACE-SHAP-min | ms brute force | ms ExactExplainer | Brute force ÷ PACE | ExactExplainer ÷ PACE |
|---|---|---|---|---|---|---|
| S1 | Online GFMM | 8796.3 | 94713 | 95105 | 10.8× | 10.8× |
| S1 | IOL-GFMM | 8828.6 | 95020 | 94950 | 10.8× | 10.8× |
| S2 | Online GFMM | 9048.2 | 97038 | 96555 | 10.7× | 10.7× |
| S2 | IOL-GFMM | 9041.1 | 96385 | 96733 | 10.7× | 10.7× |
| S3 | Online GFMM | 8988.5 | 96696 | 97023 | 10.8× | 10.8× |
| S3 | IOL-GFMM | 9009.0 | 96900 | 96806 | 10.8× | 10.7× |
| shuttle | Online GFMM | 11.2 | 630 | 544 | 56.1× | 48.4× |
| shuttle | IOL-GFMM | 225.2 | 21199 | 18133 | 94.2× | 80.5× |
| magic | Online GFMM | 236.8 | 13149 | 13171 | 55.5× | 55.6× |
| magic | IOL-GFMM | 2676.0 | 89692 | 89287 | 33.5× | 33.4× |
| page_blocks | Online GFMM | 39.0 | 3515 | 3523 | 90.1× | 90.3× |
| page_blocks | IOL-GFMM | 154.8 | 7814 | 7785 | 50.5× | 50.3× |
