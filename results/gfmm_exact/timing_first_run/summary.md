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
| shuttle | Online GFMM | 11.2 | 625 | 551 | 55.6× | 49.0× |
| shuttle | IOL-GFMM | 46.9 | 2303 | 2337 | 49.2× | 49.9× |
| page_blocks | Online GFMM | 38.8 | 3480 | 3659 | 89.7× | 94.3× |
| page_blocks | IOL-GFMM | 160.2 | 7705 | 8319 | 48.1× | 51.9× |
