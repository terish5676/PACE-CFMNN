# Phase 5 pilot: selective refinement (exploratory)

Seeds [0, 1, 2]. Not a test of any hypothesis: it sets up the confirmatory run. Mean ± 95% CI over seeds, test rows. Budget = min(300, a quarter of the pool) labels in 10 rounds; gate on the acceptance rows (macro-F1 may fall at most 0.5 points, a class's recall at most 1 point; boxes at most 1.1 × the start). Times are wall-clock per arm and grow if the machine sleeps or runs share the CPU.

## Test macro-F1 (%)

| Dataset | B0 no refinement | B9 refit on all pool labels | order_ensemble, A+B | order_ensemble, A+B+C | order_ensemble, A+B+C'-random | order_ensemble, A+B+C'-volume | random, A+B | margin, A+B | pd_u_star, A+B |
|---|---|---|---|---|---|---|---|---|---|
| S2 | 80.18 ± 4.56 | 80.20 ± 1.98 | 80.25 ± 3.74 | 80.25 ± 3.74 | 80.25 ± 3.74 | 80.25 ± 3.74 | 80.05 ± 4.02 | 80.19 ± 5.04 | 80.32 ± 3.02 |
| thyroid | 84.73 ± 7.88 | 85.12 ± 8.66 | 86.84 ± 7.44 | 87.90 ± 0.96 | 89.75 ± 10.19 | 90.84 ± 5.18 | 86.08 ± 4.42 | 85.19 ± 6.10 | 85.19 ± 6.10 |
| landsat | 82.67 ± 5.25 | 82.58 ± 5.72 | 82.30 ± 7.03 | 83.14 ± 1.29 | 83.48 ± 2.99 | 83.24 ± 6.13 | 82.75 ± 4.47 | 82.33 ± 5.73 | 82.30 ± 4.70 |
| page_blocks | 76.90 ± 5.26 | 70.66 ± 21.90 | 75.13 ± 9.89 | 79.32 ± 2.70 | 80.35 ± 2.65 | 78.14 ± 6.96 | 76.87 ± 4.48 | 77.09 ± 8.35 | 76.76 ± 9.88 |
| magic | 73.33 ± 4.65 | 73.92 ± 4.89 | 73.54 ± 4.08 | 76.83 ± 1.96 | 77.11 ± 2.20 | 78.32 ± 4.57 | 73.19 ± 5.31 | 73.42 ± 4.46 | 73.44 ± 5.04 |

## Change in test macro-F1 vs B0 (points), and box growth

| Dataset | Arm | Δ macro-F1 | Δ min class recall | Boxes start → end |
|---|---|---|---|---|
| S2 | B9 refit on all pool labels | +0.01 ± 4.36 | +0.71 | 988 → 1357 |
| S2 | order_ensemble, A+B | +0.07 ± 1.57 | +0.41 | 988 → 1085 |
| S2 | order_ensemble, A+B+C | +0.07 ± 1.57 | +0.41 | 988 → 1085 |
| S2 | order_ensemble, A+B+C'-random | +0.07 ± 1.57 | +0.41 | 988 → 1085 |
| S2 | order_ensemble, A+B+C'-volume | +0.07 ± 1.57 | +0.41 | 988 → 1085 |
| S2 | random, A+B | -0.13 ± 0.57 | +0.46 | 988 → 1083 |
| S2 | margin, A+B | +0.00 ± 1.80 | +0.41 | 988 → 1086 |
| S2 | pd_u_star, A+B | +0.14 ± 2.08 | +0.55 | 988 → 1087 |
| thyroid | B9 refit on all pool labels | +0.38 ± 0.98 | -2.08 | 47 → 56 |
| thyroid | order_ensemble, A+B | +2.11 ± 1.64 | +2.78 | 47 → 51 |
| thyroid | order_ensemble, A+B+C | +3.17 ± 7.18 | +13.45 | 47 → 50 |
| thyroid | order_ensemble, A+B+C'-random | +5.01 ± 7.79 | +15.74 | 47 → 47 |
| thyroid | order_ensemble, A+B+C'-volume | +6.11 ± 5.72 | +15.74 | 47 → 50 |
| thyroid | random, A+B | +1.34 ± 3.45 | +1.85 | 47 → 49 |
| thyroid | margin, A+B | +0.45 ± 2.42 | -1.16 | 47 → 48 |
| thyroid | pd_u_star, A+B | +0.45 ± 2.42 | -1.16 | 47 → 48 |
| landsat | B9 refit on all pool labels | -0.09 ± 0.60 | -2.65 | 147 → 189 |
| landsat | order_ensemble, A+B | -0.37 ± 1.78 | -3.70 | 147 → 156 |
| landsat | order_ensemble, A+B+C | +0.47 ± 4.90 | -0.53 | 147 → 154 |
| landsat | order_ensemble, A+B+C'-random | +0.81 ± 2.99 | -1.06 | 147 → 154 |
| landsat | order_ensemble, A+B+C'-volume | +0.57 ± 2.24 | -3.17 | 147 → 155 |
| landsat | random, A+B | +0.08 ± 1.31 | +1.59 | 147 → 157 |
| landsat | margin, A+B | -0.34 ± 0.81 | -2.65 | 147 → 148 |
| landsat | pd_u_star, A+B | -0.37 ± 1.33 | -3.70 | 147 → 149 |
| page_blocks | B9 refit on all pool labels | -6.24 ± 23.04 | +1.18 | 25 → 27 |
| page_blocks | order_ensemble, A+B | -1.77 ± 6.24 | +0.00 | 25 → 26 |
| page_blocks | order_ensemble, A+B+C | +2.42 ± 2.88 | +0.93 | 25 → 26 |
| page_blocks | order_ensemble, A+B+C'-random | +3.45 ± 3.60 | -4.63 | 25 → 26 |
| page_blocks | order_ensemble, A+B+C'-volume | +1.24 ± 4.85 | -7.41 | 25 → 26 |
| page_blocks | random, A+B | -0.03 ± 0.89 | +0.00 | 25 → 25 |
| page_blocks | margin, A+B | +0.19 ± 3.27 | +0.00 | 25 → 25 |
| page_blocks | pd_u_star, A+B | -0.14 ± 4.62 | +0.00 | 25 → 25 |
| magic | B9 refit on all pool labels | +0.59 ± 0.25 | +0.99 | 231 → 284 |
| magic | order_ensemble, A+B | +0.22 ± 1.17 | +0.54 | 231 → 235 |
| magic | order_ensemble, A+B+C | +3.50 ± 4.20 | -1.17 | 231 → 230 |
| magic | order_ensemble, A+B+C'-random | +3.79 ± 5.58 | -1.61 | 231 → 231 |
| magic | order_ensemble, A+B+C'-volume | +4.99 ± 5.03 | +1.68 | 231 → 233 |
| magic | random, A+B | -0.14 ± 0.72 | -0.69 | 231 → 236 |
| magic | margin, A+B | +0.10 ± 0.36 | +0.20 | 231 → 231 |
| magic | pd_u_star, A+B | +0.11 ± 0.40 | +0.05 | 231 → 231 |

## What the loop did (mean per seed)

| Dataset | Arm | Budget | n_wrong_selected | n_shrink_possible | n_shrink_tried | n_shrink_accepted | n_add_accepted | n_points_left_out | n_rejected_box_cap | n_rejected_f1_or_recall | mean_share_cut | mean_rows_cut | time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | order_ensemble, A+B | 250 | 162.3 | 0.0 | 0.0 | 0.0 | 246.0 | 4.0 | 3.7 | 0.3 | nan | nan | 56.4 |
| S2 | order_ensemble, A+B+C | 250 | 162.3 | 0.0 | 0.0 | 0.0 | 246.0 | 4.0 | 3.7 | 0.3 | nan | nan | 51.1 |
| S2 | order_ensemble, A+B+C'-random | 250 | 162.3 | 0.0 | 0.0 | 0.0 | 246.0 | 4.0 | 3.7 | 0.3 | nan | nan | 55.5 |
| S2 | order_ensemble, A+B+C'-volume | 250 | 162.3 | 0.0 | 0.0 | 0.0 | 246.0 | 4.0 | 3.7 | 0.3 | nan | nan | 53.8 |
| S2 | random, A+B | 250 | 51.3 | 0.0 | 0.0 | 0.0 | 249.7 | 0.3 | 0.3 | 0.0 | nan | nan | 29.8 |
| S2 | margin, A+B | 250 | 104.7 | 0.0 | 0.0 | 0.0 | 247.7 | 2.3 | 2.3 | 0.0 | nan | nan | 30.3 |
| S2 | pd_u_star, A+B | 250 | 107.7 | 0.0 | 0.0 | 0.0 | 249.3 | 0.7 | 0.7 | 0.0 | nan | nan | 207.9 |
| thyroid | order_ensemble, A+B | 300 | 40.7 | 29.3 | 0.0 | 0.0 | 296.3 | 3.7 | 2.0 | 1.7 | nan | nan | 16.0 |
| thyroid | order_ensemble, A+B+C | 300 | 39.0 | 23.7 | 23.7 | 14.7 | 282.0 | 3.3 | 0.3 | 12.0 | 0.16 | 62.37 | 16.0 |
| thyroid | order_ensemble, A+B+C'-random | 300 | 38.3 | 15.3 | 15.3 | 12.3 | 283.3 | 4.3 | 0.0 | 7.3 | 0.57 | 108.77 | 14.5 |
| thyroid | order_ensemble, A+B+C'-volume | 300 | 31.0 | 16.0 | 16.0 | 14.3 | 256.7 | 29.0 | 0.7 | 30.0 | 0.00 | 116.96 | 16.0 |
| thyroid | random, A+B | 300 | 9.7 | 7.7 | 0.0 | 0.0 | 298.0 | 2.0 | 0.0 | 2.0 | nan | nan | 6.2 |
| thyroid | margin, A+B | 300 | 26.7 | 24.3 | 0.0 | 0.0 | 291.7 | 8.3 | 0.0 | 8.3 | nan | nan | 6.4 |
| thyroid | pd_u_star, A+B | 300 | 26.0 | 23.7 | 0.0 | 0.0 | 288.0 | 12.0 | 0.0 | 12.0 | nan | nan | 29.8 |
| landsat | order_ensemble, A+B | 300 | 150.3 | 113.3 | 0.0 | 0.0 | 287.7 | 12.3 | 0.0 | 12.3 | nan | nan | 35.3 |
| landsat | order_ensemble, A+B+C | 300 | 157.3 | 114.7 | 114.7 | 60.7 | 223.3 | 16.0 | 0.0 | 70.0 | 0.31 | 9.95 | 38.1 |
| landsat | order_ensemble, A+B+C'-random | 300 | 147.7 | 105.0 | 105.0 | 51.3 | 233.3 | 15.3 | 0.0 | 69.0 | 0.32 | 10.00 | 39.7 |
| landsat | order_ensemble, A+B+C'-volume | 300 | 142.7 | 102.7 | 102.7 | 73.3 | 209.3 | 17.3 | 0.0 | 46.7 | 0.05 | 3.95 | 39.6 |
| landsat | random, A+B | 300 | 46.0 | 35.0 | 0.0 | 0.0 | 296.3 | 3.7 | 0.0 | 3.7 | nan | nan | 24.3 |
| landsat | margin, A+B | 300 | 113.0 | 103.7 | 0.0 | 0.0 | 293.7 | 6.3 | 0.0 | 6.3 | nan | nan | 26.1 |
| landsat | pd_u_star, A+B | 300 | 120.7 | 105.0 | 0.0 | 0.0 | 288.3 | 11.7 | 0.0 | 11.7 | nan | nan | 127.4 |
| page_blocks | order_ensemble, A+B | 273 | 73.3 | 64.7 | 0.0 | 0.0 | 262.3 | 10.7 | 0.3 | 10.3 | nan | nan | 15.8 |
| page_blocks | order_ensemble, A+B+C | 273 | 63.7 | 53.3 | 53.3 | 12.0 | 254.7 | 6.3 | 0.0 | 47.7 | 0.23 | 90.23 | 12.6 |
| page_blocks | order_ensemble, A+B+C'-random | 273 | 61.3 | 51.0 | 51.0 | 13.0 | 246.3 | 13.7 | 0.0 | 51.7 | 0.17 | 98.42 | 11.9 |
| page_blocks | order_ensemble, A+B+C'-volume | 273 | 60.3 | 51.3 | 51.3 | 11.7 | 243.3 | 18.0 | 0.0 | 57.7 | 0.01 | 30.43 | 11.7 |
| page_blocks | random, A+B | 273 | 21.7 | 20.0 | 0.0 | 0.0 | 272.3 | 0.7 | 0.0 | 0.7 | nan | nan | 4.1 |
| page_blocks | margin, A+B | 273 | 42.3 | 42.3 | 0.0 | 0.0 | 270.3 | 2.7 | 0.0 | 2.7 | nan | nan | 4.3 |
| page_blocks | pd_u_star, A+B | 273 | 62.3 | 62.3 | 0.0 | 0.0 | 270.3 | 2.7 | 0.0 | 2.7 | nan | nan | 13.6 |
| magic | order_ensemble, A+B | 300 | 225.7 | 208.7 | 0.0 | 0.0 | 300.0 | 0.0 | 0.0 | 0.0 | nan | nan | 452.9 |
| magic | order_ensemble, A+B+C | 300 | 190.7 | 164.3 | 164.3 | 157.0 | 142.0 | 1.0 | 0.0 | 8.3 | 0.28 | 24.85 | 127.2 |
| magic | order_ensemble, A+B+C'-random | 300 | 190.7 | 166.0 | 166.0 | 153.3 | 146.7 | 0.0 | 0.0 | 12.7 | 0.26 | 30.93 | 132.7 |
| magic | order_ensemble, A+B+C'-volume | 300 | 195.7 | 174.0 | 174.0 | 171.0 | 129.0 | 0.0 | 0.0 | 3.0 | 0.07 | 10.16 | 147.8 |
| magic | random, A+B | 300 | 73.7 | 70.7 | 0.0 | 0.0 | 300.0 | 0.0 | 0.0 | 0.0 | nan | nan | 78.0 |
| magic | margin, A+B | 300 | 87.0 | 87.0 | 0.0 | 0.0 | 300.0 | 0.0 | 0.0 | 0.0 | nan | nan | 79.5 |
| magic | pd_u_star, A+B | 300 | 150.3 | 150.3 | 0.0 | 0.0 | 300.0 | 0.0 | 0.0 | 0.0 | nan | nan | 415.4 |
