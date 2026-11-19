# Which datasets can the shrink repair? (pilot seeds 0-2)

Repairable = pool points the starting model gets wrong while they lie inside the winning wrong-class box. Eligible: at least 25 in every pilot seed. No refinement result is used.

| Dataset | Pool rows | Pool errors (per seed) | Repairable (per seed) | Eligible |
|---|---|---|---|---|
| S1 | 1000 | 138, 121, 108 | 0, 0, 0 | no |
| S2 | 1000 | 202, 219, 195 | 0, 0, 0 | no |
| S3 | 1000 | 137, 137, 153 | 0, 0, 0 | no |
| S2-imbalanced | 1000 | 91, 95, 89 | 0, 0, 0 | no |
| S2-label-noise | 1000 | 272, 272, 252 | 0, 0, 0 | no |
| thyroid | 1440 | 46, 37, 46 | 34, 31, 29 | yes |
| shuttle | 11600 | 22, 25, 20 | 20, 19, 18 | no |
| landsat | 1287 | 180, 207, 196 | 130, 153, 138 | yes |
| page_blocks | 1094 | 77, 93, 98 | 70, 86, 81 | yes |
| magic | 3804 | 1061, 881, 933 | 997, 838, 901 | yes |
| letter | 4000 | 1027, 1072, 1090 | 916, 950, 965 | yes |
| skin | 49011 | 4410, 2029, 2016 | 4409, 2028, 2014 | yes |
