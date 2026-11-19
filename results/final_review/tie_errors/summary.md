# Are all CFMNN training errors cross-class ties?

Written by `research_audit/scripts/check_tie_errors.py` (final review, post hoc check of a claim in the paper's Section 6). Split: `scripts/run_refinement.split` (the repair runs' role split); model: `CFMNN()` (θ = 0.3, γ = 4, fixed centroid, search expansion).

A cross-class tie: the row has membership 1 in a box of its own class and in a box of the wrong class, and the nearest-centroid rule picks the wrong one.

| Dataset | Seed | Training rows | Errors | Inside own-class box | Inside wrong-class box | Cross-class tie | Errors that are not |
|---|---|---|---|---|---|---|---|
| thyroid | 0 | 3,600 | 66 | 100.0% | 100.0% | 100.0% | 0 |
| thyroid | 10 | 3,600 | 85 | 100.0% | 100.0% | 100.0% | 0 |
| thyroid | 50 | 3,600 | 73 | 100.0% | 100.0% | 100.0% | 0 |
| landsat | 0 | 3,217 | 254 | 100.0% | 100.0% | 100.0% | 0 |
| landsat | 10 | 3,217 | 287 | 100.0% | 100.0% | 100.0% | 0 |
| landsat | 50 | 3,217 | 322 | 100.0% | 100.0% | 100.0% | 0 |
| page_blocks | 0 | 2,736 | 176 | 100.0% | 100.0% | 100.0% | 0 |
| page_blocks | 10 | 2,736 | 514 | 100.0% | 100.0% | 100.0% | 0 |
| page_blocks | 50 | 2,736 | 215 | 100.0% | 100.0% | 100.0% | 0 |
| magic | 0 | 9,510 | 2,348 | 100.0% | 100.0% | 100.0% | 0 |
| magic | 10 | 9,510 | 2,400 | 100.0% | 100.0% | 100.0% | 0 |
| magic | 50 | 9,510 | 1,970 | 100.0% | 100.0% | 100.0% | 0 |
| letter | 0 | 10,000 | 2,036 | 100.0% | 100.0% | 100.0% | 0 |
| letter | 10 | 10,000 | 2,021 | 100.0% | 100.0% | 100.0% | 0 |
| letter | 50 | 10,000 | 2,071 | 100.0% | 100.0% | 100.0% | 0 |
| skin | 0 | 122,528 | 10,980 | 100.0% | 100.0% | 100.0% | 0 |
| skin | 10 | 122,528 | 5,398 | 100.0% | 100.0% | 100.0% | 0 |
| skin | 50 | 122,528 | 4,669 | 100.0% | 100.0% | 100.0% | 0 |
