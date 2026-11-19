# Run P6: a certificate for PACE-SHAP explanations

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Self-repair at θ = 0.3 on the refinement run's split, seeds 330–339, 100 explained test rows, 100 explainer rows. Written by `scripts/run_shap_certificate.py`.

**Check: PASS.** Over 6,085 accepted edits and 608,500 (edit, row) pairs, no row outside the certificate changed by more than 1e-12.

| Dataset | d | Mode | Accepted edits | Certified share (mean per edit) | Changed share | Certified rows that changed | Median relative change (changed rows) | Largest relative change |
|---|---|---|---|---|---|---|---|---|
| thyroid | 21 | sampled | 117 | 99.7% | 82.7% | 83.0% | 2.1e-04 | 6.6e+00 |
| landsat | 36 | sampled | 1,728 | 100.0% | 80.3% | 80.3% | 2.6e-04 | 1.5e+00 |
| page_blocks | 10 | exact | 101 | 99.1% | 81.0% | 81.8% | 2.0e-03 | 2.3e+01 |
| magic | 10 | exact | 1,850 | 100.0% | 80.8% | 80.8% | 1.0e-03 | 2.4e+00 |
| letter | 16 | sampled | 1,983 | 99.9% | 31.7% | 31.8% | 2.0e-05 | 2.2e+00 |
| skin | 3 | exact | 306 | 100.0% | 46.2% | 46.2% | 5.9e-04 | 2.2e+00 |

Certified share: the share of the explained rows the certificate marks, per edit. Changed share: the rows whose values really changed. Certified rows that changed: of all certified (edit, row) pairs, the share whose values changed (how tight the certificate is). Relative change: a row's largest value change divided by its largest value before the edit; the median is over changed rows per edit, then over edits.
