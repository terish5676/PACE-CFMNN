# Run P (descriptive): where PACE-SHAP's speed comes from

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and 200 explainer rows, seeds 240, 241, 242, 2 points per seed, one task at a time. Milliseconds per explanation (mean over seeds), each including its explainer's set-up. Written by `scripts/run_speed_source.py`.

**Check: PASS.** PACE-SHAP without pruning agrees with PACE-SHAP within 1e-12, and shap's ExactExplainer within 1e-09, on every point.

| Dataset | d | Boxes | ms ExactExplainer | ms no pruning | ms PACE-SHAP | Structure (ExactExplainer ÷ no pruning) | Pruning (no pruning ÷ PACE-SHAP) | Total (ExactExplainer ÷ PACE-SHAP) | (Row, box) pairs skipped | Largest difference: no pruning | Largest difference: ExactExplainer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 1523 | 27819.8 | 3823.1 | 1948.7 | 7.3× | 2.0× | 14.3× | 0.624 | 3.1e-17 | 1.1e-16 |
| S2 | 10 | 1539 | 28329.5 | 3845.7 | 1957.4 | 7.4× | 2.0× | 14.5× | 0.623 | 1.4e-17 | 1.2e-16 |
| S3 | 10 | 1523 | 27695.2 | 3813.7 | 1889.5 | 7.3× | 2.0× | 14.7× | 0.635 | 2.8e-17 | 1.2e-16 |
| shuttle | 9 | 13 | 162.9 | 15.2 | 8.7 | 10.7× | 1.8× | 18.8× | 0.719 | 0.0e+00 | 2.5e-16 |
| magic | 10 | 285 | 5800.2 | 730.0 | 256.0 | 7.9× | 2.9× | 22.7× | 0.759 | 3.5e-18 | 4.3e-17 |
| page_blocks | 10 | 27 | 669.0 | 77.7 | 29.4 | 8.6× | 2.6× | 22.7× | 0.681 | 2.2e-19 | 6.0e-17 |

Boxes: mean over seeds. (Row, box) pairs skipped: PACE-SHAP's own count (`stats_['share_of_boxes_skipped']`), the share of explainer-row and box pairs that pruning removed. Largest difference: the largest absolute difference from PACE-SHAP over every point, feature and seed.
