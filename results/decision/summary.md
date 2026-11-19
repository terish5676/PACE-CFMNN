# Run P5: explaining the decision at ties

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and rows, seeds 320–329, 50 points per seed. Written by `scripts/run_decision.py`.

**Check: PASS.** Where exact, PACE-decision equals the brute force (the model's prediction on every mixed input) within 1e-09 on the first 2 points of each seed.

Tied explained points per seed: letter 22–35; magic 36–45; page_blocks 22–46.

## Primary: decision deletion AUC, tied points (lower is better)

| Dataset | Seeds | PACE-decision | PACE-SHAP | Difference | Seeds better / worse | p | p (Holm) | Verdict for PACE-decision |
|---|---|---|---|---|---|---|---|---|
| magic | 10 | 0.4528 | 0.5034 | -0.0506 | 10 / 0 | 0.0020 | 0.0059 | better |
| letter | 10 | 0.0979 | 0.1140 | -0.0161 | 10 / 0 | 0.0020 | 0.0059 | better |
| page_blocks | 10 | 0.6382 | 0.7467 | -0.1085 | 10 / 0 | 0.0020 | 0.0059 | better |

## Secondary: decision deletion AUC, untied points (lower is better)

| Dataset | Seeds | PACE-decision | PACE-SHAP | Difference | Seeds better / worse | p | p (Holm) | Verdict for PACE-decision |
|---|---|---|---|---|---|---|---|---|
| magic | 10 | 0.4706 | 0.5239 | -0.0533 | 9 / 1 | 0.0059 | 0.0117 | better |
| letter | 10 | 0.1378 | 0.1493 | -0.0116 | 9 / 1 | 0.0059 | 0.0117 | better |
| page_blocks | 10 | 0.6966 | 0.7971 | -0.1005 | 10 / 0 | 0.0020 | 0.0059 | better |

## Secondary: decision insertion AUC, tied points (higher is better)

| Dataset | Seeds | PACE-decision | PACE-SHAP | Difference | Seeds better / worse | p | p (Holm) | Verdict for PACE-decision |
|---|---|---|---|---|---|---|---|---|
| magic | 10 | 0.7916 | 0.7385 | +0.0531 | 10 / 0 | 0.0020 | 0.0059 | better |
| letter | 10 | 0.4602 | 0.4161 | +0.0441 | 10 / 0 | 0.0020 | 0.0059 | better |
| page_blocks | 10 | 0.9030 | 0.8155 | +0.0875 | 10 / 0 | 0.0020 | 0.0059 | better |

## Secondary: margin deletion AUC, tied points (lower is better)

| Dataset | Seeds | PACE-decision | PACE-SHAP | Difference | Seeds better / worse | p | p (Holm) | Verdict for PACE-decision |
|---|---|---|---|---|---|---|---|---|
| magic | 10 | -0.0471 | -0.0657 | +0.0186 | 2 / 8 | 0.0488 | 0.0977 | no significant difference |
| letter | 10 | -0.7236 | -0.7504 | +0.0268 | 0 / 10 | 0.0020 | 0.0059 | worse |
| page_blocks | 10 | 0.0094 | -0.0122 | +0.0216 | 4 / 6 | 0.3223 | 0.3223 | no significant difference |

## Secondary: margin insertion AUC, tied points (higher is better)

| Dataset | Seeds | PACE-decision | PACE-SHAP | Difference | Seeds better / worse | p | p (Holm) | Verdict for PACE-decision |
|---|---|---|---|---|---|---|---|---|
| magic | 10 | 0.1952 | 0.2088 | -0.0136 | 3 / 7 | 0.1309 | 0.1309 | no significant difference |
| letter | 10 | -0.2908 | -0.2661 | -0.0246 | 1 / 9 | 0.0137 | 0.0410 | worse |
| page_blocks | 10 | 0.1520 | 0.1815 | -0.0296 | 2 / 8 | 0.0645 | 0.1289 | no significant difference |
