"""Check the data embedded in the paper's figure widgets (doc rev 262) against the raw CSVs.

Figure 2 rows were checked in claims_rq1.py (all 22 rows match to 4 decimals).
This script checks Figure 3 (time per explanation) and Figure 4 (interaction lead).
The values below are copied from the widgets' data blocks.

    python -B research_audit/scripts/claims_figures.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RES

sys.stdout.reconfigure(encoding="utf-8")

NAME = {"S1": "S1", "S2": "S2", "S3": "S3", "Wine": "wine", "Breast Cancer": "breast_cancer", "Thyroid": "thyroid",
        "Shuttle": "shuttle", "MAGIC": "magic", "Letter": "letter", "Landsat": "landsat", "Page Blocks": "page_blocks"}
FIG3 = {  # dataset: (PACE-multi, PACE-flip, Occlusion, LIME, KernelSHAP) ms per explanation
    "S1": (0.4680, 0.8940, 78.5528, 265.3339, 1067.9005), "S2": (0.4690, 0.8884, 84.4421, 281.0254, 1048.9263),
    "S3": (0.4740, 0.8987, 83.9622, 256.8743, 1030.9182), "Wine": (0.0325, 0.1138, 5.5963, 11.6135, 33.2208),
    "Breast Cancer": (0.0474, 0.2514, 22.7617, 24.0265, 84.2678),
    "Thyroid": (0.0431, 0.2252, 14.7419, 22.9788, 77.5006), "Shuttle": (0.0203, 0.0469, 3.7318, 8.1427, 17.5657),
    "MAGIC": (0.1065, 0.6370, 18.4049, 56.5473, 234.0713), "Letter": (0.3100, 1.3669, 103.3814, 196.4892, 793.8463),
    "Landsat": (0.1702, 0.8211, 116.0077, 107.6438, 384.4276),
    "Page Blocks": (0.0228, 0.1067, 4.0055, 12.5347, 33.7622)}
FIG4 = {
    "S3": [2.3496, 2.0908, 1.7831, 2.1207, 2.1139, 1.9816, 1.8201, 1.9520, 1.5809, 2.8046],
    "S1": [1.5669, 1.9119, 1.4933, 1.2732, 2.0140, 1.1558, 1.8512, 1.6570, 1.6837, 1.6337],
    "S2": [1.3229, 1.4019, 1.3543, 1.2948, 1.3563, 1.3185, 1.5283, 1.3357, 1.2908, 1.2937]}

tm = pd.read_csv(RES / "faithfulness/timing.csv")
T = tm.groupby(["dataset", "method"]).ms_per_explanation.mean().unstack()
cols = ["PACE-multi", "PACE-flip", "Occlusion (brute force)", "LIME", "KernelSHAP"]
worst = 0.0
for name, vals in FIG3.items():
    raw = T.loc[NAME[name], cols].to_numpy(float)
    diff = np.abs(np.array(vals) - raw)  # the widget rounds to 4 decimals
    worst = max(worst, diff.max())
    if diff.max() > 5e-5:
        print(f"Figure 3 MISMATCH {name}: figure {vals} raw {raw.round(4)}")
print(f"Figure 3: 55 values, largest absolute difference from faithfulness/timing.csv = {worst:.1e} ms")
ratio = [FIG3[n][2] / FIG3[n][1] for n in FIG3]
print(f"Figure 3 occlusion/PACE-flip ratios: {min(ratio):.0f}-{max(ratio):.0f}")

it = pd.read_csv(RES / "extensions_v1/interactions/runs.csv")
worst = 0.0
for ds, vals in FIG4.items():
    for seed, v in zip(range(40, 50), vals):
        g = it[(it.dataset == ds) & (it.seed == seed)].sort_values("strength", ascending=False)
        s01 = g[(g.i == 0) & (g.j == 1)].strength.iloc[0]
        lead = s01 / g[~((g.i == 0) & (g.j == 1))].strength.iloc[0]
        worst = max(worst, abs(lead - v))
    print(f"Figure 4 {ds}: mean {np.mean(vals):.2f}")
print(f"Figure 4: 30 values, largest absolute difference from interactions/runs.csv = {worst:.1e}")
