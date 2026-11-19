# Smoke runs

Every experiment script run once by `research_audit/scripts/smoke_all.py`: one dataset, one seed and a
few points per run, with every output written to a scratch folder outside the repository. These are
small checks that the scripts run end to end; their numbers mean nothing.

**Result: 36 of 36 runs exited 0.**

| Run | Exit | Minutes | Output files |
|---|---|---|---|
| accuracy | 0 | 0.13 | runs.csv summary.md |
| cf | 0 | 0.47 | runs.csv summary.md |
| cf_ext | 0 | 0.49 | runs.csv summary.md |
| cf_table | 0 | 0.14 | review_v1\counterfactual_table.md |
| ed2_conf | 0 | 0.37 | runs.csv summary.md |
| ed2_ens | 0 | 0.42 | runs.csv summary.md |
| ed2_explore | 0 | 1.10 | runs.csv summary.md |
| ed_conf | 0 | 1.01 | runs.csv summary.md |
| ed_explore | 0 | 0.90 | runs.csv summary.md |
| exact | 0 | 0.15 | runs.csv summary.md |
| faith_cond | 0 | 0.22 | extra.csv realism.csv runs.csv summary.md |
| faith_v1 | 0 | 0.15 | flip_mass.csv runs.csv summary.md |
| faith_v2 | 0 | 0.16 | flip_mass.csv runs.csv summary.md |
| faith_v2_timing | 0 | 0.15 | timing.csv timing.md |
| faith_v3 | 0 | 0.29 | flip_mass.csv runs.csv summary.md |
| faith_v3_timing | 0 | 0.23 | timing.csv timing.md |
| g0 | 0 | 0.18 | runs.csv summary.md |
| gi | 0 | 0.17 | runs.csv summary.md |
| gi_ext | 0 | 0.23 | runs.csv summary.md |
| interactions | 0 | 0.42 | runs.csv summary.md |
| knn_reference | 0 | 4.00 | error_detection_confirmatory\knn_reference.csv error_detection_confirmatory\knn_reference.md |
| occl_batched | 0 | 0.11 | runs.csv summary.md |
| pace_shap_settings | 0 | 0.35 | faithfulness_v3\settings.csv faithfulness_v3\settings.md |
| posthoc_review | 0 | 1.10 | extensions_v1\global_importance\posthoc_points_page_blocks.md extensions_v1\posthoc_review.md |
| posthoc_rows | 0 | 2.38 | posthoc_rows.csv posthoc_rows.md |
| posthoc_shuttle | 0 | 2.00 | extensions_v1\counterfactuals\posthoc_shuttle.md |
| posthoc_sigma | 0 | 0.73 | posthoc_sigma.csv posthoc_sigma.md |
| reanalysis | 0 | 0.15 | review_v1\reanalysis.md |
| ref_conf | 0 | 1.33 | edits.csv rounds.csv runs.csv summary.md |
| ref_elig | 0 | 1.11 | eligibility.csv eligibility.md |
| ref_explore | 0 | 3.98 | edits.csv.gz parts\page_blocks__0.pkl runs.csv summary.md |
| ref_pilot | 0 | 1.68 | edits.csv rounds.csv runs.csv summary.md |
| ref_v2 | 0 | 0.84 | edits.csv.gz parts\page_blocks__50.pkl runs.csv summary.md |
| review_rq1 | 0 | 0.27 | runs.csv summary.md |
| speed | 0 | 0.11 | runs.csv summary.md |
| summarise_v1 | 0 | 0.24 | refinement_confirmatory\summary_v1.md |

## Scripts run in full: outputs against the committed files

These scripts have no `--out` option, so `smoke_redirect.py` ran them in full and sent their writes to
the scratch folder. Their outputs can be compared with the committed ones.

| Run | Output | Same as committed |
|---|---|---|
| summarise_v1 | `results/refinement_confirmatory/summary_v1.md` | yes |
| posthoc_shuttle | `results/extensions_v1/counterfactuals/posthoc_shuttle.md` | yes |
| posthoc_review | `results/extensions_v1/global_importance/posthoc_points_page_blocks.md` | yes |
| posthoc_review | `results/extensions_v1/posthoc_review.md` | yes |
| reanalysis | `results/review_v1/reanalysis.md` | yes |
| knn_reference | `results/error_detection_confirmatory/knn_reference.csv` | yes |
| knn_reference | `results/error_detection_confirmatory/knn_reference.md` | yes |
| cf_table | `results/review_v1/counterfactual_table.md` | yes |

Notes:
- `pace_shap_settings` ran on Wine only (`--only-datasets wine`), so it is a smoke run, not a reproduction.
- `reanalysis` resumes from the committed `results/review_v1/ties_v3.csv`, so its tie counts were read, not recomputed; the rest of the analysis ran in full.
