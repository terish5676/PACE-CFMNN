OK    python -m unittest discover -s tests -t .  (unittest; run separately)
OK    python scripts/accuracy_table.py
OK    python scripts/counterfactual_table_seeds.py
OK    python scripts/exact_baseline.py
OK    python scripts/knn_reference.py
OK    python scripts/pace_shap_settings.py
OK    python scripts/posthoc_review.py --points page_blocks
OK    python scripts/posthoc_rows.py --jobs 6
OK    python scripts/posthoc_shuttle_sparse.py
OK    python scripts/posthoc_sigma.py --jobs 5
OK    python scripts/review_reanalysis.py --jobs 6
OK    python scripts/run_counterfactuals.py
OK    python scripts/run_counterfactuals.py --extended --jobs 6
OK    python scripts/run_error_detection.py
OK    python scripts/run_error_detection.py --protocol confirmatory
OK    python scripts/run_error_detection_v2.py --protocol confirmatory --jobs 4
OK    python scripts/run_error_detection_v2.py --protocol ensembles --jobs 4
OK    python scripts/run_error_detection_v2.py --protocol exploratory --jobs 2
OK    python scripts/run_faithfulness.py --jobs 3
OK    python scripts/run_faithfulness.py --protocol cond --jobs 6
OK    python scripts/run_faithfulness.py --protocol v1
OK    python scripts/run_faithfulness.py --protocol v3 --jobs 6
OK    python scripts/run_faithfulness.py --protocol v3 --timing --seeds 30 31 32
OK    python scripts/run_faithfulness.py --timing
OK    python scripts/run_g0.py
OK    python scripts/run_global_importance.py --extended --jobs 6
OK    python scripts/run_global_importance.py --jobs 4
OK    python scripts/run_interactions.py --jobs 6
OK    python scripts/run_readable_baselines.py --jobs 6
OK    python scripts/run_refinement.py
OK    python scripts/run_refinement.py --protocol confirmatory --jobs 4
OK    python scripts/run_refinement.py --protocol eligibility
OK    python scripts/run_refinement_explore.py --jobs 6
OK    python scripts/run_refinement_v2.py --jobs 6
OK    python scripts/run_repair_order.py --jobs 12
OK    python scripts/run_repair_theta.py --jobs 6
OK    python scripts/run_review_rq1.py --jobs 6
OK    python scripts/sr_method_tables.py
OK    python scripts/summarise_refinement_v1.py
OK    python scripts/time_occlusion_batched.py
OK    python scripts/time_pace_flip.py
OK    python scripts/time_pace_flip.py --datasets magic letter landsat page_blocks --out results/speed_uci
OK    python scripts/time_repair_arms.py

43 commands, 0 with problems
