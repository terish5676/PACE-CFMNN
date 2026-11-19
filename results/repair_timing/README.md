# Run T: time of each repair setup on one machine

A descriptive timing check, made after the RQ3 results. It is not pre-registered and tests
nothing. It times, on the home PC, seed 50 of RQ3's split, one task at a time in one process:
train CFMNN, SR with the old gate, SR with the cached gate, AL, and SR+AL. Every call is the one
RQ3 made (`scripts/run_refinement_v2.py`), without the certificate audit. Each setup's test
macro-F1 must equal the committed RQ3 value; the summary says **Verdict: PASS** only if all do.

Why: AL and SR+AL were only ever timed on the laptop (4 jobs, with the audit), and no run saved
CFMNN's training time on the home PC, so the time table had no single-machine comparison.

```
python -B scripts/time_repair_arms.py
python -B scripts/time_repair_arms.py --summary-only
```

Outputs: `runs.csv` (30 rows: 6 datasets × 5 setups), `summary.md`, `run_log.txt`. The tables in
`results/sr_method/tables.md` (`scripts/sr_method_tables.py`) read `runs.csv`. Runbook:
`research_audit/HOME_PC_RUNBOOK.md`, section 4.
