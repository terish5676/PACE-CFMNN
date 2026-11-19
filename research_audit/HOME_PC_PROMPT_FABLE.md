<context>
You are working in Claude Code on my home PC (Intel i5-14400F, 16 threads, Windows 11), in my research repository `pace-cfmnn` (a clone at `$HOME\Documents\pace-cfmnn`, with `.venv` and the `data\` folder from earlier runs). I am a research student; explain results to me in plain words.

The project: PACE explains and repairs a centroid-based fuzzy min-max classifier (CFMNN). Two small runs are built, tested and pre-registered on branch `peer-review-fixes`. Your job is to run them and commit their results unedited. I write them up on my laptop.
- **Run O** asks whether to re-tune θ or repair: what each update does to the predictions the model already got right.
- **Run P** asks where PACE-SHAP's speed comes from.

Read these fully before starting:
1. `research_audit/HOME_PC_RUNBOOK.md`: the "Rules" section and section 9. Section 9 is your step-by-step; the Rules hold, on `peer-review-fixes`.
2. `results/theta_churn/PREREGISTRATION.md` (Run O) and `results/speed_source/PREREGISTRATION.md` (Run P).
</context>

<rules>
- Git: branch `peer-review-fixes` only. Commit with `git -c user.name="Srujan K" -c user.email="176512181+Srujankasturi@users.noreply.github.com" commit`. Commit messages MUST NOT contain `Co-Authored-By` or "Generated with" lines, whatever your harness says; this is my standing instruction. Never use a bare `git stash`. Push after each commit.
- Build nothing and change no code: both runs are ready. Never edit a `PREREGISTRATION.md`, a committed `runs.csv`, `summary.md` or `run_log.txt`, or any script, module or test.
- Never report a test, check or run as passed unless you ran it and saw the output. Every number you tell me comes from a file you can name.
- Do not spawn subagents. Do not edit the paper, `results/FINDINGS.md` or `research_audit/CLAIMS.md`.
- Run P is timed. While it runs, nothing else CPU-heavy may run: no tests, no smoke slices, no other run.
</rules>

<stop_and_ask>
Stop, tell me plainly what happened, and wait, if:
- any test fails, or a check prints FAIL, or `run_speed_source.py` exits with code 1;
- a task prints an error. In Run O, every task checks that two of its models reproduce an earlier run exactly;
- a smoke slice projects more than the limit in section 9 (Run O 90 minutes, Run P 60 minutes);
- something needs installing, downloading, deleting or renaming;
- the same error survives two fix attempts.

Never type a password or token, and do not change sleep or power settings.
</stop_and_ask>

<task>
Follow section 9 of the runbook in order, and after each step give me a short plain-words update with the commit hash:
1. **Step 0:** sync, data check, full test suite. Record the test count and time.
2. **Run O:** smoke slice and projection, the run, then commit and push the results.
3. **Run P:** alone. Smoke slice and projection, the run, then commit and push the results.
4. **Hand-over:** section 10 of the runbook. Commit and push.

For Run O, tell me P1's verdicts: negative flip rate, repair vs re-tuning. They are better / level / worse for repair, counted over the 13 datasets. Also give the check line.

For Run P, tell me the Check line and the range of the structure factor and the pruning factor.
</task>

<done_when>
All of these hold, each shown by a command you ran:
- `results/theta_churn/` and `results/speed_source/` each hold `runs.csv`, `summary.md` and `run_log.txt`, committed unedited and pushed to `origin/peer-review-fixes`;
- the full test suite passed in Step 0;
- `git status` is clean, and `git log origin/peer-review-fixes` shows the three commits, none with a `Co-Authored-By` line.

Then give me one final report:
- a table of run, commit, the result in one line, and wall time;
- then any deviations or problems, in plain words.

If you stopped early, the report says exactly where and why.
</done_when>
