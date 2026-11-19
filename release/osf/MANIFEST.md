# Pre-registrations for the OSF deposit (tough review R2)

This folder is the bundle for an OSF registration of every plan behind the paper's tests. It was
prepared on 2026-11-16, at commit `a89a179` of `tough-review-fixes`, when the last plan (batch 3)
had been pushed and before any batch 1–3 result was on the laptop.

## What the deposit proves, and what it does not

- **Git orders every plan before its results.** Each plan was committed and pushed before its
  run, and its results were committed after it. Commit dates, however, come from the committer's
  own clock, so they are not independent evidence of time.
- **OSF adds a timestamp that the team does not control**, and fixes the plans' content from the
  deposit date on.
- **It cannot prove when the older plans were written.** For runs that finished before the
  deposit, the evidence of order is the git history. The paper should say exactly that:
  "registered on OSF on <date>; each plan was pushed to the repository before its run".
- **For the 11 tough-review runs (P1–P11)**, the deposit comes before the results reach the
  paper. For any of them still to run on the home PC, it also comes before the run.

## Files

- `sha256sums.txt`: the SHA-256 of each file as git stores it at `a89a179` (38 files).
- The table below:
  - **First pushed:** the commit that added the file.
  - **Commits:** how many commits have touched the file.
  - **SHA-256 now** and **SHA-256 when first pushed:** first 16 hex digits; "same" means the file
    has not changed since.

## How to check a hash

Hash the bytes git stores, not the file on disk. On Windows, git may write CRLF line endings,
which change the hash:

```bash
git show a89a179:results/heldout/PREREGISTRATION.md | sha256sum
```

## Plans changed after their first push

Six plans were changed after they were first pushed. Every change was committed before that
run's results, and each commit says so in its message:

| Plan | Change | Results first committed |
|---|---|---|
| `contraction` (Run F) | b3eb20e, 2026-11-14 18:26, "amendment 1: review before the run" | f6d8c26, 18:51 |
| `covertype` (Run K) | 59f1a40, 2026-11-14 23:37, "amendment 1 … before the run" | 00d73ce, 23:48 |
| `error_detection_v2` (RQ2 v2) | 38eb8e3, 2026-10-29 21:34, the final plan after the design run, "before the confirmatory run" | 0180d49, 21:46 |
| `fast_gate_check` (Run E) | ab5a166 was reverted (fa2f890) and pushed again as a4c08f1, 2026-11-11 19:23 | 5144051, 19:28 |
| `repair_order` (Run H) | 06b040b, 2026-11-14 12:16, "amendment 1 … before the run" | 498784c, 13:20 |
| `systems` (Run G) | b3eb20e, 2026-11-14 18:26 (amendment 1); e3033c8, 20:05 (amendment 2, "before the rerun") | f6486fd, 21:42 |

## The table

| File | First pushed | Date | Commits | SHA-256 now | SHA-256 when first pushed |
|---|---|---|---|---|---|
| `results/cannot_hurt/PREREGISTRATION.md` | 94dd0ed | 2026-11-16 | 1 | `9729090d89b5780d…` | same |
| `results/cf_robustness/PREREGISTRATION.md` | 8134422 | 2026-11-11 | 1 | `9e7b3cec43a45705…` | same |
| `results/conditional_settings/PREREGISTRATION.md` | 7dbdc91 | 2026-11-11 | 1 | `942e5658e984a9f9…` | same |
| `results/contraction/PREREGISTRATION.md` | dcc8b99 | 2026-11-13 | 2 | `80b41b354b8c8f3f…` | 90fe80a23caed7b7… |
| `results/covertype/PREREGISTRATION.md` | fc228ef | 2026-11-14 | 2 | `fef2f450ef66836f…` | 8d69762d476c91ee… |
| `results/covertype_theta/PREREGISTRATION.md` | bd0d023 | 2026-11-15 | 1 | `26fac52de6c98f3b…` | same |
| `results/decision/PREREGISTRATION.md` | 94dd0ed | 2026-11-16 | 1 | `b06eb2b2962d8382…` | same |
| `results/error_detection_confirmatory/PREREGISTRATION.md` | fd271ed | 2026-10-29 | 1 | `7e561ac2185be493…` | same |
| `results/error_detection_v2/PREREGISTRATION.md` | 0e2c940 | 2026-10-29 | 2 | `7392afd5380dddc3…` | 82c0780aa6b741ab… |
| `results/extensions_v1/PREREGISTRATION.md` | 31583dd | 2026-10-31 | 1 | `ea64b3a6593f7759…` | same |
| `results/faithfulness_v3/PREREGISTRATION.md` | 3d33e98 | 2026-10-30 | 1 | `6178a12343aa19ab…` | same |
| `results/fast_gate_check/PREREGISTRATION.md` | ab5a166 | 2026-11-11 | 3 | `4c63069ce4e4deef…` | 55e56acb1084f8e7… |
| `results/flip_intervals/PREREGISTRATION.md` | 816d863 | 2026-11-16 | 1 | `d1c3b996db9e3410…` | same |
| `results/fmm_baselines/PREREGISTRATION.md` | 94dd0ed | 2026-11-16 | 1 | `2887fd9fe4edd43f…` | same |
| `results/generality/PREREGISTRATION.md` | fc228ef | 2026-11-14 | 1 | `cb1505f74c9f601b…` | same |
| `results/gfmm_exact/PREREGISTRATION.md` | 94dd0ed | 2026-11-16 | 1 | `96803cde39d2508d…` | same |
| `results/heldout/PREREGISTRATION.md` | 4ede92b | 2026-11-16 | 1 | `abb33443c1f1373a…` | same |
| `results/heldout/RULE.md` | 94dd0ed | 2026-11-16 | 1 | `762bf6e95eb21b8b…` | same |
| `results/readable_baselines/PREREGISTRATION.md` | 42a71d4 | 2026-11-06 | 1 | `18a2523748328e37…` | same |
| `results/recurrence/PREREGISTRATION.md` | 816d863 | 2026-11-16 | 1 | `d8b0992b2916d271…` | same |
| `results/refinement_confirmatory/PREREGISTRATION.md` | 0a7462e | 2026-10-29 | 1 | `fc0ce3bc0877ee31…` | same |
| `results/refinement_confirmatory/PROTOCOL_AMENDMENT_1.md` | b2352f1 | 2026-11-06 | 1 | `0c9264a5e4b8c3ae…` | same |
| `results/refinement_v2_confirmatory/PREREGISTRATION.md` | 1cf58c9 | 2026-11-06 | 1 | `624f65cf12d3c1e6…` | same |
| `results/repair_order/PREREGISTRATION.md` | 87aa227 | 2026-11-14 | 2 | `4694a1715c83a265…` | e5e027a38604746c… |
| `results/repair_sensitivity/PREREGISTRATION.md` | 5b2e076 | 2026-11-11 | 1 | `942d488e50332dcb…` | same |
| `results/repair_theta/PREREGISTRATION.md` | 22e687d | 2026-11-07 | 1 | `7baa1bd7ff41b7a9…` | same |
| `results/review_v1/PREREGISTRATION.md` | 265e050 | 2026-11-01 | 1 | `73f146a525e2bef8…` | same |
| `results/rq1_more_points/PREREGISTRATION.md` | fdcc1cc | 2026-11-11 | 1 | `20b53d3bf116c5c0…` | same |
| `results/sampled_baselines/PREREGISTRATION.md` | fc228ef | 2026-11-14 | 1 | `d0a636c996d39a13…` | same |
| `results/sampled_baselines_covertype/PREREGISTRATION.md` | 075cefa | 2026-11-15 | 1 | `ad5a3f1ebb3add24…` | same |
| `results/shap_certificate/PREREGISTRATION.md` | 94dd0ed | 2026-11-16 | 1 | `35676d407f937a9b…` | same |
| `results/simpson_fmm/PREREGISTRATION.md` | 53c3f81 | 2026-11-15 | 1 | `ad5b60539089601a…` | same |
| `results/skin_dedup/PREREGISTRATION.md` | 816d863 | 2026-11-16 | 1 | `5dd6d37b6d034bc5…` | same |
| `results/speed_source/PREREGISTRATION.md` | 0c5dff4 | 2026-11-15 | 1 | `7721812252475cb4…` | same |
| `results/systems/PREREGISTRATION.md` | 0bc2562 | 2026-11-13 | 3 | `06c8bda2729aba4b…` | 97268230c69bd130… |
| `results/theta_churn/PREREGISTRATION.md` | 0c5dff4 | 2026-11-15 | 1 | `22e83daf4a266596…` | same |
| `results/time_ordered/PREREGISTRATION.md` | 4ede92b | 2026-11-16 | 1 | `a14b74576331ea0b…` | same |
| `results/time_to_accuracy/PREREGISTRATION.md` | 816d863 | 2026-11-16 | 1 | `609d5c35f1527ede…` | same |

## Steps for the user (the account is yours)

1. Make a zip of the plans as git stores them (one git command, nothing runs):
   ```bash
   git archive --format=zip -o osf_preregistrations.zip a89a179 $(cut -d' ' -f3 release/osf/sha256sums.txt)
   ```
2. On osf.io, create a project (it can stay private until submission), and upload the zip,
   `sha256sums.txt` and this file.
3. Choose "Register", with the **Open-Ended Registration** template. In the summary, paste the
   section "What the deposit proves, and what it does not".
4. Send me the registration's DOI. I will add it to the paper (§6 and Appendix E), the README and
   this file.

**For double-blind review (TMLR):** an OSF registration shows the project's contributors. Use
OSF's "View-only link" with "Anonymize" ticked for reviewers, and cite the DOI only in the
camera-ready version.

**If a plan gets an amendment before its run**, rerun step 1 at the new commit and add an update
to the registration. Do not replace the original. P2's Skin decision is not an amendment: its
plan already fixes the rule, and the pilot only applies it.
