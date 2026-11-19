# Run P10, held-out datasets: the rule, frozen before any download

Written on 2026-11-16 and pushed before any of these datasets was downloaded (tough review B11).
Only OpenML's public metadata was read: the suite listing and each dataset's description. No data
file was fetched. The listing as read, with each dataset's counts, is in
[`cc18_listing_2026-11-16.txt`](cc18_listing_2026-11-16.txt).

## Why a rule

The paper's real datasets were chosen by the team over time. A reader can ask whether they were
picked because the methods work on them. A rule fixed in advance answers that: every dataset that
meets it is used, and none is added or dropped after its results are seen.

## The rule

Every dataset of the **OpenML-CC18** benchmark suite (the 73 datasets tagged `OpenML-CC18` on
2026-11-16) that meets all five conditions:

1. every feature is numeric (the class is the only symbolic attribute);
2. at most **12 features**, the range where PACE-SHAP is exact by default;
3. at least **2,000 rows**;
4. no missing values;
5. not already used in the paper. Letter, satimage (the paper's Landsat) and wdbc (the paper's
   Breast Cancer) are in CC18 and are excluded.

## The list it gives

| OpenML id | Name | Rows | Features | Classes | Smallest class | Licence (OpenML) | ARFF file |
|---|---|---|---|---|---|---|---|
| 1489 | phoneme | 5,404 | 5 | 2 | 1,586 | Public | 267 KB |
| 40983 | wilt | 4,839 | 5 | 2 | 261 | Public | 257 KB |
| 18 | mfeat-morphological | 2,000 | 6 | 10 | 200 | Public | 149 KB |
| 41027 | jungle_chess_2pcs_raw_endgame_complete | 44,819 | 6 | 3 | 4,335 | Public | 628 KB |

These four are the held-out datasets, no more and no fewer. A dataset is left out only if its file
fails the MD5 checksum OpenML publishes, or fails to load. Either would be reported, and neither
depends on results.

Close misses, for the record:
- banknote-authentication (1,372 rows) and the other datasets under 2,000 rows;
- electricity (one symbolic feature, the day of the week; it is Run P9's time-ordered dataset:
  OpenML id 151, 45,312 rows, 7 numeric features and the day, 2 classes, licence Public on
  OpenML, ARFF file 3,092,013 bytes (read from the response headers, not downloaded), MD5
  8ca97867d960ae029ae3a9ac2c923d34);
- segment (19 features) and pendigits (16).

## What comes next

Each download needs the user's approval first, giving its name, source, size and licence. Each
approved file then gets a SHA-256 pinned in `scripts/download_data.py`. After that, Run P10's
pre-registration is written and pushed before any study seed runs:
- **RQ1 (seeds 360–369):** deletion AUC of PACE-SHAP against like-for-like KernelSHAP;
- **RQ2 (seeds 370–379):** macro-F1 of self-repair against the starting model, and broken correct
  predictions (NFR) of repair against re-tuning θ;
- each a paired Wilcoxon test over the 10 seeds, with Holm correction across the four datasets.
