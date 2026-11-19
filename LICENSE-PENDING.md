# Licence: not chosen yet

The team chooses the licence before the code is released (tough review R1). Until then, no licence
is granted: all rights reserved.

Points for the team:

- **The core package** (`pace_cfmnn`) needs only NumPy, SciPy, scikit-learn and pandas, all under
  BSD-3-Clause. A permissive licence (MIT, BSD-3-Clause or Apache-2.0) fits them.
- **hyperbox-brain is GPL-3.0.** The code imports it only for the baseline runs
  (`fmm_baselines.py`, `gfmm_explain.py`'s adapters, `run_readable_baselines.py`), and it is an
  optional dependency (`pip install .[baselines]`); the repository does not include its code.
  Whether a permissive licence is fine for code that imports a GPL library at run time is a
  question for the guide or the university, not one this file settles.
- **Datasets** keep their own licences (UCI and OpenML pages); `scripts/download_data.py` fetches
  them and pins their hashes, and the repository does not redistribute them.

When the team has chosen: replace this file with `LICENSE`, add the `license` field to
`pyproject.toml`, and update the README's "License" section.
