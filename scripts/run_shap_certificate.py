"""Run P6 (tough review B7): a certificate for PACE-SHAP explanations after an edit.

Rules: results/shap_certificate/PREREGISTRATION.md (planned in advance).

``audit.pace_shap_certificate(before, after, X, background)`` marks the rows whose PACE-SHAP values
an edit can have changed: a row is outside it when its predicted class is the same before and
after and no changed box survives pruning (``PaceShap._prune``) for any background row, before or
after the edit. Then every coalition's margin is a maximum over the same unchanged boxes, so the
values are identical (exact mode), or identical with the same seed, points and order (sampled
mode). This run checks that on every accepted self-repair edit and measures how large the
certificate is.

For each dataset and seed: the refinement run's split and self-repair on the training rows (as
``run_refinement_v2.run_one``'s SR arm). The explained rows are 100 test rows; the explainer rows
are 100 training rows (fewer than the paper's 200, for time; the certificate's logic does not
depend on how many). PACE-SHAP is exact where d <= 12, else sampled with 8 permutations and a
fixed seed. After every accepted edit an ``on_accept`` hook computes the certificate and the new
values, and compares them with the values before the edit.

Check (pass/fail): no explained row outside the certificate has any value change larger than
1e-12, over every accepted edit.

    python -B scripts/run_shap_certificate.py --jobs 6
    python -B scripts/run_shap_certificate.py --jobs 6 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_cfmnn import CFMNN  # noqa: E402
from pace_cfmnn.active import run_self_repair  # noqa: E402
from pace_cfmnn.audit import pace_shap_certificate  # noqa: E402
from pace_cfmnn.explain import PaceShap  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_refinement import COST, MAX_BUDGET, N_ROUNDS, split  # noqa: E402

DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = list(range(330, 340))
N_EXPLAIN, N_BG, N_PERM = 100, 100, 8
TOL = 1e-12
OUT = ROOT / "results" / "shap_certificate"


def _key(m):
    return (m.V_.tobytes(), m.W_.tobytes(), m.centroid_.tobytes(), m.box_class_.tobytes())


class CertLog:
    """``on_accept`` hook: the PACE-SHAP certificate and the real value changes of every accepted edit."""

    def __init__(self, X, B, seed):
        self.X, self.B, self.seed = X, B, seed
        self.rows: list[dict] = []
        self._cache = None

    def values(self, model):
        k = _key(model)
        if self._cache is not None and self._cache[0] == k:
            return self._cache[1]
        ps = PaceShap(model, self.B, n_permutations=N_PERM, seed=self.seed)
        phi = ps.explain(self.X)
        self._cache = (k, phi, ps.exact)
        return phi

    def __call__(self, before, after, removed):
        t0 = time.perf_counter()
        phi0 = self.values(before)
        cert = pace_shap_certificate(before, after, self.X, self.B, removed)
        t_cert = time.perf_counter() - t0
        phi1 = self.values(after)
        exact = self._cache[2]
        change = np.abs(phi1 - phi0).max(axis=1)
        moved = change > TOL
        rel = change / np.maximum(np.abs(phi0).max(axis=1), 1e-300)     # relative to the row's largest value
        self.rows.append({"edit": len(self.rows), "removed": removed is not None, "mode": "exact" if exact else "sampled",
                          "n": len(self.X), "certified": int(cert.sum()), "changed": int(moved.sum()),
                          "changed_outside": int((moved & ~cert).sum()),
                          "max_change_outside": float(change[~cert].max()) if (~cert).any() else 0.0,
                          "changed_inside": int((moved & cert).sum()),
                          "rel_change_median": float(np.median(rel[moved])) if moved.any() else 0.0,
                          "rel_change_max": float(rel.max()), "cert_s": t_cert,
                          "seconds": time.perf_counter() - t0})


def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    P = split(name, seed)
    (Xtr, ytr), (Xp, _), (Xt, _) = P["train"], P["pool"], P["test"]
    rng = np.random.default_rng(seed + 11)
    Xe = Xt[rng.permutation(len(Xt))[:N_EXPLAIN]]
    B = Xtr[rng.choice(len(Xtr), min(N_BG, len(Xtr)), replace=False)]
    model = CFMNN().fit(Xtr, ytr)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    log = CertLog(Xe, B, seed)
    run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=budget, n_rounds=N_ROUNDS, seed=seed,
                    gate_labels_are_new=False, on_accept=log)
    base = {"dataset": name, "seed": seed, "d": Xtr.shape[1], "boxes_start": model.n_boxes_,
            "task_s": time.perf_counter() - t0}
    empty = {"edit": -1, "removed": False, "mode": "", "n": len(Xe), "certified": 0, "changed": 0,
             "changed_outside": 0, "max_change_outside": 0.0, "changed_inside": 0, "rel_change_median": 0.0,
             "rel_change_max": 0.0, "cert_s": np.nan,
             "seconds": np.nan}
    rows = [{**base, **r} for r in log.rows] or [{**base, **empty}]
    print(f"{name} seed {seed}: {len(log.rows)} accepted edits, certified {sum(r['certified'] for r in log.rows)}, "
          f"changed {sum(r['changed'] for r in log.rows)}, changed outside "
          f"{sum(r['changed_outside'] for r in log.rows)} ({base['task_s']:.0f}s)", flush=True)
    return rows


def check(df: pd.DataFrame) -> tuple[bool, list[str]]:
    g = df[(df.edit >= 0) & (df.changed_outside > 0)]
    return g.empty, [f"{r.dataset} seed {r.seed} edit {r.edit}: {r.changed_outside} rows outside the certificate "
                     f"changed (largest {r.max_change_outside:.1e})" for _, r in g.iterrows()]


def summarise(df: pd.DataFrame, out: Path) -> str:
    ok, bad = check(df)
    e = df[df.edit >= 0]
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# Run P6: a certificate for PACE-SHAP explanations", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Self-repair at θ = 0.3 on the refinement run's "
         f"split, seeds {seeds[0]}–{seeds[-1]}, {N_EXPLAIN} explained test rows, {N_BG} explainer rows. "
         "Written by `scripts/run_shap_certificate.py`.", "",
         f"**Check: {'PASS' if ok else 'FAIL'}.** Over {len(e):,} accepted edits and {int(e.n.sum()):,} "
         f"(edit, row) pairs, no row outside the certificate changed by more than {TOL:g}."]
    L += [f"- {b}" for b in bad]
    L += ["", "| Dataset | d | Mode | Accepted edits | Certified share (mean per edit) | Changed share | "
          "Certified rows that changed | Median relative change (changed rows) | Largest relative change |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds in [d for d in DATASETS if d in set(df.dataset)]:
        g = e[e.dataset == ds]
        if g.empty:
            L.append(f"| {ds} | – | – | 0 | – | – | – | – | – |")
            continue
        cert = g.certified / g.n
        ch = g.changed / g.n
        inside = g.changed_inside.sum() / max(1, g.certified.sum())
        L.append(f"| {ds} | {g.d.iloc[0]} | {g["mode"].iloc[0]} | {len(g):,} | {cert.mean():.1%} | {ch.mean():.1%} | "
                 f"{inside:.1%} | {g.rel_change_median.median():.1e} | {g.rel_change_max.max():.1e} |")
    L += ["", "Certified share: the share of the explained rows the certificate marks, per edit. Changed share: "
          "the rows whose values really changed. Certified rows that changed: of all certified (edit, row) "
          "pairs, the share whose values changed (how tight the certificate is). Relative change: a row's "
          "largest value change divided by its largest value before the edit; the median is over changed "
          "rows per edit, then over edits."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P6: PACE-SHAP certificate")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, help="smoke runs only")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "edits.csv"
    if not a.summary_only:
        tasks = [(d, s) for d in a.datasets or DATASETS for s in a.seeds or SEEDS]
        run_all(run_one, pending(csv, tasks, a.resume, a.fresh), csv, jobs=a.jobs, cost=COST)
    df = pd.read_csv(csv, float_precision="round_trip", keep_default_na=False, na_values=[""])
    print(summarise(df, out))
    sys.exit(0 if check(df)[0] else 1)


if __name__ == "__main__":
    main()
