"""Run P11 (tough review B6, part 2): how many self-repair edits provably cannot hurt.

Rules: results/cannot_hurt/PREREGISTRATION.md (planned in advance).

Condition S (``audit.training_safe_shrink``) says a shrink cannot break a correct training
prediction when (i) no training row of the box's class lies in the part of the old box the shrink
cut away, and (ii) no own-class row outside the old box whose membership fell was won by the box
before. If many accepted edits met S, the gate could be skipped for them (a fast path). The
reanalysis of saved edit logs suggests S rarely holds, because a box's faces sit on its own
members, so a shrink of a face usually cuts one away. This run counts it directly.

For each dataset and seed: the refinement run's split, model, budget and rounds
(``run_refinement.split``), then gated self-repair on the training rows exactly as
``run_refinement_v2.run_one``'s SR arm. After every gated edit, accepted or not, an ``on_try``
hook records:

* whether S holds, and why not (own-class rows cut, own-class rows won and dropped, centroid
  moved, other boxes changed);
* whether the gate accepted it, and the gate's time;
* how many correct training predictions the edit broke and how many errors it fixed.

Primary: the share of accepted edits that meet S. Check: no edit meeting S breaks a correct
training prediction (Proposition B6). If the gate never rejects an edit meeting S, skipping the
gate for those edits changes no decision, so the fast path gives the same model; the fast path is
run (a second stage, code pushed first) only where S holds for at least 10% of accepted edits and
the gate rejected some edit meeting S.

    python -B scripts/run_cannot_hurt.py --jobs 6
    python -B scripts/run_cannot_hurt.py --jobs 6 --resume
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
from pace_cfmnn.audit import changed_boxes, training_safe_shrink  # noqa: E402
from pace_cfmnn.evaluate import classification_metrics  # noqa: E402
from pace_cfmnn.resume import pending, run_all  # noqa: E402
from run_refinement import COST, MAX_BUDGET, N_ROUNDS, split  # noqa: E402

DATASETS = ["thyroid", "landsat", "page_blocks", "magic", "letter", "skin"]
SEEDS = list(range(390, 400))
STAGE2_SHARE = 0.10
OUT = ROOT / "results" / "cannot_hurt"


class SafetyLog:
    """``on_try`` hook: condition S, the gate's verdict and the training churn of every gated edit."""

    def __init__(self, X, y):
        self.X, self.y = X, np.asarray(y)
        self.rows: list[dict] = []
        self._pred = None

    def __call__(self, before, after, info, accepted, gate_s):
        yi = np.searchsorted(before.classes_, self.y)
        box = int(info["box"])
        removed = bool(info.get("removed", False))
        ch_b, ch_a = changed_boxes(before, after, box if removed else None)
        s = training_safe_shrink(before, after, self.X, yi, box, removed=removed)
        p0 = self._pred if self._pred is not None else before.predict(self.X)
        p1 = after.predict(self.X)
        if accepted:
            self._pred = p1
        elif self._pred is None:
            self._pred = p0
        right0, right1 = p0 == self.y, p1 == self.y
        self.rows.append({"edit": len(self.rows), "box": box, "removed": removed, "accepted": bool(accepted),
                          "gate_s": float(gate_s), "boxes_changed_before": len(ch_b),
                          "boxes_changed_after": len(ch_a), **s,
                          "train_broken": int((right0 & ~right1).sum()),
                          "train_fixed": int((~right0 & right1).sum())})


def run_one(name: str, seed: int):
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    P = split(name, seed)
    (Xtr, ytr), (Xp, _), (Xt, yt) = P["train"], P["pool"], P["test"]
    model = CFMNN().fit(Xtr, ytr)
    budget = min(MAX_BUDGET, len(Xp) // 4)
    log = SafetyLog(Xtr, ytr)
    sr, _, el = run_self_repair(model, Xtr, ytr, Xtr, ytr, chooser="volume", budget=budget,
                                n_rounds=N_ROUNDS, seed=seed, gate_labels_are_new=False, on_try=log)
    f0 = classification_metrics(yt, model.predict(Xt))["macro_f1"]
    f1 = classification_metrics(yt, sr.predict(Xt))["macro_f1"]
    base = {"dataset": name, "seed": seed, "n_train": len(Xtr), "budget": budget,
            "edits_logged": len(el), "edits_gated": len(log.rows),
            "edits_accepted": int(sum(r["accepted"] for r in log.rows)),
            "b0_macro_f1": f0, "sr_macro_f1": f1, "seconds": time.perf_counter() - t0}
    if base["edits_accepted"] != sum(bool(e.get("accepted")) for e in el):
        raise RuntimeError(f"{name} seed {seed}: the hook missed an accepted edit")
    empty = {"edit": -1, "box": -1, "removed": False, "accepted": False, "gate_s": np.nan,
             "boxes_changed_before": 0, "boxes_changed_after": 0, "safe": False, "n_slab": 0,
             "n_won_dropped": 0, "n_above_winner": 0, "same_centroid": True, "only_this_box": True,
             "train_broken": 0,
             "train_fixed": 0}
    rows = [{**base, **r} for r in log.rows] or [{**base, **empty}]
    acc = [r for r in log.rows if r["accepted"]]
    print(f"{name} seed {seed}: {len(log.rows)} gated, {len(acc)} accepted, "
          f"{sum(r['safe'] for r in acc)} accepted meet S; SR {f1:.2f} vs B0 {f0:.2f} "
          f"({base['seconds']:.0f}s)", flush=True)
    return rows


def check(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """PASS if no edit meeting S broke a correct training prediction."""
    e = df[(df.edit >= 0) & df.safe.astype(bool) & (df.train_broken > 0)]
    return e.empty, [f"{r.dataset} seed {r.seed} edit {r.edit}: meets S but broke {r.train_broken} "
                     "correct training predictions" for _, r in e.iterrows()]


def summarise(df: pd.DataFrame, out: Path) -> str:
    ok, bad = check(df)
    e = df[df.edit >= 0].copy()
    for c in ("safe", "accepted", "same_centroid", "only_this_box"):
        e[c] = e[c].astype(bool)
    seeds = sorted(int(s) for s in df.seed.unique())
    L = ["# Run P11: how many self-repair edits provably cannot hurt", "",
         "Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The refinement run's split and self-repair "
         f"(gated on the training rows), seeds {seeds[0]}–{seeds[-1]}. Written by `scripts/run_cannot_hurt.py`.",
         "", f"**Check: {'PASS' if ok else 'FAIL'}.** No edit meeting condition S broke a correct training "
         f"prediction ({int(e.safe.sum()):,} such edits, accepted or not)."]
    L += [f"- {b}" for b in bad]
    L += ["", "| Dataset | Accepted edits | Meet S | Share | Gated edits meeting S that the gate rejected | "
          "Own-class rows cut | Won and dropped | Above the winner (near-tie) | Centroid moved | "
          "Other boxes changed | Accepted edits that broke a correct training prediction | "
          "Gate time on accepted S edits (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    stage2 = []
    for ds in [d for d in DATASETS if d in set(df.dataset)]:
        g = e[e.dataset == ds]
        a = g[g.accepted]
        n, k = len(a), int(a.safe.sum())
        share = k / n if n else 0.0
        rej = int((g.safe & ~g.accepted).sum())
        if share >= STAGE2_SHARE and rej > 0:
            stage2.append(ds)
        pct = lambda m: f"{int(m.sum()):,} ({m.mean():.0%})" if len(m) else "–"   # noqa: E731
        L.append(f"| {ds} | {n:,} | {k:,} | {share:.1%} | {rej:,} | {pct(a.n_slab > 0)} | "
                 f"{pct(a.n_won_dropped > 0)} | {pct(a.n_above_winner > 0)} | {pct(~a.same_centroid)} | "
                 f"{pct(~a.only_this_box)} | "
                 f"{pct(a.train_broken > 0)} | {a[a.safe].gate_s.sum():.1f} |")
    a = e[e.accepted]
    n, k = len(a), int(a.safe.sum())
    rej = int((e.safe & ~e.accepted).sum())
    L += ["", f"Overall: {k:,} of {n:,} accepted edits meet S ({k / max(1, n):.1%}); the gate rejected "
          f"{rej:,} gated edits that meet S. The five columns after it count the accepted edits failing "
          "each part of S (an edit can fail several). The gate bounds the drop in macro-F1 on the gate "
          "rows, not each row, so an accepted edit can still break some correct training predictions.", "",
          f"**Second stage (the fast path):** {'run on ' + ', '.join(stage2) if stage2 else 'not needed'}. "
          f"It runs only where at least {STAGE2_SHARE:.0%} of accepted edits meet S and the gate rejected an "
          "edit meeting S; elsewhere skipping the gate for those edits changes no decision, so it gives the "
          "same model, and the saving is the gate time in the last column."]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Run P11: edits that cannot hurt")
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
    df = pd.read_csv(csv, float_precision="round_trip")
    print(summarise(df, out))
    sys.exit(0 if check(df)[0] else 1)


if __name__ == "__main__":
    main()
