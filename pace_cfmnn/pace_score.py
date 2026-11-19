"""PACE Refinement Score (blueprint 6.4).

PD, EI and U sit on different scales, so each is turned into its percentile
rank against an inner-validation set, and the ranks are averaged with equal
weights. Where EI is unavailable, the PD + U fallback (weights 1/2 each) is
used and recorded. The ranks are fitted on inner-validation data only.
"""

from __future__ import annotations

import numpy as np

from .signals import Signals

NAMES = ("pd", "ei", "u")
VARIANT = ("pd", "ei", "u_star")   # PRS*: the same with the corrected U* (post-hoc variant)


class PRS:
    def fit(self, inner: Signals) -> "PRS":
        """Store each signal's sorted inner-validation values (its empirical CDF)."""
        self.ref_ = {}
        for name in NAMES + ("u_star",):
            if getattr(inner, name, None) is None:
                continue
            v = np.asarray(getattr(inner, name), dtype=float)
            v = np.sort(v[np.isfinite(v)])
            if len(v) == 0:
                raise ValueError(f"no finite {name} values in the inner-validation set")
            self.ref_[name] = v
        return self

    def rank(self, name: str, values) -> np.ndarray:
        """Percentile rank in [0, 1]: share of inner values <= each value (nan stays nan)."""
        ref = self.ref_[name]
        v = np.asarray(values, dtype=float)
        r = np.searchsorted(ref, np.where(np.isfinite(v), v, 0.0), side="right") / len(ref)
        return np.where(np.isfinite(v), r, np.nan)

    def score(self, sig: Signals, use=NAMES):
        """Mean percentile rank of the chosen signals.

        Returns (score, fallback): with EI among ``use``, points without EI are
        scored by the remaining signals and flagged in ``fallback``.
        """
        ranks = np.stack([self.rank(n, getattr(sig, n)) for n in use])
        fallback = np.isnan(ranks).any(axis=0)
        return np.nanmean(ranks, axis=0), fallback
