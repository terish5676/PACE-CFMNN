"""Datasets, splits and preprocessing (blueprint section 9).

* Synthetic S1 and S2: ground-truth sets where only features 0-2 decide the class.
* Wine, Breast Cancer, Digits: bundled with scikit-learn (no download).
* Thyroid, Shuttle, Skin, Sepsis: the paper's UCI datasets, read from
  ``data/<name>/`` once the files have been downloaded there.

Scaling is always fitted on training rows only, then frozen; values outside
the training range are clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skds
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Ground truth for the synthetic sets: class score = 2*x0 + 1.5*x1 - x2.
SYNTH_WEIGHTS = np.array([2.0, 1.5, -1.0])
SYNTH_RELEVANT = (0, 1, 2)


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------- synthetic
def synth_score(X: np.ndarray) -> np.ndarray:
    return X[:, list(SYNTH_RELEVANT)] @ SYNTH_WEIGHTS


def synth_threshold(minority: float | None = None) -> float:
    """Decision threshold on the clean score, the same for every seed and split.

    Balanced classes use the exact median 1.25 (the score is symmetric about
    its mean). For a minority share q, the (1 - q) quantile is estimated once
    from a fixed sample of uniform points.
    """
    if minority is None:
        return 1.25
    ref = np.random.default_rng(20261028).uniform(size=(400_000, len(SYNTH_WEIGHTS)))
    return float(np.quantile(ref @ SYNTH_WEIGHTS, 1.0 - minority))


def make_synthetic(kind: str = "S1", n: int = 5000, d: int = 10, seed: int = 0, *,
                   noise: float = 0.25, margin: float = 0.1, minority: float | None = None,
                   label_noise: float = 0.0, correlated: bool = False,
                   shift: float = 0.0) -> Dataset:
    """Synthetic data where only features 0-2 carry signal.

    S1 (separable): label = score > t, and points within ``margin`` of the
    boundary are dropped, so classes do not overlap.
    S2 (overlapping): label = score + N(0, noise^2) > t, so classes overlap.

    Variants (any kind): ``minority`` sets the share of class 1 (e.g. 0.1);
    ``label_noise`` flips that share of labels; ``correlated`` makes feature 3
    a noisy copy of feature 0 (a correlated distractor); ``shift`` moves
    feature 0 towards 1 (use it for a shifted test set).
    """
    if kind == "S3":
        return make_box_synthetic(n=n, d=d, seed=seed)
    if kind not in ("S1", "S2"):
        raise ValueError("kind must be 'S1', 'S2' or 'S3'")
    if d < 4:
        raise ValueError("need d >= 4")
    rng = np.random.default_rng(seed)
    t = synth_threshold(minority)
    need = n
    Xs, ys = [], []
    while need > 0:
        m = 4 * need + 100
        X = rng.uniform(size=(m, d))
        if shift:
            X[:, 0] = X[:, 0] ** (1.0 / (1.0 + shift))  # skews feature 0 towards 1
        if correlated:
            X[:, 3] = np.clip(X[:, 0] + rng.normal(0, 0.05, m), 0, 1)
        score = synth_score(X)
        if kind == "S2":
            score = score + rng.normal(0, noise, m)
        keep = np.ones(m, bool) if kind == "S2" else np.abs(score - t) > margin
        Xs.append(X[keep][:need])
        ys.append((score[keep] > t).astype(int)[:need])
        need -= len(Xs[-1])
    X = np.vstack(Xs)[:n]
    y = np.concatenate(ys)[:n]
    if label_noise:
        flip = rng.uniform(size=n) < label_noise
        y = np.where(flip, 1 - y, y)
    names = [f"x{i}" for i in range(d)]
    info = {"relevant": list(SYNTH_RELEVANT), "weights": SYNTH_WEIGHTS.tolist(), "kind": kind,
            "minority": minority, "label_noise": label_noise, "correlated": correlated,
            "shift": shift}
    return Dataset(kind, X, y, names, info)


def s3_rule(X: np.ndarray) -> np.ndarray:
    """S3 ground truth: class 1 iff (x0 > 0.5 and x1 < 0.4) or x2 > 0.8."""
    return (((X[:, 0] > 0.5) & (X[:, 1] < 0.4)) | (X[:, 2] > 0.8)).astype(int)


def make_box_synthetic(n: int = 5000, d: int = 10, seed: int = 0, margin: float = 0.02) -> Dataset:
    """S3 (added before results, as an extra test, not a replacement for S1/S2):
    an axis-aligned rule on features 0-2 that box models can represent. Points
    within ``margin`` of a threshold on a relevant feature are dropped."""
    rng = np.random.default_rng(seed)
    Xs = []
    need = n
    while need > 0:
        X = rng.uniform(size=(4 * need + 100, d))
        near = ((np.abs(X[:, 0] - 0.5) < margin) | (np.abs(X[:, 1] - 0.4) < margin)
                | (np.abs(X[:, 2] - 0.8) < margin))
        Xs.append(X[~near][:need])
        need -= len(Xs[-1])
    X = np.vstack(Xs)[:n]
    info = {"relevant": list(SYNTH_RELEVANT), "kind": "S3", "rule": "(x0>0.5 & x1<0.4) | x2>0.8"}
    return Dataset("S3", X, s3_rule(X), [f"x{i}" for i in range(d)], info)


# --------------------------------------------------------------------- bundled
def _from_sklearn(name: str, bunch) -> Dataset:
    names = list(getattr(bunch, "feature_names", [f"x{i}" for i in range(bunch.data.shape[1])]))
    return Dataset(name, np.asarray(bunch.data, float), np.asarray(bunch.target), names)


# ------------------------------------------------------------------------- UCI
# Files expected under data/<name>/. Download them from the UCI repository
# (archive.ics.uci.edu) and unzip into those folders.
UCI_FILES = {
    "thyroid": "ann-train.data and ann-test.data (UCI 'Thyroid Disease')",
    "shuttle": "shuttle.trn and shuttle.tst (UCI 'Statlog (Shuttle)')",
    "skin": "Skin_NonSkin.txt (UCI 'Skin Segmentation')",
    "sepsis": "the primary cohort CSV (UCI 'Sepsis Survival Minimal Clinical Records')",
}


def _need(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put {UCI_FILES[name]} in {path.parent}.")
    return path


def _whitespace_table(paths: list[Path]) -> np.ndarray:
    return np.vstack([np.loadtxt(p) for p in paths])


def _load_uci(name: str, data_dir: Path) -> Dataset:
    folder = data_dir / name
    if name == "thyroid":
        arr = _whitespace_table([_need(folder / "ann-train.data", name),
                                 _need(folder / "ann-test.data", name)])
        X, y = arr[:, :-1], arr[:, -1].astype(int)
        names = [f"a{i}" for i in range(X.shape[1])]
    elif name == "shuttle":
        arr = _whitespace_table([_need(folder / "shuttle.trn", name),
                                 _need(folder / "shuttle.tst", name)])
        X, y = arr[:, :-1], arr[:, -1].astype(int)
        names = [f"a{i}" for i in range(X.shape[1])]
    elif name == "skin":
        arr = _whitespace_table([_need(folder / "Skin_NonSkin.txt", name)])
        X, y = arr[:, :-1], arr[:, -1].astype(int)
        names = ["B", "G", "R"]
    elif name == "sepsis":
        files = sorted(folder.glob("*primary*cohort*.csv")) if folder.exists() else []
        df = pd.read_csv(_need(files[0] if files else folder / "primary_cohort.csv", name))
        X, y = df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(int)
        names = list(df.columns[:-1])
    else:
        raise KeyError(name)
    return Dataset(name, X, y, names)


# ---------------------------------------------------------------------- loader
BUILTIN = ("S1", "S2", "S3", "wine", "breast_cancer", "digits")
UCI = tuple(UCI_FILES)


def load(name: str, seed: int = 0, data_dir: Path | None = None, **kwargs) -> Dataset:
    """Load a dataset by name. Synthetic sets use ``seed`` and ``kwargs``."""
    if name in ("S1", "S2", "S3"):
        return make_synthetic(name, seed=seed, **kwargs)
    if name == "wine":
        return _from_sklearn(name, skds.load_wine())
    if name == "breast_cancer":
        return _from_sklearn(name, skds.load_breast_cancer())
    if name == "digits":
        return _from_sklearn(name, skds.load_digits())
    if name in UCI:
        return _load_uci(name, data_dir or DATA_DIR)
    raise KeyError(f"unknown dataset {name!r}")


# ---------------------------------------------------------------------- splits
ROLE_SHARES = {"train": 0.5, "pool": 0.2, "inner": 0.1, "accept": 0.1, "test": 0.1}


def paper_split(y: np.ndarray, seed: int, test_size: float = 0.2) -> dict[str, np.ndarray]:
    """The paper's protocol: one stratified 80/20 train/test split."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return {"train": np.sort(tr), "test": np.sort(te)}


def role_split(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Blueprint 9.1: stratified 50/20/10/10/10 split into the five data roles.

    Rows are carved off one role at a time, each stratified on the remaining rows.
    """
    rest = np.arange(len(y))
    out, left = {}, 1.0
    roles = list(ROLE_SHARES)
    for i, role in enumerate(roles[:-1]):
        share = ROLE_SHARES[role] / left
        sss = StratifiedShuffleSplit(n_splits=1, train_size=share, random_state=seed + i)
        take, keep = next(sss.split(rest, y[rest]))
        out[role] = np.sort(rest[take])
        rest = rest[keep]
        left -= ROLE_SHARES[role]
    out[roles[-1]] = np.sort(rest)
    return out


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    """Min-max scaler fitted on training rows only; clips unseen values to [0, 1]."""
    return MinMaxScaler(clip=True).fit(X_train)
