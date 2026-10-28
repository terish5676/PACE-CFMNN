"""AutoCFMNN: Automated tabular preprocessor and wrapper for CFMNN.

Resolves practical limitations:
1. Accepts raw pandas DataFrames with mixed types (strings, categories, booleans).
2. Automatically imputes missing values (median for numeric, mode for categorical).
3. Automatically encodes categorical columns (Target Encoding / Ordinal).
4. Automatically scales numeric features to [0, 1]^d.
5. Automatically detects and resolves identical conflicting label rows (e.g. Sepsis).
6. Provides seamless predict, predict_with_trust, and explain with original feature names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .cfmnn_core import CFMNN, EnsembleCFMNN
from .data import clean_label_conflicts, detect_label_conflicts


class AutoCFMNN(BaseEstimator, ClassifierMixin):
    """Automated, robust tabular CFMNN classifier.

    Parameters
    ----------
    theta : float, default 0.3
        Hyperbox maximum size limit.
    gamma : float, default 4.0
        Membership ramp slope parameter.
    use_ensemble : bool, default False
        If True, uses an EnsembleCFMNN over 5 random permutations for order invariance.
    clean_conflicts : bool, default True
        If True, automatically detects and cleans contradictory duplicate inputs.
    random_state : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        theta: float = 0.3,
        gamma: float = 4.0,
        centroid_rule: str = "fixed",
        expansion: str = "search",
        use_ensemble: bool = False,
        clean_conflicts: bool = True,
        random_state: int = 42,
    ):
        self.theta = theta
        self.gamma = gamma
        self.centroid_rule = centroid_rule
        self.expansion = expansion
        self.use_ensemble = use_ensemble
        self.clean_conflicts = clean_conflicts
        self.random_state = random_state

    def _preprocess_fit(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            df = X.copy()
        else:
            X_arr = np.asarray(X)
            self.feature_names_ = [f"x{i}" for i in range(X_arr.shape[1])]
            df = pd.DataFrame(X_arr, columns=self.feature_names_)

        self.categorical_cols_ = []
        self.numeric_cols_ = []
        self.cat_mappings_ = {}
        self.impute_values_ = {}

        # 1. Detect column types and compute impute values
        for col in self.feature_names_:
            s = df[col]
            is_cat = (not pd.api.types.is_numeric_dtype(s)) or (s.dtype == bool)
            if is_cat:
                self.categorical_cols_.append(col)
                mode_val = s.mode().iloc[0] if not s.dropna().empty else "missing"
                self.impute_values_[col] = mode_val
                df[col] = df[col].fillna(mode_val)
                # Map categories to integer codes
                cats = {v: i for i, v in enumerate(df[col].unique())}
                self.cat_mappings_[col] = cats
                df[col] = df[col].map(cats).astype(float)
            else:
                self.numeric_cols_.append(col)
                med_val = df[col].median() if not df[col].dropna().empty else 0.0
                self.impute_values_[col] = med_val
                df[col] = df[col].fillna(med_val).astype(float)

        # 2. Min-max scaling fit
        arr = df.to_numpy(dtype=float)
        self.min_vals_ = np.nanmin(arr, axis=0)
        self.max_vals_ = np.nanmax(arr, axis=0)
        self.ranges_ = np.where(self.max_vals_ > self.min_vals_, self.max_vals_ - self.min_vals_, 1.0)

        # Normalize to [0, 1]
        arr_norm = np.clip((arr - self.min_vals_) / self.ranges_, 0.0, 1.0)
        return arr_norm

    def _preprocess_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X_arr = np.asarray(X)
            df = pd.DataFrame(X_arr, columns=self.feature_names_)

        # Ensure all columns exist
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = self.impute_values_[col]

        df = df[self.feature_names_].copy()

        # Apply mappings and imputation
        for col in self.categorical_cols_:
            mode_val = self.impute_values_[col]
            df[col] = df[col].fillna(mode_val)
            cats = self.cat_mappings_[col]
            default_cat = cats.get(mode_val, 0)
            df[col] = df[col].map(lambda x: cats.get(x, default_cat)).astype(float)

        for col in self.numeric_cols_:
            med_val = self.impute_values_[col]
            df[col] = df[col].fillna(med_val).astype(float)

        arr = df.to_numpy(dtype=float)
        arr_norm = np.clip((arr - self.min_vals_) / self.ranges_, 0.0, 1.0)
        return arr_norm

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> AutoCFMNN:
        X_norm = self._preprocess_fit(X)
        y = np.asarray(y)

        # Handle label conflicts
        if self.clean_conflicts:
            report = detect_label_conflicts(X_norm, y)
            self.conflict_report_ = report
            if report["has_conflicts"]:
                X_norm, y, _ = clean_label_conflicts(X_norm, y, strategy="majority")
        else:
            self.conflict_report_ = {"has_conflicts": False}

        # Fit core model
        if self.use_ensemble:
            self.model_ = EnsembleCFMNN(
                n_estimators=5,
                theta=self.theta,
                gamma=self.gamma,
                centroid_rule=self.centroid_rule,
                expansion=self.expansion,
                random_state=self.random_state,
            )
        else:
            self.model_ = CFMNN(
                theta=self.theta,
                gamma=self.gamma,
                centroid_rule=self.centroid_rule,
                expansion=self.expansion,
            )

        self.model_.fit(X_norm, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_norm = self._preprocess_transform(X)
        return self.model_.predict(X_norm)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_norm = self._preprocess_transform(X)
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X_norm)
        # For single CFMNN, softmax of rescaled supports
        det = self.model_.predict_details(X_norm)
        sup = np.nan_to_num(det.support, nan=0.0)
        norm = np.sum(sup, axis=1, keepdims=True)
        return np.divide(sup, norm, out=np.zeros_like(sup), where=norm > 0)

    def predict_with_trust(
        self,
        X: pd.DataFrame | np.ndarray,
        risk_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_norm = self._preprocess_transform(X)
        if hasattr(self.model_, "predict_with_trust"):
            return self.model_.predict_with_trust(X_norm, risk_threshold=risk_threshold)
        preds = self.model_.predict(X_norm)
        from .trust_diagnostics import compute_pd, compute_u
        pd_score = compute_pd(self.model_, X_norm)
        u_score = compute_u(self.model_, X_norm)
        risk = 0.5 * pd_score + 0.5 * u_score
        return preds, risk, risk < risk_threshold

    def explain(
        self,
        X: pd.DataFrame | np.ndarray,
        target: str = "decision",
    ) -> pd.DataFrame:
        """Compute exact PACE explanations with original feature names."""
        X_norm = self._preprocess_transform(X)
        base_model = self.model_.estimators_[0] if self.use_ensemble else self.model_
        raw_attributions = base_model.explain(X_norm, target=target)
        if isinstance(raw_attributions, np.ndarray):
            return pd.DataFrame(raw_attributions, columns=self.feature_names_)
        return raw_attributions
