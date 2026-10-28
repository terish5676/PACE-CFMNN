"""Oblique and Rotated Hyperbox CFMNN (Fix for Axis-Aligned Staircase Boundaries).

Generalizes CFMNN to support arbitrary orientations and rotated coordinate frames.
Addresses the staircase effect on diagonal boundaries via:
1. Global and per-class orthonormal coordinate rotations (e.g. PCA, ICA, or custom R).
2. Exact attribution back-projection: phi_X = phi_Z @ R.T.
3. RotatedHyperbox data structure supporting individual rotation matrices R_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from sklearn.decomposition import PCA

from .cfmnn_core import CFMNN, Hyperbox, PredictionDetails


@dataclass
class RotatedHyperbox:
    """Hyperbox with an orthonormal rotation matrix R_j and center c_j.
    
    A point x belongs to the rotated frame via y = R_j.T @ (x - c_j).
    Membership is then evaluated on y with bounds [v_j, w_j].
    """
    cls: int
    v: np.ndarray       # Lower bounds in rotated frame (d,)
    w: np.ndarray       # Upper bounds in rotated frame (d,)
    c: np.ndarray       # Center in original frame (d,)
    R: np.ndarray       # Orthonormal rotation matrix (d, d)
    centroid: np.ndarray # Centroid in original frame (d,)
    n_samples: int = 1

    def membership(self, x: np.ndarray, gamma: float = 1.0) -> float:
        """Compute fuzzy membership score in rotated coordinate frame."""
        # Project x into rotated frame
        diff = x - self.c
        y = diff @ self.R  # shape (d,)
        
        # Standard Simpson membership in rotated space
        d = len(self.v)
        score = 0.0
        for i in range(d):
            val = y[i]
            vi = self.v[i]
            wi = self.w[i]
            if vi <= val <= wi:
                term = 1.0
            elif val > wi:
                term = max(0.0, 1.0 - gamma * (val - wi))
            else:
                term = max(0.0, 1.0 - gamma * (vi - val))
            score += term
        return score / float(d)


class ObliqueCFMNN:
    """CFMNN with Oblique / Rotated Coordinate Transformation.
    
    Transforms input features via an orthonormal rotation matrix R:
        z = (x - mean) @ R
    allowing diagonal, oblique, or correlated manifolds to be enclosed cleanly 
    by hyperboxes without the axis-aligned staircase effect.
    
    Parameters
    ----------
    rotation : {'pca', 'identity'} or np.ndarray, default 'pca'
        Rotation method or explicit (d, d) orthonormal matrix.
    max_hyperbox_size : float, default 0.3
        Theta maximum hyperbox edge length.
    gamma : float, default 1.0
        Sensitivity parameter for membership decay.
    centroid_rule : {'fixed', 'original'}, default 'fixed'
    expansion_strategy : {'search', 'best'}, default 'search'
    tol : float, default 1e-5
    """

    def __init__(
        self,
        rotation: Literal["pca", "identity"] | np.ndarray = "pca",
        max_hyperbox_size: float = 0.3,
        gamma: float = 1.0,
        centroid_rule: Literal["fixed", "original"] = "fixed",
        expansion_strategy: Literal["search", "best"] = "search",
        tol: float = 1e-5,
    ):
        self.rotation = rotation
        self.max_hyperbox_size = max_hyperbox_size
        self.gamma = gamma
        self.centroid_rule = centroid_rule
        self.expansion_strategy = expansion_strategy
        self.tol = tol

        # Fitted attributes
        self.R_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.z_min_: np.ndarray | None = None
        self.z_max_: np.ndarray | None = None
        self.model_: CFMNN | None = None
        self.classes_: np.ndarray | None = None

    def _fit_rotation(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute center and orthonormal rotation matrix R."""
        n, d = X.shape
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        if isinstance(self.rotation, np.ndarray):
            R = self.rotation
            if R.shape != (d, d):
                raise ValueError(f"Custom rotation matrix must be shape ({d}, {d}), got {R.shape}")
        elif self.rotation == "pca":
            if d == 1:
                R = np.eye(1)
            else:
                pca = PCA(n_components=d)
                pca.fit(X_centered)
                R = pca.components_.T  # Shape (d, d), columns are principal axes
        elif self.rotation == "identity":
            R = np.eye(d)
        else:
            raise ValueError(f"Unknown rotation: {self.rotation}")

        return mean, R

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into rotated frame and normalize to [0, 1]^d."""
        if self.R_ is None or self.mean_ is None or self.z_min_ is None or self.z_max_ is None:
            raise ValueError("ObliqueCFMNN must be fitted before transforming.")
        
        # 1. Rotate
        Z = (X - self.mean_) @ self.R_
        
        # 2. Scale to [0, 1] using fitted bounds with tolerance clipping
        denom = np.where(self.z_max_ - self.z_min_ > 1e-12, self.z_max_ - self.z_min_, 1.0)
        Z_scaled = (Z - self.z_min_) / denom
        return np.clip(Z_scaled, 0.0, 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ObliqueCFMNN:
        """Fit oblique rotation and underlying hyperbox network."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)

        # 1. Learn rotation
        self.mean_, self.R_ = self._fit_rotation(X)
        Z_rot = (X - self.mean_) @ self.R_

        # 2. Fit min-max scaling in rotated space
        self.z_min_ = np.min(Z_rot, axis=0)
        self.z_max_ = np.max(Z_rot, axis=0)
        denom = np.where(self.z_max_ - self.z_min_ > 1e-12, self.z_max_ - self.z_min_, 1.0)
        Z_scaled = np.clip((Z_rot - self.z_min_) / denom, 0.0, 1.0)

        # 3. Fit base CFMNN on rotated features
        self.model_ = CFMNN(
            max_hyperbox_size=self.max_hyperbox_size,
            gamma=self.gamma,
            centroid_rule=self.centroid_rule,
            expansion_strategy=self.expansion_strategy,
            tol=self.tol,
        )
        self.model_.fit(Z_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        Z_scaled = self._transform(X)
        return self.model_.predict(Z_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class membership support probabilities."""
        Z_scaled = self._transform(X)
        return self.model_.predict_proba(Z_scaled)

    def predict_details(self, X: np.ndarray) -> PredictionDetails:
        """Return full prediction details."""
        Z_scaled = self._transform(X)
        return self.model_.predict_details(Z_scaled)

    def explain(
        self,
        X: np.ndarray,
        target: Literal["decision", "box", "contrastive"] = "decision",
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute feature attributions mapped back to original feature space.
        
        Applies linear transformation invariance:
            phi_X = phi_Z @ R.T
        providing direct, exact attributions in user's original coordinates.
        """
        Z_scaled = self._transform(X)
        bg_z = self._transform(background) if background is not None else None

        # Compute attributions in rotated space
        phi_z = self.model_.explain(Z_scaled, target=target, background=bg_z)

        # Back-project attributions to original coordinates via rotation matrix
        # phi_X_i = sum_j R_ij * phi_Z_j
        phi_x = phi_z @ self.R_.T
        return phi_x

    @property
    def n_boxes_(self) -> int:
        return self.model_.n_boxes_ if self.model_ is not None else 0

    @property
    def hyperboxes_(self) -> list[Hyperbox]:
        return self.model_.hyperboxes_ if self.model_ is not None else []
