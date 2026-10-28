"""Pretrained Vision & Text Embedding Hooks for PACE-CFMNN.

Enables CFMNN to act as an interpretable hyperbox decision head directly on top
of deep learning vision/text encoders (PyTorch nn.Module, Hugging Face transformers,
or custom feature extractors).

Workflow:
    Raw Inputs (Images, Texts, Audio)
                |
                v  (User-specified encoder callable / PyTorch module)
       Embedding Vector (d-dimensional continuous)
                |
                v  (Fitted Min-Max scaling with [0, 1] tolerance)
       Normalized Embedding Space
                |
                v
       PACE-CFMNN Hyperboxes & Explanations
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np

from .cfmnn_core import CFMNN, Hyperbox, PredictionDetails
from .oblique import ObliqueCFMNN


class EmbeddedCFMNN:
    """Pretrained Vision & Text Embedding Wrapper for CFMNN.

    Accepts arbitrary raw inputs (e.g., text strings, image tensors, or multimodal inputs)
    along with an encoder function or neural module. Automatically handles tensor detachment,
    array conversion, [0, 1] domain projection, and hyperbox classification.

    Parameters
    ----------
    encoder : Callable[[Any], Any] | None, default None
        Callable mapping raw inputs X to fixed-size continuous embedding representations.
        Can be:
        - PyTorch nn.Module or torch callable (tensors are automatically detached to numpy)
        - Hugging Face pipeline or tokenizer+model wrapper
        - Scikit-learn transformer (e.g. TfidfVectorizer, SVD)
        - Any custom function returning (n, d) ndarray or list of vectors.
    max_hyperbox_size : float, default 0.3
        Theta maximum hyperbox size in normalized embedding space.
    gamma : float, default 1.0
        Membership ramp sensitivity.
    use_oblique : bool, default False
        If True, applies ObliqueCFMNN (PCA coordinate rotation) on embeddings to prevent
        staircase effects along correlated embedding dimensions.
    expansion_strategy : {'search', 'best'}, default 'search'
    tol : float, default 1e-5
    """

    def __init__(
        self,
        encoder: Callable[[Any], Any] | None = None,
        max_hyperbox_size: float = 0.3,
        gamma: float = 1.0,
        use_oblique: bool = False,
        expansion_strategy: str = "search",
        tol: float = 1e-5,
    ):
        self.encoder = encoder
        self.max_hyperbox_size = max_hyperbox_size
        self.gamma = gamma
        self.use_oblique = use_oblique
        self.expansion_strategy = expansion_strategy
        self.tol = tol

        # Fitted attributes
        self.emb_min_: np.ndarray | None = None
        self.emb_max_: np.ndarray | None = None
        self.model_: CFMNN | ObliqueCFMNN | None = None
        self.classes_: np.ndarray | None = None
        self.embedding_dim_: int | None = None

    def _extract_embeddings(self, X: Any, is_fit: bool = False) -> np.ndarray:
        """Extract continuous feature embeddings from raw inputs."""
        if self.encoder is None:
            # Direct numeric features
            emb = np.asarray(X, dtype=float)
        else:
            # Check for scikit-learn style fit_transform
            if is_fit and hasattr(self.encoder, "fit_transform"):
                res = self.encoder.fit_transform(X)
            elif hasattr(self.encoder, "transform"):
                res = self.encoder.transform(X)
            else:
                res = self.encoder(X)

            # Handle PyTorch Tensor output
            if hasattr(res, "detach"):
                res = res.detach().cpu().numpy()
            elif hasattr(res, "toarray"):  # scipy sparse matrix
                res = res.toarray()

            emb = np.asarray(res, dtype=float)

        if emb.ndim != 2:
            raise ValueError(f"Encoder must return 2D array of shape (n_samples, d), got shape {emb.shape}")
        return emb

    def _scale(self, emb: np.ndarray) -> np.ndarray:
        """Scale embeddings to [0, 1] using fitted bounds with tolerant clipping."""
        if self.emb_min_ is None or self.emb_max_ is None:
            raise ValueError("EmbeddedCFMNN must be fitted before transforming.")
        denom = np.where(self.emb_max_ - self.emb_min_ > 1e-12, self.emb_max_ - self.emb_min_, 1.0)
        scaled = (emb - self.emb_min_) / denom
        return np.clip(scaled, 0.0, 1.0)

    def fit(self, X: Any, y: np.ndarray) -> EmbeddedCFMNN:
        """Extract embeddings from raw inputs X and fit hyperbox model.

        Parameters
        ----------
        X : raw inputs (e.g. list of text strings, PIL images, or numeric arrays)
        y : class labels, shape (n,)
        """
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)

        # 1. Extract embeddings
        emb = self._extract_embeddings(X, is_fit=True)
        self.embedding_dim_ = emb.shape[1]

        # 2. Fit Min-Max scaling
        self.emb_min_ = np.min(emb, axis=0)
        self.emb_max_ = np.max(emb, axis=0)
        denom = np.where(self.emb_max_ - self.emb_min_ > 1e-12, self.emb_max_ - self.emb_min_, 1.0)
        scaled_emb = np.clip((emb - self.emb_min_) / denom, 0.0, 1.0)

        # 3. Fit classifier (standard or oblique)
        if self.use_oblique:
            self.model_ = ObliqueCFMNN(
                rotation="pca",
                max_hyperbox_size=self.max_hyperbox_size,
                gamma=self.gamma,
                expansion_strategy=self.expansion_strategy,
                tol=self.tol,
            )
        else:
            self.model_ = CFMNN(
                max_hyperbox_size=self.max_hyperbox_size,
                gamma=self.gamma,
                expansion_strategy=self.expansion_strategy,
                tol=self.tol,
            )

        self.model_.fit(scaled_emb, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels for raw inputs X."""
        emb = self._extract_embeddings(X, is_fit=False)
        scaled_emb = self._scale(emb)
        return self.model_.predict(scaled_emb)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for raw inputs X."""
        emb = self._extract_embeddings(X, is_fit=False)
        scaled_emb = self._scale(emb)
        return self.model_.predict_proba(scaled_emb)

    def predict_details(self, X: Any) -> PredictionDetails:
        """Return full prediction details."""
        emb = self._extract_embeddings(X, is_fit=False)
        scaled_emb = self._scale(emb)
        return self.model_.predict_details(scaled_emb)

    def explain(
        self,
        X: Any,
        target: str = "decision",
        background: Any | None = None,
    ) -> np.ndarray:
        """Compute PACE attributions across embedding dimensions."""
        emb = self._extract_embeddings(X, is_fit=False)
        scaled_emb = self._scale(emb)
        scaled_bg = self._scale(self._extract_embeddings(background)) if background is not None else None
        return self.model_.explain(scaled_emb, target=target, background=scaled_bg)

    @property
    def n_boxes_(self) -> int:
        return self.model_.n_boxes_ if self.model_ is not None else 0

    @property
    def hyperboxes_(self) -> list[Hyperbox]:
        return self.model_.hyperboxes_ if self.model_ is not None else []
