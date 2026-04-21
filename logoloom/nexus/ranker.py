"""
NexusRanker — the core alignment engine.

For a given entity of interest (title + text), computes cosine similarity
against all framework controls, passes the full similarity vector through a
logistic regression model to produce an overall relevance probability, and
uses SHAP values to decompose that probability into per-control contributions
for ranking. Returns the top-k controls ranked by SHAP contribution (or by
raw cosine similarity when no model is trained).

Architecture mirrors the original NEXUS system, adapted for the nuclear
regulatory domain (NUREG-0800 Standard Review Plan criteria).

Design note on the LR architecture
-----------------------------------
The LR takes the full N_controls-dimensional similarity vector as input and
predicts a single binary label: P(this entity is relevant to the framework).
SHAP's LinearExplainer then decomposes that prediction into per-control
contributions — shap_vals[i] represents how much control i's similarity score
drove the overall relevance prediction. Controls are ranked by this SHAP
contribution, not by an incorrectly vectorised per-control LR call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .embedder import Embedder

logger = logging.getLogger(__name__)

# Number of background samples to save for SHAP (capped to avoid large files)
_SHAP_BACKGROUND_MAX = 100


class NexusRanker:
    """
    Full NEXUS pipeline: embed → similarity → logistic regression → SHAP → rank.

    Parameters
    ----------
    embedder : Embedder
        Pre-configured embedding model.
    controls : list[dict]
        List of control dicts, each with at minimum:
            - "id"    : str  (e.g. "SRP-15.2")
            - "title" : str
            - "text"  : str
        Optional fields: "chapter", "section", "framework"
    """

    def __init__(self, embedder: Embedder, controls: list[dict]):
        if not controls:
            raise ValueError("controls list must not be empty")
        self.embedder = embedder
        self.controls = controls
        self._control_embeddings: np.ndarray | None = None
        self._lr: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._explainer: shap.LinearExplainer | None = None
        self._shap_background: np.ndarray | None = None  # saved for reload

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def build_control_index(self, show_progress: bool = True) -> None:
        """Pre-compute and cache embeddings for all framework controls."""
        pairs = [(c["title"], c["text"]) for c in self.controls]
        self._control_embeddings = self.embedder.encode_batch(
            pairs, show_progress=show_progress
        )
        logger.info("Built control index: %d controls, dim=%d",
                    len(self.controls), self._control_embeddings.shape[1])

    def get_similarity_scores(self, title: str, text: str) -> np.ndarray:
        """
        Public API: compute cosine similarity between an entity and all controls.

        Returns shape (N_controls,) array of similarity scores in [-1, 1].
        """
        entity_emb = self.embedder.encode_one(title, text)
        return self._similarity_features(entity_emb)

    def _similarity_features(self, entity_embedding: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between entity and every control.
        Since embeddings are L2-normalised, dot product == cosine similarity.
        """
        if self._control_embeddings is None:
            raise RuntimeError(
                "Control index not built. Call build_control_index() first."
            )
        return self._control_embeddings @ entity_embedding  # (N,)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        """
        Fit the logistic regression ranker.

        Parameters
        ----------
        X : np.ndarray, shape (N_samples, N_controls)
            Each row is the full similarity feature vector for one training
            entity — cosine similarity between that entity and every control.
        y : np.ndarray, shape (N_samples,)
            Binary labels: 1 = entity is framework-relevant, 0 = not.
        C : float
            Inverse regularisation strength for LogisticRegression.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != len(y):
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)} labels"
            )
        if X.shape[1] != len(self.controls):
            raise ValueError(
                f"X has {X.shape[1]} features but catalog has {len(self.controls)} controls"
            )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._lr = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self._lr.fit(X_scaled, y)

        # Save a representative background set for SHAP
        n_bg = min(_SHAP_BACKGROUND_MAX, len(X_scaled))
        self._shap_background = X_scaled[:n_bg]

        self._explainer = shap.LinearExplainer(
            self._lr,
            self._shap_background,
            feature_perturbation="interventional",
        )
        logger.info(
            "Fitted LR ranker: %d samples, %d features, SHAP background=%d",
            len(X_scaled), X_scaled.shape[1], n_bg,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def rank(
        self,
        title: str,
        text: str,
        top_k: int = 10,
        explain: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Rank all controls by relevance to the given entity.

        When a trained LR model is available, controls are ranked by their
        SHAP contribution to the overall relevance probability — i.e. how
        much control i's similarity score drove the model's decision.

        Without a trained model, controls are ranked by raw cosine similarity.

        Returns
        -------
        list of dicts (length min(top_k, N_controls)), each containing:
            "rank"          : int
            "id"            : str
            "title"         : str
            "text"          : str
            "chapter"       : str
            "section"       : str
            "framework"     : str
            "cosine_sim"    : float   — raw semantic similarity
            "probability"   : float   — ranking score (SHAP or cosine_sim)
            "overall_prob"  : float   — P(relevant) from LR [only if model fitted]
            "shap_value"    : float   — per-control SHAP contribution [if explain=True]
        """
        if not title.strip() and not text.strip():
            logger.warning("rank() called with empty title and text")

        top_k = max(1, min(top_k, len(self.controls)))

        entity_emb = self.embedder.encode_one(title, text)
        sims = self._similarity_features(entity_emb)  # (N_controls,)

        overall_prob: float | None = None
        shap_vals: np.ndarray | None = None

        if self._lr is not None and self._scaler is not None:
            X_row = sims.reshape(1, -1)                     # (1, N_controls)
            X_scaled = self._scaler.transform(X_row)        # (1, N_controls)

            # Single overall relevance probability for this entity
            overall_prob = float(self._lr.predict_proba(X_scaled)[0, 1])

            if explain and self._explainer is not None:
                # SHAP decomposes overall_prob into per-control contributions.
                # shap_vals[i] > 0 means control i's similarity pushed the
                # prediction toward "relevant".
                shap_vals = self._explainer.shap_values(X_scaled)[0]  # (N_controls,)
                ranking_scores = shap_vals
            else:
                ranking_scores = sims
        else:
            # No trained model: rank by raw cosine similarity
            ranking_scores = sims

        ranked_indices = np.argsort(ranking_scores)[::-1][:top_k]

        results = []
        for rank_pos, idx in enumerate(ranked_indices, start=1):
            control = self.controls[idx]
            result: dict[str, Any] = {
                "rank": rank_pos,
                "id": control.get("id", str(idx)),
                "title": control.get("title", ""),
                "text": control.get("text", ""),
                "chapter": control.get("chapter", ""),
                "section": control.get("section", ""),
                "framework": control.get("framework", "NUREG-0800"),
                "cosine_sim": float(sims[idx]),
                "probability": float(ranking_scores[idx]),
            }
            if overall_prob is not None:
                result["overall_prob"] = overall_prob
            if shap_vals is not None:
                result["shap_value"] = float(shap_vals[idx])
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the ranker state (model + scaler + embeddings + SHAP background) to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._lr is not None:
            joblib.dump(self._lr, path / "lr_model.joblib")
        if self._scaler is not None:
            joblib.dump(self._scaler, path / "scaler.joblib")
        if self._control_embeddings is not None:
            np.save(path / "control_embeddings.npy", self._control_embeddings)
        if self._shap_background is not None:
            np.save(path / "shap_background.npy", self._shap_background)
        with open(path / "controls.json", "w") as f:
            json.dump(self.controls, f, indent=2)
        logger.info("Saved ranker to %s", path)

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "NexusRanker":
        """Load a saved ranker from disk."""
        path = Path(path)

        controls_path = path / "controls.json"
        if not controls_path.exists():
            raise FileNotFoundError(f"No controls.json found at {path}")

        with open(controls_path) as f:
            controls = json.load(f)

        ranker = cls(embedder=embedder, controls=controls)

        emb_path = path / "control_embeddings.npy"
        if emb_path.exists():
            ranker._control_embeddings = np.load(emb_path)

        lr_path = path / "lr_model.joblib"
        scaler_path = path / "scaler.joblib"
        if lr_path.exists() and scaler_path.exists():
            ranker._lr = joblib.load(lr_path)
            ranker._scaler = joblib.load(scaler_path)

            # Restore SHAP explainer using saved background data (not dummy zeros)
            bg_path = path / "shap_background.npy"
            if bg_path.exists():
                background = np.load(bg_path)
            else:
                logger.warning(
                    "No shap_background.npy found — SHAP explainer will use "
                    "scaler mean as background (less accurate). Re-fit to fix."
                )
                # Best available fallback: use scaler mean (all-zeros in scaled space)
                background = np.zeros((1, len(controls)))

            ranker._shap_background = background
            ranker._explainer = shap.LinearExplainer(
                ranker._lr,
                background,
                feature_perturbation="interventional",
            )

        logger.info("Loaded ranker from %s (%d controls)", path, len(controls))
        return ranker
