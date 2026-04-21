"""
Embedder — wraps a SentenceTransformer model and provides
encode methods for entities and framework controls.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Thin wrapper around SentenceTransformer that encodes
    (title, text) pairs as a single concatenated embedding.

    The title receives slightly more weight by being prepended
    with a separator token, consistent with the original NEXUS
    design where the section heading carries strong topical signal.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def _format(self, title: str, text: str) -> str:
        """Combine title and text into a single string for encoding."""
        title = title.strip()
        text = text.strip()
        if title:
            return f"{title}: {text}"
        return text

    def encode_one(self, title: str, text: str) -> np.ndarray:
        """Encode a single (title, text) pair."""
        return self._model.encode(
            self._format(title, text),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of (title, text) pairs.

        Returns an (N, D) float32 array of L2-normalized embeddings.
        """
        sentences = [self._format(t, tx) for t, tx in pairs]
        return self._model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
