"""Embedding service for code similarity using Sentence Transformers."""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service for generating embeddings and computing similarity.

    Uses Sentence Transformers for embedding generation and cosine similarity
    for finding similar codes in the codebook.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.

        Args:
            model_name: Name of the Sentence Transformer model to use.
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32)

        embeddings: NDArray[np.float32] = self.model.encode(
            texts, convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding array with shape (embedding_dim,).
        """
        embeddings: NDArray[np.float32] = self.model.encode(
            [text], convert_to_numpy=True
        )
        return embeddings[0].astype(np.float32)

    @staticmethod
    def cosine_similarity(
        query: NDArray[np.float32], candidates: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between query and candidate embeddings.

        Args:
            query: Query embedding with shape (embedding_dim,).
            candidates: Candidate embeddings with shape (n, embedding_dim).

        Returns:
            Array of similarity scores with shape (n,).
        """
        if len(candidates) == 0:
            return np.array([], dtype=np.float32)

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        candidates_norm = candidates / (
            np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8
        )

        similarities: NDArray[np.float32] = candidates_norm @ query_norm
        return similarities.astype(np.float32)

    def find_similar(
        self,
        query_embedding: NDArray[np.float32],
        candidate_embeddings: NDArray[np.float32],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find the top-k most similar candidates to the query.

        Args:
            query_embedding: Query embedding with shape (embedding_dim,).
            candidate_embeddings: Candidate embeddings with shape (n, embedding_dim).
            top_k: Number of top similar candidates to return.

        Returns:
            List of (index, similarity_score) tuples sorted by similarity descending.
        """
        if len(candidate_embeddings) == 0:
            return []

        similarities = self.cosine_similarity(query_embedding, candidate_embeddings)
        top_k = min(top_k, len(similarities))

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
