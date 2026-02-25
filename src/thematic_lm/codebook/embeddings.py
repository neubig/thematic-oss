"""Embedding service for code similarity using Sentence Transformers."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class MockEmbeddingService:
    """Mock embedding service for fast unit tests.

    Uses deterministic hash-based embeddings instead of real model inference.
    """

    def __init__(self, embedding_dim: int = 384):
        """Initialize mock service.

        Args:
            embedding_dim: Dimension of mock embeddings.
        """
        self.embedding_dim = embedding_dim

    def _hash_to_embedding(self, text: str) -> NDArray[np.float32]:
        """Convert text to deterministic embedding via hashing."""
        # Create deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Expand hash to embedding dimension
        np.random.seed(int.from_bytes(hash_bytes[:4], "big"))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        # Normalize
        norm = float(np.linalg.norm(embedding)) + 1e-8
        return (embedding / norm).astype(np.float32)

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate mock embeddings for texts."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        return np.array([self._hash_to_embedding(t) for t in texts], dtype=np.float32)

    def embed_single(self, text: str) -> NDArray[np.float32]:
        """Generate mock embedding for single text."""
        return self._hash_to_embedding(text)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        emb1 = self.embed_single(text1)
        emb2 = self.embed_single(text2)
        return float(EmbeddingService.cosine_similarity(emb1, emb2.reshape(1, -1))[0])

    def find_similar(
        self,
        query_embedding: NDArray[np.float32],
        candidate_embeddings: NDArray[np.float32],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find similar embeddings."""
        return EmbeddingService.find_similar_static(
            query_embedding, candidate_embeddings, top_k
        )


class EmbeddingService:
    """Service for generating embeddings and computing similarity.

    Uses Sentence Transformers for embedding generation and cosine similarity
    for finding similar codes in the codebook.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_mock: bool = False,
    ):
        """Initialize the embedding service.

        Args:
            model_name: Name of the Sentence Transformer model to use.
            use_mock: If True, use mock embeddings (fast, for testing).
        """
        self.model_name = model_name
        self.use_mock = use_mock
        self._model: SentenceTransformer | None = None
        self._mock: MockEmbeddingService | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def mock(self) -> MockEmbeddingService:
        """Get mock embedding service."""
        if self._mock is None:
            self._mock = MockEmbeddingService()
        return self._mock

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        if self.use_mock:
            return self.mock.embed(texts)

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
        if self.use_mock:
            return self.mock.embed_single(text)

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

    @staticmethod
    def find_similar_static(
        query_embedding: NDArray[np.float32],
        candidate_embeddings: NDArray[np.float32],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Static method to find similar embeddings.

        Args:
            query_embedding: Query embedding.
            candidate_embeddings: Candidate embeddings.
            top_k: Number of top results.

        Returns:
            List of (index, similarity) tuples.
        """
        if len(candidate_embeddings) == 0:
            return []

        similarities = EmbeddingService.cosine_similarity(
            query_embedding, candidate_embeddings
        )
        top_k = min(top_k, len(similarities))

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

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
        return self.find_similar_static(query_embedding, candidate_embeddings, top_k)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two text strings.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0 and 1.
        """
        if self.use_mock:
            return self.mock.compute_similarity(text1, text2)

        emb1 = self.embed_single(text1)
        emb2 = self.embed_single(text2)
        return float(self.cosine_similarity(emb1, emb2.reshape(1, -1))[0])
