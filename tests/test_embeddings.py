"""Tests for EmbeddingService."""

import numpy as np
import pytest

from thematic_lm.codebook.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    @pytest.fixture
    def service(self) -> EmbeddingService:
        """Create an embedding service for tests."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")

    def test_embed_single(self, service: EmbeddingService):
        """Test embedding a single text."""
        embedding = service.embed_single("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0

    def test_embed_batch(self, service: EmbeddingService):
        """Test embedding multiple texts."""
        texts = ["first text", "second text", "third text"]
        embeddings = service.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0

    def test_embed_empty_list(self, service: EmbeddingService):
        """Test embedding empty list returns empty array."""
        embeddings = service.embed([])

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 0

    def test_cosine_similarity(self, service: EmbeddingService):
        """Test cosine similarity computation."""
        query = service.embed_single("climate change")
        candidates = service.embed(
            ["global warming", "ice cream", "environmental impact"]
        )

        similarities = service.cosine_similarity(query, candidates)

        assert len(similarities) == 3
        # "global warming" should be more similar to "climate change" than "ice cream"
        assert similarities[0] > similarities[1]

    def test_cosine_similarity_empty_candidates(self, service: EmbeddingService):
        """Test cosine similarity with empty candidates."""
        query = service.embed_single("test")
        empty = np.array([], dtype=np.float32).reshape(0, 384)

        similarities = service.cosine_similarity(query, empty)

        assert len(similarities) == 0

    def test_find_similar(self, service: EmbeddingService):
        """Test finding similar embeddings."""
        candidates = service.embed(
            ["apple fruit", "banana fruit", "car vehicle", "climate change"]
        )
        query = service.embed_single("orange fruit")

        results = service.find_similar(query, candidates, top_k=2)

        assert len(results) == 2
        # Results should be sorted by similarity descending
        assert results[0][1] >= results[1][1]
        # Fruit items should be most similar
        assert results[0][0] in [0, 1]

    def test_find_similar_top_k_larger_than_candidates(self, service: EmbeddingService):
        """Test find_similar when top_k exceeds number of candidates."""
        candidates = service.embed(["one", "two"])
        query = service.embed_single("one")

        results = service.find_similar(query, candidates, top_k=10)

        assert len(results) == 2

    def test_find_similar_empty_candidates(self, service: EmbeddingService):
        """Test find_similar with empty candidates."""
        query = service.embed_single("test")
        empty = np.array([], dtype=np.float32).reshape(0, 384)

        results = service.find_similar(query, empty, top_k=5)

        assert results == []

    def test_lazy_model_loading(self):
        """Test that model is loaded lazily."""
        service = EmbeddingService()

        assert service._model is None
        _ = service.model
        assert service._model is not None
