"""Tests for EmbeddingService."""

import numpy as np
import pytest

from thematic_analysis.codebook.embeddings import EmbeddingService, MockEmbeddingService


class TestMockEmbeddingService:
    """Test cases for MockEmbeddingService (fast unit tests)."""

    @pytest.fixture
    def service(self) -> MockEmbeddingService:
        """Create a mock embedding service."""
        return MockEmbeddingService()

    def test_embed_single(self, service: MockEmbeddingService):
        """Test embedding a single text."""
        embedding = service.embed_single("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 384  # Default dim

    def test_embed_batch(self, service: MockEmbeddingService):
        """Test embedding multiple texts."""
        texts = ["first text", "second text", "third text"]
        embeddings = service.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384

    def test_embed_empty_list(self, service: MockEmbeddingService):
        """Test embedding empty list returns empty array."""
        embeddings = service.embed([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 384)

    def test_deterministic_embeddings(self, service: MockEmbeddingService):
        """Test that same text produces same embedding."""
        emb1 = service.embed_single("test")
        emb2 = service.embed_single("test")

        np.testing.assert_array_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self, service: MockEmbeddingService):
        """Test that different texts produce different embeddings."""
        emb1 = service.embed_single("apple")
        emb2 = service.embed_single("banana")

        assert not np.allclose(emb1, emb2)


class TestEmbeddingServiceMockMode:
    """Test EmbeddingService with mock mode enabled (fast)."""

    @pytest.fixture
    def service(self) -> EmbeddingService:
        """Create an embedding service in mock mode."""
        return EmbeddingService(use_mock=True)

    def test_embed_single(self, service: EmbeddingService):
        """Test embedding a single text."""
        embedding = service.embed_single("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1

    def test_embed_batch(self, service: EmbeddingService):
        """Test embedding multiple texts."""
        texts = ["first text", "second text", "third text"]
        embeddings = service.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3

    def test_compute_similarity(self, service: EmbeddingService):
        """Test compute_similarity method."""
        sim = service.compute_similarity("test", "test")

        # Same text should have similarity close to 1
        assert sim > 0.99

    def test_find_similar(self, service: EmbeddingService):
        """Test finding similar embeddings."""
        candidates = service.embed(["one", "two", "three"])
        query = service.embed_single("query")

        results = service.find_similar(query, candidates, top_k=2)

        assert len(results) == 2
        # Results should be sorted by similarity descending
        assert results[0][1] >= results[1][1]


class TestEmbeddingServiceStatic:
    """Test static methods that don't require model loading."""

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32
        )

        similarities = EmbeddingService.cosine_similarity(query, candidates)

        assert len(similarities) == 3
        assert similarities[0] > similarities[1]  # First is most similar

    def test_cosine_similarity_empty_candidates(self):
        """Test cosine similarity with empty candidates."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        empty = np.array([], dtype=np.float32).reshape(0, 2)

        similarities = EmbeddingService.cosine_similarity(query, empty)

        assert len(similarities) == 0

    def test_find_similar_static(self):
        """Test static find_similar method."""
        candidates = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)

        results = EmbeddingService.find_similar_static(query, candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == 0  # Most similar is the identical vector


@pytest.mark.integration
class TestEmbeddingServiceReal:
    """Integration tests using real Sentence Transformer model (slow)."""

    @pytest.fixture
    def service(self) -> EmbeddingService:
        """Create an embedding service with real model."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2", use_mock=False)

    def test_embed_single(self, service: EmbeddingService):
        """Test embedding a single text with real model."""
        embedding = service.embed_single("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0

    def test_semantic_similarity(self, service: EmbeddingService):
        """Test that semantically similar texts have higher similarity."""
        sim_related = service.compute_similarity("climate change", "global warming")
        sim_unrelated = service.compute_similarity("climate change", "banana fruit")

        assert sim_related > sim_unrelated

    def test_lazy_model_loading(self):
        """Test that model is loaded lazily."""
        service = EmbeddingService(use_mock=False)

        assert service._model is None
        _ = service.model
        assert service._model is not None
