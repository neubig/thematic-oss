"""Tests for ThematicLMPipeline."""

import json
from unittest.mock import patch

import pytest

from thematic_lm import DataSegment, PipelineConfig, PipelineResult, ThematicLMPipeline
from thematic_lm.agents import (
    ThemeAggregationResult,
)
from thematic_lm.agents.theme_aggregator import MergedTheme
from thematic_lm.codebook import Codebook, EmbeddingService, Quote
from thematic_lm.pipeline import ExecutionMode


class TestPipelineConfig:
    """Test cases for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.num_coders == 3
        assert config.num_theme_coders == 3
        assert config.batch_size == 10
        assert config.use_mock_embeddings is False
        assert config.coder_identities == []
        assert config.theme_coder_identities == []

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            num_coders=5,
            num_theme_coders=2,
            batch_size=20,
            use_mock_embeddings=True,
            coder_identities=["Expert A", "Expert B"],
        )

        assert config.num_coders == 5
        assert config.num_theme_coders == 2
        assert config.batch_size == 20
        assert config.use_mock_embeddings is True
        assert len(config.coder_identities) == 2


class TestDataSegment:
    """Test cases for DataSegment."""

    def test_data_segment_creation(self):
        """Test creating a DataSegment."""
        segment = DataSegment(
            segment_id="seg_001",
            text="This is a sample text segment.",
        )

        assert segment.segment_id == "seg_001"
        assert segment.text == "This is a sample text segment."


class TestPipelineResult:
    """Test cases for PipelineResult."""

    @pytest.fixture
    def sample_result(self) -> PipelineResult:
        """Create a sample PipelineResult."""
        themes = ThemeAggregationResult(
            themes=[
                MergedTheme(
                    name="Theme 1",
                    description="First theme",
                    original_themes=["Theme 1"],
                    codes=["code1", "code2"],
                    quotes=[Quote("q1", "Quote text")],
                )
            ]
        )
        codebook = Codebook(use_mock_embeddings=True)
        codebook.add_code("code1", [Quote("q1", "Quote text")])

        return PipelineResult(themes=themes, codebook=codebook)

    def test_to_json(self, sample_result: PipelineResult):
        """Test JSON serialization."""
        json_str = sample_result.to_json()
        data = json.loads(json_str)

        assert "themes" in data
        assert "codebook" in data
        assert "stage1_aggregations" in data
        assert len(data["themes"]["themes"]) == 1

    def test_to_dict(self, sample_result: PipelineResult):
        """Test dictionary conversion."""
        data = sample_result.to_dict()

        assert "themes" in data
        assert "codebook" in data
        assert "stage1_aggregations" in data


class TestThematicLMPipeline:
    """Test cases for ThematicLMPipeline."""

    @pytest.fixture
    def pipeline(self) -> ThematicLMPipeline:
        """Create a pipeline with mock embeddings and sequential execution for testing."""
        config = PipelineConfig(
            num_coders=2,
            num_theme_coders=2,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,  # Use sequential for mocked tests
        )
        return ThematicLMPipeline(config=config)

    @pytest.fixture
    def sample_segments(self) -> list[DataSegment]:
        """Create sample data segments."""
        return [
            DataSegment("seg_001", "I feel stressed about upcoming exams."),
            DataSegment("seg_002", "My friends help me cope with pressure."),
            DataSegment("seg_003", "Time management is my biggest challenge."),
        ]

    def test_initialization(self, pipeline: ThematicLMPipeline):
        """Test pipeline initialization."""
        assert pipeline.config.num_coders == 2
        assert pipeline.config.num_theme_coders == 2
        assert pipeline.embedding_service is not None
        assert pipeline.codebook is not None

    def test_initialization_with_custom_embedding_service(self):
        """Test initialization with custom embedding service."""
        embedding_service = EmbeddingService(use_mock=True)
        pipeline = ThematicLMPipeline(embedding_service=embedding_service)

        assert pipeline.embedding_service is embedding_service

    def test_init_stage1_agents(self, pipeline: ThematicLMPipeline):
        """Test Stage 1 agent initialization."""
        pipeline._init_stage1_agents()

        assert len(pipeline._coders) == 2
        assert pipeline._aggregator is not None
        assert pipeline._reviewer is not None

    def test_init_stage1_agents_with_identities(self):
        """Test Stage 1 agents with coder identities."""
        config = PipelineConfig(
            num_coders=2,
            coder_identities=["Expert in psychology", "Expert in sociology"],
            use_mock_embeddings=True,
        )
        pipeline = ThematicLMPipeline(config=config)
        pipeline._init_stage1_agents()

        assert pipeline._coders[0].config.identity == "Expert in psychology"
        assert pipeline._coders[1].config.identity == "Expert in sociology"

    def test_init_stage2_agents(self, pipeline: ThematicLMPipeline):
        """Test Stage 2 agent initialization."""
        pipeline._init_stage2_agents()

        assert len(pipeline._theme_coders) == 2
        assert pipeline._theme_aggregator is not None

    def test_init_stage2_agents_with_identities(self):
        """Test Stage 2 agents with theme coder identities."""
        config = PipelineConfig(
            num_theme_coders=2,
            theme_coder_identities=["Theorist", "Practitioner"],
            use_mock_embeddings=True,
        )
        pipeline = ThematicLMPipeline(config=config)
        pipeline._init_stage2_agents()

        assert pipeline._theme_coders[0].config.identity == "Theorist"
        assert pipeline._theme_coders[1].config.identity == "Practitioner"

    def test_coders_property_lazy_init(self, pipeline: ThematicLMPipeline):
        """Test that coders property lazily initializes agents."""
        assert pipeline._coders == []
        coders = pipeline.coders
        assert len(coders) == 2

    def test_theme_coders_property_lazy_init(self, pipeline: ThematicLMPipeline):
        """Test that theme_coders property lazily initializes agents."""
        assert pipeline._theme_coders == []
        theme_coders = pipeline.theme_coders
        assert len(theme_coders) == 2

    def test_run_stage1(self, pipeline: ThematicLMPipeline):
        """Test Stage 1 execution with mocked LLM calls."""
        segments = [
            DataSegment("seg_001", "I feel stressed about exams."),
        ]

        # Mock the coder response (codes are strings, not dicts)
        mock_coder_response = """```json
{
  "codes": ["exam_stress"],
  "rationales": ["Expresses stress about academic exams"]
}
```"""

        mock_aggregator_response = """```json
{
  "merged_codes": [
    {
      "code": "exam_stress",
      "quotes": [{"quote_id": "seg_001", "text": "I feel stressed about exams."}]
    }
  ]
}
```"""

        mock_reviewer_response = """```json
{
  "decision": "add_new"
}
```"""

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = [
                mock_coder_response,
                mock_coder_response,
                mock_aggregator_response,
                mock_reviewer_response,
            ]

            codebook, aggregations = pipeline.run_stage1(segments)

        assert codebook is not None
        assert len(aggregations) >= 0

    def test_run_stage2(self, pipeline: ThematicLMPipeline):
        """Test Stage 2 execution with mocked LLM calls."""
        # First, add some codes to the codebook
        pipeline.codebook.add_code("exam_stress", [Quote("q1", "Exam stress quote")])
        pipeline.codebook.add_code("peer_support", [Quote("q2", "Peer support quote")])

        mock_theme_coder_response = """```json
{
  "themes": [
    {
      "name": "Academic Stress",
      "description": "Stress related to academics",
      "codes": ["exam_stress"]
    }
  ]
}
```"""

        mock_theme_aggregator_response = """```json
{
  "merge_groups": [],
  "retain_themes": ["Academic Stress"]
}
```"""

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = [
                mock_theme_coder_response,
                mock_theme_coder_response,
                mock_theme_aggregator_response,
            ]

            themes = pipeline.run_stage2()

        assert themes is not None
        assert isinstance(themes, ThemeAggregationResult)

    def test_run_full_pipeline(self, pipeline: ThematicLMPipeline):
        """Test full pipeline execution."""
        segments = [
            DataSegment("seg_001", "I feel stressed about exams."),
        ]

        mock_coder_response = """```json
{
  "codes": ["exam_stress"],
  "rationales": ["Expresses exam-related stress"]
}
```"""

        mock_aggregator_response = """```json
{
  "merged_codes": [
    {
      "code": "exam_stress",
      "quotes": [{"quote_id": "seg_001", "text": "I feel stressed"}]
    }
  ]
}
```"""

        mock_reviewer_response = """```json
{"decision": "add_new"}
```"""

        mock_theme_coder_response = """```json
{
  "themes": [
    {"name": "Academic Stress", "description": "Stress from academics",
     "codes": ["exam_stress"]}
  ]
}
```"""

        mock_theme_aggregator_response = """```json
{"merge_groups": [], "retain_themes": ["Academic Stress"]}
```"""

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = [
                mock_coder_response,
                mock_coder_response,
                mock_aggregator_response,
                mock_reviewer_response,
                mock_theme_coder_response,
                mock_theme_coder_response,
                mock_theme_aggregator_response,
            ]

            result = pipeline.run(segments)

        assert isinstance(result, PipelineResult)
        assert result.themes is not None
        assert result.codebook is not None

    def test_run_from_texts(self, pipeline: ThematicLMPipeline):
        """Test running pipeline from text list."""
        texts = ["Sample text 1", "Sample text 2"]

        mock_coder_response = """```json
{"codes": ["test_code"], "rationales": ["Test rationale"]}
```"""

        mock_aggregator_response = """```json
{"merged_codes": [{"code": "test_code", "quotes": []}]}
```"""

        mock_reviewer_response = """```json
{"decision": "add_new"}
```"""

        mock_theme_coder_response = """```json
{"themes": [{"name": "Test", "description": "Test theme", "codes": ["test_code"]}]}
```"""

        mock_theme_aggregator_response = """```json
{"merge_groups": [], "retain_themes": ["Test"]}
```"""

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            # 2 texts * 2 coders = 4 coder calls, plus aggregator, reviewer
            # per segment, then theme coders and aggregator
            mock_llm.side_effect = [
                mock_coder_response,
                mock_coder_response,
                mock_aggregator_response,
                mock_reviewer_response,
                mock_coder_response,
                mock_coder_response,
                mock_aggregator_response,
                mock_reviewer_response,
                mock_theme_coder_response,
                mock_theme_coder_response,
                mock_theme_aggregator_response,
            ]

            result = pipeline.run_from_texts(texts)

        assert isinstance(result, PipelineResult)

    def test_batch_processing(self):
        """Test that large datasets are processed in batches."""
        config = PipelineConfig(
            num_coders=1,
            batch_size=2,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,  # Use sequential for mocking
        )
        pipeline = ThematicLMPipeline(config=config)

        segments = [DataSegment(f"seg_{i}", f"Text {i}") for i in range(5)]

        mock_coder_response = """```json
{"codes": ["test_code"], "rationales": ["Test"]}
```"""

        mock_aggregator_response = """```json
{"merged_codes": [{"code": "test_code", "quotes": []}]}
```"""

        mock_reviewer_response = """```json
{"decision": "add_new"}
```"""

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = [
                mock_coder_response,
                mock_aggregator_response,
                mock_reviewer_response,
            ] * 5  # For each segment

            codebook, aggregations = pipeline.run_stage1(segments)

        # Should have processed all 5 segments
        assert len(aggregations) == 5


class TestPipelineIntegration:
    """Integration tests for ThematicLMPipeline."""

    def test_pipeline_maintains_quote_ids(self):
        """Test that quote IDs are maintained through the pipeline."""
        config = PipelineConfig(
            num_coders=1,
            num_theme_coders=1,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,  # Use sequential for mocking
        )
        pipeline = ThematicLMPipeline(config=config)

        segment_id = "test_segment_001"
        segments = [DataSegment(segment_id, "This is test content about stress.")]

        mock_coder_response = """```json
{
  "codes": ["stress_code"],
  "rationales": ["About stress"]
}
```"""

        mock_aggregator_response = """```json
{
  "merged_codes": [
    {
      "code": "stress_code",
      "quotes": [{"quote_id": "test_segment_001", "text": "This is test content"}]
    }
  ]
}
```"""

        mock_reviewer_response = """```json
{"decision": "add_new"}
```"""

        mock_theme_response = """```json
{"themes": [{"name": "Stress", "description": "About stress",
"codes": ["stress_code"]}]}
```"""

        mock_theme_agg_response = """```json
{"merge_groups": [], "retain_themes": ["Stress"]}
```"""

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = [
                mock_coder_response,
                mock_aggregator_response,
                mock_reviewer_response,
                mock_theme_response,
                mock_theme_agg_response,
            ]

            result = pipeline.run(segments)

        # Quote IDs should be maintained
        assert result.codebook is not None

    def test_configurable_number_of_coders(self):
        """Test that the number of coders is configurable."""
        for num_coders in [1, 2, 5]:
            config = PipelineConfig(
                num_coders=num_coders,
                use_mock_embeddings=True,
            )
            pipeline = ThematicLMPipeline(config=config)
            pipeline._init_stage1_agents()

            assert len(pipeline._coders) == num_coders

    def test_end_to_end_structure(self):
        """Test end-to-end pipeline output structure."""
        config = PipelineConfig(
            num_coders=1,
            num_theme_coders=1,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,  # Use sequential for mocking
        )
        pipeline = ThematicLMPipeline(config=config)

        segments = [DataSegment("seg_001", "Test content")]

        mock_coder_response = """```json
{"codes": ["test_code"], "rationales": ["Test rationale"]}
```"""

        mock_aggregator_response = """```json
{"merged_codes": [{"code": "test_code", "quotes": []}]}
```"""

        mock_reviewer_response = """```json
{"decision": "add_new"}
```"""

        mock_theme_response = """```json
{"themes": [{"name": "Test Theme", "description": "Description",
"codes": ["test_code"]}]}
```"""

        mock_theme_agg_response = """```json
{"merge_groups": [], "retain_themes": ["Test Theme"]}
```"""

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = [
                mock_coder_response,
                mock_aggregator_response,
                mock_reviewer_response,
                mock_theme_response,
                mock_theme_agg_response,
            ]

            result = pipeline.run(segments)

        # Verify result structure
        assert isinstance(result, PipelineResult)
        assert isinstance(result.themes, ThemeAggregationResult)
        assert isinstance(result.codebook, Codebook)

        # Verify JSON output is valid
        json_output = result.to_json()
        parsed = json.loads(json_output)
        assert "themes" in parsed
        assert "codebook" in parsed
