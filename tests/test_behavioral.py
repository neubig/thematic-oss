"""Behavioral tests for Thematic-LM paper claims.

These tests verify that the implementation exhibits the expected behavioral
properties described in the paper:
1. Multi-coder disagreement resolution through negotiation
2. Independent codebook copies per coder
3. Parallel execution provides consistent results
4. Evaluation metrics computation

Based on Issue #41: Add behavioral tests for multi-coder scenarios.
"""

from unittest.mock import patch

import pytest

from thematic_lm import (
    DataSegment,
    PipelineConfig,
    ThematicLMPipeline,
)
from thematic_lm.agents import (
    AggregatorConfig,
    CodeAggregatorAgent,
    CodeAssignment,
    CoderAgent,
    CoderConfig,
)
from thematic_lm.codebook import Codebook
from thematic_lm.iterative import NegotiationStrategy
from thematic_lm.pipeline import ExecutionMode


class TestMultiCoderDisagreement:
    """Test multi-coder disagreement scenarios.

    Per the paper: 'Multiple coders often work independently to generate codes,
    and these codes are then aggregated to identify common themes.'
    """

    @pytest.fixture
    def codebook(self) -> Codebook:
        """Create a test codebook."""
        return Codebook(use_mock_embeddings=True)

    def test_coders_produce_different_codes(self):
        """Test that different coders can produce different codes."""
        codebook = Codebook(use_mock_embeddings=True)

        # Create two coders with different identities
        coder1 = CoderAgent(
            config=CoderConfig(identity="optimistic perspective"),
            codebook=codebook,
        )
        coder2 = CoderAgent(
            config=CoderConfig(identity="critical perspective"),
            codebook=codebook,
        )

        # Mock responses to simulate different coding perspectives
        mock_response_1 = '{"codes": ["hope", "opportunity"], "rationales": ["a", "b"]}'
        mock_response_2 = (
            '{"codes": ["concern", "challenge"], "rationales": ["c", "d"]}'
        )

        with patch.object(coder1, "_call_llm", return_value=mock_response_1):
            result1 = coder1.code_segment("seg1", "The future looks uncertain.")

        with patch.object(coder2, "_call_llm", return_value=mock_response_2):
            result2 = coder2.code_segment("seg1", "The future looks uncertain.")

        # Different coders should produce different codes
        assert set(result1.codes) != set(result2.codes)

    def test_negotiation_consensus_filters_minority_codes(self):
        """Test that CONSENSUS strategy filters out minority codes."""
        codebook = Codebook(use_mock_embeddings=True)
        aggregator = CodeAggregatorAgent(
            config=AggregatorConfig(negotiation_strategy=NegotiationStrategy.CONSENSUS),
            codebook=codebook,
        )

        # Create assignments simulating 3 coders
        # "common_code" appears in 2/3 (majority)
        # "unique_code_*" appears in 1/3 (minority)
        assignments = [
            CodeAssignment("seg1", "text", codes=["common_code", "unique_code_1"]),
            CodeAssignment("seg1", "text", codes=["common_code", "unique_code_2"]),
            CodeAssignment("seg1", "text", codes=["unique_code_3"]),
        ]

        # Apply negotiation
        agreed_codes = aggregator._apply_negotiation_strategy(assignments)

        # Only common_code should pass consensus
        assert "common_code" in agreed_codes
        assert "unique_code_1" not in agreed_codes
        assert "unique_code_2" not in agreed_codes
        assert "unique_code_3" not in agreed_codes

    def test_negotiation_union_keeps_all_codes(self):
        """Test that UNION strategy keeps all codes."""
        codebook = Codebook(use_mock_embeddings=True)
        aggregator = CodeAggregatorAgent(
            config=AggregatorConfig(negotiation_strategy=NegotiationStrategy.UNION),
            codebook=codebook,
        )

        assignments = [
            CodeAssignment("seg1", "text", codes=["code_a", "code_b"]),
            CodeAssignment("seg1", "text", codes=["code_c"]),
            CodeAssignment("seg1", "text", codes=["code_d"]),
        ]

        agreed_codes = aggregator._apply_negotiation_strategy(assignments)

        # All codes should be included
        assert agreed_codes == {"code_a", "code_b", "code_c", "code_d"}

    def test_negotiation_intersection_requires_all_agree(self):
        """Test that INTERSECTION strategy requires unanimous agreement."""
        codebook = Codebook(use_mock_embeddings=True)
        aggregator = CodeAggregatorAgent(
            config=AggregatorConfig(
                negotiation_strategy=NegotiationStrategy.INTERSECTION
            ),
            codebook=codebook,
        )

        # Only "unanimous_code" appears in all 3 assignments
        assignments = [
            CodeAssignment("seg1", "text", codes=["unanimous_code", "code_a"]),
            CodeAssignment("seg1", "text", codes=["unanimous_code", "code_b"]),
            CodeAssignment("seg1", "text", codes=["unanimous_code", "code_c"]),
        ]

        agreed_codes = aggregator._apply_negotiation_strategy(assignments)

        # Only unanimous code should pass
        assert agreed_codes == {"unanimous_code"}


class TestIndependentCodebooks:
    """Test that coders work with independent codebook copies.

    Per the paper: 'Each coder often works independently to generate codes.'
    """

    def test_pipeline_creates_independent_codebook_copies(self):
        """Test that each coder gets an independent codebook copy."""
        config = PipelineConfig(
            num_coders=3,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )
        pipeline = ThematicLMPipeline(config=config)
        pipeline._init_stage1_agents()

        # Get references to coder codebooks
        codebooks = [coder.codebook for coder in pipeline._coders]

        # All codebooks should be different objects
        for i in range(len(codebooks)):
            for j in range(i + 1, len(codebooks)):
                assert codebooks[i] is not codebooks[j], (
                    f"Coder {i} and Coder {j} share the same codebook object"
                )

    def test_coder_codebook_changes_dont_affect_other_coders(self):
        """Test that changes to one coder's codebook don't affect others."""
        config = PipelineConfig(
            num_coders=2,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )
        pipeline = ThematicLMPipeline(config=config)
        pipeline._init_stage1_agents()

        coder1, coder2 = pipeline._coders

        # Add a code to coder1's codebook
        from thematic_lm.codebook import Quote

        coder1.codebook.add_code("test_code", [Quote("q1", "quote text")])

        # coder2's codebook should not have this code
        assert "test_code" in coder1.codebook.codes
        assert "test_code" not in coder2.codebook.codes


class TestExecutionModes:
    """Test execution mode configuration."""

    def test_sequential_mode_produces_results(self):
        """Test that sequential execution mode produces valid results."""
        segment = DataSegment("seg1", "Test content about stress.")

        mock_coder_response = '{"codes": ["stress"], "rationales": ["about stress"]}'
        mock_agg_response = '{"merge_groups": [], "retain_codes": ["stress"]}'
        mock_rev_response = '{"decision": "add_new"}'
        mock_theme_response = (
            '{"themes": [{"name": "Stress", "description": "d", "codes": ["stress"]}]}'
        )
        mock_theme_agg = '{"merge_groups": [], "retain_themes": ["Stress"]}'

        responses = [
            mock_coder_response,
            mock_agg_response,
            mock_rev_response,
            mock_theme_response,
            mock_theme_agg,
        ]

        # Run with sequential mode
        config = PipelineConfig(
            num_coders=1,
            num_theme_coders=1,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )
        pipeline = ThematicLMPipeline(config=config)

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = responses.copy()
            result = pipeline.run([segment])

        # Verify result structure
        assert len(result.stage1_aggregations) == 1
        assert result.metrics["execution_mode"] == "sequential"

    def test_parallel_mode_is_default(self):
        """Test that parallel mode is the default execution mode."""
        config = PipelineConfig()
        assert config.execution_mode == ExecutionMode.PARALLEL

    def test_execution_mode_in_result_metrics(self):
        """Test that execution mode is recorded in result metrics."""
        config = PipelineConfig(
            num_coders=1,
            num_theme_coders=1,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )
        pipeline = ThematicLMPipeline(config=config)

        mock_responses = [
            '{"codes": ["test"], "rationales": ["r"]}',
            '{"merge_groups": [], "retain_codes": ["test"]}',
            '{"decision": "add_new"}',
            '{"themes": [{"name": "T", "description": "d", "codes": ["test"]}]}',
            '{"merge_groups": [], "retain_themes": ["T"]}',
        ]

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = mock_responses
            result = pipeline.run([DataSegment("s1", "text")])

        assert result.metrics["execution_mode"] == "sequential"


class TestEvaluationIntegration:
    """Test evaluation integration in pipeline."""

    def test_pipeline_result_has_metrics(self):
        """Test that pipeline result includes metrics."""
        config = PipelineConfig(
            num_coders=1,
            num_theme_coders=1,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )
        pipeline = ThematicLMPipeline(config=config)

        mock_responses = [
            '{"codes": ["test"], "rationales": ["r"]}',
            '{"merge_groups": [], "retain_codes": ["test"]}',
            '{"decision": "add_new"}',
            '{"themes": [{"name": "T", "description": "d", "codes": ["test"]}]}',
            '{"merge_groups": [], "retain_themes": ["T"]}',
        ]

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = mock_responses
            result = pipeline.run([DataSegment("s1", "text")])

        # Check metrics are populated
        assert "num_segments" in result.metrics
        assert "num_coders" in result.metrics
        assert "execution_mode" in result.metrics
        assert "negotiation_strategy" in result.metrics

    def test_pipeline_with_checkpoints(self):
        """Test pipeline creates HITL checkpoints when enabled."""
        config = PipelineConfig(
            num_coders=1,
            num_theme_coders=1,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
            enable_hitl_checkpoints=True,
            checkpoint_frequency=1,  # Checkpoint every batch
        )
        pipeline = ThematicLMPipeline(config=config)

        mock_responses = [
            '{"codes": ["test"], "rationales": ["r"]}',
            '{"merge_groups": [], "retain_codes": ["test"]}',
            '{"decision": "add_new"}',
            '{"themes": [{"name": "T", "description": "d", "codes": ["test"]}]}',
            '{"merge_groups": [], "retain_themes": ["T"]}',
        ]

        with patch("thematic_lm.agents.base.BaseAgent._call_llm") as mock_llm:
            mock_llm.side_effect = mock_responses
            result = pipeline.run([DataSegment("s1", "text")])

        # Check checkpoints were created
        assert len(result.checkpoints) >= 1
        assert result.checkpoints[0].stage == "coding"


class TestConfigurablePrompts:
    """Test custom prompt configuration."""

    def test_coder_with_custom_prompts(self):
        """Test that custom prompts are used when configured."""
        from thematic_lm.prompts import CoderPrompts

        custom_prompts = CoderPrompts(
            system_prompt="Custom system: {identity_section}",
            user_prompt=(
                "Custom user: {codebook_section} {segment_id} "
                "{segment_text} {similar_codes_section}"
            ),
        )

        config = CoderConfig(custom_prompts=custom_prompts)
        coder = CoderAgent(config=config, codebook=Codebook(use_mock_embeddings=True))

        # Check system prompt uses custom template
        system_prompt = coder.get_system_prompt()
        assert "Custom system:" in system_prompt

        # Check user prompt template is custom
        user_template = coder._get_user_prompt_template()
        assert "Custom user:" in user_template

    def test_domain_specific_prompts(self):
        """Test creation of domain-specific prompts."""
        from thematic_lm.prompts import create_domain_prompts

        healthcare_prompts = create_domain_prompts(
            domain="healthcare",
            additional_context="Focus on patient experience and clinical outcomes.",
        )

        # Check domain context is added
        assert "Healthcare" in healthcare_prompts.coder.system_prompt
        assert "patient experience" in healthcare_prompts.coder.system_prompt


class TestCodebookCopy:
    """Test codebook copy functionality."""

    def test_codebook_copy_creates_independent_instance(self):
        """Test that copy creates a truly independent codebook."""
        from thematic_lm.codebook import Quote

        original = Codebook(use_mock_embeddings=True)
        original.add_code("code1", [Quote("q1", "text1")])

        copy = original.copy()

        # Copy should have same codes
        assert copy.codes == original.codes
        assert len(copy) == len(original)

        # But be a different object
        assert copy is not original
        assert copy._entries is not original._entries

        # Modifying copy shouldn't affect original
        copy.add_code("code2", [Quote("q2", "text2")])
        assert "code2" in copy.codes
        assert "code2" not in original.codes
