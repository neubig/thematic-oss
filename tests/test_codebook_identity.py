"""Tests for codebook object identity (regression tests for falsy codebook bug)."""

import pytest

from thematic_analysis.agents import (
    CodeAggregatorAgent,
    CoderAgent,
    ReviewerAgent,
    ThemeCoderAgent,
)
from thematic_analysis.codebook import Codebook


class TestCodebookIdentityBug:
    """
    Regression tests for the falsy codebook bug.

    The bug: When an empty Codebook was passed to an agent, the expression
    `self.codebook = codebook or Codebook()` would create a NEW Codebook
    because empty codebooks are falsy (len=0, so bool=False).

    This caused the pipeline's codebook to not be updated by the reviewer.
    """

    def test_coder_agent_preserves_codebook_identity(self):
        """CoderAgent should use the exact codebook passed to it."""
        codebook = Codebook(use_mock_embeddings=True)
        original_id = id(codebook)

        agent = CoderAgent(codebook=codebook)

        assert id(agent.codebook) == original_id
        assert agent.codebook is codebook

    def test_reviewer_agent_preserves_codebook_identity(self):
        """ReviewerAgent should use the exact codebook passed to it."""
        codebook = Codebook(use_mock_embeddings=True)
        original_id = id(codebook)

        agent = ReviewerAgent(codebook=codebook)

        assert id(agent.codebook) == original_id
        assert agent.codebook is codebook

    def test_aggregator_agent_preserves_codebook_identity(self):
        """CodeAggregatorAgent should use the exact codebook passed to it."""
        codebook = Codebook(use_mock_embeddings=True)
        original_id = id(codebook)

        agent = CodeAggregatorAgent(codebook=codebook)

        assert id(agent.codebook) == original_id
        assert agent.codebook is codebook

    def test_theme_coder_agent_preserves_codebook_identity(self):
        """ThemeCoderAgent should use the exact codebook passed to it."""
        codebook = Codebook(use_mock_embeddings=True)
        original_id = id(codebook)

        agent = ThemeCoderAgent(codebook=codebook)

        assert id(agent.codebook) == original_id
        assert agent.codebook is codebook

    def test_empty_codebook_is_falsy(self):
        """Verify that empty codebooks are indeed falsy (the root cause)."""
        codebook = Codebook(use_mock_embeddings=True)
        assert len(codebook) == 0
        assert bool(codebook) is False

    def test_pipeline_shares_codebook_with_reviewer(self):
        """Pipeline and reviewer should share the same codebook object."""
        from thematic_analysis import PipelineConfig, ThematicLMPipeline
        from thematic_analysis.pipeline import ExecutionMode

        config = PipelineConfig(
            num_coders=2,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )

        pipeline = ThematicLMPipeline(config=config)
        pipeline._init_stage1_agents()

        assert pipeline.codebook is pipeline._reviewer.codebook
        assert id(pipeline.codebook) == id(pipeline._reviewer.codebook)

    def test_reviewer_updates_shared_codebook(self):
        """When reviewer adds codes, they should appear in pipeline's codebook."""
        from thematic_analysis import PipelineConfig, ThematicLMPipeline
        from thematic_analysis.agents import ReviewDecision, ReviewResult
        from thematic_analysis.codebook import Quote
        from thematic_analysis.pipeline import ExecutionMode

        config = PipelineConfig(
            num_coders=2,
            use_mock_embeddings=True,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )

        pipeline = ThematicLMPipeline(config=config)
        pipeline._init_stage1_agents()

        # Verify initial state
        assert len(pipeline.codebook) == 0
        assert len(pipeline._reviewer.codebook) == 0

        # Simulate reviewer adding a code
        review_result = ReviewResult(
            code="test_code",
            decision=ReviewDecision.ADD_NEW,
            rationale="Test",
            quotes=[Quote(quote_id="seg1", text="test quote")],
        )
        pipeline._reviewer.apply_review(review_result)

        # Both should now have 1 code
        assert len(pipeline._reviewer.codebook) == 1
        assert len(pipeline.codebook) == 1

        # They should be the same object
        assert pipeline.codebook.entries[0].code == "test_code"
