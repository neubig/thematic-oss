"""Tests for credibility and confirmability evaluation."""

import json
from unittest.mock import patch

import pytest

from thematic_analysis.evaluation import (
    CredibilityConfig,
    CredibilityResult,
    EvaluatorAgent,
    QuoteConsistency,
    ThemeConsistency,
)


class TestCredibilityConfig:
    """Test cases for CredibilityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CredibilityConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CredibilityConfig(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


class TestQuoteConsistency:
    """Test cases for QuoteConsistency."""

    def test_creation(self):
        """Test QuoteConsistency creation."""
        result = QuoteConsistency(
            quote_id="q1",
            quote_text="Sample quote text",
            is_consistent=True,
            reasoning="Quote supports the theme",
        )
        assert result.quote_id == "q1"
        assert result.quote_text == "Sample quote text"
        assert result.is_consistent is True
        assert "supports" in result.reasoning


class TestThemeConsistency:
    """Test cases for ThemeConsistency."""

    def test_creation(self):
        """Test ThemeConsistency creation."""
        result = ThemeConsistency(
            theme_name="Test Theme",
            theme_description="A test theme description",
            consistent_count=3,
            total_count=5,
        )
        assert result.theme_name == "Test Theme"
        assert result.consistency_score == 0.6

    def test_consistency_score_empty(self):
        """Test consistency score with no quotes."""
        result = ThemeConsistency(
            theme_name="Empty Theme",
            theme_description="No quotes",
            consistent_count=0,
            total_count=0,
        )
        assert result.consistency_score == 0.0

    def test_consistency_score_all_consistent(self):
        """Test consistency score with all consistent quotes."""
        result = ThemeConsistency(
            theme_name="Perfect Theme",
            theme_description="All consistent",
            consistent_count=10,
            total_count=10,
        )
        assert result.consistency_score == 1.0


class TestCredibilityResult:
    """Test cases for CredibilityResult."""

    @pytest.fixture
    def sample_result(self) -> CredibilityResult:
        """Create a sample result for testing."""
        return CredibilityResult(
            theme_results=[
                ThemeConsistency(
                    theme_name="Theme 1",
                    theme_description="First theme",
                    consistent_count=8,
                    total_count=10,
                ),
                ThemeConsistency(
                    theme_name="Theme 2",
                    theme_description="Second theme",
                    consistent_count=6,
                    total_count=10,
                ),
            ]
        )

    def test_overall_score(self, sample_result: CredibilityResult):
        """Test overall credibility score calculation."""
        # 14/20 = 0.7
        assert sample_result.overall_score == 0.7

    def test_overall_score_empty(self):
        """Test overall score with no themes."""
        result = CredibilityResult()
        assert result.overall_score == 0.0

    def test_num_themes(self, sample_result: CredibilityResult):
        """Test number of themes."""
        assert sample_result.num_themes == 2

    def test_total_quotes_evaluated(self, sample_result: CredibilityResult):
        """Test total quotes count."""
        assert sample_result.total_quotes_evaluated == 20

    def test_to_dict(self, sample_result: CredibilityResult):
        """Test dictionary conversion."""
        data = sample_result.to_dict()
        assert "overall_score" in data
        assert "theme_results" in data
        assert data["overall_score"] == 0.7
        assert len(data["theme_results"]) == 2

    def test_to_json(self, sample_result: CredibilityResult):
        """Test JSON conversion."""
        json_str = sample_result.to_json()
        data = json.loads(json_str)
        assert data["overall_score"] == 0.7


class TestEvaluatorAgent:
    """Test cases for EvaluatorAgent."""

    @pytest.fixture
    def agent(self) -> EvaluatorAgent:
        """Create an evaluator agent for testing."""
        return EvaluatorAgent()

    def test_initialization(self, agent: EvaluatorAgent):
        """Test agent initialization."""
        assert agent.config.model == "gpt-4o-mini"
        assert agent.config.temperature == 0.0

    def test_build_prompt(self, agent: EvaluatorAgent):
        """Test prompt building."""
        prompt = agent._build_prompt(
            theme_name="Climate Anxiety",
            theme_desc="Feelings of worry about climate change",
            quote="I feel anxious about the future of our planet.",
        )
        assert "Climate Anxiety" in prompt
        assert "Feelings of worry" in prompt
        assert "anxious about the future" in prompt

    def test_parse_response_valid_json(self, agent: EvaluatorAgent):
        """Test parsing valid JSON response."""
        response = '{"is_consistent": true, "reasoning": "Quote supports theme"}'
        is_consistent, reasoning = agent._parse_response(response)
        assert is_consistent is True
        assert "supports" in reasoning

    def test_parse_response_markdown_json(self, agent: EvaluatorAgent):
        """Test parsing JSON in markdown code block."""
        response = """```json
{"is_consistent": false, "reasoning": "Quote is unrelated"}
```"""
        is_consistent, reasoning = agent._parse_response(response)
        assert is_consistent is False
        assert "unrelated" in reasoning

    def test_parse_response_invalid_json(self, agent: EvaluatorAgent):
        """Test parsing invalid JSON response."""
        response = "This is not valid JSON"
        is_consistent, reasoning = agent._parse_response(response)
        assert is_consistent is False
        assert "Failed to parse" in reasoning

    def test_evaluate_quote(self, agent: EvaluatorAgent):
        """Test evaluating a single quote."""
        with patch.object(
            agent,
            "_call_llm",
            return_value='{"is_consistent": true, "reasoning": "Good match"}',
        ):
            result = agent.evaluate_quote(
                theme_name="Test Theme",
                theme_description="A test description",
                quote_id="q1",
                quote_text="Sample quote",
            )

            assert result.quote_id == "q1"
            assert result.is_consistent is True
            assert "Good match" in result.reasoning

    def test_evaluate_theme(self, agent: EvaluatorAgent):
        """Test evaluating a theme with multiple quotes."""
        # Mock responses - 2 consistent, 1 inconsistent
        responses = [
            '{"is_consistent": true, "reasoning": "Good"}',
            '{"is_consistent": true, "reasoning": "Good"}',
            '{"is_consistent": false, "reasoning": "Bad"}',
        ]
        with patch.object(agent, "_call_llm", side_effect=responses):
            result = agent.evaluate_theme(
                theme_name="Test Theme",
                theme_description="A test description",
                quotes=[
                    ("q1", "Quote 1"),
                    ("q2", "Quote 2"),
                    ("q3", "Quote 3"),
                ],
            )

            assert result.theme_name == "Test Theme"
            assert result.consistent_count == 2
            assert result.total_count == 3
            assert result.consistency_score == pytest.approx(2 / 3)

    def test_evaluate_multiple_themes(self, agent: EvaluatorAgent):
        """Test evaluating multiple themes."""
        with patch.object(
            agent,
            "_call_llm",
            return_value='{"is_consistent": true, "reasoning": "Good"}',
        ):
            result = agent.evaluate(
                themes=[
                    {
                        "name": "Theme 1",
                        "description": "First theme",
                        "quotes": [("q1", "Quote 1"), ("q2", "Quote 2")],
                    },
                    {
                        "name": "Theme 2",
                        "description": "Second theme",
                        "quotes": [("q3", "Quote 3")],
                    },
                ]
            )

            assert result.num_themes == 2
            assert result.total_quotes_evaluated == 3
            assert result.overall_score == 1.0


class TestEvaluatorIntegration:
    """Integration tests for EvaluatorAgent."""

    def test_empty_themes_list(self):
        """Test evaluation with no themes."""
        agent = EvaluatorAgent()
        result = agent.evaluate(themes=[])
        assert result.overall_score == 0.0
        assert result.num_themes == 0

    def test_theme_with_no_quotes(self):
        """Test evaluation of theme with no quotes."""
        agent = EvaluatorAgent()
        result = agent.evaluate(
            themes=[
                {
                    "name": "Empty Theme",
                    "description": "Theme with no data",
                    "quotes": [],
                }
            ]
        )
        assert result.num_themes == 1
        assert result.total_quotes_evaluated == 0
        assert result.theme_results[0].consistency_score == 0.0

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Simulate mix of consistent and inconsistent
        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            is_consistent = call_count[0] % 2 == 1  # Alternate
            return json.dumps(
                {
                    "is_consistent": is_consistent,
                    "reasoning": "Test reasoning",
                }
            )

        agent = EvaluatorAgent()
        with patch.object(agent, "_call_llm", side_effect=mock_response):
            result = agent.evaluate(
                themes=[
                    {
                        "name": "Climate Anxiety",
                        "description": "Worry about climate change impacts",
                        "quotes": [
                            ("q1", "I worry about rising temperatures"),
                            ("q2", "Sea level rise concerns me"),
                            ("q3", "Climate change keeps me up at night"),
                            ("q4", "I'm anxious about extreme weather"),
                        ],
                    }
                ]
            )

            assert result.num_themes == 1
            assert result.total_quotes_evaluated == 4
            # 2 consistent out of 4 (alternating)
            assert result.overall_score == 0.5
