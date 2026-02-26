"""Tests for Theme Aggregator Agent."""

import json
from unittest.mock import patch

import pytest

from thematic_analysis.agents.theme_aggregator import (
    MergedTheme,
    ThemeAggregationResult,
    ThemeAggregatorAgent,
    ThemeAggregatorConfig,
)
from thematic_analysis.agents.theme_coder import Theme, ThemeResult
from thematic_analysis.codebook import EmbeddingService, Quote


class TestThemeAggregatorConfig:
    """Test cases for ThemeAggregatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ThemeAggregatorConfig()

        assert config.similarity_threshold == 0.75
        assert config.max_quotes_per_theme == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ThemeAggregatorConfig(
            similarity_threshold=0.85,
            max_quotes_per_theme=5,
        )

        assert config.similarity_threshold == 0.85
        assert config.max_quotes_per_theme == 5


class TestMergedTheme:
    """Test cases for MergedTheme dataclass."""

    def test_merged_theme_creation(self):
        """Test creating a MergedTheme."""
        theme = MergedTheme(
            name="Social Support",
            description="Support from social networks",
            original_themes=["Peer Support", "Family Support"],
            codes=["code1", "code2"],
            quotes=[Quote("q1", "text1")],
            merge_rationale="Both describe social connections",
        )

        assert theme.name == "Social Support"
        assert len(theme.original_themes) == 2
        assert len(theme.codes) == 2
        assert theme.merge_rationale == "Both describe social connections"

    def test_merged_theme_defaults(self):
        """Test MergedTheme default values."""
        theme = MergedTheme(
            name="Test",
            description="Test desc",
            original_themes=["Test"],
            codes=["code1"],
        )

        assert theme.quotes == []
        assert theme.merge_rationale == ""


class TestThemeAggregationResult:
    """Test cases for ThemeAggregationResult."""

    def test_aggregation_result_creation(self):
        """Test creating ThemeAggregationResult."""
        themes = [
            MergedTheme(
                name="Theme 1",
                description="First theme",
                original_themes=["Theme 1"],
                codes=["code1"],
                quotes=[Quote("q1", "text1")],
            )
        ]
        result = ThemeAggregationResult(themes=themes)

        assert len(result.themes) == 1

    def test_to_json(self):
        """Test JSON serialization."""
        themes = [
            MergedTheme(
                name="Theme 1",
                description="First theme",
                original_themes=["Original 1", "Original 2"],
                codes=["code1", "code2"],
                quotes=[Quote("q1", "text1")],
                merge_rationale="Merged for similarity",
            )
        ]
        result = ThemeAggregationResult(themes=themes)

        json_str = result.to_json()
        data = json.loads(json_str)

        assert "themes" in data
        assert len(data["themes"]) == 1
        assert data["themes"][0]["name"] == "Theme 1"
        assert data["themes"][0]["merge_rationale"] == "Merged for similarity"

    def test_to_dict(self):
        """Test dictionary conversion."""
        themes = [
            MergedTheme(
                name="Theme 1",
                description="First theme",
                original_themes=["Original 1"],
                codes=["code1"],
            )
        ]
        result = ThemeAggregationResult(themes=themes)

        data = result.to_dict()

        assert "themes" in data
        assert len(data["themes"]) == 1


class TestThemeAggregatorAgent:
    """Test cases for ThemeAggregatorAgent."""

    @pytest.fixture
    def mock_embedding_service(self) -> EmbeddingService:
        """Create a mock embedding service."""
        return EmbeddingService(use_mock=True)

    @pytest.fixture
    def agent(self, mock_embedding_service: EmbeddingService) -> ThemeAggregatorAgent:
        """Create a ThemeAggregatorAgent for tests."""
        config = ThemeAggregatorConfig()
        return ThemeAggregatorAgent(
            config=config, embedding_service=mock_embedding_service
        )

    @pytest.fixture
    def sample_theme_results(self) -> list[ThemeResult]:
        """Create sample theme results for testing."""
        return [
            ThemeResult(
                themes=[
                    Theme(
                        name="Academic Stress",
                        description="Stress related to academic performance",
                        codes=["exam_anxiety", "grade_pressure"],
                        quotes=[Quote("q1", "I feel stressed about exams")],
                    ),
                    Theme(
                        name="Social Support",
                        description="Support from peers and family",
                        codes=["peer_help", "family_support"],
                        quotes=[Quote("q2", "My friends help me cope")],
                    ),
                ]
            ),
            ThemeResult(
                themes=[
                    Theme(
                        name="Time Management",
                        description="Managing time effectively",
                        codes=["scheduling", "prioritization"],
                        quotes=[Quote("q3", "I struggle with time")],
                    ),
                ]
            ),
        ]

    def test_initialization(self, agent: ThemeAggregatorAgent):
        """Test agent initialization."""
        assert agent.aggregator_config.similarity_threshold == 0.75
        assert agent.aggregator_config.max_quotes_per_theme == 10
        assert agent.embedding_service is not None

    def test_get_system_prompt(self, agent: ThemeAggregatorAgent):
        """Test system prompt generation."""
        prompt = agent.get_system_prompt()

        assert "qualitative researcher" in prompt.lower()
        assert "merge" in prompt.lower()
        assert "JSON" in prompt

    def test_collect_all_themes(
        self, agent: ThemeAggregatorAgent, sample_theme_results: list[ThemeResult]
    ):
        """Test collecting themes from multiple results."""
        themes = agent._collect_all_themes(sample_theme_results)

        assert len(themes) == 3
        assert "Academic Stress" in themes
        assert "Social Support" in themes
        assert "Time Management" in themes

    def test_collect_all_themes_merges_duplicates(self, agent: ThemeAggregatorAgent):
        """Test that duplicate theme names merge their data."""
        results = [
            ThemeResult(
                themes=[
                    Theme(
                        name="Stress",
                        description="First description",
                        codes=["code1"],
                        quotes=[Quote("q1", "Quote 1")],
                    )
                ]
            ),
            ThemeResult(
                themes=[
                    Theme(
                        name="Stress",
                        description="Second description",
                        codes=["code2"],
                        quotes=[Quote("q2", "Quote 2")],
                    )
                ]
            ),
        ]

        themes = agent._collect_all_themes(results)

        assert len(themes) == 1
        assert "code1" in themes["Stress"].codes
        assert "code2" in themes["Stress"].codes
        assert len(themes["Stress"].quotes) == 2

    def test_collect_all_themes_avoids_duplicate_codes(
        self, agent: ThemeAggregatorAgent
    ):
        """Test that duplicate codes are not added."""
        results = [
            ThemeResult(
                themes=[
                    Theme(
                        name="Stress",
                        description="Description",
                        codes=["code1", "code2"],
                        quotes=[Quote("q1", "Quote 1")],
                    )
                ]
            ),
            ThemeResult(
                themes=[
                    Theme(
                        name="Stress",
                        description="Description",
                        codes=["code1", "code3"],  # code1 is duplicate
                        quotes=[Quote("q1", "Quote 1")],  # q1 is duplicate
                    )
                ]
            ),
        ]

        themes = agent._collect_all_themes(results)

        assert len(themes["Stress"].codes) == 3
        assert len(themes["Stress"].quotes) == 1

    def test_find_similar_groups_empty(self, agent: ThemeAggregatorAgent):
        """Test finding similar groups with no themes."""
        groups = agent._find_similar_groups({})

        assert groups == []

    def test_find_similar_groups_single(self, agent: ThemeAggregatorAgent):
        """Test finding similar groups with single theme."""
        themes = {
            "Stress": Theme(
                name="Stress",
                description="Description",
                codes=["code1"],
            )
        }

        groups = agent._find_similar_groups(themes)

        assert len(groups) == 1
        assert groups[0] == ["Stress"]

    def test_format_themes_section(
        self, agent: ThemeAggregatorAgent, sample_theme_results: list[ThemeResult]
    ):
        """Test formatting themes section."""
        themes = agent._collect_all_themes(sample_theme_results)
        section = agent._format_themes_section(themes)

        assert "Academic Stress" in section
        assert "exam_anxiety" in section
        assert "Codes:" in section

    def test_format_themes_section_truncates_codes(self, agent: ThemeAggregatorAgent):
        """Test that many codes are truncated."""
        themes = {
            "Test": Theme(
                name="Test",
                description="Description",
                codes=[f"code{i}" for i in range(10)],
            )
        }

        section = agent._format_themes_section(themes)

        assert "+5 more" in section

    def test_format_similar_groups_section(self, agent: ThemeAggregatorAgent):
        """Test formatting similar groups."""
        groups = [["Theme 1", "Theme 2"], ["Theme 3"]]

        section = agent._format_similar_groups_section(groups)

        assert "Group 1" in section
        assert "Theme 1, Theme 2" in section
        assert "Standalone: Theme 3" in section

    def test_format_similar_groups_section_empty(self, agent: ThemeAggregatorAgent):
        """Test formatting empty groups."""
        section = agent._format_similar_groups_section([])

        assert "No similar groups" in section

    def test_parse_response_valid_json(self, agent: ThemeAggregatorAgent):
        """Test parsing valid JSON response."""
        themes = {
            "Peer Support": Theme(
                name="Peer Support",
                description="Peer support desc",
                codes=["code1"],
                quotes=[Quote("q1", "text1")],
            ),
            "Family Support": Theme(
                name="Family Support",
                description="Family support desc",
                codes=["code2"],
                quotes=[Quote("q2", "text2")],
            ),
            "Academic Stress": Theme(
                name="Academic Stress",
                description="Stress desc",
                codes=["code3"],
                quotes=[Quote("q3", "text3")],
            ),
        }

        response = """```json
{
  "merge_groups": [
    {
      "merged_name": "Social Support Networks",
      "merged_description": "Combined support from peers and family",
      "original_themes": ["Peer Support", "Family Support"],
      "rationale": "Both describe social connections"
    }
  ],
  "retain_themes": ["Academic Stress"]
}
```"""

        result = agent._parse_response(response, themes)

        assert len(result.themes) == 2
        # Find the merged theme
        merged = next(t for t in result.themes if t.name == "Social Support Networks")
        assert len(merged.original_themes) == 2
        assert "code1" in merged.codes
        assert "code2" in merged.codes
        assert merged.merge_rationale == "Both describe social connections"

    def test_parse_response_raw_json(self, agent: ThemeAggregatorAgent):
        """Test parsing JSON without code block."""
        themes = {
            "Stress": Theme(
                name="Stress",
                description="Stress description",
                codes=["code1"],
            )
        }

        response = '{"merge_groups": [], "retain_themes": ["Stress"]}'

        result = agent._parse_response(response, themes)

        assert len(result.themes) == 1
        assert result.themes[0].name == "Stress"

    def test_parse_response_invalid_json(self, agent: ThemeAggregatorAgent):
        """Test fallback on invalid JSON."""
        themes = {
            "Stress": Theme(
                name="Stress",
                description="Description",
                codes=["code1"],
            )
        }

        result = agent._parse_response("invalid json response", themes)

        # Fallback should return all themes as retained
        assert len(result.themes) == 1
        assert result.themes[0].name == "Stress"

    def test_parse_response_respects_max_quotes(self, agent: ThemeAggregatorAgent):
        """Test that max_quotes_per_theme is respected."""
        agent.aggregator_config.max_quotes_per_theme = 2
        themes = {
            "Test": Theme(
                name="Test",
                description="Description",
                codes=["code1"],
                quotes=[Quote(f"q{i}", f"text{i}") for i in range(10)],
            )
        }

        response = '{"merge_groups": [], "retain_themes": ["Test"]}'
        result = agent._parse_response(response, themes)

        assert len(result.themes[0].quotes) == 2

    def test_fallback_result(self, agent: ThemeAggregatorAgent):
        """Test fallback result creation."""
        themes = {
            "Theme 1": Theme(
                name="Theme 1",
                description="Desc 1",
                codes=["code1"],
            ),
            "Theme 2": Theme(
                name="Theme 2",
                description="Desc 2",
                codes=["code2"],
            ),
        }

        result = agent._fallback_result(themes)

        assert len(result.themes) == 2
        names = {t.name for t in result.themes}
        assert names == {"Theme 1", "Theme 2"}

    def test_aggregate(
        self, agent: ThemeAggregatorAgent, sample_theme_results: list[ThemeResult]
    ):
        """Test full aggregation workflow."""
        mock_response = """```json
{
  "merge_groups": [],
  "retain_themes": ["Academic Stress", "Social Support", "Time Management"]
}
```"""

        with patch.object(agent, "_call_llm", return_value=mock_response):
            result = agent.aggregate(sample_theme_results)

        assert len(result.themes) == 3

    def test_aggregate_empty_input(self, agent: ThemeAggregatorAgent):
        """Test aggregation with empty input."""
        result = agent.aggregate([])

        assert len(result.themes) == 0

    def test_aggregate_with_merge(self, agent: ThemeAggregatorAgent):
        """Test aggregation that merges themes."""
        results = [
            ThemeResult(
                themes=[
                    Theme(
                        name="Peer Support",
                        description="Support from peers",
                        codes=["peer_code"],
                        quotes=[Quote("q1", "Peers help me")],
                    ),
                    Theme(
                        name="Family Support",
                        description="Support from family",
                        codes=["family_code"],
                        quotes=[Quote("q2", "Family helps me")],
                    ),
                ]
            )
        ]

        mock_response = """```json
{
  "merge_groups": [
    {
      "merged_name": "Social Support",
      "merged_description": "Support from social connections",
      "original_themes": ["Peer Support", "Family Support"],
      "rationale": "Both describe support systems"
    }
  ],
  "retain_themes": []
}
```"""

        with patch.object(agent, "_call_llm", return_value=mock_response):
            result = agent.aggregate(results)

        assert len(result.themes) == 1
        assert result.themes[0].name == "Social Support"
        assert len(result.themes[0].codes) == 2

    def test_aggregate_single(self, agent: ThemeAggregatorAgent):
        """Test aggregate_single method."""
        theme_result = ThemeResult(
            themes=[
                Theme(
                    name="Test Theme",
                    description="Description",
                    codes=["code1"],
                )
            ]
        )

        mock_response = '{"merge_groups": [], "retain_themes": ["Test Theme"]}'

        with patch.object(agent, "_call_llm", return_value=mock_response):
            result = agent.aggregate_single(theme_result)

        assert len(result.themes) == 1


class TestThemeAggregatorIntegration:
    """Integration tests for ThemeAggregatorAgent."""

    @pytest.fixture
    def agent(self) -> ThemeAggregatorAgent:
        """Create agent with mock embedding service."""
        config = ThemeAggregatorConfig(similarity_threshold=0.75)
        embedding_service = EmbeddingService(use_mock=True)
        return ThemeAggregatorAgent(config=config, embedding_service=embedding_service)

    def test_full_aggregation_workflow(self, agent: ThemeAggregatorAgent):
        """Test complete aggregation workflow."""
        results = [
            ThemeResult(
                themes=[
                    Theme(
                        name="Academic Pressure",
                        description="Pressure from academics",
                        codes=["exam_stress", "grade_anxiety"],
                        quotes=[Quote("q1", "Exams are stressful")],
                    ),
                    Theme(
                        name="Work-Life Balance",
                        description="Balancing work and personal life",
                        codes=["schedule", "priorities"],
                        quotes=[Quote("q2", "Hard to balance everything")],
                    ),
                ]
            ),
            ThemeResult(
                themes=[
                    Theme(
                        name="Social Support",
                        description="Support from social network",
                        codes=["friends", "family"],
                        quotes=[Quote("q3", "Friends help me cope")],
                    ),
                ]
            ),
        ]

        mock_response = """```json
{
  "merge_groups": [],
  "retain_themes": ["Academic Pressure", "Work-Life Balance", "Social Support"]
}
```"""

        with patch.object(agent, "_call_llm", return_value=mock_response):
            result = agent.aggregate(results)

        # Verify result structure
        assert isinstance(result, ThemeAggregationResult)
        assert len(result.themes) == 3

        # Verify JSON output
        json_output = result.to_json()
        parsed = json.loads(json_output)
        assert len(parsed["themes"]) == 3

    def test_theme_merging_preserves_data(self, agent: ThemeAggregatorAgent):
        """Test that merging preserves all codes and quotes."""
        results = [
            ThemeResult(
                themes=[
                    Theme(
                        name="Theme A",
                        description="Description A",
                        codes=["a1", "a2"],
                        quotes=[Quote("qa1", "Quote A1"), Quote("qa2", "Quote A2")],
                    ),
                    Theme(
                        name="Theme B",
                        description="Description B",
                        codes=["b1", "b2"],
                        quotes=[Quote("qb1", "Quote B1")],
                    ),
                ]
            )
        ]

        mock_response = """```json
{
  "merge_groups": [
    {
      "merged_name": "Combined Theme",
      "merged_description": "Merged description",
      "original_themes": ["Theme A", "Theme B"],
      "rationale": "Related themes"
    }
  ],
  "retain_themes": []
}
```"""

        with patch.object(agent, "_call_llm", return_value=mock_response):
            result = agent.aggregate(results)

        merged = result.themes[0]
        assert len(merged.codes) == 4
        assert len(merged.quotes) == 3
        assert set(merged.codes) == {"a1", "a2", "b1", "b2"}
