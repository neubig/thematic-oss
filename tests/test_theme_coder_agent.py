"""Tests for ThemeCoderAgent."""

import json
from unittest.mock import patch

import pytest

from thematic_lm.agents import Theme, ThemeCoderAgent, ThemeCoderConfig, ThemeResult
from thematic_lm.codebook import Codebook, Quote


class TestThemeCoderConfig:
    """Test cases for ThemeCoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ThemeCoderConfig()

        assert config.max_themes == 10
        assert config.max_quotes_per_theme == 10
        assert config.min_codes_per_theme == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ThemeCoderConfig(
            max_themes=5,
            max_quotes_per_theme=5,
            min_codes_per_theme=3,
        )

        assert config.max_themes == 5
        assert config.max_quotes_per_theme == 5
        assert config.min_codes_per_theme == 3


class TestTheme:
    """Test cases for Theme."""

    def test_theme_creation(self):
        """Test creating a Theme."""
        quotes = [Quote("q1", "text1"), Quote("q2", "text2")]
        theme = Theme(
            name="Test Theme",
            description="A test theme description",
            codes=["code1", "code2"],
            quotes=quotes,
        )

        assert theme.name == "Test Theme"
        assert theme.description == "A test theme description"
        assert len(theme.codes) == 2
        assert len(theme.quotes) == 2

    def test_theme_default_quotes(self):
        """Test Theme with default empty quotes."""
        theme = Theme(
            name="Theme",
            description="Description",
            codes=["code1"],
        )

        assert theme.quotes == []


class TestThemeResult:
    """Test cases for ThemeResult."""

    def test_theme_result_creation(self):
        """Test creating a ThemeResult."""
        theme = Theme(
            name="Theme 1",
            description="Description 1",
            codes=["code1", "code2"],
            quotes=[Quote("q1", "text1")],
        )
        result = ThemeResult(themes=[theme])

        assert len(result.themes) == 1

    def test_to_json(self):
        """Test JSON serialization."""
        theme = Theme(
            name="Theme 1",
            description="A theme about support",
            codes=["emotional support", "peer help"],
            quotes=[Quote("q1", "felt supported")],
        )
        result = ThemeResult(themes=[theme])

        json_str = result.to_json()
        data = json.loads(json_str)

        assert "themes" in data
        assert len(data["themes"]) == 1
        assert data["themes"][0]["name"] == "Theme 1"
        assert len(data["themes"][0]["codes"]) == 2
        assert len(data["themes"][0]["quotes"]) == 1

    def test_to_dict(self):
        """Test dictionary conversion."""
        theme = Theme(
            name="Theme 1",
            description="Description",
            codes=["code1"],
            quotes=[Quote("q1", "text")],
        )
        result = ThemeResult(themes=[theme])

        data = result.to_dict()

        assert isinstance(data, dict)
        assert "themes" in data
        assert data["themes"][0]["name"] == "Theme 1"


class TestThemeCoderAgent:
    """Test cases for ThemeCoderAgent."""

    @pytest.fixture
    def codebook(self) -> Codebook:
        """Create a sample codebook."""
        cb = Codebook()
        cb.add_code("emotional support", [Quote("q1", "I felt supported by friends")])
        cb.add_code("peer connection", [Quote("q2", "Connected with classmates")])
        cb.add_code("academic stress", [Quote("q3", "Overwhelmed by coursework")])
        cb.add_code("time pressure", [Quote("q4", "Not enough hours in the day")])
        cb.add_code("coping strategies", [Quote("q5", "Learned to manage stress")])
        return cb

    @pytest.fixture
    def agent(self, codebook: Codebook) -> ThemeCoderAgent:
        """Create a ThemeCoderAgent with codebook."""
        return ThemeCoderAgent(codebook=codebook)

    def test_initialization(self, agent: ThemeCoderAgent):
        """Test agent initialization."""
        assert agent.codebook is not None
        assert isinstance(agent.theme_config, ThemeCoderConfig)

    def test_initialization_empty_codebook(self):
        """Test agent initialization with empty codebook."""
        agent = ThemeCoderAgent()
        assert len(agent.codebook) == 0

    def test_get_system_prompt_without_identity(self, agent: ThemeCoderAgent):
        """Test system prompt generation without identity."""
        prompt = agent.get_system_prompt()

        assert "qualitative researcher" in prompt
        assert "themes" in prompt.lower()
        assert "Your Perspective" not in prompt

    def test_get_system_prompt_with_identity(self, codebook: Codebook):
        """Test system prompt generation with identity."""
        config = ThemeCoderConfig(identity="feminist researcher")
        agent = ThemeCoderAgent(config=config, codebook=codebook)

        prompt = agent.get_system_prompt()

        assert "Your Perspective" in prompt
        assert "feminist researcher" in prompt

    def test_format_codes_section(self, agent: ThemeCoderAgent):
        """Test formatting codes section."""
        section = agent._format_codes_section()

        assert "emotional support" in section
        assert "academic stress" in section
        assert "quotes" in section.lower()

    def test_compress_codebook(self, agent: ThemeCoderAgent):
        """Test codebook compression."""
        compressed = agent._compress_codebook()

        assert "emotional support" in compressed
        assert "academic stress" in compressed
        # Should be comma-separated
        assert "," in compressed

    def test_parse_response_valid_json(self, agent: ThemeCoderAgent):
        """Test parsing valid JSON response."""
        response = """```json
{
  "themes": [
    {
      "name": "Social Support",
      "description": "Support from peers and friends",
      "codes": ["emotional support", "peer connection"]
    },
    {
      "name": "Academic Challenges",
      "description": "Stress from coursework",
      "codes": ["academic stress", "time pressure"]
    }
  ]
}
```"""
        themes = agent._parse_response(response)

        assert len(themes) == 2
        assert themes[0].name == "Social Support"
        assert len(themes[0].codes) == 2

    def test_parse_response_raw_json(self, agent: ThemeCoderAgent):
        """Test parsing raw JSON without code blocks."""
        response = (
            '{"themes": [{"name": "Theme", "description": "Desc", '
            '"codes": ["c1", "c2"]}]}'
        )
        themes = agent._parse_response(response)

        assert len(themes) == 1
        assert themes[0].name == "Theme"

    def test_parse_response_invalid_json(self, agent: ThemeCoderAgent):
        """Test parsing invalid JSON returns empty list."""
        themes = agent._parse_response("Not valid JSON")

        assert themes == []

    def test_parse_response_respects_max_themes(self, codebook: Codebook):
        """Test that parsing respects max_themes config."""
        config = ThemeCoderConfig(max_themes=1)
        agent = ThemeCoderAgent(config=config, codebook=codebook)

        response = """{"themes": [
            {"name": "T1", "description": "D1", "codes": ["c1", "c2"]},
            {"name": "T2", "description": "D2", "codes": ["c3", "c4"]}
        ]}"""
        themes = agent._parse_response(response)

        assert len(themes) == 1

    def test_parse_response_skips_small_themes(self, codebook: Codebook):
        """Test that themes with too few codes are skipped."""
        config = ThemeCoderConfig(min_codes_per_theme=3)
        agent = ThemeCoderAgent(config=config, codebook=codebook)

        response = """{"themes": [
            {"name": "Small", "description": "D1", "codes": ["c1", "c2"]},
            {"name": "Big", "description": "D2", "codes": ["c1", "c2", "c3"]}
        ]}"""
        themes = agent._parse_response(response)

        assert len(themes) == 1
        assert themes[0].name == "Big"

    def test_collect_quotes_for_codes(self, agent: ThemeCoderAgent):
        """Test collecting quotes from specified codes."""
        codes = ["emotional support", "peer connection"]
        quotes = agent._collect_quotes_for_codes(codes)

        assert len(quotes) == 2
        quote_ids = [q.quote_id for q in quotes]
        assert "q1" in quote_ids
        assert "q2" in quote_ids

    def test_collect_quotes_avoids_duplicates(self, agent: ThemeCoderAgent):
        """Test that duplicate quotes are avoided."""
        # Add same quote to another code
        agent.codebook.add_code(
            "duplicate test", [Quote("q1", "I felt supported by friends")]
        )

        quotes = agent._collect_quotes_for_codes(
            ["emotional support", "duplicate test"]
        )

        # Should only have one quote with id q1
        quote_ids = [q.quote_id for q in quotes]
        assert quote_ids.count("q1") == 1

    def test_collect_quotes_handles_missing_codes(self, agent: ThemeCoderAgent):
        """Test collecting quotes with nonexistent codes."""
        quotes = agent._collect_quotes_for_codes(["nonexistent code"])

        assert quotes == []

    @patch.object(ThemeCoderAgent, "_call_llm")
    def test_develop_themes(self, mock_llm, agent: ThemeCoderAgent):
        """Test developing themes from codebook."""
        mock_llm.return_value = json.dumps(
            {
                "themes": [
                    {
                        "name": "Social Support Network",
                        "description": "Support from social connections",
                        "codes": ["emotional support", "peer connection"],
                    },
                    {
                        "name": "Academic Pressure",
                        "description": "Stress from academic demands",
                        "codes": ["academic stress", "time pressure"],
                    },
                ]
            }
        )

        result = agent.develop_themes()

        assert len(result.themes) == 2
        assert result.themes[0].name == "Social Support Network"
        # Quotes should be collected from codebook
        assert len(result.themes[0].quotes) == 2
        mock_llm.assert_called_once()

    @patch.object(ThemeCoderAgent, "_call_llm")
    def test_develop_themes_empty_codebook(self, mock_llm):
        """Test developing themes with empty codebook."""
        agent = ThemeCoderAgent()

        result = agent.develop_themes()

        assert len(result.themes) == 0
        mock_llm.assert_not_called()

    @patch.object(ThemeCoderAgent, "_call_llm")
    def test_develop_themes_fallback_on_error(self, mock_llm, agent: ThemeCoderAgent):
        """Test fallback when LLM returns invalid response."""
        mock_llm.return_value = "Invalid response"

        result = agent.develop_themes()

        assert len(result.themes) == 0

    @patch.object(ThemeCoderAgent, "_call_llm")
    def test_develop_themes_from_codebook(self, mock_llm):
        """Test developing themes from provided codebook."""
        mock_llm.return_value = json.dumps(
            {
                "themes": [
                    {
                        "name": "Test Theme",
                        "description": "Description",
                        "codes": ["code1", "code2"],
                    }
                ]
            }
        )

        codebook = Codebook()
        codebook.add_code("code1", [Quote("q1", "text1")])
        codebook.add_code("code2", [Quote("q2", "text2")])

        agent = ThemeCoderAgent()
        result = agent.develop_themes_from_codebook(codebook)

        assert len(result.themes) == 1
        assert agent.codebook == codebook


class TestThemeCoderIntegration:
    """Integration tests for ThemeCoderAgent."""

    def test_full_theme_development_workflow(self):
        """Test complete theme development with real embeddings."""
        # Create a realistic codebook
        codebook = Codebook()
        codebook.add_code(
            "emotional support",
            [Quote("q1", "My friends helped me through difficult times")],
        )
        codebook.add_code(
            "family support",
            [Quote("q2", "My family was always there for me")],
        )
        codebook.add_code(
            "work stress",
            [Quote("q3", "The workload was overwhelming")],
        )
        codebook.add_code(
            "deadline pressure",
            [Quote("q4", "Too many deadlines at once")],
        )

        agent = ThemeCoderAgent(codebook=codebook)

        # Verify codebook is set up correctly
        assert len(agent.codebook) == 4

        # Verify formatting works
        codes_section = agent._format_codes_section()
        assert "emotional support" in codes_section

    def test_quote_collection_respects_max(self):
        """Test that quote collection respects max_quotes_per_theme."""
        codebook = Codebook()
        # Add code with many quotes
        quotes = [Quote(f"q{i}", f"Quote text {i}") for i in range(15)]
        codebook.add_code("multi-quote code", quotes)
        codebook.add_code("another code", [Quote("qx", "text")])

        config = ThemeCoderConfig(max_quotes_per_theme=5)
        agent = ThemeCoderAgent(config=config, codebook=codebook)

        response = """{"themes": [{
            "name": "Test",
            "description": "Desc",
            "codes": ["multi-quote code", "another code"]
        }]}"""

        themes = agent._parse_response(response)

        assert len(themes[0].quotes) == 5
