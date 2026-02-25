"""Tests for CoderAgent."""

import json
from unittest.mock import patch

import pytest

from thematic_lm.agents import CodeAssignment, CoderAgent, CoderConfig
from thematic_lm.codebook import Codebook, Quote


class TestCoderConfig:
    """Test cases for CoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CoderConfig()

        assert config.max_codes_per_segment == 5
        assert config.similarity_threshold == 0.7
        assert config.include_rationale is True
        assert config.temperature == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CoderConfig(
            max_codes_per_segment=3,
            similarity_threshold=0.8,
            identity="researcher perspective",
        )

        assert config.max_codes_per_segment == 3
        assert config.similarity_threshold == 0.8
        assert config.identity == "researcher perspective"


class TestCodeAssignment:
    """Test cases for CodeAssignment."""

    def test_code_assignment_creation(self):
        """Test creating a CodeAssignment."""
        assignment = CodeAssignment(
            segment_id="seg1",
            segment_text="sample text",
            codes=["code1", "code2"],
            rationales=["reason1", "reason2"],
            is_new_code=[True, False],
        )

        assert assignment.segment_id == "seg1"
        assert len(assignment.codes) == 2
        assert len(assignment.rationales) == 2

    def test_code_assignment_defaults(self):
        """Test CodeAssignment default values."""
        assignment = CodeAssignment(
            segment_id="seg1", segment_text="text", codes=["code1"]
        )

        assert assignment.rationales == []
        assert assignment.is_new_code == []


class TestCoderAgent:
    """Test cases for CoderAgent."""

    @pytest.fixture
    def agent(self) -> CoderAgent:
        """Create a CoderAgent for tests."""
        return CoderAgent()

    @pytest.fixture
    def agent_with_codebook(self) -> CoderAgent:
        """Create a CoderAgent with pre-populated codebook."""
        codebook = Codebook()
        codebook.add_code("emotional support", [Quote("q1", "felt supported")])
        codebook.add_code("peer connection", [Quote("q2", "connected with peers")])
        return CoderAgent(codebook=codebook)

    def test_initialization(self, agent: CoderAgent):
        """Test agent initialization."""
        assert agent.codebook is not None
        assert len(agent.codebook) == 0
        assert isinstance(agent.coder_config, CoderConfig)

    def test_initialization_with_codebook(self, agent_with_codebook: CoderAgent):
        """Test agent initialization with existing codebook."""
        assert len(agent_with_codebook.codebook) == 2

    def test_get_system_prompt_without_identity(self, agent: CoderAgent):
        """Test system prompt generation without identity."""
        prompt = agent.get_system_prompt()

        assert "qualitative researcher" in prompt
        assert "Guidelines for Coding" in prompt
        assert "Your Perspective" not in prompt

    def test_get_system_prompt_with_identity(self):
        """Test system prompt generation with identity."""
        config = CoderConfig(identity="feminist researcher")
        agent = CoderAgent(config=config)
        prompt = agent.get_system_prompt()

        assert "Your Perspective" in prompt
        assert "feminist researcher" in prompt

    def test_format_codebook_section_empty(self, agent: CoderAgent):
        """Test formatting empty codebook."""
        section = agent._format_codebook_section()

        assert "empty" in section.lower()

    def test_format_codebook_section_with_codes(self, agent_with_codebook: CoderAgent):
        """Test formatting codebook with codes."""
        section = agent_with_codebook._format_codebook_section()

        assert "emotional support" in section
        assert "peer connection" in section
        assert "2 total" in section

    def test_format_similar_codes_section_empty_codebook(self, agent: CoderAgent):
        """Test similar codes section with empty codebook."""
        section = agent._format_similar_codes_section("test text")

        assert section == ""

    def test_parse_response_valid_json(self, agent: CoderAgent):
        """Test parsing valid JSON response."""
        response = """```json
{
  "codes": ["code1", "code2"],
  "rationales": ["reason1", "reason2"],
  "is_new": [true, false]
}
```"""
        result = agent._parse_response(response, "seg1")

        assert result is not None
        assert result.codes == ["code1", "code2"]
        assert result.rationales == ["reason1", "reason2"]
        assert result.is_new_code == [True, False]

    def test_parse_response_raw_json(self, agent: CoderAgent):
        """Test parsing raw JSON without code blocks."""
        response = '{"codes": ["code1"], "rationales": ["reason1"], "is_new": [true]}'
        result = agent._parse_response(response, "seg1")

        assert result is not None
        assert result.codes == ["code1"]

    def test_parse_response_invalid_json(self, agent: CoderAgent):
        """Test parsing invalid JSON returns None."""
        response = "This is not valid JSON"
        result = agent._parse_response(response, "seg1")

        assert result is None

    def test_parse_response_missing_fields(self, agent: CoderAgent):
        """Test parsing response with missing optional fields."""
        response = '{"codes": ["code1"]}'
        result = agent._parse_response(response, "seg1")

        assert result is not None
        assert result.codes == ["code1"]
        assert len(result.rationales) >= len(result.codes)
        assert len(result.is_new_code) >= len(result.codes)

    def test_parse_response_truncates_to_max_codes(self):
        """Test that parsing respects max_codes_per_segment."""
        config = CoderConfig(max_codes_per_segment=2)
        agent = CoderAgent(config=config)

        response = '{"codes": ["c1", "c2", "c3", "c4"], "rationales": [], "is_new": []}'
        result = agent._parse_response(response, "seg1")

        assert result is not None
        assert len(result.codes) == 2

    @patch.object(CoderAgent, "_call_llm")
    def test_code_segment(self, mock_llm, agent: CoderAgent):
        """Test coding a single segment."""
        mock_llm.return_value = json.dumps(
            {"codes": ["test code"], "rationales": ["test reason"], "is_new": [True]}
        )

        result = agent.code_segment("seg1", "This is test text")

        assert result.segment_id == "seg1"
        assert result.segment_text == "This is test text"
        assert result.codes == ["test code"]
        mock_llm.assert_called_once()

    @patch.object(CoderAgent, "_call_llm")
    def test_code_segment_fallback_on_parse_error(self, mock_llm, agent: CoderAgent):
        """Test coding returns empty assignment on parse error."""
        mock_llm.return_value = "Invalid response"

        result = agent.code_segment("seg1", "Test text")

        assert result.segment_id == "seg1"
        assert result.codes == []

    @patch.object(CoderAgent, "_call_llm")
    def test_code_segments_updates_codebook(self, mock_llm, agent: CoderAgent):
        """Test that coding multiple segments updates the codebook."""
        mock_llm.return_value = json.dumps(
            {"codes": ["new code"], "rationales": ["reason"], "is_new": [True]}
        )

        segments = [("seg1", "text1"), ("seg2", "text2")]
        results = agent.code_segments(segments, update_codebook=True)

        assert len(results) == 2
        # Codebook should have entries (may be deduplicated)
        assert len(agent.codebook) >= 1

    @patch.object(CoderAgent, "_call_llm")
    def test_code_segments_no_update(self, mock_llm, agent: CoderAgent):
        """Test coding without updating codebook."""
        mock_llm.return_value = json.dumps(
            {"codes": ["code"], "rationales": ["reason"], "is_new": [True]}
        )

        segments = [("seg1", "text1")]
        agent.code_segments(segments, update_codebook=False)

        assert len(agent.codebook) == 0

    def test_update_codebook_new_code(self, agent: CoderAgent):
        """Test adding new code to codebook."""
        assignment = CodeAssignment(
            segment_id="seg1",
            segment_text="sample text",
            codes=["brand new code"],
            rationales=["reason"],
            is_new_code=[True],
        )

        agent._update_codebook(assignment)

        assert len(agent.codebook) == 1
        assert agent.codebook.codes[0] == "brand new code"

    def test_update_codebook_existing_code(self, agent_with_codebook: CoderAgent):
        """Test updating existing code in codebook."""
        assignment = CodeAssignment(
            segment_id="seg2",
            segment_text="new quote text",
            codes=["emotional support"],
            rationales=["similar meaning"],
            is_new_code=[False],
        )

        agent_with_codebook._update_codebook(assignment)

        # Should not have added a new code
        assert len(agent_with_codebook.codebook) == 2
