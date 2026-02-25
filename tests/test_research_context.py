"""Tests for research context module."""


from thematic_lm.research_context import (
    CODE_6RS,
    CONCEPTUALIZATION_GUIDANCE,
    KEYWORD_6RS,
    THEME_DEVELOPMENT_GUIDANCE,
    ResearchContext,
    ResearchParadigm,
    TheoreticalFramework,
    create_climate_research_context,
    create_healthcare_research_context,
    create_methodology_prompt,
)


class TestResearchParadigm:
    """Tests for ResearchParadigm enum."""

    def test_paradigm_values(self):
        """Test that all paradigms have expected values."""
        assert ResearchParadigm.INTERPRETIVIST.value == "interpretivist"
        assert ResearchParadigm.CONSTRUCTIVIST.value == "constructivist"
        assert ResearchParadigm.CRITICAL.value == "critical"
        assert ResearchParadigm.PRAGMATIC.value == "pragmatic"
        assert ResearchParadigm.POSITIVIST.value == "positivist"
        assert ResearchParadigm.PHENOMENOLOGICAL.value == "phenomenological"


class TestTheoreticalFramework:
    """Tests for TheoreticalFramework enum."""

    def test_framework_values(self):
        """Test that all frameworks have expected values."""
        assert TheoreticalFramework.GROUNDED_THEORY.value == "grounded_theory"
        assert TheoreticalFramework.PHENOMENOLOGY.value == "phenomenology"
        assert TheoreticalFramework.THEMATIC_ANALYSIS.value == "thematic_analysis"
        assert TheoreticalFramework.CONTENT_ANALYSIS.value == "content_analysis"


class TestResearchContext:
    """Tests for ResearchContext dataclass."""

    def test_empty_context(self):
        """Test creating an empty research context."""
        ctx = ResearchContext()
        assert ctx.is_empty()
        assert ctx.title == ""
        assert ctx.aim == ""
        assert ctx.research_questions == []

    def test_non_empty_context(self):
        """Test that context is not empty when fields are set."""
        ctx = ResearchContext(title="Test Study")
        assert not ctx.is_empty()

        ctx2 = ResearchContext(aim="To understand something")
        assert not ctx2.is_empty()

        ctx3 = ResearchContext(research_questions=["RQ1", "RQ2"])
        assert not ctx3.is_empty()

    def test_to_prompt_section_empty(self):
        """Test prompt section for empty context."""
        ctx = ResearchContext()
        assert ctx.to_prompt_section() == ""

    def test_to_prompt_section_with_title(self):
        """Test prompt section includes title."""
        ctx = ResearchContext(title="My Study")
        prompt = ctx.to_prompt_section()
        assert "## Research Study: My Study" in prompt

    def test_to_prompt_section_with_aim(self):
        """Test prompt section includes aim."""
        ctx = ResearchContext(aim="To understand X")
        prompt = ctx.to_prompt_section()
        assert "### Research Aim" in prompt
        assert "To understand X" in prompt

    def test_to_prompt_section_with_questions(self):
        """Test prompt section includes research questions."""
        ctx = ResearchContext(research_questions=["What is X?", "Why Y?"])
        prompt = ctx.to_prompt_section()
        assert "### Research Questions" in prompt
        assert "1. What is X?" in prompt
        assert "2. Why Y?" in prompt

    def test_to_prompt_section_with_framework(self):
        """Test prompt section includes theoretical framework."""
        ctx = ResearchContext(theoretical_framework="Social constructionism")
        prompt = ctx.to_prompt_section()
        assert "### Theoretical Framework" in prompt
        assert "Social constructionism" in prompt

    def test_to_prompt_section_with_paradigm(self):
        """Test prompt section includes paradigm."""
        ctx = ResearchContext(paradigm="interpretivist")
        prompt = ctx.to_prompt_section()
        assert "### Research Paradigm" in prompt
        assert "interpretivist" in prompt

    def test_to_prompt_section_with_domain(self):
        """Test prompt section includes domain."""
        ctx = ResearchContext(domain="Climate change")
        prompt = ctx.to_prompt_section()
        assert "### Domain Context" in prompt
        assert "Climate change" in prompt

    def test_to_prompt_section_with_background(self):
        """Test prompt section includes background."""
        ctx = ResearchContext(background="Some background info")
        prompt = ctx.to_prompt_section()
        assert "### Background" in prompt
        assert "Some background info" in prompt

    def test_to_prompt_section_with_keywords(self):
        """Test prompt section includes keywords."""
        ctx = ResearchContext(keywords=["climate", "anxiety", "hope"])
        prompt = ctx.to_prompt_section()
        assert "### Key Concepts" in prompt
        assert "climate, anxiety, hope" in prompt

    def test_full_context(self):
        """Test a fully populated context."""
        ctx = ResearchContext(
            title="Full Study",
            aim="Comprehensive aim",
            research_questions=["RQ1", "RQ2"],
            theoretical_framework="Framework X",
            paradigm="interpretivist",
            methodology="thematic_analysis",
            domain="Test domain",
            background="Background info",
            keywords=["key1", "key2"],
        )
        assert not ctx.is_empty()
        prompt = ctx.to_prompt_section()
        assert "Full Study" in prompt
        assert "Comprehensive aim" in prompt
        assert "RQ1" in prompt
        assert "Framework X" in prompt
        assert "interpretivist" in prompt
        assert "Test domain" in prompt
        assert "Background info" in prompt
        assert "key1, key2" in prompt


class TestPredefinedContexts:
    """Tests for predefined research contexts."""

    def test_climate_context_default(self):
        """Test default climate research context."""
        ctx = create_climate_research_context()
        assert not ctx.is_empty()
        assert "Climate" in ctx.title
        assert len(ctx.research_questions) == 4
        assert ctx.paradigm == "interpretivist"
        assert len(ctx.keywords) > 0
        assert any("climate" in k.lower() for k in ctx.keywords)

    def test_climate_context_custom_questions(self):
        """Test climate context with custom questions."""
        custom_questions = ["Custom question 1", "Custom question 2"]
        ctx = create_climate_research_context(research_questions=custom_questions)
        assert ctx.research_questions == custom_questions

    def test_healthcare_context_default(self):
        """Test default healthcare research context."""
        ctx = create_healthcare_research_context()
        assert not ctx.is_empty()
        assert "Healthcare" in ctx.title
        assert len(ctx.research_questions) == 4
        assert ctx.paradigm == "phenomenological"
        assert len(ctx.keywords) > 0
        assert "patient" in [k.lower() for k in ctx.keywords]

    def test_healthcare_context_custom_questions(self):
        """Test healthcare context with custom questions."""
        custom_questions = ["How do patients feel?"]
        ctx = create_healthcare_research_context(research_questions=custom_questions)
        assert ctx.research_questions == custom_questions


class TestMethodologyPrompt:
    """Tests for methodology prompt generation."""

    def test_empty_prompt(self):
        """Test prompt with no options."""
        prompt = create_methodology_prompt(
            research_context=None,
            include_6rs_keywords=False,
            include_6rs_codes=False,
            include_theme_guidance=False,
            include_conceptualization=False,
        )
        assert prompt == ""

    def test_prompt_with_6rs_codes(self):
        """Test prompt includes 6Rs for codes by default."""
        prompt = create_methodology_prompt()
        assert "## 6 Rs Framework for Code Quality" in prompt
        assert "Reciprocal" in prompt
        assert "Recognizable" in prompt
        assert "Responsive" in prompt
        assert "Resourceful" in prompt

    def test_prompt_with_6rs_keywords(self):
        """Test prompt includes 6Rs for keywords."""
        prompt = create_methodology_prompt(include_6rs_keywords=True)
        assert "## 6 Rs Framework for Keyword Selection" in prompt
        assert "Realness" in prompt
        assert "Richness" in prompt
        assert "Repetition" in prompt
        assert "Rationale" in prompt
        assert "Repartee" in prompt
        assert "Regal" in prompt

    def test_prompt_with_theme_guidance(self):
        """Test prompt includes theme development guidance."""
        prompt = create_methodology_prompt(include_theme_guidance=True)
        assert "## Theme Development Guidance" in prompt
        assert "Organizing codes" in prompt
        assert "Considering theory" in prompt

    def test_prompt_with_conceptualization(self):
        """Test prompt includes conceptualization guidance."""
        prompt = create_methodology_prompt(include_conceptualization=True)
        assert "## Conceptualization Guidance" in prompt
        assert "Build theory" in prompt

    def test_prompt_with_research_context(self):
        """Test prompt includes research context."""
        ctx = ResearchContext(title="Test Study", aim="To test")
        prompt = create_methodology_prompt(
            research_context=ctx, include_6rs_codes=False
        )
        assert "Test Study" in prompt
        assert "To test" in prompt

    def test_prompt_ignores_empty_context(self):
        """Test that empty context is ignored."""
        ctx = ResearchContext()
        prompt = create_methodology_prompt(
            research_context=ctx,
            include_6rs_codes=False,
        )
        assert prompt == ""

    def test_full_methodology_prompt(self):
        """Test full methodology prompt with all options."""
        ctx = create_climate_research_context()
        prompt = create_methodology_prompt(
            research_context=ctx,
            include_6rs_keywords=True,
            include_6rs_codes=True,
            include_theme_guidance=True,
            include_conceptualization=True,
        )
        # Should include all sections
        assert "Climate" in prompt
        assert "6 Rs Framework for Keyword Selection" in prompt
        assert "6 Rs Framework for Code Quality" in prompt
        assert "Theme Development Guidance" in prompt
        assert "Conceptualization Guidance" in prompt


class TestGuidanceConstants:
    """Tests for guidance constant strings."""

    def test_keyword_6rs_content(self):
        """Test KEYWORD_6RS contains all 6 Rs."""
        assert "Realness" in KEYWORD_6RS
        assert "Richness" in KEYWORD_6RS
        assert "Repetition" in KEYWORD_6RS
        assert "Rationale" in KEYWORD_6RS
        assert "Repartee" in KEYWORD_6RS
        assert "Regal" in KEYWORD_6RS
        assert "Naeem et al. 2025" in KEYWORD_6RS

    def test_code_6rs_content(self):
        """Test CODE_6RS contains all 4 Rs."""
        assert "Reciprocal" in CODE_6RS
        assert "Recognizable" in CODE_6RS
        assert "Responsive" in CODE_6RS
        assert "Resourceful" in CODE_6RS
        assert "Naeem et al. 2025" in CODE_6RS

    def test_theme_development_content(self):
        """Test THEME_DEVELOPMENT_GUIDANCE has key sections."""
        assert "Organizing codes" in THEME_DEVELOPMENT_GUIDANCE
        assert "Considering theory" in THEME_DEVELOPMENT_GUIDANCE
        assert "Looking for patterns" in THEME_DEVELOPMENT_GUIDANCE
        assert "Building coherence" in THEME_DEVELOPMENT_GUIDANCE
        assert "Staying grounded" in THEME_DEVELOPMENT_GUIDANCE

    def test_conceptualization_content(self):
        """Test CONCEPTUALIZATION_GUIDANCE has key sections."""
        assert "Interpret coherently" in CONCEPTUALIZATION_GUIDANCE
        assert "Build theory" in CONCEPTUALIZATION_GUIDANCE
        assert "Connect to literature" in CONCEPTUALIZATION_GUIDANCE
        assert "Synthesize" in CONCEPTUALIZATION_GUIDANCE
