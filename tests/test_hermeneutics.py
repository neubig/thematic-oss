"""Tests for hermeneutics module."""

from thematic_analysis.hermeneutics import (
    CHAIN_OF_THOUGHT_TEMPLATE,
    ONE_CODE_PER_PROMPT_TEMPLATE,
    RATIONALE_TEMPLATE,
    AdaptedCodebook,
    CodeDefinition,
    CodingRationale,
    DirectiveType,
    RationaleAnalysis,
    ScopeType,
    analyze_rationales,
    create_climate_adapted_codebook,
    create_cot_prompt,
    create_single_code_prompt,
    suggest_codebook_improvements,
)


class TestScopeType:
    """Tests for ScopeType enum."""

    def test_scope_values(self):
        """Test that all scope types have expected values."""
        assert ScopeType.EXPLICIT.value == "explicit"
        assert ScopeType.IMPLICIT.value == "implicit"
        assert ScopeType.BOTH.value == "both"


class TestDirectiveType:
    """Tests for DirectiveType enum."""

    def test_directive_values(self):
        """Test that directive types have expected values."""
        assert DirectiveType.MANDATORY.value == "mandatory"
        assert DirectiveType.PROHIBITORY.value == "prohibitory"


class TestCodeDefinition:
    """Tests for CodeDefinition dataclass."""

    def test_basic_creation(self):
        """Test creating a basic code definition."""
        code_def = CodeDefinition(
            code="test_code",
            description="A test code for testing",
        )
        assert code_def.code == "test_code"
        assert code_def.description == "A test code for testing"
        assert code_def.scope == ScopeType.BOTH

    def test_full_creation(self):
        """Test creating a fully specified code definition."""
        code_def = CodeDefinition(
            code="anxiety",
            description="Expressions of worry or fear",
            scope=ScopeType.EXPLICIT,
            inclusion_criteria=["Must express worry", "Must be personal"],
            exclusion_criteria=["General statements about anxiety"],
            examples=["I am worried", "This scares me"],
            counter_examples=["Some people worry about this"],
            theoretical_grounding="Based on affect theory",
        )
        assert code_def.scope == ScopeType.EXPLICIT
        assert len(code_def.inclusion_criteria) == 2
        assert len(code_def.exclusion_criteria) == 1
        assert len(code_def.examples) == 2
        assert len(code_def.counter_examples) == 1
        assert "affect theory" in code_def.theoretical_grounding

    def test_to_prompt_section_basic(self):
        """Test prompt section generation for basic code."""
        code_def = CodeDefinition(
            code="test_code",
            description="A test description",
        )
        prompt = code_def.to_prompt_section()
        assert "### Code: test_code" in prompt
        assert "**Description:** A test description" in prompt
        assert "explicit statements and implied meanings" in prompt

    def test_to_prompt_section_explicit_scope(self):
        """Test prompt section with explicit scope."""
        code_def = CodeDefinition(
            code="explicit_code",
            description="Explicit only",
            scope=ScopeType.EXPLICIT,
        )
        prompt = code_def.to_prompt_section()
        assert "ONLY when this concept is explicitly stated" in prompt

    def test_to_prompt_section_implicit_scope(self):
        """Test prompt section with implicit scope."""
        code_def = CodeDefinition(
            code="implicit_code",
            description="Implicit allowed",
            scope=ScopeType.IMPLICIT,
        )
        prompt = code_def.to_prompt_section()
        assert "implied, even if not directly stated" in prompt

    def test_to_prompt_section_with_criteria(self):
        """Test prompt section with inclusion/exclusion criteria."""
        code_def = CodeDefinition(
            code="criteria_code",
            description="Code with criteria",
            inclusion_criteria=["Criterion 1", "Criterion 2"],
            exclusion_criteria=["Exclude when X"],
        )
        prompt = code_def.to_prompt_section()
        assert "**MUST apply this code when:**" in prompt
        assert "Criterion 1" in prompt
        assert "Criterion 2" in prompt
        assert "**Do NOT apply when:**" in prompt
        assert "Exclude when X" in prompt

    def test_to_prompt_section_with_examples(self):
        """Test prompt section with examples."""
        code_def = CodeDefinition(
            code="example_code",
            description="Code with examples",
            examples=["Example 1", "Example 2"],
            counter_examples=["Counter example"],
        )
        prompt = code_def.to_prompt_section()
        assert "**Examples (should receive this code):**" in prompt
        assert '"Example 1"' in prompt
        assert "**Counter-examples (should NOT receive this code):**" in prompt
        assert '"Counter example"' in prompt

    def test_to_prompt_section_with_theory(self):
        """Test prompt section with theoretical grounding."""
        code_def = CodeDefinition(
            code="theory_code",
            description="Code with theory",
            theoretical_grounding="Based on social constructionism",
        )
        prompt = code_def.to_prompt_section()
        assert "**Theoretical basis:**" in prompt
        assert "social constructionism" in prompt


class TestAdaptedCodebook:
    """Tests for AdaptedCodebook class."""

    def test_basic_creation(self):
        """Test creating a basic codebook."""
        codebook = AdaptedCodebook(
            name="Test Codebook",
            description="A test codebook",
        )
        assert codebook.name == "Test Codebook"
        assert codebook.description == "A test codebook"
        assert len(codebook.codes) == 0

    def test_add_code(self):
        """Test adding codes to codebook."""
        codebook = AdaptedCodebook(name="Test", description="Test")
        code_def = CodeDefinition(code="code1", description="First code")
        codebook.add_code(code_def)
        assert len(codebook.codes) == 1
        assert codebook.codes[0].code == "code1"

    def test_get_code(self):
        """Test retrieving code by name."""
        codebook = AdaptedCodebook(name="Test", description="Test")
        code_def = CodeDefinition(code="TestCode", description="A code")
        codebook.add_code(code_def)

        # Exact match
        result = codebook.get_code("TestCode")
        assert result is not None
        assert result.code == "TestCode"

        # Case insensitive
        result = codebook.get_code("testcode")
        assert result is not None

        # Not found
        result = codebook.get_code("nonexistent")
        assert result is None

    def test_to_full_prompt(self):
        """Test generating full codebook prompt."""
        codebook = AdaptedCodebook(
            name="Test Codebook",
            description="Description of codebook",
            theoretical_framework="Based on theory X",
        )
        codebook.add_code(CodeDefinition(code="code1", description="First"))
        codebook.add_code(CodeDefinition(code="code2", description="Second"))

        prompt = codebook.to_full_prompt()
        assert "# Codebook: Test Codebook" in prompt
        assert "Description of codebook" in prompt
        assert "## Theoretical Framework" in prompt
        assert "Based on theory X" in prompt
        assert "## Code Definitions" in prompt
        assert "code1" in prompt
        assert "code2" in prompt

    def test_get_single_code_prompt(self):
        """Test getting prompt for single code."""
        codebook = AdaptedCodebook(name="Test", description="Test")
        code_def = CodeDefinition(code="single", description="Single code")
        codebook.add_code(code_def)

        prompt = codebook.get_single_code_prompt("single")
        assert "### Code: single" in prompt
        assert "Single code" in prompt

        # Nonexistent code
        prompt = codebook.get_single_code_prompt("nonexistent")
        assert prompt == ""


class TestCodingRationale:
    """Tests for CodingRationale dataclass."""

    def test_creation(self):
        """Test creating a coding rationale."""
        rationale = CodingRationale(
            code="test_code",
            applies=True,
            confidence=0.85,
            rationale="This applies because of X",
            evidence=["quote 1", "quote 2"],
        )
        assert rationale.code == "test_code"
        assert rationale.applies is True
        assert rationale.confidence == 0.85
        assert len(rationale.evidence) == 2


class TestRationaleAnalysis:
    """Tests for RationaleAnalysis dataclass."""

    def test_creation(self):
        """Test creating rationale analysis."""
        analysis = RationaleAnalysis(code="test")
        assert analysis.code == "test"
        assert analysis.total_decisions == 0
        assert analysis.agreement_count == 0

    def test_agreement_rate(self):
        """Test agreement rate calculation."""
        analysis = RationaleAnalysis(
            code="test",
            total_decisions=10,
            agreement_count=8,
        )
        assert analysis.agreement_rate == 0.8

        # Zero divisions
        analysis2 = RationaleAnalysis(code="test")
        assert analysis2.agreement_rate == 0.0

    def test_add_ambiguity(self):
        """Test adding ambiguity indicators."""
        analysis = RationaleAnalysis(code="test")
        analysis.add_ambiguity("indicator 1")
        analysis.add_ambiguity("indicator 2")
        analysis.add_ambiguity("indicator 1")  # Duplicate

        assert len(analysis.ambiguity_indicators) == 2

    def test_add_refinement(self):
        """Test adding refinement suggestions."""
        analysis = RationaleAnalysis(code="test")
        analysis.add_refinement("suggestion 1")
        analysis.add_refinement("suggestion 2")
        analysis.add_refinement("suggestion 1")  # Duplicate

        assert len(analysis.suggested_refinements) == 2


class TestAnalyzeRationales:
    """Tests for analyze_rationales function."""

    def test_basic_analysis(self):
        """Test basic rationale analysis."""
        rationales = [
            CodingRationale(
                code="code1", applies=True, confidence=0.9, rationale="Clear match"
            ),
            CodingRationale(
                code="code1", applies=False, confidence=0.8, rationale="No match"
            ),
        ]

        analyses = analyze_rationales(rationales)
        assert "code1" in analyses
        assert analyses["code1"].total_decisions == 2

    def test_low_confidence_ambiguity(self):
        """Test that low confidence triggers ambiguity indicator."""
        rationales = [
            CodingRationale(
                code="code1", applies=True, confidence=0.5, rationale="Unclear..."
            ),
        ]

        analyses = analyze_rationales(rationales)
        assert (
            "Low confidence suggests unclear criteria"
            in analyses["code1"].ambiguity_indicators
        )

    def test_hedging_language_detection(self):
        """Test detection of hedging language in rationales."""
        rationales = [
            CodingRationale(
                code="code1",
                applies=True,
                confidence=0.7,
                rationale="This might apply because it possibly relates to X",
            ),
        ]

        analyses = analyze_rationales(rationales)
        # Should detect hedging
        assert any("Hedging" in ind for ind in analyses["code1"].ambiguity_indicators)

    def test_gold_standard_comparison(self):
        """Test comparison with gold standard."""
        rationales = [
            CodingRationale(
                code="code1", applies=True, confidence=0.9, rationale="Match"
            ),
            CodingRationale(
                code="code1", applies=False, confidence=0.9, rationale="No match"
            ),
        ]
        gold_standard = [("code1", True)]

        analyses = analyze_rationales(rationales, gold_standard)
        # One agreement, one disagreement
        assert analyses["code1"].agreement_count == 1
        assert len(analyses["code1"].suggested_refinements) >= 1


class TestSuggestCodebookImprovements:
    """Tests for suggest_codebook_improvements function."""

    def test_low_agreement_suggestion(self):
        """Test suggestion for low agreement rate."""
        analyses = {
            "code1": RationaleAnalysis(
                code="code1",
                total_decisions=10,
                agreement_count=5,
            )
        }

        suggestions = suggest_codebook_improvements(analyses)
        assert any("low agreement" in s.lower() for s in suggestions)

    def test_ambiguity_suggestion(self):
        """Test suggestion for ambiguity indicators."""
        analysis = RationaleAnalysis(code="code1")
        analysis.add_ambiguity("Issue 1")
        analysis.add_ambiguity("Issue 2")
        analysis.add_ambiguity("Issue 3")

        suggestions = suggest_codebook_improvements({"code1": analysis})
        assert any("ambiguity" in s.lower() for s in suggestions)

    def test_refinement_suggestion(self):
        """Test that refinements are included in suggestions."""
        analysis = RationaleAnalysis(code="code1")
        analysis.add_refinement("Specific refinement needed")

        suggestions = suggest_codebook_improvements({"code1": analysis})
        assert any("Specific refinement needed" in s for s in suggestions)


class TestPromptFunctions:
    """Tests for prompt generation functions."""

    def test_create_cot_prompt_basic(self):
        """Test basic CoT prompt creation."""
        prompt = create_cot_prompt()
        assert "Chain-of-Thought" in prompt
        assert "Rationale Requirement" in prompt

    def test_create_cot_prompt_without_rationales(self):
        """Test CoT prompt without rationale requirement."""
        prompt = create_cot_prompt(require_rationales=False)
        assert "Chain-of-Thought" in prompt
        assert "Rationale Requirement" not in prompt

    def test_create_cot_prompt_with_codes(self):
        """Test CoT prompt with code definitions."""
        codes = [
            CodeDefinition(code="code1", description="First code"),
            CodeDefinition(code="code2", description="Second code"),
        ]
        prompt = create_cot_prompt(code_definitions=codes)
        assert "## Available Codes" in prompt
        assert "code1" in prompt
        assert "code2" in prompt

    def test_create_single_code_prompt(self):
        """Test single code prompt creation."""
        code_def = CodeDefinition(code="test", description="Test code")
        prompt = create_single_code_prompt(code_def, "Sample text to evaluate")
        assert "Single Code Evaluation" in prompt
        assert "test" in prompt
        assert "Test code" in prompt
        assert "Sample text to evaluate" in prompt


class TestClimateAdaptedCodebook:
    """Tests for climate adapted codebook factory."""

    def test_creation(self):
        """Test creating climate codebook."""
        codebook = create_climate_adapted_codebook()
        assert codebook.name == "Climate Change Perceptions"
        assert len(codebook.codes) >= 2

    def test_has_anxiety_code(self):
        """Test that codebook has climate anxiety code."""
        codebook = create_climate_adapted_codebook()
        anxiety_code = codebook.get_code("climate_anxiety")
        assert anxiety_code is not None
        assert len(anxiety_code.inclusion_criteria) > 0
        assert len(anxiety_code.examples) > 0
        assert len(anxiety_code.counter_examples) > 0

    def test_has_hope_code(self):
        """Test that codebook has hope/agency code."""
        codebook = create_climate_adapted_codebook()
        hope_code = codebook.get_code("hope_agency")
        assert hope_code is not None
        assert len(hope_code.inclusion_criteria) > 0
        assert hope_code.theoretical_grounding != ""


class TestTemplates:
    """Tests for prompt templates."""

    def test_chain_of_thought_template_content(self):
        """Test that CoT template has key elements."""
        assert "Initial Reading" in CHAIN_OF_THOUGHT_TEMPLATE
        assert "Quote Identification" in CHAIN_OF_THOUGHT_TEMPLATE
        assert "Justification" in CHAIN_OF_THOUGHT_TEMPLATE
        assert "REASONING" in CHAIN_OF_THOUGHT_TEMPLATE

    def test_rationale_template_content(self):
        """Test that rationale template has key elements."""
        assert "MUST provide a rationale" in RATIONALE_TEMPLATE
        assert "evidence" in RATIONALE_TEMPLATE.lower()
        assert "codes" in RATIONALE_TEMPLATE
        assert "rationales" in RATIONALE_TEMPLATE

    def test_one_code_prompt_template_content(self):
        """Test that one-code prompt template has key elements."""
        assert "Single Code Evaluation" in ONE_CODE_PER_PROMPT_TEMPLATE
        assert "{code_definition}" in ONE_CODE_PER_PROMPT_TEMPLATE
        assert "{text}" in ONE_CODE_PER_PROMPT_TEMPLATE
        assert "applies" in ONE_CODE_PER_PROMPT_TEMPLATE
        assert "confidence" in ONE_CODE_PER_PROMPT_TEMPLATE
