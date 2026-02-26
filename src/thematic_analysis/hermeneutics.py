"""Hermeneutic techniques for deeper thematic analysis.

Implements techniques from Dunivin (2025) 'Scaling Hermeneutics: A Guide to
Qualitative Coding with LLMs for Reflexive Content Analysis' to achieve
deeper, theory-connected interpretations.

Key techniques:
- Chain-of-thought prompting with rationale generation
- Codebook adaptation for LLM comprehension
- One-code-per-prompt approach
- Explicit/implicit scope control
- Iterative codebook refinement

Reference: Dunivin, Z. O. (2025). Scaling hermeneutics: A guide to qualitative
coding with LLMs for reflexive content analysis. EPJ Data Science, 14(1), 28.
"""

from dataclasses import dataclass, field
from enum import Enum


class ScopeType(Enum):
    """Scope control types for code definitions.

    Controls how broadly or narrowly the LLM should interpret a code.
    """

    EXPLICIT = "explicit"  # Code explicitly stated in text
    IMPLICIT = "implicit"  # Code implied but not directly stated
    BOTH = "both"  # Accept both explicit and implicit mentions


class DirectiveType(Enum):
    """Types of directives for code definitions.

    Mandatory phrasing is more effective than prohibitory for LLMs.
    """

    MANDATORY = "mandatory"  # "This code MUST be applied when..."
    PROHIBITORY = "prohibitory"  # "Do NOT apply this code when..."


@dataclass
class CodeDefinition:
    """Enhanced code definition optimized for LLM comprehension.

    Codebooks designed for humans need adaptation for LLMs. This class
    captures the additional detail needed for reliable LLM coding.

    Attributes:
        code: The code label.
        description: Clear, precise description of what the code captures.
        scope: Whether code applies to explicit, implicit, or both mentions.
        inclusion_criteria: Mandatory criteria for when to apply the code.
        exclusion_criteria: Criteria for when NOT to apply the code.
        examples: Positive examples of text that should receive this code.
        counter_examples: Negative examples that should NOT receive this code.
        theoretical_grounding: How this code connects to theory.
    """

    code: str
    description: str
    scope: ScopeType = ScopeType.BOTH
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    counter_examples: list[str] = field(default_factory=list)
    theoretical_grounding: str = ""

    def to_prompt_section(self) -> str:
        """Convert code definition to a prompt section.

        Returns:
            Formatted string for inclusion in prompts.
        """
        sections = [f"### Code: {self.code}", f"**Description:** {self.description}"]

        # Scope instructions
        if self.scope == ScopeType.EXPLICIT:
            sections.append(
                "**Scope:** Apply ONLY when this concept is "
                "explicitly stated in the text."
            )
        elif self.scope == ScopeType.IMPLICIT:
            sections.append(
                "**Scope:** Apply when this concept is implied, "
                "even if not directly stated."
            )
        else:
            sections.append(
                "**Scope:** Apply to both explicit statements and implied meanings."
            )

        # Inclusion criteria (mandatory phrasing)
        if self.inclusion_criteria:
            criteria = "\n".join(f"  - {c}" for c in self.inclusion_criteria)
            sections.append(f"**MUST apply this code when:**\n{criteria}")

        # Exclusion criteria
        if self.exclusion_criteria:
            criteria = "\n".join(f"  - {c}" for c in self.exclusion_criteria)
            sections.append(f"**Do NOT apply when:**\n{criteria}")

        # Examples
        if self.examples:
            examples = "\n".join(f'  - "{ex}"' for ex in self.examples)
            sections.append(f"**Examples (should receive this code):**\n{examples}")

        if self.counter_examples:
            examples = "\n".join(f'  - "{ex}"' for ex in self.counter_examples)
            sections.append(
                f"**Counter-examples (should NOT receive this code):**\n{examples}"
            )

        # Theoretical grounding
        if self.theoretical_grounding:
            sections.append(f"**Theoretical basis:** {self.theoretical_grounding}")

        return "\n".join(sections)


@dataclass
class AdaptedCodebook:
    """A codebook adapted for LLM comprehension.

    Contains enhanced code definitions with clear criteria,
    examples, and theoretical connections.
    """

    name: str
    description: str
    codes: list[CodeDefinition] = field(default_factory=list)
    theoretical_framework: str = ""

    def add_code(self, code_def: CodeDefinition) -> None:
        """Add a code definition to the codebook."""
        self.codes.append(code_def)

    def get_code(self, code_name: str) -> CodeDefinition | None:
        """Get a code definition by name."""
        for code in self.codes:
            if code.code.lower() == code_name.lower():
                return code
        return None

    def to_full_prompt(self) -> str:
        """Generate full codebook prompt section."""
        sections = [
            f"# Codebook: {self.name}",
            f"{self.description}",
        ]

        if self.theoretical_framework:
            sections.append(f"\n## Theoretical Framework\n{self.theoretical_framework}")

        sections.append("\n## Code Definitions\n")
        for code_def in self.codes:
            sections.append(code_def.to_prompt_section())
            sections.append("")  # Blank line between codes

        return "\n".join(sections)

    def get_single_code_prompt(self, code_name: str) -> str:
        """Get prompt section for a single code (one-code-per-prompt approach).

        Args:
            code_name: Name of the code to get.

        Returns:
            Prompt section for that code only.
        """
        code_def = self.get_code(code_name)
        if code_def is None:
            return ""

        return code_def.to_prompt_section()


# Chain-of-thought prompting templates

CHAIN_OF_THOUGHT_TEMPLATE = """
## Coding Approach: Chain-of-Thought

When coding this text, you MUST follow this reasoning process:

1. **Initial Reading**: Read the text carefully and identify the main topic/content.

2. **Quote Identification**: Identify the specific words/phrases relevant to coding.

3. **Code Consideration**: For each potential code, ask yourself:
   - Does this text explicitly mention this concept?
   - Does this text implicitly suggest this concept?
   - What evidence supports applying this code?

4. **Justification**: Before assigning any code, write out your reasoning.

5. **Final Decision**: Only after justifying, decide whether to apply each code.

Provide your reasoning in this format:
```
REASONING:
- Initial observation: [what you notice first]
- Key phrases: [relevant quotes from the text]
- Code consideration: [why each code does or doesn't apply]
- Conclusion: [final coding decision with justification]

CODES: [list of assigned codes]
```
"""

RATIONALE_TEMPLATE = """
## Rationale Requirement

For EVERY code you assign, you MUST provide a rationale explaining:
1. What specific evidence in the text supports this code
2. How this evidence meets the code's criteria
3. Why alternative codes are less appropriate

Format your response as:
```json
{{
  "codes": ["code1", "code2"],
  "rationales": [
    "Evidence: [quote]. This meets criterion X because...",
    "Evidence: [quote]. This demonstrates concept Y by..."
  ]
}}
```

Rationales help identify ambiguities in code definitions and ensure
consistent, justified coding decisions.
"""

ONE_CODE_PER_PROMPT_TEMPLATE = """
## Single Code Evaluation

You are evaluating whether the following code applies to the given text.
Focus ONLY on this code - do not consider other potential codes.

{code_definition}

## Text to Evaluate
"{text}"

## Your Task
1. Carefully read the text
2. Consider the code definition above
3. Determine if this code should be applied
4. Provide your reasoning

Respond with:
```json
{{
  "applies": true/false,
  "confidence": 0.0-1.0,
  "rationale": "Your detailed reasoning here"
}}
```
"""


@dataclass
class CodingRationale:
    """A rationale for a coding decision.

    Captures the reasoning behind applying or not applying a code,
    which is essential for understanding LLM behavior and improving
    codebook definitions.
    """

    code: str
    applies: bool
    confidence: float
    rationale: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class RationaleAnalysis:
    """Analysis of rationales across multiple coding decisions.

    Used for iterative codebook improvement by identifying patterns
    in LLM reasoning that reveal ambiguities or issues.
    """

    code: str
    total_decisions: int = 0
    agreement_count: int = 0
    ambiguity_indicators: list[str] = field(default_factory=list)
    suggested_refinements: list[str] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        """Calculate agreement rate."""
        if self.total_decisions == 0:
            return 0.0
        return self.agreement_count / self.total_decisions

    def add_ambiguity(self, indicator: str) -> None:
        """Add an ambiguity indicator from rationale analysis."""
        if indicator not in self.ambiguity_indicators:
            self.ambiguity_indicators.append(indicator)

    def add_refinement(self, suggestion: str) -> None:
        """Add a suggested refinement for the code definition."""
        if suggestion not in self.suggested_refinements:
            self.suggested_refinements.append(suggestion)


def create_cot_prompt(
    code_definitions: list[CodeDefinition] | None = None,
    require_rationales: bool = True,
    one_code_at_time: bool = False,
) -> str:
    """Create a chain-of-thought prompting section.

    Args:
        code_definitions: Code definitions to include.
        require_rationales: Whether to require rationales for each code.
        one_code_at_time: Whether to use one-code-per-prompt approach.

    Returns:
        Formatted prompt section with CoT instructions.
    """
    sections = [CHAIN_OF_THOUGHT_TEMPLATE.strip()]

    if require_rationales:
        sections.append(RATIONALE_TEMPLATE.strip())

    if code_definitions and not one_code_at_time:
        codes_section = "\n\n".join(cd.to_prompt_section() for cd in code_definitions)
        sections.append(f"\n## Available Codes\n\n{codes_section}")

    return "\n\n".join(sections)


def create_single_code_prompt(code_def: CodeDefinition, text: str) -> str:
    """Create a prompt for single-code evaluation.

    Args:
        code_def: The code definition to evaluate.
        text: The text to evaluate against this code.

    Returns:
        Formatted prompt for single-code evaluation.
    """
    return ONE_CODE_PER_PROMPT_TEMPLATE.format(
        code_definition=code_def.to_prompt_section(),
        text=text,
    )


def analyze_rationales(
    rationales: list[CodingRationale],
    gold_standard: list[tuple[str, bool]] | None = None,
) -> dict[str, RationaleAnalysis]:
    """Analyze coding rationales to identify issues.

    This helps with iterative codebook improvement by revealing:
    - Ambiguities in code definitions
    - Common misunderstandings
    - Areas needing clarification

    Args:
        rationales: List of coding rationales to analyze.
        gold_standard: Optional list of (code, should_apply) tuples for
            comparison with human decisions.

    Returns:
        Dictionary mapping code names to their analysis.
    """
    analyses: dict[str, RationaleAnalysis] = {}

    # Group rationales by code
    for rationale in rationales:
        if rationale.code not in analyses:
            analyses[rationale.code] = RationaleAnalysis(code=rationale.code)

        analysis = analyses[rationale.code]
        analysis.total_decisions += 1

        # Check for low confidence (potential ambiguity)
        if 0.4 <= rationale.confidence <= 0.6:
            analysis.add_ambiguity("Low confidence suggests unclear criteria")

        # Check for hedging language in rationale
        hedging_words = ["might", "possibly", "unclear", "ambiguous", "could be"]
        for word in hedging_words:
            if word in rationale.rationale.lower():
                analysis.add_ambiguity(f"Hedging language used: '{word}'")
                break

    # Compare with gold standard if provided
    if gold_standard:
        gold_lookup = {code: applies for code, applies in gold_standard}
        for rationale in rationales:
            if rationale.code in gold_lookup:
                expected = gold_lookup[rationale.code]
                if rationale.applies == expected:
                    analyses[rationale.code].agreement_count += 1
                else:
                    # Disagreement - analyze why
                    msg = (
                        f"Disagreement with human: LLM said {rationale.applies}, "
                        f"expected {expected}. Rationale: "
                        f"{rationale.rationale[:100]}..."
                    )
                    analyses[rationale.code].add_refinement(msg)

    return analyses


def suggest_codebook_improvements(
    analyses: dict[str, RationaleAnalysis],
) -> list[str]:
    """Suggest improvements to codebook based on rationale analysis.

    Args:
        analyses: Analysis results from analyze_rationales.

    Returns:
        List of improvement suggestions.
    """
    suggestions = []

    for code, analysis in analyses.items():
        # Low agreement rate
        if analysis.total_decisions > 5 and analysis.agreement_rate < 0.7:
            suggestions.append(
                f"Code '{code}' has low agreement ({analysis.agreement_rate:.0%}). "
                "Consider clarifying inclusion/exclusion criteria."
            )

        # Many ambiguity indicators
        if len(analysis.ambiguity_indicators) > 2:
            suggestions.append(
                f"Code '{code}' shows ambiguity: "
                f"{', '.join(analysis.ambiguity_indicators[:3])}. "
                "Consider adding more examples or clarifying scope."
            )

        # Specific refinements
        for refinement in analysis.suggested_refinements[:2]:
            suggestions.append(f"Code '{code}': {refinement}")

    return suggestions


# Example adapted codes for common domains


def create_climate_adapted_codebook() -> AdaptedCodebook:
    """Create an LLM-adapted codebook for climate change analysis."""
    codebook = AdaptedCodebook(
        name="Climate Change Perceptions",
        description="Codes for analyzing public perceptions of climate change",
        theoretical_framework=(
            "Drawing on social constructionism and risk perception theory, "
            "this codebook captures how individuals construct meaning around "
            "climate change through emotional, cognitive, and social lenses."
        ),
    )

    codebook.add_code(
        CodeDefinition(
            code="climate_anxiety",
            description=(
                "Expressions of worry, fear, or distress specifically related "
                "to climate change and its consequences."
            ),
            scope=ScopeType.BOTH,
            inclusion_criteria=[
                "Text expresses worry about climate-related outcomes",
                "Text mentions fear or distress about environmental future",
                "Text describes feeling overwhelmed by climate issues",
            ],
            exclusion_criteria=[
                "General anxiety not connected to climate",
                "Mere acknowledgment of climate change without emotional content",
                "Discussion of others' anxiety without personal expression",
            ],
            examples=[
                "I can't stop worrying about what the planet will look like",
                "Climate change keeps me up at night",
                "I feel helpless watching the glaciers melt",
            ],
            counter_examples=[
                "Climate change is a real phenomenon",
                "Scientists say temperatures are rising",
                "Some people are worried about climate change",
            ],
            theoretical_grounding=(
                "Connects to eco-anxiety literature (Hickman et al., 2021) "
                "and risk perception theory's affect heuristic."
            ),
        )
    )

    codebook.add_code(
        CodeDefinition(
            code="hope_agency",
            description=(
                "Expressions of hope, optimism, or sense of personal/collective "
                "agency to address climate change."
            ),
            scope=ScopeType.BOTH,
            inclusion_criteria=[
                "Text expresses belief that positive change is possible",
                "Text mentions personal or collective ability to make a difference",
                "Text shows optimism about climate solutions",
            ],
            exclusion_criteria=[
                "Passive acknowledgment of potential solutions",
                "Cynical or sarcastic mentions of hope",
                "Discussion of hope in others without personal expression",
            ],
            examples=[
                "I believe we can still turn this around if we act together",
                "Every small action counts towards a better future",
                "The youth climate movement gives me hope",
            ],
            counter_examples=[
                "There are various climate solutions being proposed",
                "Some people think we can fix this",
                "Hope? Yeah right, good luck with that",
            ],
            theoretical_grounding=(
                "Connects to self-efficacy theory (Bandura) and "
                "constructive hope literature in environmental psychology."
            ),
        )
    )

    return codebook
