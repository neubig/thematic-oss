"""Credibility and Confirmability Evaluation.

Implements LLM-as-judge evaluation for assessing whether themes accurately
represent the data (credibility) and are data-driven rather than bias-driven
(confirmability).

Based on Section 3.1 of the Thematic-LM paper (WWW '25).
"""

import json
from dataclasses import dataclass, field

from thematic_analysis.agents.base import AgentConfig, BaseAgent


@dataclass
class CredibilityConfig(AgentConfig):
    """Configuration for the credibility evaluator.

    Inherits from AgentConfig but uses different defaults suitable
    for evaluation tasks (lower temperature for more consistent judgments).
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class QuoteConsistency:
    """Result of evaluating a single quote's consistency with a theme."""

    quote_id: str
    quote_text: str
    is_consistent: bool
    reasoning: str


@dataclass
class ThemeConsistency:
    """Result of evaluating a theme's credibility and confirmability."""

    theme_name: str
    theme_description: str
    quote_results: list[QuoteConsistency] = field(default_factory=list)
    consistent_count: int = 0
    total_count: int = 0

    @property
    def consistency_score(self) -> float:
        """Percentage of quotes consistent with the theme (0.0 to 1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.consistent_count / self.total_count


@dataclass
class CredibilityResult:
    """Overall credibility and confirmability evaluation result."""

    theme_results: list[ThemeConsistency] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Overall credibility score across all themes (0.0 to 1.0)."""
        if not self.theme_results:
            return 0.0
        total_consistent = sum(t.consistent_count for t in self.theme_results)
        total_quotes = sum(t.total_count for t in self.theme_results)
        if total_quotes == 0:
            return 0.0
        return total_consistent / total_quotes

    @property
    def num_themes(self) -> int:
        """Number of themes evaluated."""
        return len(self.theme_results)

    @property
    def total_quotes_evaluated(self) -> int:
        """Total number of quotes evaluated."""
        return sum(t.total_count for t in self.theme_results)

    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "overall_score": self.overall_score,
            "num_themes": self.num_themes,
            "total_quotes_evaluated": self.total_quotes_evaluated,
            "theme_results": [
                {
                    "theme_name": t.theme_name,
                    "theme_description": t.theme_description,
                    "consistency_score": t.consistency_score,
                    "consistent_count": t.consistent_count,
                    "total_count": t.total_count,
                    "quote_results": [
                        {
                            "quote_id": q.quote_id,
                            "quote_text": q.quote_text,
                            "is_consistent": q.is_consistent,
                            "reasoning": q.reasoning,
                        }
                        for q in t.quote_results
                    ],
                }
                for t in self.theme_results
            ],
        }

    def to_json(self) -> str:
        """Convert result to JSON format."""
        return json.dumps(self.to_dict(), indent=2)


EVALUATOR_SYSTEM_PROMPT = """\
You are an expert qualitative research evaluator assessing the credibility and
confirmability of thematic analysis results.

Your task is to evaluate whether a given quote is consistent with its assigned
theme. A quote is consistent if:
1. The quote genuinely supports or illustrates the theme
2. The connection between quote and theme is clear and logical
3. There is no evidence of hallucination (making up content not in the data)
4. There is no evidence of bias (forcing data to fit a predetermined narrative)

Respond in JSON format:
{
    "is_consistent": true/false,
    "reasoning": "Brief explanation of your assessment"
}
"""


class EvaluatorAgent(BaseAgent):
    """LLM-as-judge evaluator for credibility and confirmability.

    Assesses whether themes accurately represent the data and are data-driven
    rather than driven by biases or hallucinations. Inherits LLM handling from
    BaseAgent.
    """

    def __init__(self, config: CredibilityConfig | None = None):
        """Initialize the evaluator agent.

        Args:
            config: Evaluator configuration.
        """
        super().__init__(config or CredibilityConfig())

    def get_system_prompt(self) -> str:
        """Get the system prompt for evaluation."""
        return EVALUATOR_SYSTEM_PROMPT

    def _build_prompt(self, theme_name: str, theme_desc: str, quote: str) -> str:
        """Build evaluation prompt for a quote-theme pair.

        Args:
            theme_name: Name of the theme.
            theme_desc: Description of the theme.
            quote: The quote text to evaluate.

        Returns:
            Formatted prompt for evaluation.
        """
        return f"""\
Evaluate whether the following quote is consistent with the theme.

THEME: {theme_name}
DESCRIPTION: {theme_desc}

QUOTE: "{quote}"

Is this quote consistent with the theme? Respond in JSON format."""

    def _parse_response(self, response: str) -> tuple[bool, str]:
        """Parse the evaluator's response.

        Args:
            response: Raw response from the LLM.

        Returns:
            Tuple of (is_consistent, reasoning).
        """
        try:
            # Try to extract JSON from response
            text = response.strip()

            # Handle markdown code blocks
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                text = text[start:end].strip()

            data = json.loads(text)
            return data.get("is_consistent", False), data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            # Default to inconsistent if parsing fails
            return False, f"Failed to parse response: {response[:100]}"

    def evaluate_quote(
        self,
        theme_name: str,
        theme_description: str,
        quote_id: str,
        quote_text: str,
    ) -> QuoteConsistency:
        """Evaluate a single quote's consistency with a theme.

        Args:
            theme_name: Name of the theme.
            theme_description: Description of the theme.
            quote_id: ID of the quote.
            quote_text: Text of the quote.

        Returns:
            QuoteConsistency result.
        """
        prompt = self._build_prompt(theme_name, theme_description, quote_text)

        content = self._call_llm(EVALUATOR_SYSTEM_PROMPT, prompt)
        is_consistent, reasoning = self._parse_response(content)

        return QuoteConsistency(
            quote_id=quote_id,
            quote_text=quote_text,
            is_consistent=is_consistent,
            reasoning=reasoning,
        )

    def evaluate_theme(
        self,
        theme_name: str,
        theme_description: str,
        quotes: list[tuple[str, str]],
    ) -> ThemeConsistency:
        """Evaluate a theme's credibility based on its associated quotes.

        Args:
            theme_name: Name of the theme.
            theme_description: Description of the theme.
            quotes: List of (quote_id, quote_text) tuples.

        Returns:
            ThemeConsistency result.
        """
        quote_results: list[QuoteConsistency] = []
        consistent_count = 0

        for quote_id, quote_text in quotes:
            result = self.evaluate_quote(
                theme_name, theme_description, quote_id, quote_text
            )
            quote_results.append(result)
            if result.is_consistent:
                consistent_count += 1

        return ThemeConsistency(
            theme_name=theme_name,
            theme_description=theme_description,
            quote_results=quote_results,
            consistent_count=consistent_count,
            total_count=len(quotes),
        )

    def evaluate(
        self,
        themes: list[dict],
    ) -> CredibilityResult:
        """Evaluate credibility and confirmability of themes.

        Args:
            themes: List of theme dictionaries with keys:
                - name: Theme name
                - description: Theme description
                - quotes: List of (quote_id, quote_text) tuples

        Returns:
            CredibilityResult with overall and per-theme scores.
        """
        theme_results: list[ThemeConsistency] = []

        for theme in themes:
            result = self.evaluate_theme(
                theme_name=theme["name"],
                theme_description=theme["description"],
                quotes=theme.get("quotes", []),
            )
            theme_results.append(result)

        return CredibilityResult(theme_results=theme_results)

    def evaluate_from_pipeline_result(self, pipeline_result) -> CredibilityResult:
        """Evaluate credibility from a PipelineResult.

        Args:
            pipeline_result: A PipelineResult from ThematicLMPipeline.

        Returns:
            CredibilityResult with evaluation scores.
        """
        themes = []
        for merged_theme in pipeline_result.themes.themes:
            quotes = [(q.quote_id, q.text) for q in merged_theme.quotes if q.quote_id]
            themes.append(
                {
                    "name": merged_theme.name,
                    "description": merged_theme.description,
                    "quotes": quotes,
                }
            )
        return self.evaluate(themes)
