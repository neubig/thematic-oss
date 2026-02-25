"""Coder agent for assigning codes to text segments."""

import json
import re
from dataclasses import dataclass, field

from thematic_lm.agents.base import AgentConfig, BaseAgent
from thematic_lm.codebook import Codebook, Quote


@dataclass
class CoderConfig(AgentConfig):
    """Configuration for the Coder agent."""

    max_codes_per_segment: int = 5
    similarity_threshold: float = 0.7
    include_rationale: bool = True  # Chain-of-thought for better alignment


@dataclass
class CodeAssignment:
    """Result of coding a text segment."""

    segment_id: str
    segment_text: str
    codes: list[str]
    rationales: list[str] = field(default_factory=list)
    is_new_code: list[bool] = field(default_factory=list)


CODER_SYSTEM_PROMPT = """\
You are an expert qualitative researcher performing thematic coding.
Your task is to analyze text segments and assign meaningful codes
that capture the key concepts, themes, and patterns.

## Guidelines for Coding:
1. **Read carefully**: Understand the full meaning and context of the text
2. **Identify key concepts**: Look for important ideas, experiences, or patterns
3. **Create descriptive codes**: Codes should be concise but meaningful labels
4. **Consider existing codes**: When possible, use or adapt existing codes
5. **Be consistent**: Apply codes consistently across similar content

## Code Quality Criteria (6 Rs):
- **Reciprocal**: Codes should relate meaningfully to the data
- **Recognizable**: Codes should be clear and understandable
- **Responsive**: Codes should address the research questions
- **Resourceful**: Codes should capture nuanced meanings

{identity_section}

## Output Format:
Respond with a JSON object containing:
- "codes": List of code labels assigned to this segment
- "rationales": List of brief explanations for each code assignment
- "is_new": List of booleans indicating if each code is new (not in codebook)

Example:
```json
{{
  "codes": ["emotional support", "peer connection"],
  "rationales": ["Comfort from others", "Building peer relationships"],
  "is_new": [false, true]
}}
```"""

CODER_USER_PROMPT = """\
## Current Codebook:
{codebook_section}

## Text Segment to Code:
ID: {segment_id}
Text: "{segment_text}"

{similar_codes_section}

Please analyze this text segment and assign appropriate codes.
Provide your response as JSON."""


class CoderAgent(BaseAgent):
    """Agent that assigns codes to text segments.

    The Coder agent analyzes text and assigns codes from an existing codebook
    or creates new codes when needed. It uses semantic similarity to find
    relevant existing codes.
    """

    def __init__(
        self,
        config: CoderConfig | None = None,
        codebook: Codebook | None = None,
    ):
        """Initialize the Coder agent.

        Args:
            config: Coder configuration.
            codebook: Initial codebook to use. Created if not provided.
        """
        super().__init__(config or CoderConfig())
        self.coder_config: CoderConfig = self.config  # type: ignore
        self.codebook = codebook or Codebook()

    def get_system_prompt(self) -> str:
        """Get the system prompt with optional identity."""
        identity_section = ""
        if self.config.identity:
            identity_section = f"""
## Your Perspective:
You are coding from the following perspective: {self.config.identity}
Let this perspective inform how you interpret and code the data, while maintaining
analytical rigor and staying grounded in the text."""

        return CODER_SYSTEM_PROMPT.format(identity_section=identity_section)

    def _format_codebook_section(self) -> str:
        """Format the current codebook for the prompt."""
        if len(self.codebook) == 0:
            return "The codebook is currently empty. Create new codes as needed."

        codes_list = "\n".join(
            f"- {entry.code}" for entry in self.codebook.entries[:50]
        )
        return f"Existing codes ({len(self.codebook)} total):\n{codes_list}"

    def _format_similar_codes_section(self, text: str) -> str:
        """Find and format similar codes for the given text."""
        if len(self.codebook) == 0:
            return ""

        similar = self.codebook.find_similar_codes(text, top_k=5)
        if not similar:
            return ""

        similar_list = "\n".join(
            f"- {entry.code} (similarity: {score:.2f})"
            for entry, score in similar
            if score >= self.coder_config.similarity_threshold
        )

        if not similar_list:
            return ""

        return (
            "## Similar Existing Codes:\n"
            f"Consider using these relevant codes:\n{similar_list}"
        )

    def _parse_response(self, response: str, segment_id: str) -> CodeAssignment | None:
        """Parse the LLM response into a CodeAssignment.

        Args:
            response: The raw LLM response.
            segment_id: The ID of the segment being coded.

        Returns:
            CodeAssignment if parsing succeeds, None otherwise.
        """
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        try:
            data = json.loads(json_str)
            codes = data.get("codes", [])
            rationales = data.get("rationales", [])
            is_new = data.get("is_new", [True] * len(codes))

            # Ensure lists are the same length
            while len(rationales) < len(codes):
                rationales.append("")
            while len(is_new) < len(codes):
                is_new.append(True)

            return CodeAssignment(
                segment_id=segment_id,
                segment_text="",  # Will be filled by caller
                codes=codes[: self.coder_config.max_codes_per_segment],
                rationales=rationales[: self.coder_config.max_codes_per_segment],
                is_new_code=is_new[: self.coder_config.max_codes_per_segment],
            )
        except json.JSONDecodeError:
            return None

    def code_segment(self, segment_id: str, text: str) -> CodeAssignment:
        """Code a single text segment.

        Args:
            segment_id: Unique identifier for the segment.
            text: The text to code.

        Returns:
            CodeAssignment with assigned codes.
        """
        user_prompt = CODER_USER_PROMPT.format(
            codebook_section=self._format_codebook_section(),
            segment_id=segment_id,
            segment_text=text,
            similar_codes_section=self._format_similar_codes_section(text),
        )

        response = self._call_llm(self.get_system_prompt(), user_prompt)
        assignment = self._parse_response(response, segment_id)

        if assignment is None:
            # Fallback: return empty assignment
            assignment = CodeAssignment(
                segment_id=segment_id,
                segment_text=text,
                codes=[],
                rationales=[],
                is_new_code=[],
            )
        else:
            assignment.segment_text = text

        return assignment

    def code_segments(
        self, segments: list[tuple[str, str]], update_codebook: bool = True
    ) -> list[CodeAssignment]:
        """Code multiple text segments.

        Args:
            segments: List of (segment_id, text) tuples.
            update_codebook: Whether to add new codes to the codebook.

        Returns:
            List of CodeAssignments.
        """
        assignments = []

        for segment_id, text in segments:
            assignment = self.code_segment(segment_id, text)
            assignments.append(assignment)

            if update_codebook:
                self._update_codebook(assignment)

        return assignments

    def _update_codebook(self, assignment: CodeAssignment) -> None:
        """Update the codebook with new codes from an assignment.

        Args:
            assignment: The code assignment to process.
        """
        for i, code in enumerate(assignment.codes):
            is_new = (
                assignment.is_new_code[i] if i < len(assignment.is_new_code) else True
            )

            if is_new:
                # Check if a very similar code exists
                similar = self.codebook.find_similar_codes(code, top_k=1)
                if similar and similar[0][1] >= 0.95:
                    # Use existing code instead
                    existing_entry = similar[0][0]
                    idx = self.codebook.entries.index(existing_entry)
                    quote = Quote(
                        quote_id=assignment.segment_id, text=assignment.segment_text
                    )
                    self.codebook.add_quotes_to_code(idx, [quote])
                else:
                    # Add new code
                    quote = Quote(
                        quote_id=assignment.segment_id, text=assignment.segment_text
                    )
                    self.codebook.add_code(code, [quote])
            else:
                # Find and update existing code
                similar = self.codebook.find_similar_codes(code, top_k=1)
                if similar:
                    existing_entry = similar[0][0]
                    idx = self.codebook.entries.index(existing_entry)
                    quote = Quote(
                        quote_id=assignment.segment_id, text=assignment.segment_text
                    )
                    self.codebook.add_quotes_to_code(idx, [quote])
