"""Research Context for Qualitative Analysis.

Implements research context management for thematic analysis as described
in Naeem et al. (2025). Agents should be familiarized with research context
before coding, including:
- Research aim and questions
- Theoretical framework
- Methodology and philosophical underpinnings
- Keyword selection criteria (6 Rs framework)

Based on: Naeem, M., Smith, T., & Thomas, L. (2025). Thematic Analysis and
Artificial Intelligence: A Step-by-Step Process for Using ChatGPT in
Thematic Analysis. International Journal of Qualitative Methods, 24, 1-18.
"""

from dataclasses import dataclass, field
from enum import Enum


class ResearchParadigm(Enum):
    """Research paradigms/philosophies for qualitative research."""

    INTERPRETIVIST = "interpretivist"
    CONSTRUCTIVIST = "constructivist"
    CRITICAL = "critical"
    PRAGMATIC = "pragmatic"
    POSITIVIST = "positivist"
    PHENOMENOLOGICAL = "phenomenological"


class TheoreticalFramework(Enum):
    """Common theoretical frameworks in qualitative research."""

    GROUNDED_THEORY = "grounded_theory"
    PHENOMENOLOGY = "phenomenology"
    ETHNOGRAPHY = "ethnography"
    NARRATIVE = "narrative"
    CASE_STUDY = "case_study"
    THEMATIC_ANALYSIS = "thematic_analysis"
    CONTENT_ANALYSIS = "content_analysis"
    DISCOURSE_ANALYSIS = "discourse_analysis"


@dataclass
class ResearchContext:
    """Research context for guiding thematic analysis.

    Captures the essential elements that should inform coding decisions:
    - Research aim and questions
    - Theoretical framework and philosophy
    - Methodological approach
    - Domain-specific background

    Attributes:
        title: Title of the research study.
        aim: Primary aim or purpose of the research.
        research_questions: List of specific research questions.
        theoretical_framework: The theoretical lens guiding analysis.
        paradigm: Philosophical paradigm (interpretivist, critical, etc.).
        methodology: Methodological approach being used.
        domain: The subject domain (e.g., climate change, healthcare).
        background: Additional background context for the study.
        keywords: Key terms or concepts relevant to the research.
    """

    title: str = ""
    aim: str = ""
    research_questions: list[str] = field(default_factory=list)
    theoretical_framework: str = ""
    paradigm: str = ""
    methodology: str = "thematic_analysis"
    domain: str = ""
    background: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """Convert research context to a formatted prompt section.

        Returns:
            Formatted string for inclusion in agent prompts.
        """
        sections = []

        if self.title:
            sections.append(f"## Research Study: {self.title}")

        if self.aim:
            sections.append(f"### Research Aim\n{self.aim}")

        if self.research_questions:
            rq_list = "\n".join(
                f"{i+1}. {q}" for i, q in enumerate(self.research_questions)
            )
            sections.append(f"### Research Questions\n{rq_list}")

        if self.theoretical_framework:
            sections.append(
                f"### Theoretical Framework\n{self.theoretical_framework}"
            )

        if self.paradigm:
            sections.append(f"### Research Paradigm\n{self.paradigm}")

        if self.domain:
            sections.append(f"### Domain Context\n{self.domain}")

        if self.background:
            sections.append(f"### Background\n{self.background}")

        if self.keywords:
            kw_list = ", ".join(self.keywords)
            sections.append(f"### Key Concepts\n{kw_list}")

        return "\n\n".join(sections) if sections else ""

    def is_empty(self) -> bool:
        """Check if the context is essentially empty."""
        return not any([
            self.title,
            self.aim,
            self.research_questions,
            self.theoretical_framework,
            self.paradigm,
            self.domain,
            self.background,
            self.keywords,
        ])


# The 6 Rs Framework for Keyword and Code Selection (Naeem et al. 2025)

KEYWORD_6RS = """
## 6 Rs Framework for Keyword Selection (Naeem et al. 2025)

When selecting keywords from the data, apply these criteria:

1. **Realness**: Keywords should represent genuine, authentic expressions from
   participants, not researcher-imposed terms

2. **Richness**: Keywords should capture the depth and nuance of meaning,
   not just surface-level descriptions

3. **Repetition**: Keywords that appear frequently across the data may
   indicate important patterns (but frequency alone is not sufficient)

4. **Rationale**: There should be a clear justification for why this
   keyword captures something meaningful about the research question

5. **Repartee**: Keywords should maintain the "voice" of participants,
   preserving the interactional quality of the data

6. **Regal**: Keywords should be elevated to represent broader conceptual
   significance while staying grounded in the data
"""

CODE_6RS = """
## 6 Rs Framework for Code Quality (Naeem et al. 2025)

When assigning and evaluating codes, ensure they meet these criteria:

1. **Reciprocal**: Codes should have a mutual, bidirectional relationship
   with the data - the code illuminates the data and the data supports the code

2. **Recognizable**: Codes should be clear and understandable to other
   researchers; avoid jargon or overly abstract labels

3. **Responsive**: Codes should directly address and relate to the research
   questions being investigated

4. **Resourceful**: Codes should be analytically productive, capturing
   nuanced meanings that contribute to deeper understanding
"""

THEME_DEVELOPMENT_GUIDANCE = """
## Theme Development Guidance (Naeem et al. 2025)

Themes should be developed by:

1. **Organizing codes**: Group related codes into categories based on
   their inter-relationships, not just surface similarity

2. **Considering theory**: Let the theoretical framework guide how you
   understand relationships between codes

3. **Looking for patterns**: Identify recurring patterns that speak to
   the research questions

4. **Building coherence**: Themes should tell a coherent story about
   the data that advances understanding

5. **Staying grounded**: While abstracting from codes to themes,
   maintain connection to the original data
"""

CONCEPTUALIZATION_GUIDANCE = """
## Conceptualization Guidance (Naeem et al. 2025)

Beyond identifying themes, work toward conceptualization:

1. **Interpret coherently**: Bring together codes and themes into a
   coherent interpretation that defines new concepts

2. **Build theory**: Move from description to explanation - what do
   the themes tell us about the phenomenon?

3. **Connect to literature**: How do emerging concepts relate to
   existing theoretical frameworks?

4. **Synthesize**: Develop a conceptual model that integrates the
   findings into a coherent framework
"""


def create_methodology_prompt(
    research_context: ResearchContext | None = None,
    include_6rs_keywords: bool = False,
    include_6rs_codes: bool = True,
    include_theme_guidance: bool = False,
    include_conceptualization: bool = False,
) -> str:
    """Create a comprehensive methodology prompt section.

    Args:
        research_context: The research context to include.
        include_6rs_keywords: Include keyword selection 6Rs.
        include_6rs_codes: Include code quality 6Rs.
        include_theme_guidance: Include theme development guidance.
        include_conceptualization: Include conceptualization guidance.

    Returns:
        Formatted prompt section with methodology guidance.
    """
    sections = []

    if research_context and not research_context.is_empty():
        sections.append(research_context.to_prompt_section())

    if include_6rs_keywords:
        sections.append(KEYWORD_6RS.strip())

    if include_6rs_codes:
        sections.append(CODE_6RS.strip())

    if include_theme_guidance:
        sections.append(THEME_DEVELOPMENT_GUIDANCE.strip())

    if include_conceptualization:
        sections.append(CONCEPTUALIZATION_GUIDANCE.strip())

    return "\n\n".join(sections)


# Predefined research contexts for common domains

def create_climate_research_context(
    research_questions: list[str] | None = None,
) -> ResearchContext:
    """Create a research context for climate change studies.

    Args:
        research_questions: Specific research questions to include.

    Returns:
        ResearchContext configured for climate research.
    """
    return ResearchContext(
        title="Climate Change Perceptions Study",
        aim=(
            "To understand public perceptions, attitudes, and emotional "
            "responses to climate change and their implications for "
            "climate communication and policy"
        ),
        research_questions=research_questions or [
            "How do people perceive and make sense of climate change?",
            "What emotional responses does climate change evoke?",
            "How do perceptions vary across different demographic groups?",
            "What factors influence attitudes toward climate action?",
        ],
        theoretical_framework=(
            "Social constructionism - understanding how people construct "
            "meaning around climate change through social interaction and "
            "cultural context"
        ),
        paradigm="interpretivist",
        methodology="thematic_analysis",
        domain="Climate change, environmental psychology, public opinion",
        background=(
            "Climate change is one of the most significant challenges facing "
            "humanity. Understanding how people perceive and respond to this "
            "issue is crucial for effective communication and policy-making."
        ),
        keywords=[
            "climate change",
            "global warming",
            "environment",
            "sustainability",
            "anxiety",
            "hope",
            "action",
            "policy",
            "future",
            "responsibility",
        ],
    )


def create_healthcare_research_context(
    research_questions: list[str] | None = None,
) -> ResearchContext:
    """Create a research context for healthcare studies.

    Args:
        research_questions: Specific research questions to include.

    Returns:
        ResearchContext configured for healthcare research.
    """
    return ResearchContext(
        title="Healthcare Experience Study",
        aim=(
            "To understand patient experiences, perspectives, and needs "
            "in healthcare settings to inform patient-centered care"
        ),
        research_questions=research_questions or [
            "How do patients experience healthcare services?",
            "What factors influence patient satisfaction?",
            "How do patients navigate the healthcare system?",
            "What are patients' unmet needs?",
        ],
        theoretical_framework=(
            "Patient-centered care framework - focusing on the individual "
            "patient's experience, values, and preferences in healthcare"
        ),
        paradigm="phenomenological",
        methodology="thematic_analysis",
        domain="Healthcare, patient experience, medical sociology",
        background=(
            "Understanding patient experiences is essential for improving "
            "healthcare quality and outcomes. Qualitative research provides "
            "insights into the lived experiences of patients."
        ),
        keywords=[
            "patient",
            "care",
            "experience",
            "treatment",
            "communication",
            "support",
            "access",
            "quality",
            "satisfaction",
            "needs",
        ],
    )
