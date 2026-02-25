"""Enhanced persona system with user-uploaded context.

This module extends the identity perspective system to support personalized
personas based on researcher-provided materials such as:
- Prior publications
- Analytical memos
- Research artifacts (codebooks, theme maps)
- Theoretical frameworks
- Domain expertise descriptions

The system extracts relevant context from uploaded materials and generates
personas that reflect the researcher's unique perspective and methodology.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from thematic_lm.identity import IdentityPerspective


class ContextType(Enum):
    """Types of context that can be uploaded."""

    PAPER = "paper"  # Academic paper or publication
    MEMO = "memo"  # Analytical or research memo
    CODEBOOK = "codebook"  # Prior codebook or coding scheme
    FRAMEWORK = "framework"  # Theoretical framework description
    EXPERTISE = "expertise"  # Domain expertise description
    ARTIFACT = "artifact"  # Other research artifact


@dataclass
class ContextDocument:
    """A document providing context for persona generation.

    Attributes:
        content: The text content of the document.
        context_type: Type of context (paper, memo, etc.).
        title: Optional title for the document.
        summary: Optional pre-computed summary.
        keywords: Extracted keywords or key concepts.
        methodology_notes: Extracted methodology-related content.
        theoretical_notes: Extracted theory-related content.
    """

    content: str
    context_type: ContextType = ContextType.ARTIFACT
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    methodology_notes: str = ""
    theoretical_notes: str = ""

    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)

    def get_word_count(self) -> int:
        """Return approximate word count."""
        return len(self.content.split())


@dataclass
class PersonaContext:
    """Aggregated context for persona generation.

    Combines multiple context documents into a unified representation
    that can be used to generate personalized personas.

    Attributes:
        documents: List of context documents.
        researcher_name: Name of the researcher (optional).
        research_domain: Primary research domain.
        methodological_approach: Summary of methodological preferences.
        theoretical_orientation: Summary of theoretical leanings.
        key_concepts: Important concepts from the researcher's work.
        writing_style_notes: Notes about analytical writing style.
    """

    documents: list[ContextDocument] = field(default_factory=list)
    researcher_name: str = ""
    research_domain: str = ""
    methodological_approach: str = ""
    theoretical_orientation: str = ""
    key_concepts: list[str] = field(default_factory=list)
    writing_style_notes: str = ""

    def add_document(self, doc: ContextDocument) -> None:
        """Add a context document."""
        self.documents.append(doc)

    def get_total_word_count(self) -> int:
        """Get total word count across all documents."""
        return sum(doc.get_word_count() for doc in self.documents)

    def is_empty(self) -> bool:
        """Check if context is empty."""
        return len(self.documents) == 0 and not any(
            [
                self.research_domain,
                self.methodological_approach,
                self.theoretical_orientation,
                self.key_concepts,
            ]
        )


class PersonaGenerator:
    """Generates personalized personas from context documents.

    This class extracts relevant information from uploaded documents
    and generates identity perspectives that reflect the researcher's
    unique analytical approach.
    """

    def __init__(self, max_context_tokens: int = 2000):
        """Initialize the persona generator.

        Args:
            max_context_tokens: Maximum tokens for context in prompts.
        """
        self.max_context_tokens = max_context_tokens

    def extract_context(self, doc: ContextDocument) -> ContextDocument:
        """Extract relevant context from a document.

        This performs basic extraction. In a production system, this would
        use NLP or LLM-based extraction for better results.

        Args:
            doc: The document to process.

        Returns:
            The document with extracted context added.
        """
        # Extract methodology-related content
        methodology_markers = [
            "methodology",
            "method",
            "approach",
            "qualitative",
            "quantitative",
            "analysis",
            "coding",
            "thematic",
            "grounded theory",
            "phenomenolog",
            "ethnograph",
        ]
        methodology_sentences = []
        for sentence in doc.content.split("."):
            if any(marker in sentence.lower() for marker in methodology_markers):
                methodology_sentences.append(sentence.strip())
        if methodology_sentences:
            doc.methodology_notes = ". ".join(methodology_sentences[:5]) + "."

        # Extract theory-related content
        theory_markers = [
            "theory",
            "theoretic",
            "framework",
            "conceptual",
            "paradigm",
            "perspective",
            "lens",
            "approach",
        ]
        theory_sentences = []
        for sentence in doc.content.split("."):
            if any(marker in sentence.lower() for marker in theory_markers):
                theory_sentences.append(sentence.strip())
        if theory_sentences:
            doc.theoretical_notes = ". ".join(theory_sentences[:5]) + "."

        # Extract keywords (simple frequency-based for now)
        words = doc.content.lower().split()
        # Filter common words and short words
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "their",
            "them",
            "we",
            "our",
            "us",
            "i",
            "my",
            "me",
            "you",
            "your",
        }
        word_freq: dict[str, int] = {}
        for word in words:
            word = word.strip(".,!?;:'\"()[]{}").lower()
            if len(word) > 4 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        doc.keywords = [w for w, _ in sorted_words[:20]]

        return doc

    def generate_summary(self, doc: ContextDocument) -> str:
        """Generate a summary of the document.

        For production use, this should use an LLM for better summaries.
        This implementation provides a basic extractive summary.

        Args:
            doc: The document to summarize.

        Returns:
            A summary string.
        """
        sentences = doc.content.split(".")
        # Take first few sentences as summary (simple approach)
        summary_sentences = sentences[:3]
        return ". ".join(s.strip() for s in summary_sentences if s.strip()) + "."

    def compress_context(
        self, context: PersonaContext, max_tokens: int | None = None
    ) -> str:
        """Compress context for inclusion in prompts.

        Args:
            context: The persona context to compress.
            max_tokens: Maximum tokens (uses default if not specified).

        Returns:
            Compressed context string suitable for prompts.
        """
        max_tokens = max_tokens or self.max_context_tokens
        # Estimate ~4 chars per token
        max_chars = max_tokens * 4

        sections = []

        if context.researcher_name:
            sections.append(f"Researcher: {context.researcher_name}")

        if context.research_domain:
            sections.append(f"Domain: {context.research_domain}")

        if context.methodological_approach:
            sections.append(
                f"Methodology: {context.methodological_approach[:500]}..."
                if len(context.methodological_approach) > 500
                else f"Methodology: {context.methodological_approach}"
            )

        if context.theoretical_orientation:
            sections.append(
                f"Theory: {context.theoretical_orientation[:500]}..."
                if len(context.theoretical_orientation) > 500
                else f"Theory: {context.theoretical_orientation}"
            )

        if context.key_concepts:
            sections.append(f"Key concepts: {', '.join(context.key_concepts[:10])}")

        # Add document summaries
        for doc in context.documents[:3]:
            if doc.summary:
                doc_section = f"From {doc.title or doc.context_type.value}: "
                doc_section += (
                    doc.summary[:200] + "..." if len(doc.summary) > 200 else doc.summary
                )
                sections.append(doc_section)

        result = "\n".join(sections)

        # Truncate if needed
        if len(result) > max_chars:
            result = result[:max_chars] + "..."

        return result

    def generate_persona(
        self,
        context: PersonaContext,
        persona_name: str | None = None,
    ) -> IdentityPerspective:
        """Generate an identity perspective from context.

        Args:
            context: The persona context to use.
            persona_name: Optional name for the persona.

        Returns:
            An IdentityPerspective reflecting the context.
        """
        name = persona_name or f"custom_{context.researcher_name or 'researcher'}"

        # Build description from context
        description_parts = []

        if context.researcher_name:
            description_parts.append(
                f"You embody the analytical perspective of {context.researcher_name}, "
                "a researcher with specific expertise and methodological preferences."
            )
        else:
            description_parts.append(
                "You embody a personalized analytical perspective based on the "
                "provided research context and materials."
            )

        if context.research_domain:
            description_parts.append(
                f"Your expertise is in {context.research_domain}. "
                "Apply domain-specific knowledge when analyzing data."
            )

        if context.methodological_approach:
            description_parts.append(
                f"Methodological approach: {context.methodological_approach}"
            )

        if context.theoretical_orientation:
            description_parts.append(
                f"Theoretical orientation: {context.theoretical_orientation}"
            )

        if context.key_concepts:
            concepts = ", ".join(context.key_concepts[:10])
            description_parts.append(
                f"Pay particular attention to concepts such as: {concepts}."
            )

        if context.writing_style_notes:
            description_parts.append(f"Analytical style: {context.writing_style_notes}")

        # Add notes from documents
        for doc in context.documents[:2]:
            if doc.methodology_notes:
                description_parts.append(
                    f"From prior work: {doc.methodology_notes[:200]}"
                )

        description = " ".join(description_parts)

        return IdentityPerspective(name=name, description=description)


def load_document_from_text(
    content: str,
    context_type: ContextType = ContextType.ARTIFACT,
    title: str = "",
) -> ContextDocument:
    """Load a context document from text content.

    Args:
        content: The text content.
        context_type: Type of context.
        title: Optional title.

    Returns:
        A ContextDocument instance.
    """
    return ContextDocument(content=content, context_type=context_type, title=title)


def load_document_from_file(
    path: str | Path,
    context_type: ContextType | None = None,
) -> ContextDocument:
    """Load a context document from a file.

    Supports .txt and .md files. PDF support would require additional
    dependencies in a production implementation.

    Args:
        path: Path to the file.
        context_type: Type of context (inferred from extension if not provided).

    Returns:
        A ContextDocument instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file type is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Infer context type from filename if not provided
    if context_type is None:
        filename_lower = path.name.lower()
        if "memo" in filename_lower:
            context_type = ContextType.MEMO
        elif "codebook" in filename_lower:
            context_type = ContextType.CODEBOOK
        elif "framework" in filename_lower:
            context_type = ContextType.FRAMEWORK
        elif "paper" in filename_lower or path.suffix == ".pdf":
            context_type = ContextType.PAPER
        else:
            context_type = ContextType.ARTIFACT

    # Read content based on file type
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        content = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .md")

    return ContextDocument(
        content=content,
        context_type=context_type,
        title=path.stem,
    )


def create_persona_from_documents(
    documents: list[ContextDocument],
    researcher_name: str = "",
    research_domain: str = "",
) -> IdentityPerspective:
    """Convenience function to create a persona from documents.

    Args:
        documents: List of context documents.
        researcher_name: Optional researcher name.
        research_domain: Optional research domain.

    Returns:
        An IdentityPerspective reflecting the documents.
    """
    generator = PersonaGenerator()

    # Extract context from each document
    processed_docs = [generator.extract_context(doc) for doc in documents]

    # Build persona context
    context = PersonaContext(
        documents=processed_docs,
        researcher_name=researcher_name,
        research_domain=research_domain,
    )

    # Aggregate methodology notes
    methodology_parts = [
        doc.methodology_notes for doc in processed_docs if doc.methodology_notes
    ]
    if methodology_parts:
        context.methodological_approach = " ".join(methodology_parts[:3])

    # Aggregate theoretical notes
    theory_parts = [
        doc.theoretical_notes for doc in processed_docs if doc.theoretical_notes
    ]
    if theory_parts:
        context.theoretical_orientation = " ".join(theory_parts[:3])

    # Aggregate keywords
    all_keywords: dict[str, int] = {}
    for doc in processed_docs:
        for kw in doc.keywords:
            all_keywords[kw] = all_keywords.get(kw, 0) + 1
    sorted_kw = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
    context.key_concepts = [kw for kw, _ in sorted_kw[:15]]

    return generator.generate_persona(context)


def create_domain_expert_persona(
    domain: str,
    expertise_description: str,
    methodological_preference: str = "",
    theoretical_lens: str = "",
) -> IdentityPerspective:
    """Create a domain expert persona without uploading documents.

    Args:
        domain: The research domain (e.g., "healthcare", "education").
        expertise_description: Description of domain expertise.
        methodological_preference: Preferred methodology (optional).
        theoretical_lens: Theoretical perspective (optional).

    Returns:
        An IdentityPerspective for a domain expert.
    """
    context = PersonaContext(
        research_domain=domain,
        methodological_approach=methodological_preference,
        theoretical_orientation=theoretical_lens,
        key_concepts=[],
    )

    # Parse expertise description for key concepts
    words = expertise_description.lower().split()
    # Simple extraction of multi-word terms
    for i in range(len(words) - 1):
        if len(words[i]) > 4 and len(words[i + 1]) > 4:
            context.key_concepts.append(f"{words[i]} {words[i + 1]}")

    context.key_concepts = context.key_concepts[:10]

    # Build custom description
    description = f"You are an expert in {domain}. {expertise_description} "

    if methodological_preference:
        description += f"Your methodological approach: {methodological_preference}. "

    if theoretical_lens:
        description += f"You analyze data through the lens of {theoretical_lens}. "

    description += (
        "Apply your domain expertise when coding and interpreting data, "
        "bringing specialized knowledge to identify patterns and themes "
        "that a generalist might miss."
    )

    return IdentityPerspective(
        name=f"expert_{domain.lower().replace(' ', '_')}",
        description=description,
    )
