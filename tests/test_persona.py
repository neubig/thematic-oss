"""Tests for persona module."""

import tempfile
from pathlib import Path

import pytest

from thematic_lm.persona import (
    ContextDocument,
    ContextType,
    PersonaContext,
    PersonaGenerator,
    create_domain_expert_persona,
    create_persona_from_documents,
    load_document_from_file,
    load_document_from_text,
)


class TestContextType:
    """Tests for ContextType enum."""

    def test_context_type_values(self):
        """Test that all context types have expected values."""
        assert ContextType.PAPER.value == "paper"
        assert ContextType.MEMO.value == "memo"
        assert ContextType.CODEBOOK.value == "codebook"
        assert ContextType.FRAMEWORK.value == "framework"
        assert ContextType.EXPERTISE.value == "expertise"
        assert ContextType.ARTIFACT.value == "artifact"


class TestContextDocument:
    """Tests for ContextDocument dataclass."""

    def test_basic_creation(self):
        """Test creating a basic document."""
        doc = ContextDocument(content="Sample content")
        assert doc.content == "Sample content"
        assert doc.context_type == ContextType.ARTIFACT
        assert doc.title == ""

    def test_full_creation(self):
        """Test creating a fully specified document."""
        doc = ContextDocument(
            content="Research paper content",
            context_type=ContextType.PAPER,
            title="My Research Paper",
            summary="A summary",
            keywords=["research", "analysis"],
            methodology_notes="Qualitative approach",
            theoretical_notes="Grounded theory",
        )
        assert doc.context_type == ContextType.PAPER
        assert doc.title == "My Research Paper"
        assert len(doc.keywords) == 2

    def test_len(self):
        """Test document length."""
        doc = ContextDocument(content="Hello world")
        assert len(doc) == 11

    def test_word_count(self):
        """Test word count calculation."""
        doc = ContextDocument(content="This is a test sentence with seven words")
        assert doc.get_word_count() == 8


class TestPersonaContext:
    """Tests for PersonaContext dataclass."""

    def test_empty_context(self):
        """Test creating empty context."""
        ctx = PersonaContext()
        assert ctx.is_empty()
        assert len(ctx.documents) == 0

    def test_non_empty_context_with_documents(self):
        """Test context with documents is not empty."""
        ctx = PersonaContext()
        ctx.add_document(ContextDocument(content="test"))
        assert not ctx.is_empty()

    def test_non_empty_context_with_domain(self):
        """Test context with domain is not empty."""
        ctx = PersonaContext(research_domain="healthcare")
        assert not ctx.is_empty()

    def test_add_document(self):
        """Test adding documents."""
        ctx = PersonaContext()
        doc1 = ContextDocument(content="First")
        doc2 = ContextDocument(content="Second")
        ctx.add_document(doc1)
        ctx.add_document(doc2)
        assert len(ctx.documents) == 2

    def test_total_word_count(self):
        """Test total word count across documents."""
        ctx = PersonaContext()
        ctx.add_document(ContextDocument(content="one two three"))
        ctx.add_document(ContextDocument(content="four five"))
        assert ctx.get_total_word_count() == 5


class TestPersonaGenerator:
    """Tests for PersonaGenerator class."""

    @pytest.fixture
    def generator(self) -> PersonaGenerator:
        """Create a generator for tests."""
        return PersonaGenerator()

    @pytest.fixture
    def sample_doc(self) -> ContextDocument:
        """Create a sample document for testing."""
        content = """
        This research paper presents a qualitative methodology for studying
        healthcare experiences. Our theoretical framework draws on social
        constructionism and phenomenology to understand patient perspectives.
        
        We used thematic analysis as our primary method of data analysis,
        following the approach outlined by Braun and Clarke. The coding
        process involved iterative refinement of themes.
        
        Key findings include the importance of communication, trust, and
        empathy in healthcare interactions. Patients valued being heard
        and understood by their healthcare providers.
        """
        return ContextDocument(
            content=content,
            context_type=ContextType.PAPER,
            title="Healthcare Study",
        )

    def test_initialization(self, generator: PersonaGenerator):
        """Test generator initialization."""
        assert generator.max_context_tokens == 2000

    def test_custom_max_tokens(self):
        """Test custom max tokens."""
        gen = PersonaGenerator(max_context_tokens=1000)
        assert gen.max_context_tokens == 1000

    def test_extract_context(self, generator: PersonaGenerator, sample_doc):
        """Test context extraction from document."""
        processed = generator.extract_context(sample_doc)

        # Should extract methodology notes
        assert processed.methodology_notes != ""
        assert any(
            word in processed.methodology_notes.lower()
            for word in ["methodology", "qualitative", "analysis"]
        )

        # Should extract theoretical notes
        assert processed.theoretical_notes != ""
        assert any(
            word in processed.theoretical_notes.lower()
            for word in ["theoretical", "framework", "theory"]
        )

        # Should extract keywords
        assert len(processed.keywords) > 0

    def test_generate_summary(self, generator: PersonaGenerator, sample_doc):
        """Test summary generation."""
        summary = generator.generate_summary(sample_doc)
        assert len(summary) > 0
        # Should be shorter than original
        assert len(summary) < len(sample_doc.content)

    def test_compress_context(self, generator: PersonaGenerator):
        """Test context compression."""
        ctx = PersonaContext(
            researcher_name="Dr. Smith",
            research_domain="Healthcare",
            methodological_approach="Qualitative thematic analysis",
            theoretical_orientation="Social constructionism",
            key_concepts=["patient", "experience", "communication"],
        )

        compressed = generator.compress_context(ctx)

        assert "Dr. Smith" in compressed
        assert "Healthcare" in compressed
        assert "thematic" in compressed.lower() or "Qualitative" in compressed

    def test_compress_context_respects_max_tokens(self):
        """Test that compression respects token limit."""
        gen = PersonaGenerator(max_context_tokens=50)  # Very small
        ctx = PersonaContext(
            methodological_approach="A " * 500,  # Very long
        )

        compressed = gen.compress_context(ctx)
        # Should be truncated (50 tokens * 4 chars ≈ 200 chars)
        assert len(compressed) <= 250  # Some margin for formatting

    def test_generate_persona_basic(self, generator: PersonaGenerator):
        """Test basic persona generation."""
        ctx = PersonaContext(research_domain="healthcare")

        persona = generator.generate_persona(ctx)

        assert persona.name.startswith("custom_")
        assert "healthcare" in persona.description.lower()

    def test_generate_persona_with_name(self, generator: PersonaGenerator):
        """Test persona generation with researcher name."""
        ctx = PersonaContext(
            researcher_name="Dr. Jane Smith",
            research_domain="education",
        )

        persona = generator.generate_persona(ctx)

        assert "Dr. Jane Smith" in persona.description
        assert "education" in persona.description.lower()

    def test_generate_persona_with_methodology(self, generator: PersonaGenerator):
        """Test persona generation with methodology."""
        ctx = PersonaContext(
            methodological_approach="Grounded theory with constant comparison",
        )

        persona = generator.generate_persona(ctx)

        assert "grounded theory" in persona.description.lower()

    def test_generate_persona_with_theory(self, generator: PersonaGenerator):
        """Test persona generation with theoretical orientation."""
        ctx = PersonaContext(
            theoretical_orientation="Critical realism and systems thinking",
        )

        persona = generator.generate_persona(ctx)

        assert (
            "critical realism" in persona.description.lower()
            or "systems thinking" in persona.description.lower()
            or "Theoretical" in persona.description
        )

    def test_generate_persona_with_concepts(self, generator: PersonaGenerator):
        """Test persona generation with key concepts."""
        ctx = PersonaContext(
            key_concepts=["resilience", "adaptation", "coping"],
        )

        persona = generator.generate_persona(ctx)

        assert any(
            concept in persona.description.lower()
            for concept in ["resilience", "adaptation", "coping"]
        )


class TestLoadFunctions:
    """Tests for document loading functions."""

    def test_load_document_from_text(self):
        """Test loading document from text."""
        doc = load_document_from_text(
            content="Sample content here",
            context_type=ContextType.MEMO,
            title="My Memo",
        )

        assert doc.content == "Sample content here"
        assert doc.context_type == ContextType.MEMO
        assert doc.title == "My Memo"

    def test_load_document_from_file_txt(self):
        """Test loading document from .txt file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content from a file.")
            temp_path = f.name

        try:
            doc = load_document_from_file(temp_path)
            assert "test content" in doc.content
        finally:
            Path(temp_path).unlink()

    def test_load_document_from_file_md(self):
        """Test loading document from .md file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Heading\n\nMarkdown content here.")
            temp_path = f.name

        try:
            doc = load_document_from_file(temp_path)
            assert "Markdown content" in doc.content
        finally:
            Path(temp_path).unlink()

    def test_load_document_infers_type_from_filename(self):
        """Test that context type is inferred from filename."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="memo_", delete=False
        ) as f:
            f.write("Analytical memo content")
            temp_path = f.name

        try:
            doc = load_document_from_file(temp_path)
            assert doc.context_type == ContextType.MEMO
        finally:
            Path(temp_path).unlink()

    def test_load_document_from_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            load_document_from_file("/nonexistent/path/file.txt")

    def test_load_document_from_file_unsupported_type(self):
        """Test error for unsupported file type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document_from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestCreatePersonaFromDocuments:
    """Tests for create_persona_from_documents function."""

    def test_single_document(self):
        """Test persona creation from single document."""
        doc = ContextDocument(
            content="""
            Our methodology uses thematic analysis following Braun and Clarke's
            six-phase approach. The theoretical framework is grounded in
            phenomenology and social constructionism.
            """,
            context_type=ContextType.PAPER,
        )

        persona = create_persona_from_documents([doc])

        assert persona.name.startswith("custom_")
        # Should extract some methodology content
        assert len(persona.description) > 50

    def test_multiple_documents(self):
        """Test persona creation from multiple documents."""
        docs = [
            ContextDocument(
                content="Research on patient experience using qualitative methods",
                context_type=ContextType.PAPER,
            ),
            ContextDocument(
                content="Analytical memo: observed themes of trust and communication",
                context_type=ContextType.MEMO,
            ),
        ]

        persona = create_persona_from_documents(docs)

        assert persona.name.startswith("custom_")
        assert len(persona.description) > 50

    def test_with_researcher_name(self):
        """Test persona includes researcher name."""
        doc = ContextDocument(content="Research content")

        persona = create_persona_from_documents(
            [doc],
            researcher_name="Dr. Research",
        )

        assert "Dr. Research" in persona.description

    def test_with_domain(self):
        """Test persona includes research domain."""
        doc = ContextDocument(content="Research content")

        persona = create_persona_from_documents(
            [doc],
            research_domain="climate psychology",
        )

        assert "climate psychology" in persona.description.lower()


class TestCreateDomainExpertPersona:
    """Tests for create_domain_expert_persona function."""

    def test_basic_expert(self):
        """Test creating basic domain expert."""
        persona = create_domain_expert_persona(
            domain="healthcare",
            expertise_description="Expert in patient-centered care and communication",
        )

        assert "healthcare" in persona.name
        assert "healthcare" in persona.description.lower()
        assert (
            "patient" in persona.description.lower() or "Expert" in persona.description
        )

    def test_expert_with_methodology(self):
        """Test expert persona with methodology."""
        persona = create_domain_expert_persona(
            domain="education",
            expertise_description="Specialist in classroom observation",
            methodological_preference="Ethnographic approach with fieldwork",
        )

        assert (
            "Ethnographic" in persona.description
            or "fieldwork" in persona.description.lower()
        )

    def test_expert_with_theory(self):
        """Test expert persona with theoretical lens."""
        persona = create_domain_expert_persona(
            domain="sociology",
            expertise_description="Expert in social movements",
            theoretical_lens="Critical theory and power dynamics",
        )

        assert (
            "Critical theory" in persona.description
            or "power" in persona.description.lower()
            or "lens" in persona.description
        )

    def test_expert_name_formatting(self):
        """Test that expert name is formatted correctly."""
        persona = create_domain_expert_persona(
            domain="Health Care Systems",
            expertise_description="Expert",
        )

        assert persona.name == "expert_health_care_systems"


class TestIntegration:
    """Integration tests for persona system."""

    def test_full_workflow(self):
        """Test full workflow from documents to persona."""
        # Create sample documents
        paper = ContextDocument(
            content="""
            This paper presents findings from a qualitative study on climate
            change perceptions. Using thematic analysis methodology based on
            Braun and Clarke's framework, we identified five key themes.
            Our theoretical approach draws on social constructionism to
            understand how people make meaning of climate information.
            Key concepts include climate anxiety, hope, agency, and resilience.
            """,
            context_type=ContextType.PAPER,
            title="Climate Perceptions Study",
        )

        memo = ContextDocument(
            content="""
            Analytical memo from coding session 3:
            Observed interesting patterns around emotional responses.
            Participants express both anxiety and hope when discussing future.
            The theme of personal agency seems central to coping strategies.
            """,
            context_type=ContextType.MEMO,
            title="Coding Memo 3",
        )

        # Generate persona
        persona = create_persona_from_documents(
            [paper, memo],
            researcher_name="Dr. Climate",
            research_domain="Environmental Psychology",
        )

        # Verify persona
        assert "Dr. Climate" in persona.description
        assert "Environmental Psychology" in persona.description
        # Should have extracted some concepts
        assert len(persona.description) > 100

    def test_persona_usable_in_agent(self):
        """Test that generated persona can be used as agent identity."""
        persona = create_domain_expert_persona(
            domain="education",
            expertise_description="Specialist in teacher professional development",
        )

        # Persona should have required attributes for use as identity
        assert hasattr(persona, "name")
        assert hasattr(persona, "description")
        assert isinstance(persona.name, str)
        assert isinstance(persona.description, str)
        assert len(persona.description) > 0

        # Should work as identity string
        identity_str = str(persona)
        assert len(identity_str) > 0
