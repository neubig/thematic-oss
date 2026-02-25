"""Two-stage pipeline for thematic analysis."""

import json
import uuid
from dataclasses import dataclass, field

from thematic_lm.agents import (
    AggregationResult,
    AggregatorConfig,
    CodeAggregatorAgent,
    CodeAssignment,
    CoderAgent,
    CoderConfig,
    ReviewerAgent,
    ReviewerConfig,
    ThemeAggregationResult,
    ThemeAggregatorAgent,
    ThemeAggregatorConfig,
    ThemeCoderAgent,
    ThemeCoderConfig,
    ThemeResult,
)
from thematic_lm.codebook import Codebook, EmbeddingService


@dataclass
class PipelineConfig:
    """Configuration for the thematic analysis pipeline."""

    # Stage 1: Coding configuration
    num_coders: int = 3
    coder_config: CoderConfig = field(default_factory=CoderConfig)
    aggregator_config: AggregatorConfig = field(default_factory=AggregatorConfig)
    reviewer_config: ReviewerConfig = field(default_factory=ReviewerConfig)

    # Stage 2: Theme development configuration
    num_theme_coders: int = 3
    theme_coder_config: ThemeCoderConfig = field(default_factory=ThemeCoderConfig)
    theme_aggregator_config: ThemeAggregatorConfig = field(
        default_factory=ThemeAggregatorConfig
    )

    # Processing configuration
    batch_size: int = 10
    use_mock_embeddings: bool = False

    # Coder identities (optional)
    coder_identities: list[str] = field(default_factory=list)
    theme_coder_identities: list[str] = field(default_factory=list)


@dataclass
class DataSegment:
    """A segment of data to be coded."""

    segment_id: str
    text: str


@dataclass
class PipelineResult:
    """Result of running the thematic analysis pipeline."""

    themes: ThemeAggregationResult
    codebook: Codebook
    stage1_aggregations: list[AggregationResult] = field(default_factory=list)

    def to_json(self) -> str:
        """Convert result to JSON format."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "themes": self.themes.to_dict(),
            "codebook": self.codebook.to_dict(),
            "stage1_aggregations": [a.to_dict() for a in self.stage1_aggregations],
        }


class ThematicLMPipeline:
    """Two-stage pipeline for large-scale thematic analysis.

    Stage 1 (Coding): Data → Multiple Coders → Aggregator → Reviewer → Codebook
    Stage 2 (Theme Development): Codebook → Theme Coders → Theme Aggregator → Themes

    Based on Section 3 of 'Thematic-LM: A LLM-based Multi-agent System for
    Large-scale Thematic Analysis' (WWW '25)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
            embedding_service: Service for computing embeddings.
        """
        self.config = config or PipelineConfig()
        self.embedding_service = embedding_service or EmbeddingService(
            use_mock=self.config.use_mock_embeddings
        )

        # Initialize codebook (max_quotes_per_code defaults to reasonable value)
        self.codebook = Codebook(
            embedding_service=self.embedding_service,
        )

        # Initialize Stage 1 agents
        self._coders: list[CoderAgent] = []
        self._aggregator: CodeAggregatorAgent | None = None
        self._reviewer: ReviewerAgent | None = None

        # Initialize Stage 2 agents
        self._theme_coders: list[ThemeCoderAgent] = []
        self._theme_aggregator: ThemeAggregatorAgent | None = None

    def _init_stage1_agents(self) -> None:
        """Initialize Stage 1 (coding) agents."""
        # Create coders with optional identities
        self._coders = []
        for i in range(self.config.num_coders):
            coder_config = CoderConfig(
                max_codes_per_segment=self.config.coder_config.max_codes_per_segment,
                model=self.config.coder_config.model,
                temperature=self.config.coder_config.temperature,
                max_tokens=self.config.coder_config.max_tokens,
            )

            # Set identity if available
            if i < len(self.config.coder_identities):
                coder_config.identity = self.config.coder_identities[i]

            coder = CoderAgent(config=coder_config, codebook=self.codebook)
            self._coders.append(coder)

        # Create aggregator
        self._aggregator = CodeAggregatorAgent(
            config=self.config.aggregator_config,
            codebook=self.codebook,
        )

        # Create reviewer
        self._reviewer = ReviewerAgent(
            config=self.config.reviewer_config,
            codebook=self.codebook,
        )

    def _init_stage2_agents(self) -> None:
        """Initialize Stage 2 (theme development) agents."""
        # Create theme coders with optional identities
        self._theme_coders = []
        for i in range(self.config.num_theme_coders):
            theme_config = ThemeCoderConfig(
                max_themes=self.config.theme_coder_config.max_themes,
                min_codes_per_theme=self.config.theme_coder_config.min_codes_per_theme,
                model=self.config.theme_coder_config.model,
                temperature=self.config.theme_coder_config.temperature,
                max_tokens=self.config.theme_coder_config.max_tokens,
            )

            # Set identity if available
            if i < len(self.config.theme_coder_identities):
                theme_config.identity = self.config.theme_coder_identities[i]

            theme_coder = ThemeCoderAgent(
                config=theme_config,
                codebook=self.codebook,
            )
            self._theme_coders.append(theme_coder)

        # Create theme aggregator
        self._theme_aggregator = ThemeAggregatorAgent(
            config=self.config.theme_aggregator_config,
            embedding_service=self.embedding_service,
        )

    def _process_batch_stage1(
        self, segments: list[DataSegment]
    ) -> list[AggregationResult]:
        """Process a batch of segments through Stage 1.

        Args:
            segments: List of data segments to code.

        Returns:
            List of aggregation results.
        """
        if not self._coders or not self._aggregator:
            self._init_stage1_agents()

        all_aggregations: list[AggregationResult] = []

        for segment in segments:
            # Get codes from all coders
            all_assignments: list[CodeAssignment] = []
            for coder in self._coders:
                assignment = coder.code_segment(
                    segment.segment_id,
                    segment.text,
                )
                all_assignments.append(assignment)

            # Aggregate codes
            assert self._aggregator is not None
            aggregation = self._aggregator.aggregate(all_assignments)
            all_aggregations.append(aggregation)

            # Review and update codebook
            assert self._reviewer is not None
            self._reviewer.process_aggregation_result(aggregation)

        return all_aggregations

    def run_stage1(
        self, segments: list[DataSegment]
    ) -> tuple[Codebook, list[AggregationResult]]:
        """Run Stage 1: Coding.

        Processes data segments through multiple coders, aggregates their codes,
        and reviews them to build a codebook.

        Args:
            segments: List of data segments to code.

        Returns:
            Tuple of (codebook, aggregation_results).
        """
        self._init_stage1_agents()

        all_aggregations: list[AggregationResult] = []

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            batch_results = self._process_batch_stage1(batch)
            all_aggregations.extend(batch_results)

        return self.codebook, all_aggregations

    def run_stage2(self, codebook: Codebook | None = None) -> ThemeAggregationResult:
        """Run Stage 2: Theme Development.

        Develops themes from the codebook using multiple theme coders
        and aggregates them.

        Args:
            codebook: Optional codebook to use. If None, uses the pipeline's codebook.

        Returns:
            ThemeAggregationResult with final themes.
        """
        if codebook is not None:
            self.codebook = codebook

        self._init_stage2_agents()

        # Get themes from all theme coders
        all_theme_results: list[ThemeResult] = []
        for theme_coder in self._theme_coders:
            # Update theme coder's codebook reference
            theme_coder.codebook = self.codebook
            theme_result = theme_coder.develop_themes()
            all_theme_results.append(theme_result)

        # Aggregate themes
        assert self._theme_aggregator is not None
        final_themes = self._theme_aggregator.aggregate(all_theme_results)

        return final_themes

    def run(self, segments: list[DataSegment]) -> PipelineResult:
        """Run the full two-stage pipeline.

        Args:
            segments: List of data segments to analyze.

        Returns:
            PipelineResult with themes, codebook, and intermediate results.
        """
        # Stage 1: Coding
        codebook, aggregations = self.run_stage1(segments)

        # Stage 2: Theme Development
        themes = self.run_stage2(codebook)

        return PipelineResult(
            themes=themes,
            codebook=codebook,
            stage1_aggregations=aggregations,
        )

    def run_from_texts(self, texts: list[str]) -> PipelineResult:
        """Run pipeline from a list of text strings.

        Convenience method that generates segment IDs automatically.

        Args:
            texts: List of text strings to analyze.

        Returns:
            PipelineResult with themes and codebook.
        """
        segments = [
            DataSegment(
                segment_id=f"seg_{uuid.uuid4().hex[:8]}",
                text=text,
            )
            for text in texts
        ]
        return self.run(segments)

    @property
    def coders(self) -> list[CoderAgent]:
        """Get the list of coder agents."""
        if not self._coders:
            self._init_stage1_agents()
        return self._coders

    @property
    def theme_coders(self) -> list[ThemeCoderAgent]:
        """Get the list of theme coder agents."""
        if not self._theme_coders:
            self._init_stage2_agents()
        return self._theme_coders
