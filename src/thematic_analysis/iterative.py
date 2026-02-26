"""Iterative pipeline with early negotiated agreement between coders.

This module implements a batch-based iterative pipeline that allows:
- Early negotiation rounds between coders
- Continuous reviewer feedback during coding
- Inter-coder communication and alignment
- Adaptive coding strategy based on quality metrics

Unlike the sequential approach in the original Thematic-LM paper, this
design catches problematic codes early and enables coders to learn from
each other during the coding process.

Based on: Braun & Clarke (2006) on iterative qualitative research
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum


class IterationMode(Enum):
    """Mode of iteration for the pipeline."""

    SEQUENTIAL = "sequential"  # Original: all at once, review at end
    BATCH = "batch"  # Batch-based with periodic review
    CONTINUOUS = "continuous"  # Continuous review after each item


class NegotiationStrategy(Enum):
    """Strategy for negotiating code agreement between coders."""

    CONSENSUS = "consensus"  # Codes need majority agreement
    CONFIDENCE = "confidence"  # Higher-confidence codes get priority
    UNION = "union"  # Accept all codes from all coders
    INTERSECTION = "intersection"  # Only codes agreed by all


@dataclass
class IterationMetrics:
    """Metrics tracked during iterative coding.

    Attributes:
        iteration: Current iteration number.
        batch_number: Current batch number.
        items_processed: Number of items processed so far.
        codes_added: Number of new codes added this iteration.
        codes_merged: Number of codes merged this iteration.
        codes_rejected: Number of codes rejected this iteration.
        agreement_rate: Agreement rate between coders.
        codebook_stability: How much codebook changed (0-1, 1 = no change).
    """

    iteration: int = 0
    batch_number: int = 0
    items_processed: int = 0
    codes_added: int = 0
    codes_merged: int = 0
    codes_rejected: int = 0
    agreement_rate: float = 0.0
    codebook_stability: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "iteration": self.iteration,
            "batch_number": self.batch_number,
            "items_processed": self.items_processed,
            "codes_added": self.codes_added,
            "codes_merged": self.codes_merged,
            "codes_rejected": self.codes_rejected,
            "agreement_rate": self.agreement_rate,
            "codebook_stability": self.codebook_stability,
        }


@dataclass
class NegotiationResult:
    """Result of negotiation between coders.

    Attributes:
        agreed_codes: Codes that all/most coders agreed on.
        disputed_codes: Codes with disagreement.
        rejected_codes: Codes that were rejected.
        discussion_notes: Notes from the negotiation process.
    """

    agreed_codes: list[str] = field(default_factory=list)
    disputed_codes: list[str] = field(default_factory=list)
    rejected_codes: list[str] = field(default_factory=list)
    discussion_notes: str = ""


@dataclass
class ReviewFeedback:
    """Feedback from reviewer on a batch of codes.

    Attributes:
        approved_codes: Codes approved by reviewer.
        flagged_codes: Codes that need attention.
        suggestions: Suggestions for improvement.
        quality_score: Overall quality score (0-1).
        should_continue: Whether to continue with current approach.
    """

    approved_codes: list[str] = field(default_factory=list)
    flagged_codes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    should_continue: bool = True


@dataclass
class IterativePipelineConfig:
    """Configuration for iterative pipeline.

    Attributes:
        iteration_mode: Sequential, batch, or continuous.
        batch_size: Number of items per batch.
        review_frequency: Review every N batches.
        negotiation_strategy: How to resolve disagreements.
        early_stop_threshold: Stop if quality exceeds this.
        min_agreement_rate: Minimum agreement to proceed.
        max_iterations: Maximum iterations (0 = no limit).
    """

    iteration_mode: IterationMode = IterationMode.BATCH
    batch_size: int = 10
    review_frequency: int = 1
    negotiation_strategy: NegotiationStrategy = NegotiationStrategy.CONSENSUS
    early_stop_threshold: float = 0.9
    min_agreement_rate: float = 0.6
    max_iterations: int = 0

    @classmethod
    def sequential(cls) -> "IterativePipelineConfig":
        """Create a sequential (non-iterative) configuration."""
        return cls(iteration_mode=IterationMode.SEQUENTIAL)

    @classmethod
    def standard_batch(cls, batch_size: int = 10) -> "IterativePipelineConfig":
        """Create a standard batch configuration."""
        return cls(
            iteration_mode=IterationMode.BATCH,
            batch_size=batch_size,
            review_frequency=1,
        )

    @classmethod
    def high_quality(cls) -> "IterativePipelineConfig":
        """Create a high-quality configuration with frequent review."""
        return cls(
            iteration_mode=IterationMode.BATCH,
            batch_size=5,
            review_frequency=1,
            min_agreement_rate=0.7,
            early_stop_threshold=0.95,
        )

    @classmethod
    def fast(cls) -> "IterativePipelineConfig":
        """Create a fast configuration with less review."""
        return cls(
            iteration_mode=IterationMode.BATCH,
            batch_size=20,
            review_frequency=2,
            min_agreement_rate=0.5,
        )


def chunk_data[T](data: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Split data into chunks of specified size.

    Args:
        data: List to chunk.
        chunk_size: Size of each chunk.

    Yields:
        Chunks of the data.
    """
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def calculate_agreement_rate(
    code_sets: list[set[str]],
) -> float:
    """Calculate agreement rate between multiple code sets.

    Uses Jaccard similarity averaged across all pairs.

    Args:
        code_sets: List of code sets from different coders.

    Returns:
        Agreement rate (0-1).
    """
    if len(code_sets) < 2:
        return 1.0

    total_similarity = 0.0
    pair_count = 0

    for i in range(len(code_sets)):
        for j in range(i + 1, len(code_sets)):
            set_a = code_sets[i]
            set_b = code_sets[j]

            if not set_a and not set_b:
                similarity = 1.0
            elif not set_a or not set_b:
                similarity = 0.0
            else:
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                similarity = intersection / union if union > 0 else 0.0

            total_similarity += similarity
            pair_count += 1

    return total_similarity / pair_count if pair_count > 0 else 0.0


def calculate_codebook_stability(
    previous_codes: set[str],
    current_codes: set[str],
) -> float:
    """Calculate how stable the codebook is between iterations.

    A stability of 1.0 means no change, 0.0 means completely different.

    Args:
        previous_codes: Codes from previous iteration.
        current_codes: Codes from current iteration.

    Returns:
        Stability score (0-1).
    """
    if not previous_codes and not current_codes:
        return 1.0
    if not previous_codes or not current_codes:
        return 0.0

    intersection = len(previous_codes & current_codes)
    union = len(previous_codes | current_codes)
    return intersection / union if union > 0 else 0.0


def negotiate_consensus(
    code_sets: list[set[str]],
    threshold: float = 0.5,
) -> NegotiationResult:
    """Negotiate codes using consensus strategy.

    Codes agreed by threshold proportion of coders are accepted.

    Args:
        code_sets: Code sets from different coders.
        threshold: Proportion of coders needed for agreement.

    Returns:
        NegotiationResult with agreed and disputed codes.
    """
    if not code_sets:
        return NegotiationResult()

    num_coders = len(code_sets)
    all_codes = set()
    for code_set in code_sets:
        all_codes.update(code_set)

    agreed = []
    disputed = []

    for code in all_codes:
        count = sum(1 for code_set in code_sets if code in code_set)
        proportion = count / num_coders

        if proportion >= threshold:
            agreed.append(code)
        else:
            disputed.append(code)

    return NegotiationResult(
        agreed_codes=agreed,
        disputed_codes=disputed,
        discussion_notes=f"Consensus threshold: {threshold:.0%}",
    )


def negotiate_confidence(
    code_sets: list[set[str]],
    confidences: list[dict[str, float]] | None = None,
) -> NegotiationResult:
    """Negotiate codes using confidence-weighted strategy.

    Higher-confidence codes from any coder are prioritized.

    Args:
        code_sets: Code sets from different coders.
        confidences: Optional confidence scores per coder per code.

    Returns:
        NegotiationResult with agreed codes.
    """
    if not code_sets:
        return NegotiationResult()

    all_codes = set()
    for code_set in code_sets:
        all_codes.update(code_set)

    # If no confidences provided, treat all equally
    if confidences is None:
        confidences = [{code: 1.0 for code in code_set} for code_set in code_sets]

    # Average confidence per code
    code_confidence: dict[str, float] = {}
    for code in all_codes:
        scores = [conf.get(code, 0.0) for conf in confidences if code in conf]
        code_confidence[code] = sum(scores) / len(scores) if scores else 0.0

    # Sort by confidence
    sorted_codes = sorted(code_confidence.items(), key=lambda x: x[1], reverse=True)

    agreed = [code for code, conf in sorted_codes if conf >= 0.5]
    disputed = [code for code, conf in sorted_codes if conf < 0.5]

    return NegotiationResult(
        agreed_codes=agreed,
        disputed_codes=disputed,
        discussion_notes="Confidence-weighted negotiation",
    )


def negotiate_union(code_sets: list[set[str]]) -> NegotiationResult:
    """Negotiate codes using union strategy.

    Accepts all codes from all coders.

    Args:
        code_sets: Code sets from different coders.

    Returns:
        NegotiationResult with all codes agreed.
    """
    all_codes = set()
    for code_set in code_sets:
        all_codes.update(code_set)

    return NegotiationResult(
        agreed_codes=list(all_codes),
        discussion_notes="Union: accepted all codes from all coders",
    )


def negotiate_intersection(code_sets: list[set[str]]) -> NegotiationResult:
    """Negotiate codes using intersection strategy.

    Only accepts codes agreed by all coders.

    Args:
        code_sets: Code sets from different coders.

    Returns:
        NegotiationResult with codes agreed by all.
    """
    if not code_sets:
        return NegotiationResult()

    agreed_codes = set(code_sets[0])
    for code_set in code_sets[1:]:
        agreed_codes &= code_set

    all_codes = set()
    for code_set in code_sets:
        all_codes.update(code_set)

    disputed = list(all_codes - agreed_codes)

    return NegotiationResult(
        agreed_codes=list(agreed_codes),
        disputed_codes=disputed,
        discussion_notes="Intersection: only codes agreed by all coders",
    )


def negotiate(
    code_sets: list[set[str]],
    strategy: NegotiationStrategy,
    confidences: list[dict[str, float]] | None = None,
) -> NegotiationResult:
    """Negotiate codes using the specified strategy.

    Args:
        code_sets: Code sets from different coders.
        strategy: Negotiation strategy to use.
        confidences: Optional confidence scores for confidence strategy.

    Returns:
        NegotiationResult based on the strategy.
    """
    if strategy == NegotiationStrategy.CONSENSUS:
        return negotiate_consensus(code_sets)
    elif strategy == NegotiationStrategy.CONFIDENCE:
        return negotiate_confidence(code_sets, confidences)
    elif strategy == NegotiationStrategy.UNION:
        return negotiate_union(code_sets)
    elif strategy == NegotiationStrategy.INTERSECTION:
        return negotiate_intersection(code_sets)
    else:
        return negotiate_consensus(code_sets)


@dataclass
class IterationState:
    """State of the iterative pipeline.

    Tracks the current state of iteration including metrics history.

    Attributes:
        current_iteration: Current iteration number.
        current_batch: Current batch number.
        total_items: Total items to process.
        items_processed: Items processed so far.
        current_codes: Current set of codes.
        metrics_history: History of metrics per iteration.
        should_stop: Whether iteration should stop.
        stop_reason: Reason for stopping if applicable.
    """

    current_iteration: int = 0
    current_batch: int = 0
    total_items: int = 0
    items_processed: int = 0
    current_codes: set[str] = field(default_factory=set)
    metrics_history: list[IterationMetrics] = field(default_factory=list)
    should_stop: bool = False
    stop_reason: str = ""

    def record_metrics(self, metrics: IterationMetrics) -> None:
        """Record metrics for the current iteration."""
        self.metrics_history.append(metrics)

    def check_early_stop(
        self,
        quality_score: float,
        threshold: float,
    ) -> bool:
        """Check if early stopping criteria is met.

        Args:
            quality_score: Current quality score.
            threshold: Threshold for early stopping.

        Returns:
            True if should stop early.
        """
        if quality_score >= threshold:
            self.should_stop = True
            self.stop_reason = f"Quality threshold reached: {quality_score:.2f}"
            return True
        return False

    def check_max_iterations(self, max_iterations: int) -> bool:
        """Check if maximum iterations reached.

        Args:
            max_iterations: Maximum iterations (0 = no limit).

        Returns:
            True if should stop.
        """
        if max_iterations > 0 and self.current_iteration >= max_iterations:
            self.should_stop = True
            self.stop_reason = f"Maximum iterations reached: {max_iterations}"
            return True
        return False

    def get_progress(self) -> float:
        """Get progress as a percentage.

        Returns:
            Progress (0-100).
        """
        if self.total_items == 0:
            return 0.0
        return (self.items_processed / self.total_items) * 100

    def get_summary(self) -> dict:
        """Get a summary of the iteration state.

        Returns:
            Dictionary with state summary.
        """
        return {
            "current_iteration": self.current_iteration,
            "current_batch": self.current_batch,
            "progress": f"{self.get_progress():.1f}%",
            "items_processed": self.items_processed,
            "total_items": self.total_items,
            "codes_count": len(self.current_codes),
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
        }


class IterativePipelineController:
    """Controller for managing iterative pipeline execution.

    Manages the iteration process, tracking state, and coordinating
    between coding, negotiation, and review phases.
    """

    def __init__(self, config: IterativePipelineConfig | None = None):
        """Initialize the iterative pipeline controller.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or IterativePipelineConfig()
        self.state = IterationState()

    def initialize(self, total_items: int) -> None:
        """Initialize the pipeline for a new run.

        Args:
            total_items: Total number of items to process.
        """
        self.state = IterationState(total_items=total_items)

    def get_batches[T](self, data: list[T]) -> list[list[T]]:
        """Get batches of data based on configuration.

        Args:
            data: Data to batch.

        Returns:
            List of batches.
        """
        if self.config.iteration_mode == IterationMode.SEQUENTIAL:
            return [data]  # Single batch with all data

        return list(chunk_data(data, self.config.batch_size))

    def should_review(self) -> bool:
        """Check if review should happen at current batch.

        Returns:
            True if should review.
        """
        if self.config.iteration_mode == IterationMode.SEQUENTIAL:
            return True  # Review at end

        if self.config.iteration_mode == IterationMode.CONTINUOUS:
            return True  # Always review

        # Batch mode: review every N batches
        return self.state.current_batch % self.config.review_frequency == 0

    def negotiate_codes(
        self,
        code_sets: list[set[str]],
        confidences: list[dict[str, float]] | None = None,
    ) -> NegotiationResult:
        """Negotiate codes using configured strategy.

        Args:
            code_sets: Code sets from different coders.
            confidences: Optional confidence scores.

        Returns:
            Negotiation result.
        """
        return negotiate(
            code_sets,
            self.config.negotiation_strategy,
            confidences,
        )

    def update_state(
        self,
        batch_size: int,
        agreed_codes: list[str],
        metrics: IterationMetrics,
    ) -> None:
        """Update state after processing a batch.

        Args:
            batch_size: Number of items in the batch.
            agreed_codes: Codes agreed upon in this batch.
            metrics: Metrics from this batch.
        """
        self.state.current_batch += 1
        self.state.items_processed += batch_size
        self.state.current_codes.update(agreed_codes)
        self.state.record_metrics(metrics)

    def advance_iteration(self) -> None:
        """Advance to the next iteration."""
        self.state.current_iteration += 1

    def check_stopping_conditions(
        self,
        quality_score: float,
    ) -> bool:
        """Check all stopping conditions.

        Args:
            quality_score: Current quality score.

        Returns:
            True if should stop.
        """
        if self.state.check_early_stop(quality_score, self.config.early_stop_threshold):
            return True

        if self.state.check_max_iterations(self.config.max_iterations):
            return True

        return False

    def get_state(self) -> IterationState:
        """Get the current state.

        Returns:
            Current iteration state.
        """
        return self.state

    def get_metrics_summary(self) -> dict:
        """Get a summary of metrics across all iterations.

        Returns:
            Dictionary with metrics summary.
        """
        if not self.state.metrics_history:
            return {}

        total_added = sum(m.codes_added for m in self.state.metrics_history)
        total_merged = sum(m.codes_merged for m in self.state.metrics_history)
        total_rejected = sum(m.codes_rejected for m in self.state.metrics_history)

        avg_agreement = sum(m.agreement_rate for m in self.state.metrics_history) / len(
            self.state.metrics_history
        )

        return {
            "iterations": len(self.state.metrics_history),
            "total_codes_added": total_added,
            "total_codes_merged": total_merged,
            "total_codes_rejected": total_rejected,
            "average_agreement_rate": avg_agreement,
            "final_codebook_size": len(self.state.current_codes),
        }
