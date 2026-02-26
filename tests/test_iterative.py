"""Tests for iterative pipeline module."""

from thematic_analysis.iterative import (
    IterationMetrics,
    IterationMode,
    IterationState,
    IterativePipelineConfig,
    IterativePipelineController,
    NegotiationResult,
    NegotiationStrategy,
    ReviewFeedback,
    calculate_agreement_rate,
    calculate_codebook_stability,
    chunk_data,
    negotiate,
    negotiate_confidence,
    negotiate_consensus,
    negotiate_intersection,
    negotiate_union,
)


class TestIterationMode:
    """Tests for IterationMode enum."""

    def test_mode_values(self):
        """Test that all modes have expected values."""
        assert IterationMode.SEQUENTIAL.value == "sequential"
        assert IterationMode.BATCH.value == "batch"
        assert IterationMode.CONTINUOUS.value == "continuous"


class TestNegotiationStrategy:
    """Tests for NegotiationStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategies have expected values."""
        assert NegotiationStrategy.CONSENSUS.value == "consensus"
        assert NegotiationStrategy.CONFIDENCE.value == "confidence"
        assert NegotiationStrategy.UNION.value == "union"
        assert NegotiationStrategy.INTERSECTION.value == "intersection"


class TestIterationMetrics:
    """Tests for IterationMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = IterationMetrics()
        assert metrics.iteration == 0
        assert metrics.agreement_rate == 0.0
        assert metrics.codebook_stability == 1.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = IterationMetrics(
            iteration=1,
            batch_number=2,
            items_processed=10,
            codes_added=5,
            agreement_rate=0.8,
        )
        d = metrics.to_dict()
        assert d["iteration"] == 1
        assert d["batch_number"] == 2
        assert d["items_processed"] == 10
        assert d["codes_added"] == 5
        assert d["agreement_rate"] == 0.8


class TestNegotiationResult:
    """Tests for NegotiationResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = NegotiationResult()
        assert result.agreed_codes == []
        assert result.disputed_codes == []
        assert result.rejected_codes == []

    def test_with_codes(self):
        """Test result with codes."""
        result = NegotiationResult(
            agreed_codes=["code1", "code2"],
            disputed_codes=["code3"],
            discussion_notes="Discussion happened",
        )
        assert len(result.agreed_codes) == 2
        assert len(result.disputed_codes) == 1
        assert "Discussion" in result.discussion_notes


class TestReviewFeedback:
    """Tests for ReviewFeedback dataclass."""

    def test_default_values(self):
        """Test default feedback values."""
        feedback = ReviewFeedback()
        assert feedback.approved_codes == []
        assert feedback.quality_score == 0.0
        assert feedback.should_continue

    def test_with_feedback(self):
        """Test feedback with values."""
        feedback = ReviewFeedback(
            approved_codes=["code1"],
            flagged_codes=["code2"],
            suggestions=["Clarify code2"],
            quality_score=0.85,
        )
        assert len(feedback.approved_codes) == 1
        assert feedback.quality_score == 0.85


class TestIterativePipelineConfig:
    """Tests for IterativePipelineConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = IterativePipelineConfig()
        assert config.iteration_mode == IterationMode.BATCH
        assert config.batch_size == 10
        assert config.review_frequency == 1

    def test_sequential_preset(self):
        """Test sequential configuration preset."""
        config = IterativePipelineConfig.sequential()
        assert config.iteration_mode == IterationMode.SEQUENTIAL

    def test_standard_batch_preset(self):
        """Test standard batch configuration preset."""
        config = IterativePipelineConfig.standard_batch(batch_size=15)
        assert config.iteration_mode == IterationMode.BATCH
        assert config.batch_size == 15
        assert config.review_frequency == 1

    def test_high_quality_preset(self):
        """Test high quality configuration preset."""
        config = IterativePipelineConfig.high_quality()
        assert config.batch_size == 5
        assert config.min_agreement_rate == 0.7
        assert config.early_stop_threshold == 0.95

    def test_fast_preset(self):
        """Test fast configuration preset."""
        config = IterativePipelineConfig.fast()
        assert config.batch_size == 20
        assert config.review_frequency == 2


class TestChunkData:
    """Tests for chunk_data function."""

    def test_basic_chunking(self):
        """Test basic data chunking."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        chunks = list(chunk_data(data, 3))

        assert len(chunks) == 4
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7, 8, 9]
        assert chunks[3] == [10]

    def test_exact_chunks(self):
        """Test when data divides evenly."""
        data = [1, 2, 3, 4, 5, 6]
        chunks = list(chunk_data(data, 2))

        assert len(chunks) == 3
        assert all(len(c) == 2 for c in chunks)

    def test_single_chunk(self):
        """Test when chunk size is larger than data."""
        data = [1, 2, 3]
        chunks = list(chunk_data(data, 10))

        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3]

    def test_empty_data(self):
        """Test with empty data."""
        chunks = list(chunk_data([], 5))
        assert len(chunks) == 0


class TestCalculateAgreementRate:
    """Tests for calculate_agreement_rate function."""

    def test_identical_sets(self):
        """Test agreement rate for identical sets."""
        sets = [{"a", "b", "c"}, {"a", "b", "c"}]
        rate = calculate_agreement_rate(sets)
        assert rate == 1.0

    def test_completely_different_sets(self):
        """Test agreement rate for completely different sets."""
        sets = [{"a", "b"}, {"c", "d"}]
        rate = calculate_agreement_rate(sets)
        assert rate == 0.0

    def test_partial_overlap(self):
        """Test agreement rate for partial overlap."""
        sets = [{"a", "b", "c"}, {"b", "c", "d"}]
        rate = calculate_agreement_rate(sets)
        # Jaccard: |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5
        assert rate == 0.5

    def test_single_set(self):
        """Test agreement rate with single set."""
        sets = [{"a", "b"}]
        rate = calculate_agreement_rate(sets)
        assert rate == 1.0

    def test_empty_sets(self):
        """Test agreement rate with empty sets."""
        sets = [set(), set()]
        rate = calculate_agreement_rate(sets)
        assert rate == 1.0

    def test_multiple_sets(self):
        """Test agreement rate with multiple sets."""
        sets = [{"a", "b"}, {"b", "c"}, {"c", "d"}]
        rate = calculate_agreement_rate(sets)
        # Average of pairwise Jaccard similarities
        assert 0 <= rate <= 1


class TestCalculateCodebookStability:
    """Tests for calculate_codebook_stability function."""

    def test_identical_codebooks(self):
        """Test stability for identical codebooks."""
        stability = calculate_codebook_stability({"a", "b"}, {"a", "b"})
        assert stability == 1.0

    def test_completely_different(self):
        """Test stability for completely different codebooks."""
        stability = calculate_codebook_stability({"a", "b"}, {"c", "d"})
        assert stability == 0.0

    def test_partial_overlap(self):
        """Test stability for partial overlap."""
        stability = calculate_codebook_stability({"a", "b", "c"}, {"b", "c", "d"})
        # Jaccard: 2/4 = 0.5
        assert stability == 0.5

    def test_empty_codebooks(self):
        """Test stability for empty codebooks."""
        stability = calculate_codebook_stability(set(), set())
        assert stability == 1.0

    def test_one_empty(self):
        """Test stability when one codebook is empty."""
        stability = calculate_codebook_stability({"a", "b"}, set())
        assert stability == 0.0


class TestNegotiateConsensus:
    """Tests for negotiate_consensus function."""

    def test_full_agreement(self):
        """Test negotiation with full agreement."""
        sets = [{"a", "b"}, {"a", "b"}]
        result = negotiate_consensus(sets)

        assert "a" in result.agreed_codes
        assert "b" in result.agreed_codes
        assert len(result.disputed_codes) == 0

    def test_partial_agreement(self):
        """Test negotiation with partial agreement."""
        sets = [{"a", "b"}, {"b", "c"}]
        result = negotiate_consensus(sets, threshold=0.5)

        # b is agreed by both (100%), a and c by one each (50%)
        assert "b" in result.agreed_codes
        assert "a" in result.agreed_codes  # 50% >= 0.5
        assert "c" in result.agreed_codes

    def test_high_threshold(self):
        """Test negotiation with high threshold."""
        sets = [{"a", "b"}, {"b", "c"}, {"c", "d"}]
        result = negotiate_consensus(sets, threshold=0.6)

        # Only codes with 2+ out of 3 coders pass (66.7% >= 60%)
        assert "b" in result.agreed_codes  # Appears in sets 0 and 1 (2/3)
        assert "c" in result.agreed_codes  # Appears in sets 1 and 2 (2/3)

    def test_empty_sets(self):
        """Test negotiation with empty sets."""
        result = negotiate_consensus([])
        assert len(result.agreed_codes) == 0


class TestNegotiateConfidence:
    """Tests for negotiate_confidence function."""

    def test_with_confidences(self):
        """Test negotiation with confidence scores."""
        sets = [{"a", "b"}, {"b", "c"}]
        confidences = [{"a": 0.9, "b": 0.7}, {"b": 0.8, "c": 0.3}]

        result = negotiate_confidence(sets, confidences)

        # High confidence codes should be agreed
        assert "a" in result.agreed_codes  # 0.9 avg
        assert "b" in result.agreed_codes  # (0.7+0.8)/2 = 0.75

    def test_without_confidences(self):
        """Test negotiation without confidence scores."""
        sets = [{"a", "b"}, {"b", "c"}]
        result = negotiate_confidence(sets)

        # All should be treated equally (confidence 1.0)
        assert "a" in result.agreed_codes
        assert "b" in result.agreed_codes
        assert "c" in result.agreed_codes


class TestNegotiateUnion:
    """Tests for negotiate_union function."""

    def test_union(self):
        """Test union negotiation."""
        sets = [{"a", "b"}, {"c", "d"}]
        result = negotiate_union(sets)

        assert "a" in result.agreed_codes
        assert "b" in result.agreed_codes
        assert "c" in result.agreed_codes
        assert "d" in result.agreed_codes
        assert len(result.disputed_codes) == 0


class TestNegotiateIntersection:
    """Tests for negotiate_intersection function."""

    def test_with_overlap(self):
        """Test intersection with overlapping sets."""
        sets = [{"a", "b", "c"}, {"b", "c", "d"}]
        result = negotiate_intersection(sets)

        assert "b" in result.agreed_codes
        assert "c" in result.agreed_codes
        assert "a" in result.disputed_codes
        assert "d" in result.disputed_codes

    def test_no_overlap(self):
        """Test intersection with no overlap."""
        sets = [{"a", "b"}, {"c", "d"}]
        result = negotiate_intersection(sets)

        assert len(result.agreed_codes) == 0
        assert len(result.disputed_codes) == 4


class TestNegotiate:
    """Tests for negotiate function."""

    def test_consensus_strategy(self):
        """Test negotiate with consensus strategy."""
        sets = [{"a", "b"}, {"b", "c"}]
        result = negotiate(sets, NegotiationStrategy.CONSENSUS)
        assert isinstance(result, NegotiationResult)

    def test_confidence_strategy(self):
        """Test negotiate with confidence strategy."""
        sets = [{"a", "b"}, {"b", "c"}]
        result = negotiate(sets, NegotiationStrategy.CONFIDENCE)
        assert isinstance(result, NegotiationResult)

    def test_union_strategy(self):
        """Test negotiate with union strategy."""
        sets = [{"a"}, {"b"}]
        result = negotiate(sets, NegotiationStrategy.UNION)
        assert "a" in result.agreed_codes
        assert "b" in result.agreed_codes

    def test_intersection_strategy(self):
        """Test negotiate with intersection strategy."""
        sets = [{"a", "b"}, {"b", "c"}]
        result = negotiate(sets, NegotiationStrategy.INTERSECTION)
        assert "b" in result.agreed_codes


class TestIterationState:
    """Tests for IterationState class."""

    def test_default_state(self):
        """Test default state values."""
        state = IterationState()
        assert state.current_iteration == 0
        assert state.items_processed == 0
        assert state.should_stop is False

    def test_record_metrics(self):
        """Test recording metrics."""
        state = IterationState()
        metrics = IterationMetrics(iteration=1, agreement_rate=0.8)
        state.record_metrics(metrics)

        assert len(state.metrics_history) == 1
        assert state.metrics_history[0].agreement_rate == 0.8

    def test_check_early_stop(self):
        """Test early stop check."""
        state = IterationState()

        # Below threshold
        assert not state.check_early_stop(0.8, 0.9)
        assert not state.should_stop

        # Above threshold
        assert state.check_early_stop(0.95, 0.9)
        assert state.should_stop
        assert "threshold" in state.stop_reason.lower()

    def test_check_max_iterations(self):
        """Test max iterations check."""
        state = IterationState(current_iteration=5)

        # Below max
        assert not state.check_max_iterations(10)

        # At max
        assert state.check_max_iterations(5)
        assert state.should_stop

        # No limit
        state = IterationState(current_iteration=100)
        assert not state.check_max_iterations(0)

    def test_get_progress(self):
        """Test progress calculation."""
        state = IterationState(total_items=100, items_processed=25)
        assert state.get_progress() == 25.0

        # Empty state
        state = IterationState(total_items=0)
        assert state.get_progress() == 0.0

    def test_get_summary(self):
        """Test state summary."""
        state = IterationState(
            current_iteration=2,
            current_batch=5,
            total_items=100,
            items_processed=50,
            current_codes={"a", "b", "c"},
        )
        summary = state.get_summary()

        assert summary["current_iteration"] == 2
        assert summary["current_batch"] == 5
        assert summary["codes_count"] == 3
        assert "50.0%" in summary["progress"]


class TestIterativePipelineController:
    """Tests for IterativePipelineController class."""

    def test_default_initialization(self):
        """Test default controller initialization."""
        controller = IterativePipelineController()
        assert controller.config.iteration_mode == IterationMode.BATCH
        assert controller.state.current_iteration == 0

    def test_initialize(self):
        """Test pipeline initialization."""
        controller = IterativePipelineController()
        controller.initialize(total_items=100)

        assert controller.state.total_items == 100
        assert controller.state.items_processed == 0

    def test_get_batches_sequential(self):
        """Test batch creation in sequential mode."""
        config = IterativePipelineConfig.sequential()
        controller = IterativePipelineController(config=config)

        data = [1, 2, 3, 4, 5]
        batches = controller.get_batches(data)

        assert len(batches) == 1
        assert batches[0] == data

    def test_get_batches_batch_mode(self):
        """Test batch creation in batch mode."""
        config = IterativePipelineConfig(batch_size=2)
        controller = IterativePipelineController(config=config)

        data = [1, 2, 3, 4, 5]
        batches = controller.get_batches(data)

        assert len(batches) == 3

    def test_should_review_sequential(self):
        """Test review check in sequential mode."""
        config = IterativePipelineConfig.sequential()
        controller = IterativePipelineController(config=config)

        assert controller.should_review()

    def test_should_review_continuous(self):
        """Test review check in continuous mode."""
        config = IterativePipelineConfig(iteration_mode=IterationMode.CONTINUOUS)
        controller = IterativePipelineController(config=config)

        assert controller.should_review()

    def test_should_review_batch(self):
        """Test review check in batch mode."""
        config = IterativePipelineConfig(review_frequency=2)
        controller = IterativePipelineController(config=config)

        controller.state.current_batch = 0
        assert controller.should_review()  # 0 % 2 == 0

        controller.state.current_batch = 1
        assert not controller.should_review()  # 1 % 2 != 0

        controller.state.current_batch = 2
        assert controller.should_review()  # 2 % 2 == 0

    def test_negotiate_codes(self):
        """Test code negotiation through controller."""
        config = IterativePipelineConfig(negotiation_strategy=NegotiationStrategy.UNION)
        controller = IterativePipelineController(config=config)

        code_sets = [{"a", "b"}, {"c"}]
        result = controller.negotiate_codes(code_sets)

        assert "a" in result.agreed_codes
        assert "b" in result.agreed_codes
        assert "c" in result.agreed_codes

    def test_update_state(self):
        """Test state update after batch."""
        controller = IterativePipelineController()
        controller.initialize(total_items=100)

        metrics = IterationMetrics(iteration=1, codes_added=3)
        controller.update_state(
            batch_size=10,
            agreed_codes=["code1", "code2"],
            metrics=metrics,
        )

        assert controller.state.current_batch == 1
        assert controller.state.items_processed == 10
        assert "code1" in controller.state.current_codes
        assert len(controller.state.metrics_history) == 1

    def test_advance_iteration(self):
        """Test advancing iteration."""
        controller = IterativePipelineController()

        assert controller.state.current_iteration == 0
        controller.advance_iteration()
        assert controller.state.current_iteration == 1

    def test_check_stopping_conditions(self):
        """Test stopping condition checks."""
        config = IterativePipelineConfig(
            early_stop_threshold=0.9,
            max_iterations=5,
        )
        controller = IterativePipelineController(config=config)

        # Below threshold
        assert not controller.check_stopping_conditions(0.8)

        # Above threshold
        assert controller.check_stopping_conditions(0.95)
        assert controller.state.should_stop

    def test_get_metrics_summary(self):
        """Test metrics summary."""
        controller = IterativePipelineController()

        # Add some metrics
        controller.state.record_metrics(
            IterationMetrics(codes_added=5, agreement_rate=0.8)
        )
        controller.state.record_metrics(
            IterationMetrics(codes_added=3, codes_merged=2, agreement_rate=0.9)
        )
        controller.state.current_codes = {"a", "b", "c", "d", "e"}

        summary = controller.get_metrics_summary()

        assert summary["iterations"] == 2
        assert summary["total_codes_added"] == 8
        assert summary["total_codes_merged"] == 2
        assert abs(summary["average_agreement_rate"] - 0.85) < 0.01
        assert summary["final_codebook_size"] == 5


class TestIntegration:
    """Integration tests for iterative pipeline."""

    def test_full_iteration_workflow(self):
        """Test full iteration workflow."""
        config = IterativePipelineConfig.standard_batch(batch_size=5)
        controller = IterativePipelineController(config=config)

        # Simulate data
        data = list(range(20))
        controller.initialize(total_items=len(data))

        # Process batches
        batches = controller.get_batches(data)
        assert len(batches) == 4

        for batch in batches:
            # Simulate coder outputs
            coder1_codes = {f"code_{i}" for i in batch[:3]}
            coder2_codes = {f"code_{i}" for i in batch[1:4]}

            # Negotiate
            result = controller.negotiate_codes([coder1_codes, coder2_codes])

            # Record metrics
            agreement = calculate_agreement_rate([coder1_codes, coder2_codes])
            metrics = IterationMetrics(
                iteration=controller.state.current_iteration,
                batch_number=controller.state.current_batch,
                codes_added=len(result.agreed_codes),
                agreement_rate=agreement,
            )

            # Update state
            controller.update_state(
                batch_size=len(batch),
                agreed_codes=result.agreed_codes,
                metrics=metrics,
            )

            # Check review
            if controller.should_review():
                # Simulate review
                pass

        # Final state
        assert controller.state.items_processed == 20
        assert controller.state.current_batch == 4
        assert len(controller.state.metrics_history) == 4

        summary = controller.get_metrics_summary()
        assert summary["iterations"] == 4

    def test_early_stopping(self):
        """Test early stopping when quality threshold is met."""
        config = IterativePipelineConfig(
            batch_size=5,
            early_stop_threshold=0.9,
        )
        controller = IterativePipelineController(config=config)
        controller.initialize(total_items=100)

        # Simulate high quality batch
        stopped = controller.check_stopping_conditions(0.95)

        assert stopped
        assert controller.state.should_stop
        assert "threshold" in controller.state.stop_reason.lower()
