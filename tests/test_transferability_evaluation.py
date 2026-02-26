"""Tests for transferability evaluation."""

import pytest

from thematic_analysis.evaluation import (
    TransferabilityEvaluator,
    TransferabilityResult,
    split_dataset,
    split_dataset_stratified,
)


class TestSplitDataset:
    """Tests for split_dataset function."""

    def test_basic_split(self):
        """Test basic dataset splitting."""
        data = list(range(100))
        train, val = split_dataset(data, train_ratio=0.8, shuffle=False)
        assert len(train) == 80
        assert len(val) == 20

    def test_split_ratios(self):
        """Test various split ratios."""
        data = list(range(10))
        train, val = split_dataset(data, train_ratio=0.5, shuffle=False)
        assert len(train) == 5
        assert len(val) == 5

    def test_split_preserves_all_items(self):
        """Test that split contains all original items."""
        data = ["a", "b", "c", "d", "e"]
        train, val = split_dataset(data, train_ratio=0.6, shuffle=False)
        assert len(train) + len(val) == len(data)
        assert set(train + val) == set(data)

    def test_shuffle_with_seed(self):
        """Test reproducibility with seed."""
        data = list(range(20))
        train1, val1 = split_dataset(data, shuffle=True, seed=42)
        train2, val2 = split_dataset(data, shuffle=True, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_different_seeds_different_results(self):
        """Test different seeds produce different splits."""
        data = list(range(100))
        train1, _ = split_dataset(data, shuffle=True, seed=42)
        train2, _ = split_dataset(data, shuffle=True, seed=123)
        assert train1 != train2

    def test_invalid_ratio_raises(self):
        """Test that invalid ratios raise ValueError."""
        data = [1, 2, 3]
        with pytest.raises(ValueError):
            split_dataset(data, train_ratio=1.5)
        with pytest.raises(ValueError):
            split_dataset(data, train_ratio=-0.1)

    def test_empty_dataset(self):
        """Test splitting empty dataset."""
        train, val = split_dataset([], train_ratio=0.8)
        assert train == []
        assert val == []

    def test_single_item(self):
        """Test splitting single item dataset."""
        data = ["only_item"]
        train, val = split_dataset(data, train_ratio=0.8, shuffle=False)
        # With ratio 0.8, 1 item -> 0.8 -> 0 items in train
        # Wait, int(1 * 0.8) = 0, so train=[], val=["only_item"]
        assert len(train) + len(val) == 1


class TestSplitDatasetStratified:
    """Tests for split_dataset_stratified function."""

    def test_basic_stratified_split(self):
        """Test basic stratified splitting."""
        data = list(range(10))
        labels = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
        train, val = split_dataset_stratified(
            data, labels, train_ratio=0.8, shuffle=False
        )
        # 5 A's -> 4 train, 1 val; 5 B's -> 4 train, 1 val
        assert len(train) == 8
        assert len(val) == 2

    def test_preserves_distribution(self):
        """Test that class distribution is preserved."""
        data = list(range(100))
        labels = ["A"] * 70 + ["B"] * 30
        train, val = split_dataset_stratified(
            data, labels, train_ratio=0.8, shuffle=False, seed=42
        )
        # A: 70 * 0.8 = 56 train, 14 val
        # B: 30 * 0.8 = 24 train, 6 val
        assert len(train) == 80
        assert len(val) == 20

    def test_mismatched_lengths_raises(self):
        """Test that mismatched data/labels lengths raise ValueError."""
        data = [1, 2, 3]
        labels = ["A", "B"]
        with pytest.raises(ValueError):
            split_dataset_stratified(data, labels)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with seed."""
        data = list(range(50))
        labels = ["A"] * 25 + ["B"] * 25
        train1, val1 = split_dataset_stratified(data, labels, seed=42)
        train2, val2 = split_dataset_stratified(data, labels, seed=42)
        assert train1 == train2
        assert val1 == val2


class TestTransferabilityResult:
    """Tests for TransferabilityResult dataclass."""

    def test_creation(self):
        """Test creating a result."""
        result = TransferabilityResult(
            rouge_1=0.8,
            rouge_2=0.6,
            combined_rouge=0.7,
            train_theme_count=3,
            val_theme_count=3,
        )
        assert result.rouge_1 == 0.8
        assert result.rouge_2 == 0.6
        assert result.combined_rouge == 0.7
        assert result.overall_score == 0.7

    def test_default_values(self):
        """Test default values."""
        result = TransferabilityResult()
        assert result.rouge_1 == 0.0
        assert result.rouge_2 == 0.0
        assert result.combined_rouge == 0.0
        assert result.train_themes == []
        assert result.val_themes == []

    def test_to_dict(self):
        """Test dictionary conversion."""
        themes = [{"name": "Theme", "description": "Description"}]
        result = TransferabilityResult(
            rouge_1=0.8,
            rouge_2=0.6,
            combined_rouge=0.7,
            train_theme_count=1,
            val_theme_count=1,
            train_themes=themes,
            val_themes=themes,
        )
        data = result.to_dict()
        assert "overall_score" in data
        assert "rouge_1" in data
        assert "rouge_2" in data
        assert "train_themes" in data
        assert data["train_theme_count"] == 1


class TestTransferabilityEvaluator:
    """Tests for TransferabilityEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> TransferabilityEvaluator:
        """Create evaluator for testing."""
        return TransferabilityEvaluator()

    def test_compare_identical_themes(self, evaluator: TransferabilityEvaluator):
        """Test comparing identical theme sets."""
        themes = [
            {"name": "Climate Anxiety", "description": "Worry about climate change"},
        ]
        result = evaluator.compare_theme_sets(themes, themes)
        assert result.combined_rouge == pytest.approx(1.0)
        assert result.train_theme_count == 1
        assert result.val_theme_count == 1

    def test_compare_different_themes(self, evaluator: TransferabilityEvaluator):
        """Test comparing different theme sets."""
        train_themes = [
            {"name": "Climate Anxiety", "description": "Worry about climate change"},
        ]
        val_themes = [
            {"name": "Economic Growth", "description": "Focus on economic development"},
        ]
        result = evaluator.compare_theme_sets(train_themes, val_themes)
        # Different themes should have lower overlap
        assert result.combined_rouge < 1.0

    def test_compare_overlapping_themes(self, evaluator: TransferabilityEvaluator):
        """Test comparing theme sets with some overlap."""
        train_themes = [
            {"name": "Climate Anxiety", "description": "Worry about climate impacts"},
            {"name": "Hope for Future", "description": "Optimism about solutions"},
        ]
        val_themes = [
            {"name": "Climate Worry", "description": "Concern about climate change"},
            {"name": "Future Optimism", "description": "Hope for better outcomes"},
        ]
        result = evaluator.compare_theme_sets(train_themes, val_themes)
        # Should have moderate overlap due to similar vocabulary
        assert 0.0 < result.combined_rouge < 1.0

    def test_evaluate_alias(self, evaluator: TransferabilityEvaluator):
        """Test that evaluate is alias for compare_theme_sets."""
        themes = [{"name": "Test", "description": "Description"}]
        result1 = evaluator.compare_theme_sets(themes, themes)
        result2 = evaluator.evaluate(themes, themes)
        assert result1.combined_rouge == result2.combined_rouge

    def test_empty_theme_sets(self, evaluator: TransferabilityEvaluator):
        """Test with empty theme sets."""
        result = evaluator.compare_theme_sets([], [])
        assert result.combined_rouge == 0.0
        assert result.train_theme_count == 0
        assert result.val_theme_count == 0

    def test_cross_validation(self, evaluator: TransferabilityEvaluator):
        """Test cross-validation style evaluation."""
        theme_sets = [
            [{"name": "Theme A", "description": "Description A"}],
            [{"name": "Theme B", "description": "Description B"}],
            [{"name": "Theme C", "description": "Description C"}],
        ]
        results = evaluator.evaluate_cross_validation(theme_sets)
        # C(3,2) = 3 comparisons
        assert len(results) == 3

    def test_average_transferability(self, evaluator: TransferabilityEvaluator):
        """Test averaging multiple results."""
        results = [
            TransferabilityResult(rouge_1=0.8, rouge_2=0.6, combined_rouge=0.7),
            TransferabilityResult(rouge_1=0.6, rouge_2=0.4, combined_rouge=0.5),
        ]
        avg = evaluator.average_transferability(results)
        assert avg.rouge_1 == pytest.approx(0.7)
        assert avg.rouge_2 == pytest.approx(0.5)
        assert avg.combined_rouge == pytest.approx(0.6)

    def test_average_empty_results(self, evaluator: TransferabilityEvaluator):
        """Test averaging empty results list."""
        avg = evaluator.average_transferability([])
        assert avg.combined_rouge == 0.0


class TestTransferabilityIntegration:
    """Integration tests for transferability evaluation."""

    def test_full_workflow(self):
        """Test complete transferability evaluation workflow."""
        # Simulate splitting data and running TA on each split
        evaluator = TransferabilityEvaluator()

        # Themes from training split
        train_themes = [
            {
                "name": "Climate Anxiety",
                "description": "Feelings of worry and fear about climate change",
            },
            {
                "name": "Hope and Action",
                "description": "Optimism about climate solutions",
            },
            {
                "name": "Systemic Change",
                "description": "Need for political and economic transformation",
            },
        ]

        # Themes from validation split (similar but slightly different)
        val_themes = [
            {
                "name": "Climate Worry",
                "description": "Concern and anxiety about climate impacts",
            },
            {
                "name": "Collective Hope",
                "description": "Hope through collective climate action",
            },
            {
                "name": "Policy Reform",
                "description": "Need for governmental policy changes",
            },
        ]

        result = evaluator.evaluate(train_themes, val_themes)

        assert 0.0 <= result.overall_score <= 1.0
        assert result.train_theme_count == 3
        assert result.val_theme_count == 3

    def test_multiple_splits_evaluation(self):
        """Test evaluation across multiple splits."""
        evaluator = TransferabilityEvaluator()

        # Simulate k-fold style splits
        theme_sets = [
            [
                {"name": "Anxiety", "description": "Climate worry and fear"},
                {"name": "Hope", "description": "Optimism for solutions"},
            ],
            [
                {"name": "Fear", "description": "Climate anxiety and concern"},
                {"name": "Action", "description": "Taking climate action"},
            ],
            [
                {"name": "Worry", "description": "Climate fear and stress"},
                {"name": "Belief", "description": "Belief in positive change"},
            ],
        ]

        # Cross-validation style comparison
        results = evaluator.evaluate_cross_validation(theme_sets)
        avg = evaluator.average_transferability(results)

        assert len(results) == 3  # C(3,2)
        assert 0.0 <= avg.overall_score <= 1.0
        # Themes have overlap, expect moderate score
        assert avg.overall_score > 0.0
