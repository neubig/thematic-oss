"""Tests for dependability evaluation using ROUGE scores."""

import pytest

from thematic_lm.evaluation import (
    DependabilityEvaluator,
    DependabilityResult,
    PairwiseComparison,
    rouge_combined,
    rouge_n,
    rouge_n_directional,
)


class TestRougeNDirectional:
    """Tests for directional ROUGE-N function."""

    def test_identical_texts(self):
        """ROUGE score of identical texts should be 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        assert rouge_n_directional(text, text, n=1) == 1.0

    def test_no_overlap(self):
        """ROUGE score with no common words should be 0.0."""
        text_a = "hello world"
        text_b = "foo bar"
        assert rouge_n_directional(text_a, text_b, n=1) == 0.0

    def test_partial_overlap(self):
        """ROUGE score with partial overlap."""
        text_a = "the cat sat on the mat"  # 6 words
        text_b = "the dog sat on the rug"  # shares: the(2), sat, on = 4
        # Overlap count: the, cat->X, sat, on, the, mat->X = 4/6
        score = rouge_n_directional(text_a, text_b, n=1)
        assert score == pytest.approx(4 / 6)

    def test_empty_reference(self):
        """ROUGE with empty reference should be 0.0."""
        assert rouge_n_directional("", "some text", n=1) == 0.0

    def test_empty_candidate(self):
        """ROUGE with empty candidate should be 0.0."""
        assert rouge_n_directional("some text", "", n=1) == 0.0

    def test_bigrams(self):
        """Test ROUGE-2 (bigrams)."""
        # text_a bigrams: (the,quick), (quick,brown), (brown,fox)
        text_a = "the quick brown fox"
        # text_b bigrams: (the,quick), (quick,red), (red,fox)
        text_b = "the quick red fox"
        # Overlap: (the,quick) = 1/3
        score = rouge_n_directional(text_a, text_b, n=2)
        assert score == pytest.approx(1 / 3)


class TestRougeN:
    """Tests for symmetric ROUGE-N function."""

    def test_identical_texts(self):
        """Symmetric ROUGE of identical texts should be 1.0."""
        text = "this is a test sentence"
        assert rouge_n(text, text, n=1) == 1.0

    def test_symmetric(self):
        """ROUGE should be symmetric (order doesn't matter)."""
        text_a = "the cat"
        text_b = "the dog"
        assert rouge_n(text_a, text_b, n=1) == rouge_n(text_b, text_a, n=1)

    def test_partial_overlap_symmetric(self):
        """Test symmetric score with partial overlap."""
        text_a = "one two three"  # 3 words
        text_b = "one four five"  # shares: one
        # A->B: 1/3, B->A: 1/3
        # symmetric: 0.5 * (1/3 + 1/3) = 1/3
        assert rouge_n(text_a, text_b, n=1) == pytest.approx(1 / 3)


class TestRougeCombined:
    """Tests for combined ROUGE function."""

    def test_identical_texts(self):
        """Combined ROUGE of identical texts should be 1.0."""
        text = "the quick brown fox"
        assert rouge_combined(text, text) == 1.0

    def test_combined_is_average(self):
        """Combined should be average of ROUGE-1 and ROUGE-2."""
        text_a = "the quick brown"
        text_b = "the quick red"
        r1 = rouge_n(text_a, text_b, n=1)
        r2 = rouge_n(text_a, text_b, n=2)
        expected = 0.5 * (r1 + r2)
        assert rouge_combined(text_a, text_b) == pytest.approx(expected)


class TestPairwiseComparison:
    """Tests for PairwiseComparison dataclass."""

    def test_creation(self):
        """Test creating a comparison result."""
        comp = PairwiseComparison(
            run_a_id="run_1",
            run_b_id="run_2",
            rouge_1=0.8,
            rouge_2=0.6,
            combined_rouge=0.7,
        )
        assert comp.run_a_id == "run_1"
        assert comp.run_b_id == "run_2"
        assert comp.rouge_1 == 0.8
        assert comp.rouge_2 == 0.6
        assert comp.combined_rouge == 0.7

    def test_rouge_score_alias(self):
        """Test rouge_score property is alias for combined_rouge."""
        comp = PairwiseComparison(
            run_a_id="a",
            run_b_id="b",
            rouge_1=0.5,
            rouge_2=0.4,
            combined_rouge=0.45,
        )
        assert comp.rouge_score == comp.combined_rouge


class TestDependabilityResult:
    """Tests for DependabilityResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> DependabilityResult:
        """Create a sample result for testing."""
        return DependabilityResult(
            pairwise_comparisons=[
                PairwiseComparison(
                    run_a_id="run_1",
                    run_b_id="run_2",
                    rouge_1=0.8,
                    rouge_2=0.6,
                    combined_rouge=0.7,
                ),
                PairwiseComparison(
                    run_a_id="run_1",
                    run_b_id="run_3",
                    rouge_1=0.9,
                    rouge_2=0.7,
                    combined_rouge=0.8,
                ),
                PairwiseComparison(
                    run_a_id="run_2",
                    run_b_id="run_3",
                    rouge_1=0.7,
                    rouge_2=0.5,
                    combined_rouge=0.6,
                ),
            ]
        )

    def test_average_rouge_1(self, sample_result: DependabilityResult):
        """Test average ROUGE-1 calculation."""
        expected = (0.8 + 0.9 + 0.7) / 3
        assert sample_result.average_rouge_1 == pytest.approx(expected)

    def test_average_rouge_2(self, sample_result: DependabilityResult):
        """Test average ROUGE-2 calculation."""
        expected = (0.6 + 0.7 + 0.5) / 3
        assert sample_result.average_rouge_2 == pytest.approx(expected)

    def test_overall_score(self, sample_result: DependabilityResult):
        """Test overall dependability score."""
        expected = (0.7 + 0.8 + 0.6) / 3
        assert sample_result.overall_score == pytest.approx(expected)

    def test_num_comparisons(self, sample_result: DependabilityResult):
        """Test number of comparisons."""
        assert sample_result.num_comparisons == 3

    def test_empty_result(self):
        """Test empty result returns 0.0 for all scores."""
        result = DependabilityResult()
        assert result.overall_score == 0.0
        assert result.average_rouge_1 == 0.0
        assert result.average_rouge_2 == 0.0
        assert result.num_comparisons == 0

    def test_to_dict(self, sample_result: DependabilityResult):
        """Test dictionary conversion."""
        data = sample_result.to_dict()
        assert "overall_score" in data
        assert "average_rouge_1" in data
        assert "average_rouge_2" in data
        assert "num_comparisons" in data
        assert "pairwise_comparisons" in data
        assert len(data["pairwise_comparisons"]) == 3


class TestDependabilityEvaluator:
    """Tests for DependabilityEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> DependabilityEvaluator:
        """Create evaluator for testing."""
        return DependabilityEvaluator()

    def test_compare_texts(self, evaluator: DependabilityEvaluator):
        """Test comparing two texts."""
        text_a = "climate change impacts ecosystems"
        text_b = "climate change affects environments"
        result = evaluator.compare_texts(text_a, text_b, "run_a", "run_b")
        assert result.run_a_id == "run_a"
        assert result.run_b_id == "run_b"
        assert 0.0 <= result.rouge_1 <= 1.0
        assert 0.0 <= result.rouge_2 <= 1.0

    def test_compare_themes(self, evaluator: DependabilityEvaluator):
        """Test comparing two sets of themes."""
        themes_a = [
            {"name": "Climate Anxiety", "description": "Worry about climate change"},
            {"name": "Hope for Future", "description": "Optimism about solutions"},
        ]
        themes_b = [
            {"name": "Climate Worry", "description": "Concern about climate impacts"},
            {"name": "Future Hope", "description": "Optimism about climate solutions"},
        ]
        result = evaluator.compare_themes(themes_a, themes_b)
        assert 0.0 <= result.combined_rouge <= 1.0

    def test_compare_codes(self, evaluator: DependabilityEvaluator):
        """Test comparing two sets of codes."""
        codes_a = [
            {"name": "anxiety", "description": "feeling worried"},
            {"name": "fear", "description": "feeling scared"},
        ]
        codes_b = [
            {"name": "worry", "description": "feeling anxious"},
            {"name": "concern", "description": "feeling troubled"},
        ]
        result = evaluator.compare_codes(codes_a, codes_b)
        assert 0.0 <= result.combined_rouge <= 1.0

    def test_evaluate_multiple_runs_themes(self, evaluator: DependabilityEvaluator):
        """Test evaluating multiple runs comparing themes."""
        runs = [
            {
                "id": "run_1",
                "themes": [
                    {"name": "Climate Anxiety", "description": "Worry about climate"},
                ],
            },
            {
                "id": "run_2",
                "themes": [
                    {"name": "Climate Worry", "description": "Concern about climate"},
                ],
            },
            {
                "id": "run_3",
                "themes": [
                    {"name": "Climate Fear", "description": "Anxiety about climate"},
                ],
            },
        ]
        result = evaluator.evaluate_multiple_runs(runs, compare_type="themes")
        # 3 runs -> C(3,2) = 3 comparisons
        assert result.num_comparisons == 3
        assert 0.0 <= result.overall_score <= 1.0

    def test_evaluate_multiple_runs_codes(self, evaluator: DependabilityEvaluator):
        """Test evaluating multiple runs comparing codes."""
        runs = [
            {
                "id": "run_1",
                "codes": [{"name": "code1", "description": "description 1"}],
            },
            {
                "id": "run_2",
                "codes": [{"name": "code2", "description": "description 2"}],
            },
        ]
        result = evaluator.evaluate_multiple_runs(runs, compare_type="codes")
        # 2 runs -> C(2,2) = 1 comparison
        assert result.num_comparisons == 1

    def test_evaluate_single_run(self, evaluator: DependabilityEvaluator):
        """Test with single run returns empty result."""
        runs = [
            {
                "id": "run_1",
                "themes": [{"name": "Theme", "description": "Description"}],
            },
        ]
        result = evaluator.evaluate_multiple_runs(runs)
        assert result.num_comparisons == 0
        assert result.overall_score == 0.0

    def test_evaluate_empty_runs(self, evaluator: DependabilityEvaluator):
        """Test with no runs returns empty result."""
        result = evaluator.evaluate_multiple_runs([])
        assert result.num_comparisons == 0

    def test_identical_runs(self, evaluator: DependabilityEvaluator):
        """Test identical runs have ROUGE score of 1.0."""
        themes = [
            {"name": "Climate Anxiety", "description": "Worry about climate change"},
        ]
        runs = [
            {"id": "run_1", "themes": themes},
            {"id": "run_2", "themes": themes},
        ]
        result = evaluator.evaluate_multiple_runs(runs)
        assert result.overall_score == pytest.approx(1.0)


class TestDependabilityIntegration:
    """Integration tests for dependability evaluation."""

    def test_realistic_theme_comparison(self):
        """Test with realistic theme sets from multiple runs."""
        evaluator = DependabilityEvaluator()

        # Simulate 3 runs of thematic analysis
        runs = [
            {
                "id": "run_identity_1",
                "themes": [
                    {
                        "name": "Climate Anxiety and Fear",
                        "description": (
                            "Feelings of worry, fear, and anxiety "
                            "about climate change impacts"
                        ),
                    },
                    {
                        "name": "Hope and Action",
                        "description": (
                            "Optimism about climate solutions and "
                            "motivation to take action"
                        ),
                    },
                    {
                        "name": "Systemic Change",
                        "description": (
                            "Need for large-scale political and economic transformation"
                        ),
                    },
                ],
            },
            {
                "id": "run_identity_2",
                "themes": [
                    {
                        "name": "Climate Distress",
                        "description": (
                            "Emotional distress and worry related to climate change"
                        ),
                    },
                    {
                        "name": "Collective Action Hope",
                        "description": (
                            "Belief in the power of collective action "
                            "to address climate change"
                        ),
                    },
                    {
                        "name": "Policy Change",
                        "description": (
                            "Need for governmental and systemic policy changes"
                        ),
                    },
                ],
            },
            {
                "id": "run_identity_3",
                "themes": [
                    {
                        "name": "Environmental Worry",
                        "description": (
                            "Anxiety and concern about environmental degradation"
                        ),
                    },
                    {
                        "name": "Community Action",
                        "description": (
                            "Hope through grassroots and community-based initiatives"
                        ),
                    },
                    {
                        "name": "Institutional Reform",
                        "description": (
                            "Need for reform of institutions addressing climate"
                        ),
                    },
                ],
            },
        ]

        result = evaluator.evaluate_multiple_runs(runs, compare_type="themes")

        # Verify structure
        assert result.num_comparisons == 3  # C(3,2) = 3
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.average_rouge_1 <= 1.0
        assert 0.0 <= result.average_rouge_2 <= 1.0

        # Verify each comparison has valid scores
        for comp in result.pairwise_comparisons:
            assert 0.0 <= comp.rouge_1 <= 1.0
            assert 0.0 <= comp.rouge_2 <= 1.0
            assert 0.0 <= comp.combined_rouge <= 1.0

        # The themes have overlap, so expect moderate scores
        assert result.overall_score > 0.1
