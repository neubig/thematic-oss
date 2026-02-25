"""Dependability Evaluation.

Implements inter-rater reliability evaluation using ROUGE scores to assess
the consistency of thematic analysis results across multiple runs.

Based on Section 3.1 of the Thematic-LM paper (WWW '25).
"""

from dataclasses import dataclass, field


def _get_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from text.

    Args:
        text: Input text to extract n-grams from.
        n: Size of n-grams to extract.

    Returns:
        List of n-gram tuples.
    """
    words = text.lower().split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def _count_ngram_overlap(
    ngrams_a: list[tuple[str, ...]], ngrams_b: list[tuple[str, ...]]
) -> int:
    """Count overlapping n-grams between two lists.

    Args:
        ngrams_a: First list of n-grams.
        ngrams_b: Second list of n-grams.

    Returns:
        Number of overlapping n-grams.
    """
    set_b = set(ngrams_b)
    return sum(1 for ng in ngrams_a if ng in set_b)


def rouge_n_directional(text_a: str, text_b: str, n: int = 1) -> float:
    """Calculate directional ROUGE-N score (A -> B).

    ROUGE-N_A→B = Number of overlapping n-grams in B / Total n-grams in A

    Args:
        text_a: Reference text.
        text_b: Candidate text.
        n: N-gram size (1 for unigrams, 2 for bigrams).

    Returns:
        ROUGE-N score from A to B (0.0 to 1.0).
    """
    ngrams_a = _get_ngrams(text_a, n)
    ngrams_b = _get_ngrams(text_b, n)

    if not ngrams_a:
        return 0.0

    overlap = _count_ngram_overlap(ngrams_a, ngrams_b)
    return overlap / len(ngrams_a)


def rouge_n(text_a: str, text_b: str, n: int = 1) -> float:
    """Calculate symmetric ROUGE-N score.

    ROUGE-N = 0.5 * (ROUGE-N_A→B + ROUGE-N_B→A)

    Args:
        text_a: First text.
        text_b: Second text.
        n: N-gram size (1 for unigrams, 2 for bigrams).

    Returns:
        Symmetric ROUGE-N score (0.0 to 1.0).
    """
    score_a_to_b = rouge_n_directional(text_a, text_b, n)
    score_b_to_a = rouge_n_directional(text_b, text_a, n)
    return 0.5 * (score_a_to_b + score_b_to_a)


def rouge_combined(text_a: str, text_b: str) -> float:
    """Calculate combined ROUGE score (average of ROUGE-1 and ROUGE-2).

    ROUGE = 0.5 * (ROUGE-1 + ROUGE-2)

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Combined ROUGE score (0.0 to 1.0).
    """
    rouge_1 = rouge_n(text_a, text_b, n=1)
    rouge_2 = rouge_n(text_a, text_b, n=2)
    return 0.5 * (rouge_1 + rouge_2)


@dataclass
class PairwiseComparison:
    """Result of comparing two theme/code sets."""

    run_a_id: str
    run_b_id: str
    rouge_1: float
    rouge_2: float
    combined_rouge: float

    @property
    def rouge_score(self) -> float:
        """Alias for combined_rouge for backwards compatibility."""
        return self.combined_rouge


@dataclass
class DependabilityResult:
    """Overall dependability evaluation result."""

    pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)

    @property
    def average_rouge_1(self) -> float:
        """Average ROUGE-1 score across all comparisons."""
        if not self.pairwise_comparisons:
            return 0.0
        return sum(c.rouge_1 for c in self.pairwise_comparisons) / len(
            self.pairwise_comparisons
        )

    @property
    def average_rouge_2(self) -> float:
        """Average ROUGE-2 score across all comparisons."""
        if not self.pairwise_comparisons:
            return 0.0
        return sum(c.rouge_2 for c in self.pairwise_comparisons) / len(
            self.pairwise_comparisons
        )

    @property
    def overall_score(self) -> float:
        """Overall dependability score (average combined ROUGE)."""
        if not self.pairwise_comparisons:
            return 0.0
        return sum(c.combined_rouge for c in self.pairwise_comparisons) / len(
            self.pairwise_comparisons
        )

    @property
    def num_comparisons(self) -> int:
        """Number of pairwise comparisons performed."""
        return len(self.pairwise_comparisons)

    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "overall_score": self.overall_score,
            "average_rouge_1": self.average_rouge_1,
            "average_rouge_2": self.average_rouge_2,
            "num_comparisons": self.num_comparisons,
            "pairwise_comparisons": [
                {
                    "run_a_id": c.run_a_id,
                    "run_b_id": c.run_b_id,
                    "rouge_1": c.rouge_1,
                    "rouge_2": c.rouge_2,
                    "combined_rouge": c.combined_rouge,
                }
                for c in self.pairwise_comparisons
            ],
        }


def _themes_to_text(themes: list[dict]) -> str:
    """Convert a list of themes to a single text representation.

    Concatenates theme names and descriptions for ROUGE comparison.

    Args:
        themes: List of theme dictionaries with 'name' and 'description' keys.

    Returns:
        Combined text representation of all themes.
    """
    parts = []
    for theme in themes:
        name = theme.get("name", "")
        desc = theme.get("description", "")
        parts.append(f"{name} {desc}")
    return " ".join(parts)


def _codes_to_text(codes: list[dict]) -> str:
    """Convert a list of codes to a single text representation.

    Concatenates code names and descriptions for ROUGE comparison.

    Args:
        codes: List of code dictionaries with 'name' and 'description' keys.

    Returns:
        Combined text representation of all codes.
    """
    parts = []
    for code in codes:
        name = code.get("name", "")
        desc = code.get("description", "")
        parts.append(f"{name} {desc}")
    return " ".join(parts)


class DependabilityEvaluator:
    """Evaluates dependability via inter-rater reliability using ROUGE scores.

    Compares multiple runs of thematic analysis to measure consistency.
    Higher ROUGE scores indicate more dependable (consistent) results.
    """

    def compare_texts(
        self, text_a: str, text_b: str, run_a_id: str, run_b_id: str
    ) -> PairwiseComparison:
        """Compare two text representations using ROUGE.

        Args:
            text_a: Text from first run.
            text_b: Text from second run.
            run_a_id: Identifier for first run.
            run_b_id: Identifier for second run.

        Returns:
            PairwiseComparison result.
        """
        r1 = rouge_n(text_a, text_b, n=1)
        r2 = rouge_n(text_a, text_b, n=2)
        combined = 0.5 * (r1 + r2)

        return PairwiseComparison(
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            rouge_1=r1,
            rouge_2=r2,
            combined_rouge=combined,
        )

    def compare_themes(
        self,
        themes_a: list[dict],
        themes_b: list[dict],
        run_a_id: str = "run_a",
        run_b_id: str = "run_b",
    ) -> PairwiseComparison:
        """Compare two sets of themes using ROUGE.

        Args:
            themes_a: Themes from first run.
            themes_b: Themes from second run.
            run_a_id: Identifier for first run.
            run_b_id: Identifier for second run.

        Returns:
            PairwiseComparison result.
        """
        text_a = _themes_to_text(themes_a)
        text_b = _themes_to_text(themes_b)
        return self.compare_texts(text_a, text_b, run_a_id, run_b_id)

    def compare_codes(
        self,
        codes_a: list[dict],
        codes_b: list[dict],
        run_a_id: str = "run_a",
        run_b_id: str = "run_b",
    ) -> PairwiseComparison:
        """Compare two sets of codes using ROUGE.

        Args:
            codes_a: Codes from first run.
            codes_b: Codes from second run.
            run_a_id: Identifier for first run.
            run_b_id: Identifier for second run.

        Returns:
            PairwiseComparison result.
        """
        text_a = _codes_to_text(codes_a)
        text_b = _codes_to_text(codes_b)
        return self.compare_texts(text_a, text_b, run_a_id, run_b_id)

    def evaluate_multiple_runs(
        self,
        runs: list[dict],
        compare_type: str = "themes",
    ) -> DependabilityResult:
        """Evaluate dependability across multiple thematic analysis runs.

        Performs pairwise comparisons between all runs.

        Args:
            runs: List of run dictionaries with 'id' and 'themes'/'codes' keys.
            compare_type: What to compare - 'themes' or 'codes'.

        Returns:
            DependabilityResult with all pairwise comparisons.
        """
        comparisons: list[PairwiseComparison] = []

        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                run_a = runs[i]
                run_b = runs[j]

                run_a_id = run_a.get("id", f"run_{i}")
                run_b_id = run_b.get("id", f"run_{j}")

                if compare_type == "themes":
                    data_a = run_a.get("themes", [])
                    data_b = run_b.get("themes", [])
                    comparison = self.compare_themes(data_a, data_b, run_a_id, run_b_id)
                else:
                    data_a = run_a.get("codes", [])
                    data_b = run_b.get("codes", [])
                    comparison = self.compare_codes(data_a, data_b, run_a_id, run_b_id)

                comparisons.append(comparison)

        return DependabilityResult(pairwise_comparisons=comparisons)
