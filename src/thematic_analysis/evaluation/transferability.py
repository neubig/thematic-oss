"""Transferability Evaluation.

Implements transferability evaluation by measuring theme generalization
across dataset splits. Uses ROUGE scores to assess how well themes
discovered in one split generalize to another.

Based on Section 3.1 of the Thematic-LM paper (WWW '25).
"""

import random
from dataclasses import dataclass, field

from thematic_analysis.evaluation.dependability import rouge_combined, rouge_n


def split_dataset[T](
    data: list[T],
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[list[T], list[T]]:
    """Split a dataset into training and validation sets.

    Args:
        data: List of data items to split.
        train_ratio: Fraction of data to use for training (0.0 to 1.0).
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_set, validation_set).
    """
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0.0 and 1.0")

    items = list(data)
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(items)

    split_idx = int(len(items) * train_ratio)
    return items[:split_idx], items[split_idx:]


def split_dataset_stratified[T](
    data: list[T],
    labels: list[str],
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[list[T], list[T]]:
    """Split a dataset while maintaining label distribution.

    Args:
        data: List of data items to split.
        labels: List of labels corresponding to each data item.
        train_ratio: Fraction of data to use for training (0.0 to 1.0).
        shuffle: Whether to shuffle within each stratum.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_set, validation_set).
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")

    if seed is not None:
        random.seed(seed)

    # Group by label
    groups: dict[str, list[T]] = {}
    for item, label in zip(data, labels, strict=True):
        if label not in groups:
            groups[label] = []
        groups[label].append(item)

    train_set: list[T] = []
    val_set: list[T] = []

    for items in groups.values():
        if shuffle:
            random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        train_set.extend(items[:split_idx])
        val_set.extend(items[split_idx:])

    if shuffle:
        random.shuffle(train_set)
        random.shuffle(val_set)

    return train_set, val_set


@dataclass
class TransferabilityResult:
    """Result of transferability evaluation."""

    rouge_1: float = 0.0
    rouge_2: float = 0.0
    combined_rouge: float = 0.0
    train_theme_count: int = 0
    val_theme_count: int = 0
    train_themes: list[dict] = field(default_factory=list)
    val_themes: list[dict] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Overall transferability score."""
        return self.combined_rouge

    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "overall_score": self.overall_score,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "combined_rouge": self.combined_rouge,
            "train_theme_count": self.train_theme_count,
            "val_theme_count": self.val_theme_count,
            "train_themes": self.train_themes,
            "val_themes": self.val_themes,
        }


def _themes_to_text(themes: list[dict]) -> str:
    """Convert themes to text for ROUGE comparison."""
    parts = []
    for theme in themes:
        name = theme.get("name", "")
        desc = theme.get("description", "")
        parts.append(f"{name} {desc}")
    return " ".join(parts)


class TransferabilityEvaluator:
    """Evaluates transferability by measuring theme generalization.

    Assesses how well themes discovered in a training split generalize
    to a validation split by computing ROUGE overlap scores.
    """

    def compare_theme_sets(
        self,
        train_themes: list[dict],
        val_themes: list[dict],
    ) -> TransferabilityResult:
        """Compare themes from training and validation splits.

        Args:
            train_themes: Themes discovered from training split.
            val_themes: Themes discovered from validation split.

        Returns:
            TransferabilityResult with ROUGE scores.
        """
        text_train = _themes_to_text(train_themes)
        text_val = _themes_to_text(val_themes)

        r1 = rouge_n(text_train, text_val, n=1)
        r2 = rouge_n(text_train, text_val, n=2)
        combined = rouge_combined(text_train, text_val)

        return TransferabilityResult(
            rouge_1=r1,
            rouge_2=r2,
            combined_rouge=combined,
            train_theme_count=len(train_themes),
            val_theme_count=len(val_themes),
            train_themes=train_themes,
            val_themes=val_themes,
        )

    def evaluate(
        self,
        train_themes: list[dict],
        val_themes: list[dict],
    ) -> TransferabilityResult:
        """Evaluate transferability between two theme sets.

        This is an alias for compare_theme_sets for API consistency.

        Args:
            train_themes: Themes discovered from training split.
            val_themes: Themes discovered from validation split.

        Returns:
            TransferabilityResult with evaluation scores.
        """
        return self.compare_theme_sets(train_themes, val_themes)

    def evaluate_cross_validation(
        self,
        theme_sets: list[list[dict]],
    ) -> list[TransferabilityResult]:
        """Evaluate transferability using cross-validation style comparison.

        Compares each theme set against all others to assess
        generalization capability.

        Args:
            theme_sets: List of theme sets from different splits/folds.

        Returns:
            List of TransferabilityResults for each pairwise comparison.
        """
        results: list[TransferabilityResult] = []

        for i in range(len(theme_sets)):
            for j in range(i + 1, len(theme_sets)):
                result = self.compare_theme_sets(
                    theme_sets[i],
                    theme_sets[j],
                )
                results.append(result)

        return results

    def average_transferability(
        self,
        results: list[TransferabilityResult],
    ) -> TransferabilityResult:
        """Compute average transferability across multiple comparisons.

        Args:
            results: List of TransferabilityResults to average.

        Returns:
            TransferabilityResult with averaged scores.
        """
        if not results:
            return TransferabilityResult()

        avg_r1 = sum(r.rouge_1 for r in results) / len(results)
        avg_r2 = sum(r.rouge_2 for r in results) / len(results)
        avg_combined = sum(r.combined_rouge for r in results) / len(results)

        return TransferabilityResult(
            rouge_1=avg_r1,
            rouge_2=avg_r2,
            combined_rouge=avg_combined,
            train_theme_count=0,
            val_theme_count=0,
        )
