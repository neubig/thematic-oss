"""Evaluation module for Thematic-LM.

Provides evaluation metrics based on trustworthiness principles from
qualitative research: credibility, confirmability, dependability, and
transferability.
"""

from thematic_analysis.evaluation.credibility import (
    CredibilityConfig,
    CredibilityResult,
    EvaluatorAgent,
    QuoteConsistency,
    ThemeConsistency,
)
from thematic_analysis.evaluation.dependability import (
    DependabilityEvaluator,
    DependabilityResult,
    PairwiseComparison,
    rouge_combined,
    rouge_n,
    rouge_n_directional,
)
from thematic_analysis.evaluation.transferability import (
    TransferabilityEvaluator,
    TransferabilityResult,
    split_dataset,
    split_dataset_stratified,
)


__all__ = [
    # Credibility/Confirmability
    "EvaluatorAgent",
    "CredibilityConfig",
    "CredibilityResult",
    "ThemeConsistency",
    "QuoteConsistency",
    # Dependability
    "DependabilityEvaluator",
    "DependabilityResult",
    "PairwiseComparison",
    "rouge_n",
    "rouge_n_directional",
    "rouge_combined",
    # Transferability
    "TransferabilityEvaluator",
    "TransferabilityResult",
    "split_dataset",
    "split_dataset_stratified",
]
