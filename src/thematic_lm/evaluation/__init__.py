"""Evaluation module for Thematic-LM.

Provides evaluation metrics based on trustworthiness principles from
qualitative research: credibility, confirmability, dependability, and
transferability.
"""

from thematic_lm.evaluation.credibility import (
    CredibilityConfig,
    CredibilityResult,
    EvaluatorAgent,
    QuoteConsistency,
    ThemeConsistency,
)
from thematic_lm.evaluation.dependability import (
    DependabilityEvaluator,
    DependabilityResult,
    PairwiseComparison,
    rouge_combined,
    rouge_n,
    rouge_n_directional,
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
]
