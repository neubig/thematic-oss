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


__all__ = [
    "EvaluatorAgent",
    "CredibilityConfig",
    "CredibilityResult",
    "ThemeConsistency",
    "QuoteConsistency",
]
