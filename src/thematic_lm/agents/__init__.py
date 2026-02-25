"""Agents module for thematic analysis."""

from thematic_lm.agents.aggregator import (
    AggregationResult,
    AggregatorConfig,
    CodeAggregatorAgent,
    MergedCode,
)
from thematic_lm.agents.base import AgentConfig, BaseAgent
from thematic_lm.agents.coder import CodeAssignment, CoderAgent, CoderConfig


__all__ = [
    "BaseAgent",
    "AgentConfig",
    "CoderAgent",
    "CoderConfig",
    "CodeAssignment",
    "CodeAggregatorAgent",
    "AggregatorConfig",
    "AggregationResult",
    "MergedCode",
]
