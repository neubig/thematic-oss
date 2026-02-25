"""
Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis

An open-source implementation of the Thematic-LM paper using the OpenHands SDK.
"""

from thematic_lm.pipeline import (
    DataSegment,
    PipelineConfig,
    PipelineResult,
    ThematicLMPipeline,
)


__version__ = "0.1.0"
__all__ = [
    "__version__",
    "ThematicLMPipeline",
    "PipelineConfig",
    "PipelineResult",
    "DataSegment",
]
