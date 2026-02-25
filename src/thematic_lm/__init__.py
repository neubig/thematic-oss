"""
Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis

An open-source implementation of the Thematic-LM paper using the OpenHands SDK.
"""

from thematic_lm.identity import (
    CONSERVATIVE_VIEW,
    HUMAN_DRIVEN_CLIMATE,
    INDIGENOUS_VIEW,
    NATURAL_CLIMATE,
    PREDEFINED_IDENTITIES,
    PROGRESSIVE_VIEW,
    IdentityPerspective,
    create_custom_identity,
    get_diverse_identities,
    get_identity,
    list_identities,
)
from thematic_lm.pipeline import (
    DataSegment,
    PipelineConfig,
    PipelineResult,
    ThematicLMPipeline,
)
from thematic_lm.research_context import (
    CODE_6RS,
    CONCEPTUALIZATION_GUIDANCE,
    KEYWORD_6RS,
    THEME_DEVELOPMENT_GUIDANCE,
    ResearchContext,
    ResearchParadigm,
    TheoreticalFramework,
    create_climate_research_context,
    create_healthcare_research_context,
    create_methodology_prompt,
)


__version__ = "0.1.0"
__all__ = [
    "__version__",
    # Pipeline
    "ThematicLMPipeline",
    "PipelineConfig",
    "PipelineResult",
    "DataSegment",
    # Identity perspectives
    "IdentityPerspective",
    "get_identity",
    "list_identities",
    "create_custom_identity",
    "get_diverse_identities",
    "PREDEFINED_IDENTITIES",
    "HUMAN_DRIVEN_CLIMATE",
    "NATURAL_CLIMATE",
    "PROGRESSIVE_VIEW",
    "CONSERVATIVE_VIEW",
    "INDIGENOUS_VIEW",
    # Research context (Naeem et al. 2025)
    "ResearchContext",
    "ResearchParadigm",
    "TheoreticalFramework",
    "create_methodology_prompt",
    "create_climate_research_context",
    "create_healthcare_research_context",
    "KEYWORD_6RS",
    "CODE_6RS",
    "THEME_DEVELOPMENT_GUIDANCE",
    "CONCEPTUALIZATION_GUIDANCE",
]
