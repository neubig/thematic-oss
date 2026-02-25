"""
Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis

An open-source implementation of the Thematic-LM paper using the OpenHands SDK.
"""

from thematic_lm.hermeneutics import (
    CHAIN_OF_THOUGHT_TEMPLATE,
    ONE_CODE_PER_PROMPT_TEMPLATE,
    RATIONALE_TEMPLATE,
    AdaptedCodebook,
    CodeDefinition,
    CodingRationale,
    DirectiveType,
    RationaleAnalysis,
    ScopeType,
    analyze_rationales,
    create_climate_adapted_codebook,
    create_cot_prompt,
    create_single_code_prompt,
    suggest_codebook_improvements,
)
from thematic_lm.hitl import (
    AuditTrail,
    AutomationLevel,
    HITLConfig,
    HITLController,
    Intervention,
    InterventionStage,
    InterventionType,
    ReviewCheckpoint,
    SeedInput,
    create_seed_input,
)
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
from thematic_lm.persona import (
    ContextDocument,
    ContextType,
    PersonaContext,
    PersonaGenerator,
    create_domain_expert_persona,
    create_persona_from_documents,
    load_document_from_file,
    load_document_from_text,
)
from thematic_lm.pipeline import (
    DataSegment,
    PipelineConfig,
    PipelineResult,
    ThematicLMPipeline,
)
from thematic_lm.prompts import (
    AggregatorPrompts,
    CoderPrompts,
    PromptConfig,
    ReviewerPrompts,
    ThemeAggregatorPrompts,
    ThemeCoderPrompts,
    create_domain_prompts,
    get_prompt_config,
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
    # Hermeneutics (Dunivin 2025)
    "CodeDefinition",
    "AdaptedCodebook",
    "ScopeType",
    "DirectiveType",
    "CodingRationale",
    "RationaleAnalysis",
    "create_cot_prompt",
    "create_single_code_prompt",
    "analyze_rationales",
    "suggest_codebook_improvements",
    "create_climate_adapted_codebook",
    "CHAIN_OF_THOUGHT_TEMPLATE",
    "RATIONALE_TEMPLATE",
    "ONE_CODE_PER_PROMPT_TEMPLATE",
    # Enhanced persona system
    "ContextDocument",
    "ContextType",
    "PersonaContext",
    "PersonaGenerator",
    "create_persona_from_documents",
    "create_domain_expert_persona",
    "load_document_from_text",
    "load_document_from_file",
    # Human-in-the-Loop (HITL) system
    "HITLController",
    "HITLConfig",
    "AutomationLevel",
    "InterventionStage",
    "InterventionType",
    "Intervention",
    "AuditTrail",
    "ReviewCheckpoint",
    "SeedInput",
    "create_seed_input",
    # Configurable prompts (Issue #38)
    "PromptConfig",
    "CoderPrompts",
    "AggregatorPrompts",
    "ReviewerPrompts",
    "ThemeCoderPrompts",
    "ThemeAggregatorPrompts",
    "get_prompt_config",
    "create_domain_prompts",
]
