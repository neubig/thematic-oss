"""Human-in-the-Loop (HITL) system for thematic analysis.

Implements a flexible HITL system that allows researchers to participate
at any stage of the thematic analysis process, from minimal to maximum
involvement.

Intervention Points:
1. Pre-Analysis Input: Seed codes, themes, theory injection
2. Coding Stage: Code review, editing, quote validation
3. Theme Development: Theme review, editing, merging
4. Post-Analysis: Final review, iteration requests

Configuration Levels:
1. Fully Automated: No human intervention
2. Supervised: Human reviews key outputs
3. Guided: Human provides seeds and reviews
4. Collaborative: Human intervenes at multiple points
5. Human-Led: Human makes key decisions, LLM assists

Based on: Braun & Clarke (2006), Naeem et al. (2025)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AutomationLevel(Enum):
    """Level of automation in the analysis process."""

    FULLY_AUTOMATED = "fully_automated"  # No human intervention
    SUPERVISED = "supervised"  # Human reviews but doesn't modify
    GUIDED = "guided"  # Human provides seeds and reviews
    COLLABORATIVE = "collaborative"  # Human intervenes at multiple points
    HUMAN_LED = "human_led"  # Human makes key decisions


class InterventionStage(Enum):
    """Stages where human intervention can occur."""

    PRE_ANALYSIS = "pre_analysis"
    CODING = "coding"
    CODE_AGGREGATION = "code_aggregation"
    THEME_DEVELOPMENT = "theme_development"
    THEME_AGGREGATION = "theme_aggregation"
    POST_ANALYSIS = "post_analysis"


class InterventionType(Enum):
    """Types of human interventions."""

    # Pre-analysis
    SEED_CODES = "seed_codes"
    SEED_THEMES = "seed_themes"
    THEORY_INJECTION = "theory_injection"
    RESEARCH_QUESTIONS = "research_questions"
    KEYWORDS = "keywords"

    # Coding stage
    CODE_REVIEW = "code_review"
    CODE_EDIT = "code_edit"
    CODE_MERGE = "code_merge"
    CODE_DELETE = "code_delete"
    QUOTE_VALIDATION = "quote_validation"

    # Theme development
    THEME_REVIEW = "theme_review"
    THEME_EDIT = "theme_edit"
    THEME_MERGE = "theme_merge"
    THEME_DELETE = "theme_delete"
    HIERARCHY_ADJUST = "hierarchy_adjust"

    # Post-analysis
    FINAL_REVIEW = "final_review"
    ITERATION_REQUEST = "iteration_request"
    ANNOTATION = "annotation"


@dataclass
class Intervention:
    """A record of a human intervention.

    Attributes:
        intervention_type: Type of intervention.
        stage: Stage at which intervention occurred.
        timestamp: When the intervention was made.
        user_id: Identifier for the user who made the intervention.
        data: The intervention data (varies by type).
        notes: Optional notes about the intervention.
        previous_value: Previous value before the intervention.
    """

    intervention_type: InterventionType
    stage: InterventionStage
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = ""
    data: Any = None
    notes: str = ""
    previous_value: Any = None

    def to_dict(self) -> dict:
        """Convert to dictionary format for serialization."""
        return {
            "type": self.intervention_type.value,
            "stage": self.stage.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "data": self.data,
            "notes": self.notes,
            "previous_value": self.previous_value,
        }


@dataclass
class AuditTrail:
    """Audit trail of all human interventions.

    Tracks all human modifications for transparency and reproducibility.
    """

    interventions: list[Intervention] = field(default_factory=list)
    session_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)

    def add(self, intervention: Intervention) -> None:
        """Add an intervention to the audit trail."""
        self.interventions.append(intervention)

    def get_by_stage(self, stage: InterventionStage) -> list[Intervention]:
        """Get all interventions at a specific stage."""
        return [i for i in self.interventions if i.stage == stage]

    def get_by_type(self, intervention_type: InterventionType) -> list[Intervention]:
        """Get all interventions of a specific type."""
        return [
            i for i in self.interventions if i.intervention_type == intervention_type
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary format for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "interventions": [i.to_dict() for i in self.interventions],
        }

    def __len__(self) -> int:
        return len(self.interventions)


@dataclass
class SeedInput:
    """Pre-analysis input from researchers.

    Contains seed codes, themes, and other inputs that guide the analysis.

    Attributes:
        seed_codes: Codes the researcher believes are important.
        seed_themes: Pre-defined themes to organize around.
        theory: Theoretical framework to apply.
        research_questions: Specific questions to guide coding.
        keywords: Terms to pay special attention to.
        domain_context: Domain-specific context.
    """

    seed_codes: list[str] = field(default_factory=list)
    seed_themes: list[str] = field(default_factory=list)
    theory: str = ""
    research_questions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    domain_context: str = ""

    def is_empty(self) -> bool:
        """Check if seed input is empty."""
        return not any(
            [
                self.seed_codes,
                self.seed_themes,
                self.theory,
                self.research_questions,
                self.keywords,
                self.domain_context,
            ]
        )

    def to_prompt_section(self) -> str:
        """Convert to prompt section for agent guidance."""
        sections = []

        if self.seed_codes:
            codes_list = ", ".join(self.seed_codes)
            sections.append(
                f"**Seed Codes (researcher-provided):** {codes_list}\n"
                "Pay special attention to these codes when analyzing data."
            )

        if self.seed_themes:
            themes_list = ", ".join(self.seed_themes)
            sections.append(
                f"**Expected Themes:** {themes_list}\n"
                "Consider organizing codes around these themes."
            )

        if self.theory:
            sections.append(
                f"**Theoretical Framework:** {self.theory}\n"
                "Apply this theoretical lens in your analysis."
            )

        if self.research_questions:
            rq_list = "\n".join(f"  - {rq}" for rq in self.research_questions)
            sections.append(
                f"**Research Questions:**\n{rq_list}\n"
                "Ensure coding addresses these questions."
            )

        if self.keywords:
            kw_list = ", ".join(self.keywords)
            sections.append(
                f"**Keywords of Interest:** {kw_list}\n"
                "Flag when these terms appear in the data."
            )

        if self.domain_context:
            sections.append(f"**Domain Context:** {self.domain_context}")

        return "\n\n".join(sections) if sections else ""


@dataclass
class ReviewCheckpoint:
    """A checkpoint where human review can occur.

    Represents a pause point in the analysis where humans can review
    and potentially modify results.

    Attributes:
        stage: The analysis stage.
        data: The data to review.
        is_approved: Whether the checkpoint has been approved.
        modifications: Any modifications made during review.
        reviewer_notes: Notes from the reviewer.
    """

    stage: InterventionStage
    data: Any
    is_approved: bool = False
    modifications: Any = None
    reviewer_notes: str = ""

    def approve(self, notes: str = "") -> None:
        """Approve the checkpoint without modifications."""
        self.is_approved = True
        self.reviewer_notes = notes

    def modify(self, modified_data: Any, notes: str = "") -> None:
        """Modify and approve the checkpoint."""
        self.modifications = modified_data
        self.is_approved = True
        self.reviewer_notes = notes

    def get_result(self) -> Any:
        """Get the result (modified if modifications exist, else original)."""
        return self.modifications if self.modifications is not None else self.data


@dataclass
class HITLConfig:
    """Configuration for human-in-the-loop behavior.

    Attributes:
        automation_level: The level of automation.
        review_stages: Stages requiring human review.
        auto_approve_timeout: Seconds before auto-approving (0 = no auto).
        require_approval: Whether approval is required to continue.
        callback: Optional callback function for review notifications.
    """

    automation_level: AutomationLevel = AutomationLevel.FULLY_AUTOMATED
    review_stages: list[InterventionStage] = field(default_factory=list)
    auto_approve_timeout: int = 0  # 0 means no auto-approve
    require_approval: bool = False
    callback: Callable[[ReviewCheckpoint], None] | None = None

    @classmethod
    def fully_automated(cls) -> "HITLConfig":
        """Create a fully automated configuration."""
        return cls(automation_level=AutomationLevel.FULLY_AUTOMATED)

    @classmethod
    def supervised(cls) -> "HITLConfig":
        """Create a supervised configuration with review at key stages."""
        return cls(
            automation_level=AutomationLevel.SUPERVISED,
            review_stages=[
                InterventionStage.CODE_AGGREGATION,
                InterventionStage.THEME_AGGREGATION,
            ],
        )

    @classmethod
    def guided(cls) -> "HITLConfig":
        """Create a guided configuration with seeds and review."""
        return cls(
            automation_level=AutomationLevel.GUIDED,
            review_stages=[
                InterventionStage.PRE_ANALYSIS,
                InterventionStage.THEME_DEVELOPMENT,
                InterventionStage.POST_ANALYSIS,
            ],
        )

    @classmethod
    def collaborative(cls) -> "HITLConfig":
        """Create a collaborative configuration with multiple checkpoints."""
        return cls(
            automation_level=AutomationLevel.COLLABORATIVE,
            review_stages=list(InterventionStage),
            require_approval=True,
        )

    @classmethod
    def human_led(cls) -> "HITLConfig":
        """Create a human-led configuration where humans drive decisions."""
        return cls(
            automation_level=AutomationLevel.HUMAN_LED,
            review_stages=list(InterventionStage),
            require_approval=True,
        )

    def requires_review_at(self, stage: InterventionStage) -> bool:
        """Check if review is required at a specific stage."""
        if self.automation_level == AutomationLevel.FULLY_AUTOMATED:
            return False
        return stage in self.review_stages


class HITLController:
    """Controller for managing human-in-the-loop interactions.

    Manages checkpoints, interventions, and the audit trail.
    """

    def __init__(
        self,
        config: HITLConfig | None = None,
        seed_input: SeedInput | None = None,
    ):
        """Initialize the HITL controller.

        Args:
            config: HITL configuration.
            seed_input: Pre-analysis seed input.
        """
        self.config = config or HITLConfig.fully_automated()
        self.seed_input = seed_input or SeedInput()
        self.audit_trail = AuditTrail()
        self.checkpoints: dict[InterventionStage, ReviewCheckpoint] = {}

        # Record seed input if provided
        if not self.seed_input.is_empty():
            self.audit_trail.add(
                Intervention(
                    intervention_type=InterventionType.SEED_CODES,
                    stage=InterventionStage.PRE_ANALYSIS,
                    data=self.seed_input,
                    notes="Initial seed input provided",
                )
            )

    def set_seed_input(self, seed_input: SeedInput) -> None:
        """Set or update seed input.

        Args:
            seed_input: The new seed input.
        """
        old_input = self.seed_input
        self.seed_input = seed_input

        self.audit_trail.add(
            Intervention(
                intervention_type=InterventionType.SEED_CODES,
                stage=InterventionStage.PRE_ANALYSIS,
                data=seed_input,
                previous_value=old_input,
                notes="Seed input updated",
            )
        )

    def add_seed_codes(self, codes: list[str]) -> None:
        """Add seed codes to the input."""
        self.seed_input.seed_codes.extend(codes)
        self.audit_trail.add(
            Intervention(
                intervention_type=InterventionType.SEED_CODES,
                stage=InterventionStage.PRE_ANALYSIS,
                data=codes,
                notes=f"Added {len(codes)} seed codes",
            )
        )

    def add_seed_themes(self, themes: list[str]) -> None:
        """Add seed themes to the input."""
        self.seed_input.seed_themes.extend(themes)
        self.audit_trail.add(
            Intervention(
                intervention_type=InterventionType.SEED_THEMES,
                stage=InterventionStage.PRE_ANALYSIS,
                data=themes,
                notes=f"Added {len(themes)} seed themes",
            )
        )

    def inject_theory(self, theory: str) -> None:
        """Inject theoretical framework."""
        old_theory = self.seed_input.theory
        self.seed_input.theory = theory
        self.audit_trail.add(
            Intervention(
                intervention_type=InterventionType.THEORY_INJECTION,
                stage=InterventionStage.PRE_ANALYSIS,
                data=theory,
                previous_value=old_theory,
                notes="Theory injected",
            )
        )

    def create_checkpoint(
        self,
        stage: InterventionStage,
        data: Any,
    ) -> ReviewCheckpoint:
        """Create a review checkpoint.

        Args:
            stage: The analysis stage.
            data: The data to review.

        Returns:
            A ReviewCheckpoint instance.
        """
        checkpoint = ReviewCheckpoint(stage=stage, data=data)
        self.checkpoints[stage] = checkpoint

        # Auto-approve if not requiring review at this stage
        if not self.config.requires_review_at(stage):
            checkpoint.approve("Auto-approved (not a review stage)")

        # Call notification callback if set
        if self.config.callback and not checkpoint.is_approved:
            self.config.callback(checkpoint)

        return checkpoint

    def approve_checkpoint(
        self,
        stage: InterventionStage,
        notes: str = "",
    ) -> bool:
        """Approve a checkpoint without modifications.

        Args:
            stage: The stage to approve.
            notes: Optional notes.

        Returns:
            True if approval was successful.
        """
        if stage not in self.checkpoints:
            return False

        checkpoint = self.checkpoints[stage]
        checkpoint.approve(notes)

        self.audit_trail.add(
            Intervention(
                intervention_type=InterventionType.FINAL_REVIEW,
                stage=stage,
                notes=notes or "Checkpoint approved",
            )
        )

        return True

    def modify_checkpoint(
        self,
        stage: InterventionStage,
        modified_data: Any,
        notes: str = "",
    ) -> bool:
        """Modify a checkpoint.

        Args:
            stage: The stage to modify.
            modified_data: The modified data.
            notes: Optional notes.

        Returns:
            True if modification was successful.
        """
        if stage not in self.checkpoints:
            return False

        checkpoint = self.checkpoints[stage]
        old_data = checkpoint.data
        checkpoint.modify(modified_data, notes)

        self.audit_trail.add(
            Intervention(
                intervention_type=InterventionType.CODE_EDIT,  # Generic edit
                stage=stage,
                data=modified_data,
                previous_value=old_data,
                notes=notes or "Checkpoint modified",
            )
        )

        return True

    def record_intervention(
        self,
        intervention_type: InterventionType,
        stage: InterventionStage,
        data: Any,
        previous_value: Any = None,
        notes: str = "",
        user_id: str = "",
    ) -> None:
        """Record an intervention to the audit trail.

        Args:
            intervention_type: Type of intervention.
            stage: Stage of intervention.
            data: Intervention data.
            previous_value: Previous value before change.
            notes: Optional notes.
            user_id: Optional user identifier.
        """
        self.audit_trail.add(
            Intervention(
                intervention_type=intervention_type,
                stage=stage,
                data=data,
                previous_value=previous_value,
                notes=notes,
                user_id=user_id,
            )
        )

    def get_prompt_augmentation(self) -> str:
        """Get prompt augmentation from seed input.

        Returns:
            String to add to prompts based on seed input.
        """
        return self.seed_input.to_prompt_section()

    def is_checkpoint_approved(self, stage: InterventionStage) -> bool:
        """Check if a checkpoint is approved.

        Args:
            stage: The stage to check.

        Returns:
            True if approved or not requiring review.
        """
        if not self.config.requires_review_at(stage):
            return True

        if stage not in self.checkpoints:
            return False

        return self.checkpoints[stage].is_approved

    def get_checkpoint_result(self, stage: InterventionStage) -> Any:
        """Get the result from a checkpoint (original or modified).

        Args:
            stage: The stage to get result for.

        Returns:
            The checkpoint result, or None if not found.
        """
        if stage not in self.checkpoints:
            return None
        return self.checkpoints[stage].get_result()

    def get_audit_summary(self) -> dict:
        """Get a summary of the audit trail.

        Returns:
            Dictionary with audit summary statistics.
        """
        by_stage: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for intervention in self.audit_trail.interventions:
            stage = intervention.stage.value
            itype = intervention.intervention_type.value

            by_stage[stage] = by_stage.get(stage, 0) + 1
            by_type[itype] = by_type.get(itype, 0) + 1

        return {
            "total_interventions": len(self.audit_trail),
            "by_stage": by_stage,
            "by_type": by_type,
            "checkpoints_created": len(self.checkpoints),
            "checkpoints_approved": sum(
                1 for cp in self.checkpoints.values() if cp.is_approved
            ),
        }


def create_seed_input(
    codes: list[str] | None = None,
    themes: list[str] | None = None,
    theory: str = "",
    questions: list[str] | None = None,
    keywords: list[str] | None = None,
) -> SeedInput:
    """Convenience function to create seed input.

    Args:
        codes: Seed codes.
        themes: Seed themes.
        theory: Theoretical framework.
        questions: Research questions.
        keywords: Keywords of interest.

    Returns:
        A SeedInput instance.
    """
    return SeedInput(
        seed_codes=codes or [],
        seed_themes=themes or [],
        theory=theory,
        research_questions=questions or [],
        keywords=keywords or [],
    )
