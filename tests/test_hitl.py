"""Tests for Human-in-the-Loop (HITL) module."""

from thematic_analysis.hitl import (
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


class TestAutomationLevel:
    """Tests for AutomationLevel enum."""

    def test_automation_levels(self):
        """Test all automation levels have expected values."""
        assert AutomationLevel.FULLY_AUTOMATED.value == "fully_automated"
        assert AutomationLevel.SUPERVISED.value == "supervised"
        assert AutomationLevel.GUIDED.value == "guided"
        assert AutomationLevel.COLLABORATIVE.value == "collaborative"
        assert AutomationLevel.HUMAN_LED.value == "human_led"


class TestInterventionStage:
    """Tests for InterventionStage enum."""

    def test_intervention_stages(self):
        """Test all stages have expected values."""
        assert InterventionStage.PRE_ANALYSIS.value == "pre_analysis"
        assert InterventionStage.CODING.value == "coding"
        assert InterventionStage.THEME_DEVELOPMENT.value == "theme_development"
        assert InterventionStage.POST_ANALYSIS.value == "post_analysis"


class TestInterventionType:
    """Tests for InterventionType enum."""

    def test_pre_analysis_types(self):
        """Test pre-analysis intervention types."""
        assert InterventionType.SEED_CODES.value == "seed_codes"
        assert InterventionType.THEORY_INJECTION.value == "theory_injection"

    def test_coding_types(self):
        """Test coding stage intervention types."""
        assert InterventionType.CODE_REVIEW.value == "code_review"
        assert InterventionType.CODE_EDIT.value == "code_edit"
        assert InterventionType.QUOTE_VALIDATION.value == "quote_validation"

    def test_theme_types(self):
        """Test theme development intervention types."""
        assert InterventionType.THEME_REVIEW.value == "theme_review"
        assert InterventionType.THEME_MERGE.value == "theme_merge"


class TestIntervention:
    """Tests for Intervention dataclass."""

    def test_basic_creation(self):
        """Test creating a basic intervention."""
        intervention = Intervention(
            intervention_type=InterventionType.SEED_CODES,
            stage=InterventionStage.PRE_ANALYSIS,
        )
        assert intervention.intervention_type == InterventionType.SEED_CODES
        assert intervention.stage == InterventionStage.PRE_ANALYSIS
        assert intervention.data is None

    def test_full_creation(self):
        """Test creating a fully specified intervention."""
        intervention = Intervention(
            intervention_type=InterventionType.CODE_EDIT,
            stage=InterventionStage.CODING,
            user_id="user123",
            data={"code": "new_code"},
            notes="Added new code",
            previous_value={"code": "old_code"},
        )
        assert intervention.user_id == "user123"
        assert intervention.data == {"code": "new_code"}
        assert intervention.previous_value == {"code": "old_code"}

    def test_to_dict(self):
        """Test converting intervention to dictionary."""
        intervention = Intervention(
            intervention_type=InterventionType.SEED_CODES,
            stage=InterventionStage.PRE_ANALYSIS,
            data=["code1", "code2"],
        )
        d = intervention.to_dict()
        assert d["type"] == "seed_codes"
        assert d["stage"] == "pre_analysis"
        assert d["data"] == ["code1", "code2"]
        assert "timestamp" in d


class TestAuditTrail:
    """Tests for AuditTrail class."""

    def test_empty_trail(self):
        """Test empty audit trail."""
        trail = AuditTrail()
        assert len(trail) == 0
        assert trail.interventions == []

    def test_add_intervention(self):
        """Test adding interventions."""
        trail = AuditTrail()
        intervention = Intervention(
            intervention_type=InterventionType.SEED_CODES,
            stage=InterventionStage.PRE_ANALYSIS,
        )
        trail.add(intervention)
        assert len(trail) == 1

    def test_get_by_stage(self):
        """Test filtering by stage."""
        trail = AuditTrail()
        trail.add(
            Intervention(
                intervention_type=InterventionType.SEED_CODES,
                stage=InterventionStage.PRE_ANALYSIS,
            )
        )
        trail.add(
            Intervention(
                intervention_type=InterventionType.CODE_EDIT,
                stage=InterventionStage.CODING,
            )
        )
        trail.add(
            Intervention(
                intervention_type=InterventionType.THEORY_INJECTION,
                stage=InterventionStage.PRE_ANALYSIS,
            )
        )

        pre_analysis = trail.get_by_stage(InterventionStage.PRE_ANALYSIS)
        assert len(pre_analysis) == 2

        coding = trail.get_by_stage(InterventionStage.CODING)
        assert len(coding) == 1

    def test_get_by_type(self):
        """Test filtering by type."""
        trail = AuditTrail()
        trail.add(
            Intervention(
                intervention_type=InterventionType.SEED_CODES,
                stage=InterventionStage.PRE_ANALYSIS,
            )
        )
        trail.add(
            Intervention(
                intervention_type=InterventionType.SEED_CODES,
                stage=InterventionStage.PRE_ANALYSIS,
            )
        )
        trail.add(
            Intervention(
                intervention_type=InterventionType.CODE_EDIT,
                stage=InterventionStage.CODING,
            )
        )

        seed_codes = trail.get_by_type(InterventionType.SEED_CODES)
        assert len(seed_codes) == 2

    def test_to_dict(self):
        """Test converting trail to dictionary."""
        trail = AuditTrail(session_id="test123")
        trail.add(
            Intervention(
                intervention_type=InterventionType.SEED_CODES,
                stage=InterventionStage.PRE_ANALYSIS,
            )
        )

        d = trail.to_dict()
        assert d["session_id"] == "test123"
        assert "started_at" in d
        assert len(d["interventions"]) == 1


class TestSeedInput:
    """Tests for SeedInput dataclass."""

    def test_empty_seed_input(self):
        """Test empty seed input."""
        seed = SeedInput()
        assert seed.is_empty()

    def test_non_empty_with_codes(self):
        """Test seed input with codes is not empty."""
        seed = SeedInput(seed_codes=["code1", "code2"])
        assert not seed.is_empty()

    def test_non_empty_with_theory(self):
        """Test seed input with theory is not empty."""
        seed = SeedInput(theory="Social constructionism")
        assert not seed.is_empty()

    def test_to_prompt_section_empty(self):
        """Test prompt section for empty input."""
        seed = SeedInput()
        assert seed.to_prompt_section() == ""

    def test_to_prompt_section_with_codes(self):
        """Test prompt section with seed codes."""
        seed = SeedInput(seed_codes=["anxiety", "hope"])
        prompt = seed.to_prompt_section()
        assert "Seed Codes" in prompt
        assert "anxiety" in prompt
        assert "hope" in prompt

    def test_to_prompt_section_with_themes(self):
        """Test prompt section with seed themes."""
        seed = SeedInput(seed_themes=["Emotional Response", "Coping Strategies"])
        prompt = seed.to_prompt_section()
        assert "Expected Themes" in prompt
        assert "Emotional Response" in prompt

    def test_to_prompt_section_with_theory(self):
        """Test prompt section with theory."""
        seed = SeedInput(theory="Grounded theory approach")
        prompt = seed.to_prompt_section()
        assert "Theoretical Framework" in prompt
        assert "Grounded theory" in prompt

    def test_to_prompt_section_with_questions(self):
        """Test prompt section with research questions."""
        seed = SeedInput(
            research_questions=["How do people cope?", "What drives hope?"]
        )
        prompt = seed.to_prompt_section()
        assert "Research Questions" in prompt
        assert "How do people cope?" in prompt

    def test_to_prompt_section_with_keywords(self):
        """Test prompt section with keywords."""
        seed = SeedInput(keywords=["climate", "future", "children"])
        prompt = seed.to_prompt_section()
        assert "Keywords of Interest" in prompt
        assert "climate" in prompt


class TestReviewCheckpoint:
    """Tests for ReviewCheckpoint class."""

    def test_basic_creation(self):
        """Test creating a checkpoint."""
        checkpoint = ReviewCheckpoint(
            stage=InterventionStage.CODING,
            data={"codes": ["code1", "code2"]},
        )
        assert checkpoint.stage == InterventionStage.CODING
        assert not checkpoint.is_approved

    def test_approve(self):
        """Test approving a checkpoint."""
        checkpoint = ReviewCheckpoint(
            stage=InterventionStage.CODING,
            data=["code1"],
        )
        checkpoint.approve("Looks good")
        assert checkpoint.is_approved
        assert checkpoint.reviewer_notes == "Looks good"

    def test_modify(self):
        """Test modifying a checkpoint."""
        checkpoint = ReviewCheckpoint(
            stage=InterventionStage.CODING,
            data=["code1"],
        )
        checkpoint.modify(["code1", "code2"], "Added new code")
        assert checkpoint.is_approved
        assert checkpoint.modifications == ["code1", "code2"]
        assert checkpoint.reviewer_notes == "Added new code"

    def test_get_result_unmodified(self):
        """Test getting result without modifications."""
        checkpoint = ReviewCheckpoint(
            stage=InterventionStage.CODING,
            data=["original"],
        )
        checkpoint.approve()
        assert checkpoint.get_result() == ["original"]

    def test_get_result_modified(self):
        """Test getting result with modifications."""
        checkpoint = ReviewCheckpoint(
            stage=InterventionStage.CODING,
            data=["original"],
        )
        checkpoint.modify(["modified"])
        assert checkpoint.get_result() == ["modified"]


class TestHITLConfig:
    """Tests for HITLConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = HITLConfig()
        assert config.automation_level == AutomationLevel.FULLY_AUTOMATED
        assert config.review_stages == []

    def test_fully_automated(self):
        """Test fully automated preset."""
        config = HITLConfig.fully_automated()
        assert config.automation_level == AutomationLevel.FULLY_AUTOMATED
        assert not config.requires_review_at(InterventionStage.CODING)

    def test_supervised(self):
        """Test supervised preset."""
        config = HITLConfig.supervised()
        assert config.automation_level == AutomationLevel.SUPERVISED
        assert config.requires_review_at(InterventionStage.CODE_AGGREGATION)
        assert config.requires_review_at(InterventionStage.THEME_AGGREGATION)
        assert not config.requires_review_at(InterventionStage.CODING)

    def test_guided(self):
        """Test guided preset."""
        config = HITLConfig.guided()
        assert config.automation_level == AutomationLevel.GUIDED
        assert config.requires_review_at(InterventionStage.PRE_ANALYSIS)
        assert config.requires_review_at(InterventionStage.THEME_DEVELOPMENT)
        assert config.requires_review_at(InterventionStage.POST_ANALYSIS)

    def test_collaborative(self):
        """Test collaborative preset."""
        config = HITLConfig.collaborative()
        assert config.automation_level == AutomationLevel.COLLABORATIVE
        assert config.require_approval
        # Should require review at all stages
        for stage in InterventionStage:
            assert config.requires_review_at(stage)

    def test_human_led(self):
        """Test human-led preset."""
        config = HITLConfig.human_led()
        assert config.automation_level == AutomationLevel.HUMAN_LED
        assert config.require_approval


class TestHITLController:
    """Tests for HITLController class."""

    def test_default_initialization(self):
        """Test default controller initialization."""
        controller = HITLController()
        assert controller.config.automation_level == AutomationLevel.FULLY_AUTOMATED
        assert controller.seed_input.is_empty()
        assert len(controller.audit_trail) == 0

    def test_initialization_with_seed_input(self):
        """Test controller with seed input."""
        seed = SeedInput(seed_codes=["code1", "code2"])
        controller = HITLController(seed_input=seed)

        assert not controller.seed_input.is_empty()
        assert len(controller.audit_trail) == 1  # Seed input recorded

    def test_set_seed_input(self):
        """Test setting seed input."""
        controller = HITLController()
        seed = SeedInput(seed_codes=["new_code"])
        controller.set_seed_input(seed)

        assert controller.seed_input.seed_codes == ["new_code"]
        assert len(controller.audit_trail) == 1

    def test_add_seed_codes(self):
        """Test adding seed codes."""
        controller = HITLController()
        controller.add_seed_codes(["code1", "code2"])

        assert "code1" in controller.seed_input.seed_codes
        assert "code2" in controller.seed_input.seed_codes

    def test_add_seed_themes(self):
        """Test adding seed themes."""
        controller = HITLController()
        controller.add_seed_themes(["Theme A", "Theme B"])

        assert "Theme A" in controller.seed_input.seed_themes

    def test_inject_theory(self):
        """Test theory injection."""
        controller = HITLController()
        controller.inject_theory("Social constructionism")

        assert controller.seed_input.theory == "Social constructionism"
        assert len(controller.audit_trail) == 1

    def test_create_checkpoint_auto_approve(self):
        """Test checkpoint creation with auto-approval."""
        controller = HITLController()  # Fully automated
        checkpoint = controller.create_checkpoint(
            InterventionStage.CODING,
            {"codes": ["code1"]},
        )

        assert checkpoint.is_approved  # Auto-approved

    def test_create_checkpoint_requires_review(self):
        """Test checkpoint creation requiring review."""
        config = HITLConfig.collaborative()
        controller = HITLController(config=config)
        checkpoint = controller.create_checkpoint(
            InterventionStage.CODING,
            {"codes": ["code1"]},
        )

        assert not checkpoint.is_approved  # Needs review

    def test_approve_checkpoint(self):
        """Test approving a checkpoint."""
        config = HITLConfig.collaborative()
        controller = HITLController(config=config)
        controller.create_checkpoint(InterventionStage.CODING, ["code1"])

        result = controller.approve_checkpoint(InterventionStage.CODING, "Approved")

        assert result
        assert controller.is_checkpoint_approved(InterventionStage.CODING)

    def test_modify_checkpoint(self):
        """Test modifying a checkpoint."""
        config = HITLConfig.collaborative()
        controller = HITLController(config=config)
        controller.create_checkpoint(InterventionStage.CODING, ["code1"])

        result = controller.modify_checkpoint(
            InterventionStage.CODING,
            ["code1", "code2"],
            "Added code2",
        )

        assert result
        assert controller.get_checkpoint_result(InterventionStage.CODING) == [
            "code1",
            "code2",
        ]

    def test_record_intervention(self):
        """Test recording an intervention."""
        controller = HITLController()
        controller.record_intervention(
            intervention_type=InterventionType.CODE_EDIT,
            stage=InterventionStage.CODING,
            data={"edited": True},
            user_id="user123",
            notes="Manual edit",
        )

        assert len(controller.audit_trail) == 1

    def test_get_prompt_augmentation(self):
        """Test getting prompt augmentation."""
        seed = SeedInput(
            seed_codes=["anxiety", "hope"],
            theory="Risk perception theory",
        )
        controller = HITLController(seed_input=seed)

        prompt = controller.get_prompt_augmentation()

        assert "anxiety" in prompt
        assert "Risk perception theory" in prompt

    def test_is_checkpoint_approved_no_checkpoint(self):
        """Test checking approval for nonexistent checkpoint."""
        controller = HITLController()

        # Should return True for fully automated (no review needed)
        assert controller.is_checkpoint_approved(InterventionStage.CODING)

    def test_get_checkpoint_result_no_checkpoint(self):
        """Test getting result for nonexistent checkpoint."""
        controller = HITLController()

        result = controller.get_checkpoint_result(InterventionStage.CODING)
        assert result is None

    def test_get_audit_summary(self):
        """Test getting audit summary."""
        controller = HITLController()
        controller.add_seed_codes(["code1"])
        controller.add_seed_codes(["code2"])
        controller.inject_theory("Theory X")
        controller.create_checkpoint(InterventionStage.CODING, ["data"])

        summary = controller.get_audit_summary()

        assert summary["total_interventions"] == 3
        assert "seed_codes" in summary["by_type"]
        assert summary["checkpoints_created"] == 1

    def test_callback_notification(self):
        """Test callback notification on checkpoint creation."""
        notifications = []

        def callback(checkpoint):
            notifications.append(checkpoint)

        config = HITLConfig.collaborative()
        config.callback = callback

        controller = HITLController(config=config)
        controller.create_checkpoint(InterventionStage.CODING, ["data"])

        assert len(notifications) == 1
        assert notifications[0].stage == InterventionStage.CODING


class TestCreateSeedInput:
    """Tests for create_seed_input function."""

    def test_empty_input(self):
        """Test creating empty seed input."""
        seed = create_seed_input()
        assert seed.is_empty()

    def test_with_codes(self):
        """Test creating seed input with codes."""
        seed = create_seed_input(codes=["code1", "code2"])
        assert seed.seed_codes == ["code1", "code2"]

    def test_with_all_params(self):
        """Test creating seed input with all parameters."""
        seed = create_seed_input(
            codes=["anxiety"],
            themes=["Emotion"],
            theory="Risk theory",
            questions=["How?", "Why?"],
            keywords=["climate"],
        )

        assert "anxiety" in seed.seed_codes
        assert "Emotion" in seed.seed_themes
        assert seed.theory == "Risk theory"
        assert len(seed.research_questions) == 2
        assert "climate" in seed.keywords


class TestIntegration:
    """Integration tests for HITL system."""

    def test_full_workflow_collaborative(self):
        """Test full workflow in collaborative mode."""
        # Setup
        config = HITLConfig.collaborative()
        seed = SeedInput(
            seed_codes=["climate_anxiety", "hope"],
            theory="Ecological psychology",
            research_questions=["How do people emotionally respond to climate change?"],
        )
        controller = HITLController(config=config, seed_input=seed)

        # Pre-analysis checkpoint
        pre_checkpoint = controller.create_checkpoint(
            InterventionStage.PRE_ANALYSIS,
            {"initial_setup": True},
        )
        assert not pre_checkpoint.is_approved
        controller.approve_checkpoint(
            InterventionStage.PRE_ANALYSIS, "Setup looks good"
        )

        # Coding checkpoint
        coding_result = {"codes": ["anxiety", "fear", "hope"]}
        controller.create_checkpoint(
            InterventionStage.CODING,
            coding_result,
        )
        controller.modify_checkpoint(
            InterventionStage.CODING,
            {"codes": ["anxiety", "fear", "hope", "resilience"]},
            "Added resilience code",
        )

        # Verify
        final_codes = controller.get_checkpoint_result(InterventionStage.CODING)
        assert "resilience" in final_codes["codes"]

        # Check audit trail
        summary = controller.get_audit_summary()
        assert summary["total_interventions"] >= 3
        assert summary["checkpoints_created"] >= 2

    def test_prompt_augmentation_in_agent(self):
        """Test that seed input can augment agent prompts."""
        seed = SeedInput(
            seed_codes=["trust", "communication"],
            theory="Patient-centered care framework",
            keywords=["doctor", "nurse", "hospital"],
        )
        controller = HITLController(seed_input=seed)

        augmentation = controller.get_prompt_augmentation()

        # Should create useful prompt text
        assert "trust" in augmentation
        assert "communication" in augmentation
        assert "Patient-centered care" in augmentation
        assert "doctor" in augmentation
