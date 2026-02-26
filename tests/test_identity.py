"""Tests for identity perspectives module."""

import pytest

from thematic_analysis.identity import (
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


class TestIdentityPerspective:
    """Test cases for IdentityPerspective dataclass."""

    def test_creation(self):
        """Test basic creation of identity perspective."""
        identity = IdentityPerspective(
            name="test_identity",
            description="A test description for coding perspective.",
        )
        assert identity.name == "test_identity"
        assert "test description" in identity.description

    def test_str_returns_description(self):
        """Test that str() returns the description for prompt usage."""
        identity = IdentityPerspective(
            name="test",
            description="The perspective description.",
        )
        assert str(identity) == "The perspective description."


class TestPredefinedIdentities:
    """Test cases for predefined identity perspectives."""

    def test_human_driven_climate_exists(self):
        """Test HUMAN_DRIVEN_CLIMATE identity."""
        assert HUMAN_DRIVEN_CLIMATE.name == "human_driven_climate"
        assert "human activities" in HUMAN_DRIVEN_CLIMATE.description.lower()
        assert "climate" in HUMAN_DRIVEN_CLIMATE.description.lower()

    def test_natural_climate_exists(self):
        """Test NATURAL_CLIMATE identity."""
        assert NATURAL_CLIMATE.name == "natural_climate"
        assert "natural" in NATURAL_CLIMATE.description.lower()
        assert "cycles" in NATURAL_CLIMATE.description.lower()

    def test_progressive_view_exists(self):
        """Test PROGRESSIVE_VIEW identity."""
        assert PROGRESSIVE_VIEW.name == "progressive_view"
        assert "environmental justice" in PROGRESSIVE_VIEW.description.lower()
        assert "equity" in PROGRESSIVE_VIEW.description.lower()

    def test_conservative_view_exists(self):
        """Test CONSERVATIVE_VIEW identity."""
        assert CONSERVATIVE_VIEW.name == "conservative_view"
        assert "market-driven" in CONSERVATIVE_VIEW.description.lower()
        assert "economic" in CONSERVATIVE_VIEW.description.lower()

    def test_indigenous_view_exists(self):
        """Test INDIGENOUS_VIEW identity."""
        assert INDIGENOUS_VIEW.name == "indigenous_view"
        assert "traditional" in INDIGENOUS_VIEW.description.lower()
        assert "interconnected" in INDIGENOUS_VIEW.description.lower()

    def test_predefined_identities_dict(self):
        """Test PREDEFINED_IDENTITIES contains all five."""
        assert len(PREDEFINED_IDENTITIES) == 5
        assert "human_driven_climate" in PREDEFINED_IDENTITIES
        assert "natural_climate" in PREDEFINED_IDENTITIES
        assert "progressive_view" in PREDEFINED_IDENTITIES
        assert "conservative_view" in PREDEFINED_IDENTITIES
        assert "indigenous_view" in PREDEFINED_IDENTITIES


class TestGetIdentity:
    """Test cases for get_identity function."""

    def test_get_existing_identity(self):
        """Test retrieving an existing identity."""
        identity = get_identity("human_driven_climate")
        assert identity == HUMAN_DRIVEN_CLIMATE

    def test_get_all_predefined_identities(self):
        """Test that all predefined identities can be retrieved."""
        for name in list_identities():
            identity = get_identity(name)
            assert identity.name == name

    def test_get_unknown_identity_raises(self):
        """Test that unknown identity raises KeyError."""
        with pytest.raises(KeyError) as excinfo:
            get_identity("nonexistent_identity")
        assert "nonexistent_identity" in str(excinfo.value)
        assert "Available" in str(excinfo.value)


class TestListIdentities:
    """Test cases for list_identities function."""

    def test_returns_list(self):
        """Test that list_identities returns a list."""
        identities = list_identities()
        assert isinstance(identities, list)
        assert len(identities) == 5

    def test_contains_all_predefined(self):
        """Test that all predefined identities are listed."""
        identities = list_identities()
        assert "human_driven_climate" in identities
        assert "indigenous_view" in identities


class TestCreateCustomIdentity:
    """Test cases for create_custom_identity function."""

    def test_create_custom(self):
        """Test creating a custom identity."""
        identity = create_custom_identity(
            name="environmental_scientist",
            description="You are an environmental scientist with data-driven focus.",
        )
        assert identity.name == "environmental_scientist"
        assert "data-driven" in identity.description

    def test_custom_identity_usable_as_string(self):
        """Test custom identity can be used as string."""
        identity = create_custom_identity(
            name="custom",
            description="Custom perspective for analysis.",
        )
        assert str(identity) == "Custom perspective for analysis."


class TestGetDiverseIdentities:
    """Test cases for get_diverse_identities function."""

    def test_returns_five_identities(self):
        """Test that get_diverse_identities returns all five."""
        identities = get_diverse_identities()
        assert len(identities) == 5

    def test_returns_list_of_identity_perspectives(self):
        """Test that all returned items are IdentityPerspective."""
        identities = get_diverse_identities()
        for identity in identities:
            assert isinstance(identity, IdentityPerspective)

    def test_contains_all_predefined(self):
        """Test that all predefined identities are included."""
        identities = get_diverse_identities()
        names = [i.name for i in identities]
        assert "human_driven_climate" in names
        assert "natural_climate" in names
        assert "progressive_view" in names
        assert "conservative_view" in names
        assert "indigenous_view" in names


class TestIdentityIntegration:
    """Integration tests for identity perspectives with agents."""

    def test_identity_can_be_used_as_coder_identity(self):
        """Test that identity description can be used with CoderConfig."""
        from thematic_analysis.agents import CoderConfig

        identity = get_identity("progressive_view")
        config = CoderConfig(identity=str(identity))

        assert config.identity is not None
        assert "environmental justice" in config.identity

    def test_diverse_identities_create_unique_configs(self):
        """Test that diverse identities create distinct configurations."""
        from thematic_analysis.agents import CoderConfig

        configs = []
        for identity in get_diverse_identities():
            config = CoderConfig(identity=str(identity))
            configs.append(config)

        # Each config should have a unique identity
        identities_set = {c.identity for c in configs}
        assert len(identities_set) == 5

    def test_identity_integration_with_theme_coder(self):
        """Test that identity works with ThemeCoderConfig."""
        from thematic_analysis.agents import ThemeCoderConfig
        from thematic_analysis.codebook import Codebook

        identity = get_identity("indigenous_view")
        config = ThemeCoderConfig(identity=str(identity))
        codebook = Codebook(use_mock_embeddings=True)

        from thematic_analysis.agents import ThemeCoderAgent

        agent = ThemeCoderAgent(config=config, codebook=codebook)

        # The identity should be in the system prompt
        prompt = agent.get_system_prompt()
        assert "traditional" in prompt.lower() or "indigenous" in prompt.lower()
