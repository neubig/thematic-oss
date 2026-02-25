"""Identity perspectives for coder agents.

This module implements the identity perspective system described in Section 3.2
of the Thematic-LM paper. Identity perspectives allow coders to simulate diverse
viewpoints, improving the diversity and coverage of qualitative analysis.
"""

from dataclasses import dataclass


@dataclass
class IdentityPerspective:
    """An identity perspective for a coder agent.

    Attributes:
        name: Short name for the identity (e.g., "human_driven_climate")
        description: Full description of the perspective that will be provided
            to the agent as context for their analysis approach.
    """

    name: str
    description: str

    def __str__(self) -> str:
        """Return the description for use in prompts."""
        return self.description


# Predefined identity perspectives from the paper (Section 3.2)

HUMAN_DRIVEN_CLIMATE = IdentityPerspective(
    name="human_driven_climate",
    description=(
        "You adopt the widely accepted scientific view that human activities are "
        "the primary drivers of climate change. Focus on the role of "
        "industrialization, fossil fuel emissions, deforestation, and other "
        "anthropogenic activities in accelerating global warming. Interpret data "
        "through this lens, highlighting human impacts on the environment."
    ),
)

NATURAL_CLIMATE = IdentityPerspective(
    name="natural_climate",
    description=(
        "You approach climate change from the viewpoint that it is a natural "
        "phenomenon, part of Earth's long-term climatic cycles. Reflect the "
        "arguments that climate fluctuations have occurred over millennia due to "
        "factors like solar radiation, volcanic activity, and ocean currents, "
        "suggesting that current climate shifts may not be solely due to human "
        "activities."
    ),
)

PROGRESSIVE_VIEW = IdentityPerspective(
    name="progressive_view",
    description=(
        "You hold a progressive perspective rooted in environmental justice, "
        "equity, and sustainability, advocating for systemic changes that address "
        "not only environmental issues but also social inequalities exacerbated by "
        "climate impacts. Emphasize green technologies, grassroots activism, and "
        "policies that ensure vulnerable communities are not disproportionately "
        "affected."
    ),
)

CONSERVATIVE_VIEW = IdentityPerspective(
    name="conservative_view",
    description=(
        "You reflect a conservative perspective on climate change, focusing on "
        "gradual, market-driven solutions rather than large-scale regulatory "
        "interventions. Prioritize economic stability, energy independence, and "
        "limited government involvement in climate policies. From this viewpoint, "
        "climate action should not jeopardize economic growth, jobs, or individual "
        "freedoms."
    ),
)

INDIGENOUS_VIEW = IdentityPerspective(
    name="indigenous_view",
    description=(
        "You operate from the perspective that climate change is deeply intertwined "
        "with human relationships with nature and the environment. Emphasize "
        "traditional ecological knowledge, the interconnectedness of all living "
        "beings, and the sacred responsibility to care for the land. Highlight "
        "climate change's cultural, spiritual, and community-based dimensions."
    ),
)

# Collection of all predefined identities
PREDEFINED_IDENTITIES: dict[str, IdentityPerspective] = {
    "human_driven_climate": HUMAN_DRIVEN_CLIMATE,
    "natural_climate": NATURAL_CLIMATE,
    "progressive_view": PROGRESSIVE_VIEW,
    "conservative_view": CONSERVATIVE_VIEW,
    "indigenous_view": INDIGENOUS_VIEW,
}


def get_identity(name: str) -> IdentityPerspective:
    """Get a predefined identity by name.

    Args:
        name: The name of the predefined identity.

    Returns:
        The corresponding IdentityPerspective.

    Raises:
        KeyError: If the identity name is not found.
    """
    if name not in PREDEFINED_IDENTITIES:
        available = ", ".join(PREDEFINED_IDENTITIES.keys())
        raise KeyError(f"Unknown identity '{name}'. Available: {available}")
    return PREDEFINED_IDENTITIES[name]


def list_identities() -> list[str]:
    """List all available predefined identity names.

    Returns:
        List of identity names.
    """
    return list(PREDEFINED_IDENTITIES.keys())


def create_custom_identity(name: str, description: str) -> IdentityPerspective:
    """Create a custom identity perspective.

    Args:
        name: Short name for the identity.
        description: Full description that will guide the agent's analysis.

    Returns:
        A new IdentityPerspective instance.
    """
    return IdentityPerspective(name=name, description=description)


def get_diverse_identities() -> list[IdentityPerspective]:
    """Get the five diverse identities for multi-perspective analysis.

    Returns:
        List of the five predefined identity perspectives.
    """
    return [
        HUMAN_DRIVEN_CLIMATE,
        NATURAL_CLIMATE,
        PROGRESSIVE_VIEW,
        CONSERVATIVE_VIEW,
        INDIGENOUS_VIEW,
    ]
