"""Base agent class for all thematic analysis agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from openhands.sdk import LLM, Message, TextContent


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    model: str = "anthropic/claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    identity: str | None = None  # Optional identity/persona for the agent


class BaseAgent(ABC):
    """Base class for all thematic analysis agents.

    Provides common functionality for LLM-based agents including
    client management and message handling.
    """

    def __init__(self, config: AgentConfig | None = None):
        """Initialize the agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or AgentConfig()
        self._llm: LLM | None = None

    @property
    def llm(self) -> LLM:
        """Lazy load the LLM instance."""
        if self._llm is None:
            self._llm = LLM(
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
        return self._llm

    def _create_messages(self, system_prompt: str, user_prompt: str) -> list[Message]:
        """Create message list for LLM call.

        Args:
            system_prompt: The system prompt with instructions.
            user_prompt: The user prompt with the task.

        Returns:
            List of Message objects.
        """
        return [
            Message(role="system", content=[TextContent(text=system_prompt)]),
            Message(role="user", content=[TextContent(text=user_prompt)]),
        ]

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with the given prompts.

        Args:
            system_prompt: The system prompt with instructions.
            user_prompt: The user prompt with the task.

        Returns:
            The LLM response text.
        """
        messages = self._create_messages(system_prompt, user_prompt)
        response = self.llm.completion(messages=messages)

        # Extract text from response message content
        content_parts = []
        for part in response.message.content:
            if isinstance(part, TextContent):
                content_parts.append(part.text)
        return "".join(content_parts)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Returns:
            The system prompt string.
        """
        pass
