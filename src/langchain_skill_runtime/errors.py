"""Typed errors exposed by the runtime."""


class SkillRuntimeError(Exception):
    """Base error for all public runtime failures."""


class SkillNotFoundError(SkillRuntimeError):
    """The requested Skill does not exist."""


class SkillDisabledError(SkillRuntimeError):
    """The requested Skill is disabled."""


class SkillParseError(SkillRuntimeError):
    """The SKILL.md content is invalid."""


class SkillFileNotFoundError(SkillRuntimeError):
    """The requested SKILL.md file does not exist."""


class SkillReadError(SkillRuntimeError):
    """The requested SKILL.md file cannot be read safely."""


class SkillRuntimeConfigurationError(SkillRuntimeError):
    """The selected compile mode is missing required runtime dependencies."""


class ToolDefinitionError(SkillRuntimeError):
    """A Tool definition or input schema is invalid."""


class ToolAdapterNotFoundError(SkillRuntimeError):
    """No adapter is registered for a Tool type."""


class DuplicateToolAdapterError(SkillRuntimeError):
    """More than one adapter was registered for a Tool type."""


class ToolUnavailableError(SkillRuntimeError):
    """A Tool cannot be used in the current context."""


class ToolBuildError(SkillRuntimeError):
    """A Tool could not be converted to a LangChain BaseTool."""


class ToolExecutionError(SkillRuntimeError):
    """A Tool invocation failed without exposing implementation details."""


class ToolExecutionTimeoutError(SkillRuntimeError):
    """A Tool invocation exceeded its configured timeout."""


class ToolOutputValidationError(SkillRuntimeError):
    """A Tool result did not match its output schema."""


class ToolOutputTooLargeError(SkillRuntimeError):
    """A Tool result exceeded its configured output limit."""


class FunctionNotRegisteredError(ToolBuildError):
    """A configured function registry key is unknown."""


class SkillCompileError(SkillRuntimeError):
    """The Skill cannot compile because a required Tool failed."""


class ClientToolError(SkillRuntimeError):
    """Base error for client-executed Tools."""


class ClientToolTimeoutError(ClientToolError):
    """A client Tool did not finish before its deadline."""


class ClientToolConnectionLostError(ClientToolError):
    """The client connection was lost during execution."""


class ClientToolResultMismatchError(ClientToolError):
    """A client result does not match its pending invocation."""


class ClientToolExecutionError(ClientToolError):
    """The client reported a controlled execution failure."""
