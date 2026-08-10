"""Registry for deployed and trusted Python functions."""

from collections.abc import Callable
from typing import Any, Protocol


class FunctionRegistry(Protocol):
    """Resolves a configured key to a deployed callable."""

    def get(self, registry_key: str) -> Callable[..., Any] | None: ...


class InMemoryFunctionRegistry:
    """Small registry suitable for application composition and tests."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable[..., Any]] = {}

    def register(self, registry_key: str, function: Callable[..., Any]) -> None:
        if not registry_key.strip():
            raise ValueError("registry_key 不能为空")
        self._functions[registry_key] = function

    def get(self, registry_key: str) -> Callable[..., Any] | None:
        return self._functions.get(registry_key)
