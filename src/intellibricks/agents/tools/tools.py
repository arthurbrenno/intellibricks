from typing import Literal, Protocol, Any, Callable, runtime_checkable
from abc import abstractmethod
from dataclasses import dataclass, field


@runtime_checkable
class SupportsCallableConversion(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def to_callable(self) -> Callable[..., Any]: ...


@dataclass(frozen=True, kw_only=True)
class DuckDuckGoTool(SupportsCallableConversion):
    query: str
    n: int = 1
    safe_search: bool = False
    region: Literal[
        "us-en",
        "uk-en",
        "de-de",
        "es-es",
        "fr-fr",
        "it-it",
        "nl-nl",
        "pl-pl",
        "pt-br",
        "tr-tr",
    ] = field(default_factory=lambda: "us-en")
