from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    overload,
    runtime_checkable,
)

import msgspec

from intellibricks.llms.schema import PartType, Prompt, RawResponse

if TYPE_CHECKING:
    from intellibricks.llms.schema import ChatCompletion, Message

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)

@runtime_checkable
class SupportsAsyncChat(Protocol):
    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...
    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...
    @overload
    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...
