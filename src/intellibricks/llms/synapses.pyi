"""
Synapse:
> The junction between two neurons that allows a signal to pass between them.

Welcome to the synapses
"""

from __future__ import annotations

import logging
from typing import (
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    overload,
    runtime_checkable,
)

import msgspec
from architecture import log
from architecture.extensions import Maybe
from langfuse import Langfuse

from intellibricks.llms.base import FileContent
from intellibricks.llms.base import Language as TranscriptionsLanguage
from intellibricks.llms.general_web_search import WebSearchable

from .constants import (
    Language,
)
from .types import (
    CacheConfig,
    ChatCompletion,
    Message,
    PartType,
    Prompt,
    RawResponse,
    TextTranscriptionOutput,
    ToolInputType,
    TraceParams,
)
from .types import AIModel, TranscriptionModelType

debug_logger = log.create_logger(__name__, level=logging.DEBUG)
error_logger = log.create_logger(__name__, level=logging.ERROR)

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)

@runtime_checkable
class SynapseProtocol(Protocol):
    @overload
    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...
    @overload
    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...

class Synapse(msgspec.Struct, frozen=True, omit_defaults=True):
    model: AIModel = msgspec.field(
        default_factory=lambda: "google/genai/gemini-2.0-flash-exp"
    )
    api_key: Optional[str] = None
    cloud_project: Optional[str] = None
    cloud_location: Optional[str] = None
    langfuse: Maybe[Langfuse] = Maybe(None)
    web_searcher: Optional[WebSearchable] = None

    @classmethod
    def of(
        cls,
        model: AIModel,
        *,
        api_key: Optional[str] = None,
        langfuse: Optional[Langfuse] = None,
        web_searcher: Optional[WebSearchable] = None,
        cloud_project: Optional[str] = None,
        cloud_location: Optional[str] = None,
    ) -> Synapse: ...
    @overload
    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...
    @overload
    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...

class SynapseCascade:
    synapses: Sequence[Synapse | SynapseCascade]
    """A sequence of Synapse or SynapseCascade objects"""

    shuffle: bool = False
    """Indicates whether the synapses should be shuffled before trying them"""

    @classmethod
    def of(cls, *synapses: Synapse | SynapseCascade, shuffle: bool = False) -> SynapseCascade: ...
    @overload
    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...
    @overload
    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
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
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...
    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...

class TextTranscriptionSynapse(msgspec.Struct, frozen=True):
    """A synapse for audio transcriptions"""

    model: TranscriptionModelType
    api_key: Optional[str] = None
    langfuse: Optional[Langfuse] = None

    @classmethod
    def of(
        cls,
        model: TranscriptionModelType,
        api_key: Optional[str] = None,
        langfuse: Optional[Langfuse] = None,
    ) -> TextTranscriptionSynapse: ...
    def transcribe(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> TextTranscriptionOutput: ...
    async def transcribe_async(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> TextTranscriptionOutput: ...
