"""LLM engines module"""

# TODO: Create stubs file for engines
from __future__ import annotations

import asyncio
import timeit
import uuid
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import aiocache
import msgspec
from architecture import BaseModel
from architecture.extensions import Maybe
from architecture.logging import LoggerFactory
from architecture.utils.creators import DynamicInstanceCreator
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from google.oauth2 import service_account
from langfuse import Langfuse
from langfuse.client import (
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)
from langfuse.model import ModelUsage
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import LLM

from intellibricks import util
from intellibricks.llms.web_search import WebSearchable
from intellibricks.rag.contracts import RAGQueriable

from .config import CacheConfig
from .constants import (
    AIModel,
    FinishReason,
    Language,
    MessageRole,
)
from .exceptions import MaxRetriesReachedException
from .schema import (
    CompletionMessage,
    CompletionOutput,
    CompletionTokensDetails,
    Message,
    MessageChoice,
    Prompt,
    PromptTokensDetails,
    Tag,
    Usage,
)
from .types import TraceParams
from .util import count_tokens

logger = LoggerFactory.create(__name__)

T = TypeVar("T", bound=msgspec.Struct)
U = TypeVar("U", bound=msgspec.Struct | None)


@runtime_checkable
class CompletionEngineProtocol(Protocol):
    @overload
    def complete(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    def complete(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    def complete(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...

    @overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    def chat(
        self,
        *,
        messages: list[Message],
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...

    @overload
    async def complete_async(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    async def complete_async(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    async def complete_async(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...

    @overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...


class CompletionEngine(CompletionEngineProtocol):
    langfuse: Maybe[Langfuse]
    vertex_credentials: Maybe[service_account.Credentials]
    web_searcher: Optional[WebSearchable] = None
    json_encoder: msgspec.json.Encoder
    json_decoder: msgspec.json.Decoder

    def __init__(
        self,
        *,
        langfuse: Optional[Langfuse] = None,
        json_encoder: Optional[msgspec.json.Encoder] = None,
        json_decoder: Optional[msgspec.json.Decoder] = None,
        vertex_credentials: Optional[service_account.Credentials] = None,
        web_searcher: Optional[
            WebSearchable
        ] = None,  # TODO: not working yet. Tryng to manage my time for that (work + university)
    ) -> None:
        self.langfuse = Maybe(langfuse or None)
        self.json_encoder = json_encoder or msgspec.json.Encoder()
        self.json_decoder = json_decoder or msgspec.json.Decoder()
        self.vertex_credentials = Maybe(vertex_credentials or None)
        self.web_searcher = web_searcher

    @overload
    def complete(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    def complete(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    def complete(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]:
        system_prompt = (
            system_prompt
            or "You are a helpful assistant. Answer in the same language the user asked."
        )
        prompt = prompt.content if isinstance(prompt, Prompt) else prompt
        system_prompt = (
            system_prompt.content
            if isinstance(system_prompt, Prompt)
            else system_prompt
        )

        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=prompt),
        ]

        return self.chat(
            messages=messages,
            response_format=response_format,
            model=model,
            fallback_models=fallback_models,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache_config=cache_config,
            trace_params=trace_params,
            postergate_token_counting=postergate_token_counting,
            tools=tools,
            data_stores=data_stores,
            web_search=web_search,
        )

    @overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    def chat(
        self,
        *,
        messages: list[Message],
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No event loop running
            return cast(
                CompletionOutput[T] | CompletionOutput[None],
                asyncio.run(
                    self._achat(
                        messages=messages,
                        response_format=response_format,
                        model=model,
                        fallback_models=fallback_models,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        cache_config=cache_config,
                        trace_params=trace_params,
                        postergate_token_counting=postergate_token_counting,
                        tools=tools,
                        data_stores=data_stores,
                        web_search=web_search,
                    )
                ),
            )
        else:
            return cast(
                CompletionOutput[T] | CompletionOutput[None],
                loop.run_until_complete(
                    self._achat(
                        messages=messages,
                        response_format=response_format,
                        model=model,
                        fallback_models=fallback_models,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        cache_config=cache_config,
                        trace_params=trace_params,
                        postergate_token_counting=postergate_token_counting,
                        tools=tools,
                        data_stores=data_stores,
                        web_search=web_search,
                    )
                ),
            )

    @overload
    async def complete_async(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    async def complete_async(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    async def complete_async(
        self,
        prompt: Union[str, Prompt],
        *,
        system_prompt: Optional[Union[str, Prompt]] = None,
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]:
        system_prompt = (
            system_prompt
            or "You are a helpful assistant. Answer in the same language the user asked."
        )
        prompt = prompt.content if isinstance(prompt, Prompt) else prompt
        system_prompt = (
            system_prompt.content
            if isinstance(system_prompt, Prompt)
            else system_prompt
        )

        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=prompt),
        ]

        return await self.chat_async(
            messages=messages,
            response_format=response_format,
            model=model,
            fallback_models=fallback_models,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache_config=cache_config,
            trace_params=trace_params,
            postergate_token_counting=postergate_token_counting,
            tools=tools,
            data_stores=data_stores,
            web_search=web_search,
        )

    @overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]:
        return await self._achat(
            messages=messages,
            response_format=response_format,
            model=model,
            fallback_models=fallback_models,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache_config=cache_config,
            trace_params=trace_params,
            postergate_token_counting=postergate_token_counting,
            tools=tools,
            data_stores=data_stores,
            web_search=web_search,
        )

    @overload
    async def _achat(
        self,
        *,
        messages: list[Message],
        response_format: Type[T],
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T]: ...

    @overload
    async def _achat(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[None]: ...

    async def _achat(
        self,
        *,
        messages: list[Message],
        response_format: Optional[Type[T]] = None,
        model: AIModel = AIModel.STUDIO_GEMINI_1P5_FLASH,
        fallback_models: Optional[list[AIModel]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: Optional[list[Callable[..., Any]]] = None,
        data_stores: Optional[Sequence[RAGQueriable]] = None,
        web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
    ) -> CompletionOutput[T] | CompletionOutput[None]:
        start_time = timeit.default_timer()
        trace_params = trace_params or {}
        cache_config = cache_config or CacheConfig()

        trace_params["input"] = messages

        completion_id: uuid.UUID = uuid.uuid4()

        trace: Maybe[StatefulTraceClient] = self.langfuse.map(
            lambda langfuse: langfuse.trace(id=completion_id.__str__(), **trace_params)
        )

        choices: list[MessageChoice[T]] = []

        fallback_models = fallback_models or []
        n = n or 1
        temperature = temperature or 0.7
        max_tokens = max_tokens or 5000
        max_retries = max_retries or 1

        models: list[AIModel] = [model] + fallback_models

        logger.info(
            f"Starting chat completion. Main model: {model}, Fallback models: {fallback_models}"
        )

        maybe_span: Maybe[StatefulSpanClient] = Maybe(None)
        for model in models:
            for retry in range(max_retries):
                try:
                    span_id: str = f"sp-{completion_id}-{retry}"
                    maybe_span = Maybe(
                        trace.map(
                            lambda trace: trace.span(
                                id=span_id,
                                input=messages,
                                name="Response Generation",
                            )
                        ).unwrap()
                    )

                    choices, usage = await self._aget_choices(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        cache_config=cache_config,
                        trace=trace,
                        span=maybe_span,
                        postergate_token_counting=postergate_token_counting,
                        language=language,
                    )

                    logger.info(
                        f"Successfully generated completion with model {model} in retry {retry}"
                    )

                    output: CompletionOutput[T] | CompletionOutput[None] = (
                        CompletionOutput(
                            elapsed_time=timeit.default_timer() - start_time,
                            id=completion_id,
                            model=model,
                            choices=choices,
                            usage=usage,
                        )
                    )

                    maybe_span.end(output=output.choices)

                    maybe_span.score(
                        id=f"sc-{maybe_span.map(lambda span: span.id).unwrap()}",
                        name="Sucesso",
                        value=1.0,
                        comment="Choices generated successfully!",
                    )

                    trace.update(output=output.choices)
                    return output
                except Exception as e:
                    # Log the error in span and continue to the next one
                    maybe_span.end(output={})
                    maybe_span.update(status_message="Erro na geração.", level="ERROR")
                    maybe_span.score(
                        id=f"sc-{maybe_span.unwrap()}",
                        name="Sucesso",
                        value=0.0,
                        comment=f"Error while generating choices: {e}",
                    )
                    logger.error(
                        f"An error ocurred in retry {retry}",
                    )
                    logger.exception(e)
                    continue

        raise MaxRetriesReachedException()

    async def _aget_choices(
        self,
        *,
        model: AIModel,
        messages: list[Message],
        n: int,
        temperature: float,
        max_tokens: int,
        trace: Maybe[StatefulTraceClient],
        span: Maybe[StatefulSpanClient],
        cache_config: CacheConfig,
        postergate_token_counting: bool,
        response_format: Optional[Type[T]],
        language: Language,
    ) -> Tuple[list[MessageChoice[T]], Usage]:
        choices: list[MessageChoice[T]] = []
        model_input_cost, model_output_cost = model.ppm()
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0
        total_input_cost: float = 0.0
        total_output_cost: float = 0.0

        llm: LLM = await self._get_cached_llm(
            model=model,
            max_tokens=max_tokens,
            cache_config=cache_config,
        )

        for i in range(n):
            current_messages = messages.copy()

            if response_format is not None:
                current_messages = self._append_response_format_to_prompt(
                    messages=current_messages,
                    response_format=response_format,
                    language=language,
                )

            generation: Maybe[StatefulGenerationClient] = span.map(
                lambda span: span.generation(
                    id=f"gen-{uuid.uuid4()}-{i}",
                    model=model.value,
                    input=current_messages,
                    model_parameters={
                        "max_tokens": max_tokens,
                        "temperature": str(temperature),
                    },
                )
            )

            chat_response: ChatResponse = await llm.achat(
                messages=[
                    message.to_llama_index_chat_message()
                    for message in current_messages
                ]
            )

            logger.debug(
                f"Received AI response from model {model.value}: {chat_response.message.content}"
            )

            generation.end(
                output=chat_response.message.content,
            )

            usage_future = self._calculate_token_usage(
                model=model,
                messages=current_messages,
                chat_response=chat_response,
                generation=generation,
                span=span,
                index=i,
                model_input_cost=model_input_cost,
                model_output_cost=model_output_cost,
            )

            if not postergate_token_counting:
                usage = await usage_future
                total_prompt_tokens += usage.prompt_tokens or 0
                total_completion_tokens += usage.completion_tokens or 0
                total_input_cost += usage.input_cost or 0.0
                total_output_cost += usage.output_cost or 0.0
            else:
                asyncio.create_task(usage_future)

            completion_message = CompletionMessage(
                role=MessageRole(chat_response.message.role.value),
                content=chat_response.message.content,
                parsed=self._get_parsed(
                    response_format,
                    chat_response.message.content,
                    trace=trace,
                    span=span,
                ),
            )

            choices.append(
                MessageChoice(
                    index=i,
                    message=completion_message,
                    logprobs=chat_response.logprobs,
                    finish_reason=FinishReason.NONE,
                )
            )
            logger.info(f"Successfully generated choice {i+1} for model {model.value}")

        usage = self._create_usage(
            postergate_token_counting,
            total_prompt_tokens,
            total_completion_tokens,
            total_input_cost,
            total_output_cost,
        )

        return choices, usage

    async def _calculate_token_usage(
        self,
        *,
        model: AIModel,
        messages: list[Message],
        chat_response: ChatResponse,
        generation: Maybe[StatefulGenerationClient],
        span: Maybe[StatefulSpanClient],
        index: int,
        model_input_cost: float,
        model_output_cost: float,
    ) -> Usage:
        prompt_counting_span: Maybe[StatefulSpanClient] = span.map(
            lambda span: span.span(
                id=f"sp-prompt-{span.id}-{index}",
                name="Contagem de Tokens",
                input={
                    "mensagens": [
                        message.as_dict(encoder=self.json_encoder)
                        for message in messages
                    ]
                    + [chat_response.model_dump()]
                },
            )
        )

        prompt_tokens = sum(
            count_tokens(model=model, text=msg.content or "") for msg in messages
        )

        completion_tokens = count_tokens(
            model=model, text=chat_response.message.content or ""
        )

        prompt_counting_span.end(
            output={
                "model": model.value,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

        prompt_cost_span: Maybe[StatefulSpanClient] = span.map(
            lambda span: span.span(
                id=f"sp-sum-prompt-{span.id}-{index}",
                name="Determinando preço dos tokens",
                input={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model_input_cost": model_input_cost,
                    "model_output_cost": model_output_cost,
                },
            )
        )

        scale = 1 / 1_000_000

        completion_input_cost = round(prompt_tokens * model_input_cost * scale, 5)
        completion_output_cost = round(completion_tokens * model_output_cost * scale, 5)

        prompt_cost_span.end(
            output={
                "prompt_cost": completion_input_cost,
                "completion_cost": completion_output_cost,
            }
        )

        generation.update(
            usage=ModelUsage(
                unit="TOKENS",
                input=prompt_tokens,
                output=completion_tokens,
                total=prompt_tokens + completion_tokens,
                input_cost=completion_input_cost,
                output_cost=completion_output_cost,
                total_cost=completion_input_cost + completion_output_cost,
            )
        )

        return self._create_usage(
            False,
            prompt_tokens,
            completion_tokens,
            completion_input_cost,
            completion_output_cost,
        )

    def _create_usage(
        self,
        postergate_token_counting: bool,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        input_cost: Optional[float],
        output_cost: Optional[float],
    ) -> Usage:
        if postergate_token_counting:
            return Usage(
                prompt_tokens=None,
                completion_tokens=None,
                input_cost=None,
                output_cost=None,
                total_cost=None,
                total_tokens=None,
                prompt_tokens_details=PromptTokensDetails(
                    audio_tokens=None, cached_tokens=None
                ),
                completion_tokens_details=CompletionTokensDetails(
                    audio_tokens=None, reasoning_tokens=None
                ),
            )
        else:
            total_cost = (input_cost or 0.0) + (output_cost or 0.0)
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            return Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                total_tokens=total_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    audio_tokens=None, cached_tokens=None
                ),
                completion_tokens_details=CompletionTokensDetails(
                    audio_tokens=None, reasoning_tokens=None
                ),
            )

    # @overload
    # def _get_parsed(
    #     self,
    #     response_format: Type[T],
    #     content: Optional[str],
    #     trace: Maybe[StatefulTraceClient],
    #     span: Maybe[StatefulSpanClient],
    # ) -> T: ...

    # @overload
    # def _get_parsed(
    #     self,
    #     response_format: None,
    #     content: Optional[str],
    #     trace: Maybe[StatefulTraceClient],
    #     span: Maybe[StatefulSpanClient],
    # ) -> None: ...

    def _get_parsed(
        self,
        response_format: Optional[Type[T]],
        content: Optional[str],
        trace: Maybe[StatefulTraceClient],
        span: Maybe[StatefulSpanClient],
    ) -> T:
        if response_format is None:
            logger.warning("Response format is None")
            return cast(T, None)

        if content is None:
            logger.warning("Contents of the message are none")
            return cast(T, None)

        if isinstance(response_format, dict):
            LLMResponse: Type[msgspec.Struct] = util.get_struct_from_schema(
                response_format, bases=(BaseModel,), name="ResponseModel"
            )

            response_format = LLMResponse

        tag: Optional[Tag] = Tag.from_string(
            content, tag_name="structured"
        ) or Tag.from_string(content, tag_name="output")

        if tag is None:
            span.map(
                lambda span: span.event(
                    id=f"ev-{trace.id}",
                    name="Getting Structured Response",
                    input=content,
                    output=None,
                    level="ERROR",
                    metadata={"response_format": response_format, "content": content},
                )
            )
            return cast(T, None)

        structured: dict[str, Any] = tag.as_object()

        if not structured:
            raise ValueError("Tag object could not be parsed as structured content")

        model: T = msgspec.json.decode(
            msgspec.json.encode(structured), type=response_format
        )

        span.map(
            lambda span: span.event(
                id=f"ev-{trace.id}",
                name="Getting Structured Response",
                input=f"<structured>\n{tag.content}\n</structured>",
                output=model,
                level="DEBUG",
                metadata={"response_format": response_format, "content": content},
            )
        )

        return model

    def _get_structured_prompt_instructions_by_language(
        self, language: Language, schema: dict[str, Any]
    ) -> str:
        match language:
            case Language.ENGLISH:
                return f"""
                    <output>
                        Inside a "<structured>" tag, the assistant will return an output formatted in JSON, which complies with the following JSON schema:
                        <json_schema>
                        {schema}
                        </json_schema>
                        The JSON returned by the assistant, within the tag, must adhere to the schema mentioned above and take into account the instructions provided in the given task. The assistant must close the tag with </structured>.
                    </output>
                """
            case Language.SPANISH:
                return f"""
                    <output>
                        Dentro de una etiqueta "<structured>", el asistente devolverá una salida formateada en JSON, que cumple con el siguiente esquema JSON:
                        <json_schema>
                        {schema}
                        </json_schema>
                        El JSON devuelto por el asistente, dentro de la etiqueta, debe adherirse al esquema mencionado anteriormente y tener en cuenta las instrucciones proporcionadas en la tarea dada. El asistente debe cerrar la etiqueta con </structured>.
                    </output>
                """
            case Language.FRENCH:
                return f"""
                    <output>
                        À l'intérieur d'une balise "<structured>", l'assistant renverra une sortie formatée en JSON, qui est conforme au schéma JSON suivant :
                        <json_schema>
                        {schema}
                        </json_schema>
                        Le JSON renvoyé par l'assistant, à l'intérieur de la balise, doit respecter le schéma mentionné ci-dessus et prendre en compte les instructions fournies dans la tâche donnée. L'assistant doit fermer la balise avec </structured>.
                    </output>
                """
            case Language.GERMAN:
                return f"""
                    <output>
                        Innerhalb eines "<structured>"-Tags gibt der Assistent eine Ausgabe im JSON-Format zurück, die dem folgenden JSON-Schema entspricht:
                        <json_schema>
                        {schema}
                        </json_schema>
                        Das vom Assistenten zurückgegebene JSON innerhalb des Tags muss dem oben genannten Schema entsprechen und die in der gegebenen Aufgabe angegebenen Anweisungen berücksichtigen. Der Assistent muss das Tag mit </structured> schließen.
                    </output>
                """
            case Language.CHINESE:
                return f"""
                    <output>
                        在 "<structured>" 标签内，助手将以 JSON 格式返回输出，符合以下 JSON 模式：
                        <json_schema>
                        {schema}
                        </json_schema>
                        助手在标签内返回的 JSON 必须遵守上述模式，并考虑到给定任务中提供的说明。助手必须用 </structured> 关闭标签。
                    </output>
                """
            case Language.JAPANESE:
                return f"""
                    <output>
                        "<structured>" タグ内で、アシスタントは次の JSON スキーマに準拠した JSON 形式の出力を返します:
                        <json_schema>
                        {schema}
                        </json_schema>
                        タグ内でアシスタントが返す JSON は、上記のスキーマに従い、指定されたタスクで提供された指示を考慮に入れる必要があります。アシスタントはタグを </structured> で閉じる必要があります。
                    </output>
                """
            case Language.PORTUGUESE:
                return f"""
                    <output>
                        Dentro de uma tag "<structured>", o assistente retornará uma saída formatada em JSON, que está em conformidade com o seguinte esquema JSON:
                        <json_schema>
                        {schema}
                        </json_schema>
                        O JSON retornado pelo assistente, dentro da tag, deve aderir ao esquema mencionado acima e levar em conta as instruções fornecidas na tarefa dada. O assistente deve fechar a tag com </structured>.
                    </output>
                """

    def _append_response_format_to_prompt(
        self,
        *,
        messages: list[Message],
        response_format: Type[T],
        language: Language,
        prompt_role: Optional[MessageRole] = None,
    ) -> list[Message]:
        if prompt_role is None:
            prompt_role = MessageRole.SYSTEM

        basemodel_schema = msgspec.json.schema(response_format)

        new_prompt = self._get_structured_prompt_instructions_by_language(
            language=language, schema=basemodel_schema
        )

        for message in messages:
            if message.content is None:
                message.content = new_prompt
                continue

            if message.role == prompt_role:
                message.content += new_prompt
                return messages

        messages.append(Message(role=prompt_role, content=new_prompt))

        return messages

    @aiocache.cached(ttl=3600)
    async def _get_cached_llm(
        self,
        model: AIModel,
        max_tokens: int,
        cache_config: CacheConfig,
    ) -> LLM:
        constructor_params = {
            "max_tokens": max_tokens,
            "model_name": model.value,
            "project": self.vertex_credentials.map(
                lambda credentials: credentials.project_id
            ).unwrap(),
            "model": model.value,
            "credentials": self.vertex_credentials.unwrap(),
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            "cache_config": cache_config,
            "timeout": 120,
            "generate_kwargs": {"timeout": 120},
        }

        return DynamicInstanceCreator(
            AIModel.get_llama_index_model_cls(model)
        ).create_instance(**constructor_params)
