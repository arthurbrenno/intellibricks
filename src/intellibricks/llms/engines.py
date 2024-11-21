"""LLM engines module"""

# TODO: Create stubs file for engines
from __future__ import annotations

import asyncio
import typing
import uuid

import aiocache
import msgspec
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
from architecture import BaseModel, DynamicDict
from architecture.extensions import Maybe
from architecture.logging import LoggerFactory
from architecture.utils.creators import DynamicInstanceCreator

from intellibricks import util
from intellibricks.rag.contracts import RAGQueriable

from .config import CacheConfig
from .constants import (
    AIModel,
    FinishReason,
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

T = typing.TypeVar("T", bound=msgspec.Struct)
U = typing.TypeVar("U", bound=msgspec.Struct | None)


@typing.runtime_checkable
class CompletionEngineProtocol(typing.Protocol):
    @typing.overload
    def complete(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    def complete(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    def complete(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...

    @typing.overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    def chat(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...

    @typing.overload
    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...

    @typing.overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T] | CompletionOutput[None]: ...


class CompletionEngine(CompletionEngineProtocol):
    langfuse: Maybe[Langfuse]
    vertex_credentials: Maybe[service_account.Credentials]
    json_encoder: msgspec.json.Encoder
    json_decoder: msgspec.json.Decoder

    def __init__(
        self,
        *,
        langfuse: typing.Optional[Langfuse] = None,
        json_encoder: typing.Optional[msgspec.json.Encoder] = None,
        json_decoder: typing.Optional[msgspec.json.Decoder] = None,
        vertex_credentials: typing.Optional[service_account.Credentials] = None,
    ) -> None:
        self.langfuse = Maybe(langfuse or None)
        self.json_encoder = json_encoder or msgspec.json.Encoder()
        self.json_decoder = json_decoder or msgspec.json.Decoder()
        self.vertex_credentials = Maybe(vertex_credentials or None)

    @typing.overload
    def complete(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    def complete(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    def complete(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
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

    @typing.overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    def chat(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    def chat(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T] | CompletionOutput[None]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No event loop running
            return typing.cast(
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
            return typing.cast(
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

    @typing.overload
    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
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

    @typing.overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
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

    @typing.overload
    async def _achat(
        self,
        *,
        messages: list[Message],
        response_format: typing.Type[T],
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T]: ...

    @typing.overload
    async def _achat(
        self,
        *,
        messages: list[Message],
        response_format: None = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[None]: ...

    async def _achat(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[typing.Type[T]] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable[..., typing.Any]]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[T] | CompletionOutput[None]:
        trace_params = trace_params or {}
        cache_config = cache_config or CacheConfig()

        trace_params["input"] = messages

        completion_id: uuid.UUID = uuid.uuid4()

        trace: Maybe[StatefulTraceClient] = self.langfuse.map(
            lambda langfuse: langfuse.trace(id=completion_id.__str__(), **trace_params)
        )

        choices: list[MessageChoice[T]] = []

        model = model or AIModel.STUDIO_GEMINI_1P5_FLASH
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
                                name="Geração de Resposta",
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
                    )

                    logger.info(
                        f"Successfully generated completion with model {model} in retry {retry}"
                    )

                    output: CompletionOutput[T] | CompletionOutput[None] = (
                        CompletionOutput(
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
                        comment="Escolhas geradas com sucesso!",
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
                        comment=f"Erro ao gerar escolhas: {e}",
                    )
                    logger.error(
                        f"An error ocurred in retry {retry}",
                    )
                    logger.exception(e)
                    continue

        raise MaxRetriesReachedException()

    # @typing.overload
    # async def _aget_choices(
    #     self,
    #     *,
    #     model: AIModel,
    #     messages: list[Message],
    #     n: int,
    #     temperature: float,
    #     stream: bool,
    #     max_tokens: int,
    #     trace: Maybe[StatefulTraceClient],
    #     span: Maybe[StatefulSpanClient],
    #     cache_config: CacheConfig,
    #     postergate_token_counting: bool,
    #     response_format: typing.Type[T],
    # ) -> typing.Tuple[list[MessageChoice[T]], Usage]: ...

    # @typing.overload
    # async def _aget_choices(
    #     self,
    #     *,
    #     model: AIModel,
    #     messages: list[Message],
    #     n: int,
    #     temperature: float,
    #     stream: bool,
    #     max_tokens: int,
    #     trace: Maybe[StatefulTraceClient],
    #     span: Maybe[StatefulSpanClient],
    #     cache_config: CacheConfig,
    #     postergate_token_counting: bool,
    #     response_format: None,
    # ) -> typing.Tuple[list[MessageChoice[None]], Usage]: ...

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
        response_format: typing.Optional[typing.Type[T]],
    ) -> typing.Tuple[list[MessageChoice[T]], Usage]:
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
        prompt_tokens: typing.Optional[int],
        completion_tokens: typing.Optional[int],
        input_cost: typing.Optional[float],
        output_cost: typing.Optional[float],
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

    # @typing.overload
    # def _get_parsed(
    #     self,
    #     response_format: typing.Type[T],
    #     content: typing.Optional[str],
    #     trace: Maybe[StatefulTraceClient],
    #     span: Maybe[StatefulSpanClient],
    # ) -> T: ...

    # @typing.overload
    # def _get_parsed(
    #     self,
    #     response_format: None,
    #     content: typing.Optional[str],
    #     trace: Maybe[StatefulTraceClient],
    #     span: Maybe[StatefulSpanClient],
    # ) -> None: ...

    def _get_parsed(
        self,
        response_format: typing.Optional[typing.Type[T]],
        content: typing.Optional[str],
        trace: Maybe[StatefulTraceClient],
        span: Maybe[StatefulSpanClient],
    ) -> T:
        if response_format is None:
            logger.warning("Response format is None")
            return typing.cast(T, None)

        if content is None:
            logger.warning("Contents of the message are none")
            return typing.cast(T, None)

        if isinstance(response_format, dict):
            LLMResponse: typing.Type[msgspec.Struct] = util.get_struct_from_schema(
                response_format, bases=(BaseModel,), name="ResponseModel"
            )

            response_format = LLMResponse

        tag: typing.Optional[Tag] = Tag.from_string(
            content, tag_name="structured"
        ) or Tag.from_string(content, tag_name="output")

        if tag is None:
            span.map(
                lambda span: span.event(
                    id=f"ev-{trace.id}",
                    name="Obtendo resposta estruturada",
                    input=content,
                    output=None,
                    level="ERROR",
                    metadata={"response_format": response_format, "content": content},
                )
            )
            return typing.cast(T, None)

        structured: dict[str, typing.Any] = tag.as_object()

        if not structured:
            raise ValueError("Tag object could not be parsed as structured content")

        model: T = msgspec.json.decode(
            msgspec.json.encode(structured), type=response_format
        )

        span.map(
            lambda span: span.event(
                id=f"ev-{trace.id}",
                name="Obtendo resposta estruturada",
                input=f"<structured>\n{tag.content}\n</structured>",
                output=model,
                level="DEBUG",
                metadata={"response_format": response_format, "content": content},
            )
        )

        return model

    def _append_response_format_to_prompt(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[typing.Type[T]],
        prompt_role: typing.Optional[MessageRole] = None,
    ) -> list[Message]:
        if prompt_role is None:
            prompt_role = MessageRole.SYSTEM

        basemodel_schema = msgspec.json.schema(response_format)

        new_prompt: str = f"""
        <saida>
            Dentro de uma tag "<structured>" a assistente irá retornar uma saída, formatada em JSON, que esteja de acordo com o seguinte esquema JSON:
            <json_schema>
            {basemodel_schema}
            </json_schema>
            O JSON retornado pela assistente, dentro da tag, deve estar de acordo com o esquema mencionado acima e deve levar em conta as instruções dadas na tarefa estipulada. A assistente deve fechar a tag com </structured>.
        </saida>
        """

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
        constructor_params: dict[str, typing.Any] = (
            DynamicDict.having(
                "max_tokens",
                equals_to=max_tokens,
            )
            .as_well_as("model_name", equals_to=model.value)
            .also(
                "project",
                equals_to=self.vertex_credentials.map(
                    lambda credentials: credentials.project_id
                ).unwrap(),
            )
            .also("model", equals_to=model.value)
            .also(
                "credentials",
                equals_to=self.vertex_credentials.unwrap(),
            )
            .also(
                "safety_settings",
                equals_to={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            .also("cache_config", equals_to=cache_config)
            .also("timeout", equals_to=120)
            .at_last("generate_kwargs", equals_to={"timeout": 120})
        )

        return DynamicInstanceCreator(
            AIModel.get_llama_index_model_cls(model)
        ).create_instance(**constructor_params)
