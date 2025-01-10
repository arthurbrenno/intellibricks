"""
Synapse:
> The junction between two neurons that allows a signal to pass between them.

Welcome to the synapses
"""

from __future__ import annotations

import uuid
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import msgspec
from architecture.extensions import Maybe
from architecture.logging import LoggerFactory
from architecture.utils import run_sync
from langfuse import Langfuse
from langfuse.client import (
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)
from langfuse.model import ModelUsage

from intellibricks.llms.base.contracts import SupportsAsyncChat
from intellibricks.llms.factories import SupportsAsyncChatFactory
from intellibricks.llms.general_web_search import WebSearchable

from .constants import (
    Language,
)
from .schema import (
    ChatCompletion,
    Message,
    PartType,
    Part,
    Prompt,
    RawResponse,
    DeveloperMessage,
    UserMessage,
    CacheConfig,
    TraceParams,
)
from .types import AIModel

logger = LoggerFactory.create(__name__)

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)


@runtime_checkable
class SynapseProtocol(Protocol):
    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]: ...

    def chat(
        self,
        *,
        messages: Sequence[Message],
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]: ...

    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]: ...

    async def chat_async(
        self,
        *,
        messages: Sequence[Message],
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]: ...


class Synapse(msgspec.Struct, frozen=True, omit_defaults=True):
    model: AIModel = msgspec.field(
        default_factory=lambda: "google/genai/gemini-2.0-flash-exp"
    )
    api_key: Optional[str] = None
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
    ) -> Synapse:
        return cls(
            model=model,
            langfuse=Maybe(langfuse),
            api_key=api_key,
            web_searcher=web_searcher,
        )

    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        if system_prompt is None:
            system_prompt = [
                Part.from_text(
                    "You are a helpful assistant."
                    "Answer in the same language"
                    "the conversation goes."
                )
            ]

        match system_prompt:
            case str():
                system_message = DeveloperMessage(
                    contents=[Part.from_text(system_prompt)]
                )
            case Prompt():
                system_message = DeveloperMessage(
                    contents=[Part.from_text(system_prompt.as_string())]
                )
            case Part():
                system_message = DeveloperMessage(contents=[system_prompt])
            case _:
                system_message = DeveloperMessage(contents=system_prompt)

        match prompt:
            case str():
                user_message = UserMessage(contents=[Part.from_text(prompt)])
            case Prompt():
                user_message = UserMessage(
                    contents=[Part.from_text(prompt.as_string())]
                )
            case Part():
                user_message = UserMessage(contents=[prompt])
            case _:
                user_message = UserMessage(contents=prompt)

        messages: Sequence[Message] = [
            system_message,
            user_message,
        ]

        return self.chat(
            messages=messages,
            response_model=response_model,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            cache_config=cache_config,
            trace_params=trace_params,
            tools=tools,
            general_web_search=general_web_search,
            language=language,
            timeout=timeout,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        return run_sync(
            self.__achat,
            messages=messages,
            response_model=response_model,
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            max_retries=max_retries,
            cache_config=cache_config,
            trace_params=trace_params,
            tools=tools,
            general_web_search=general_web_search,
            language=language,
            timeout=timeout,
        )

    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        if system_prompt is None:
            system_prompt = [
                Part.from_text(
                    "You are a helpful assistant."
                    "Answer in the same language"
                    "the conversation goes."
                )
            ]

        match system_prompt:
            case str():
                system_message = DeveloperMessage(
                    contents=[Part.from_text(system_prompt)]
                )
            case Prompt():
                system_message = DeveloperMessage(
                    contents=[Part.from_text(system_prompt.as_string())]
                )
            case Part():
                system_message = DeveloperMessage(contents=[system_prompt])
            case _:
                system_message = DeveloperMessage(contents=system_prompt)

        match prompt:
            case str():
                user_message = DeveloperMessage(contents=[Part.from_text(prompt)])
            case Prompt():
                user_message = DeveloperMessage(
                    contents=[Part.from_text(prompt.as_string())]
                )
            case Part():
                user_message = DeveloperMessage(contents=[prompt])
            case _:
                user_message = DeveloperMessage(contents=prompt)

        messages: Sequence[Message] = [
            system_message,
            user_message,
        ]

        return await self.chat_async(
            messages=messages,
            response_model=response_model,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            cache_config=cache_config,
            trace_params=trace_params,
            tools=tools,
            general_web_search=general_web_search,
            language=language,
            timeout=timeout,
        )

    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        return await self.__achat(
            messages=messages,
            response_model=response_model,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            cache_config=cache_config,
            trace_params=trace_params,
            tools=tools,
            general_web_search=general_web_search,
            language=language,
            timeout=timeout,
        )

    async def __achat(
        self,
        *,
        messages: Sequence[Message],
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        trace_params = trace_params or {
            "name": "chat_completion",
            "user_id": "not_provided",
        }
        cache_config = cache_config or CacheConfig()

        trace_params["input"] = messages

        completion_id: uuid.UUID = uuid.uuid4()

        trace: Maybe[StatefulTraceClient] = self.langfuse.map(
            lambda langfuse: langfuse.trace(**trace_params)
        )

        ai_model: AIModel = self.model or "google/genai/gemini-2.0-flash-exp"

        max_retries = max_retries or 1

        logger.debug("Starting chat completion.")

        maybe_span: Maybe[StatefulSpanClient] = Maybe(
            trace.map(
                lambda trace: trace.span(
                    id=f"sp-{completion_id}",
                    input=messages,
                    name="Response Generation",
                )
            ).unwrap()
        )

        generation: Maybe[StatefulGenerationClient] = maybe_span.map(
            lambda span: span.generation(
                model=ai_model,
                input=messages,
                model_parameters={
                    "max_tokens": max_tokens,
                    "temperature": str(temperature),
                },
            )
        )

        chat_model: SupportsAsyncChat = SupportsAsyncChatFactory.create(
            model=ai_model,
            params={
                "model_name": ai_model.split("/")[2],
                "language": language,
                "general_web_search": general_web_search,
                "api_key": self.api_key,
                "max_retries": max_retries,
            },
        )

        try:
            completion = await chat_model.chat_async(
                messages=messages,
                response_model=response_model,
                n=n,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                tools=tools,
                timeout=timeout,
            )

            generation.end(
                output=completion.message,
            )

            generation.update(
                usage=ModelUsage(
                    unit="TOKENS",
                    input=completion.usage.prompt_tokens
                    if isinstance(completion.usage.prompt_tokens, int)
                    else None,
                    output=completion.usage.completion_tokens
                    if isinstance(completion.usage.completion_tokens, int)
                    else None,
                    total=completion.usage.total_tokens
                    if isinstance(completion.usage.total_tokens, int)
                    else None,
                    input_cost=completion.usage.input_cost or 0.0,
                    output_cost=completion.usage.output_cost or 0.0,
                    total_cost=completion.usage.total_cost or 0.0,
                )
            )

            maybe_span.score(
                id=f"sc-{maybe_span.map(lambda span: span.id).unwrap()}",
                name="Success",
                value=1.0,
                comment="Choices generated successfully!",
            )

            return completion

        except Exception as e:
            maybe_span.end(output={})
            maybe_span.update(status_message="Error in completion", level="ERROR")
            maybe_span.score(
                id=f"sc-{maybe_span.unwrap()}",
                name="Sucess",
                value=0.0,
                comment=f"Error while generating choices: {e}",
            )
            raise e


class SynapticFallbackChain(msgspec.Struct, frozen=True):
    """If one synapse fails, the next one will be used. This class
    implements the same interface as Synapse, so you can use it
    like a normal Synapse object and also with union type hints like
    synapse: Synapse | SynapticFallbackChain
    """

    synapses: Sequence[Synapse]

    @classmethod
    def from_synapses(cls, *synapses: Synapse) -> SynapticFallbackChain:
        return cls(synapses=synapses)

    def complete(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        for synapse in self.synapses:
            try:
                return synapse.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_model=response_model,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences,
                    cache_config=cache_config,
                    trace_params=trace_params,
                    tools=tools,
                    general_web_search=general_web_search,
                    language=language,
                    timeout=timeout,
                )
            except Exception as e:
                logger.warning(f"Synapse {synapse.model} failed on complete: {e}")
                last_exception = e
                continue
        if last_exception:
            raise last_exception
        raise RuntimeError("All synapses failed for complete method.")

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        for synapse in self.synapses:
            try:
                return synapse.chat(
                    messages=messages,
                    response_model=response_model,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences,
                    cache_config=cache_config,
                    trace_params=trace_params,
                    tools=tools,
                    general_web_search=general_web_search,
                    language=language,
                    timeout=timeout,
                )
            except Exception as e:
                logger.warning(f"Synapse {synapse.model} failed on chat: {e}")
                last_exception = e
                continue
        if last_exception:
            raise last_exception
        raise RuntimeError("All synapses failed for chat method.")

    async def complete_async(
        self,
        prompt: str | Prompt | PartType | Sequence[PartType],
        *,
        system_prompt: Optional[str | Prompt | PartType | Sequence[PartType]] = None,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        for synapse in self.synapses:
            try:
                return await synapse.complete_async(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_model=response_model,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences,
                    cache_config=cache_config,
                    trace_params=trace_params,
                    tools=tools,
                    general_web_search=general_web_search,
                    language=language,
                    timeout=timeout,
                )
            except Exception as e:
                logger.warning(f"Synapse {synapse.model} failed on complete_async: {e}")
                last_exception = e
                continue
        if last_exception:
            raise last_exception
        raise RuntimeError("All synapses failed for complete_async method.")

    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[Literal[1, 2, 3, 4, 5]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        trace_params: Optional[TraceParams] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        for synapse in self.synapses:
            try:
                return await synapse.chat_async(
                    messages=messages,
                    response_model=response_model,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences,
                    cache_config=cache_config,
                    trace_params=trace_params,
                    tools=tools,
                    general_web_search=general_web_search,
                    language=language,
                    timeout=timeout,
                )
            except Exception as e:
                logger.warning(f"Synapse {synapse.model} failed on chat_async: {e}")
                last_exception = e
                continue
        if last_exception:
            raise last_exception
        raise RuntimeError("All synapses failed for chat_async method.")
