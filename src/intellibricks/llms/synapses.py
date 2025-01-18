"""
Synapse:
> The junction between two neurons that allows a signal to pass between them.

Welcome to the synapses
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import (
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    cast,
    runtime_checkable,
)

import msgspec
from architecture import log
from architecture.extensions import Maybe
from architecture.utils import run_sync
from architecture.utils.functions import fire_and_forget
from langfuse import Langfuse
from langfuse.client import (
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)
from langfuse.model import ModelUsage

from intellibricks.llms.base import FileContent, LanguageModel, TranscriptionModel
from intellibricks.llms.base import Language as TranscriptionsLanguage
from intellibricks.llms.factories import LanguageModelFactory, TranscriptionModelFactory
from intellibricks.llms.general_web_search import WebSearchable

from .constants import (
    Language,
)
from .types import (
    CacheConfig,
    ChatCompletion,
    DeveloperMessage,
    Message,
    Part,
    PartType,
    Prompt,
    RawResponse,
    TextTranscriptionOutput,
    ToolInputType,
    TraceParams,
    UserMessage,
)
from .types import AIModel, TranscriptionModelType

debug_logger = log.create_logger(__name__, level=logging.DEBUG)
error_logger = log.create_logger(__name__, level=logging.ERROR)

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
        tools: Optional[Sequence[ToolInputType]] = None,
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
        tools: Optional[Sequence[ToolInputType]] = None,
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
        tools: Optional[Sequence[ToolInputType]] = None,
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
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]: ...


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
    ) -> Synapse:
        return cls(
            model=model,
            langfuse=Maybe(langfuse),
            api_key=api_key,
            web_searcher=web_searcher,
            cloud_project=cloud_project,
            cloud_location=cloud_location,
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
        tools: Optional[Sequence[ToolInputType]] = None,
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
        tools: Optional[Sequence[ToolInputType]] = None,
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
        tools: Optional[Sequence[ToolInputType]] = None,
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
        tools: Optional[Sequence[ToolInputType]] = None,
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
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        debug_logger.debug("Entering __achat method.")

        trace_params = trace_params or {
            "name": "chat_completion",
            "user_id": "not_provided",
        }

        trace_params.setdefault("user_id", "not_provided")
        trace_params.setdefault("name", "chat_completion")

        cache_config = cache_config or CacheConfig()

        trace_params["input"] = messages

        debug_logger.debug("Generating completion ID.")
        completion_id: uuid.UUID = uuid.uuid4()

        debug_logger.debug("Initializing Langfuse trace (if available).")
        trace: Maybe[StatefulTraceClient] = self.langfuse.map(
            lambda langfuse: langfuse.trace(**trace_params)  # type: ignore
        )

        ai_model: AIModel = self.model or "google/genai/gemini-2.0-flash-exp"
        debug_logger.debug(f"Using AI model: {ai_model}")

        max_retries = max_retries or 2
        debug_logger.debug(f"Maximum retries set to: {max_retries}")

        debug_logger.debug("Creating Langfuse span (if trace is available).")
        maybe_span: Maybe[StatefulSpanClient] = Maybe(
            trace.map(
                lambda trace: trace.span(  # type: ignore
                    id=f"sp-{completion_id}",
                    input=messages,
                    name="Response Generation",
                )
            ).unwrap()
        )
        debug_logger.debug("Creating Langfuse generation (if span is available).")
        generation: Maybe[StatefulGenerationClient] = maybe_span.map(
            lambda span: span.generation(  # type: ignore
                model=ai_model,
                input=messages,
                model_parameters={
                    "max_tokens": max_tokens,
                    "temperature": str(temperature),
                },
            )
        )

        debug_logger.debug("Creating Language Model instance.")
        chat_model: LanguageModel = LanguageModelFactory.create(
            model=ai_model,
            params={
                "model_name": ai_model.split("/")[2],
                "language": language,
                "general_web_search": general_web_search,
                "api_key": self.api_key,
                "max_retries": max_retries,
                "project": self.cloud_project,
                "location": self.cloud_location,
            },
        )
        debug_logger.debug("Language Model instance created.")

        try:
            debug_logger.debug("CALLING THE AI MODEL.")
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

            debug_logger.debug("chat_async method call completed successfully.")

            fire_and_forget(
                self.__end_observability_logic, generation, maybe_span, completion
            )
            debug_logger.debug("Returning completion object.")
            return completion

        except Exception as e:
            error_logger.error(
                f"An error occurred during chat completion: {e}", exc_info=True
            )
            debug_logger.debug("Ending Langfuse span due to error.")
            maybe_span.end(output={})
            debug_logger.debug("Updating Langfuse span status due to error.")
            maybe_span.update(status_message="Error in completion", level="ERROR")
            debug_logger.debug("Scoring Langfuse span as failure due to error.")
            maybe_span.score(
                id=f"sc-{maybe_span.unwrap()}",
                name="Sucess",
                value=0.0,
                comment=f"Error while generating choices: {e}",
            )
            debug_logger.debug("Langfuse span error handling completed.")
            raise e

    async def __end_observability_logic(
        self,
        generation: Maybe[StatefulGenerationClient],
        maybe_span: Maybe[StatefulSpanClient],
        completion: ChatCompletion[S] | ChatCompletion[RawResponse],
    ) -> None:
        debug_logger.debug("Ending Langfuse generation.")
        generation.end(
            output=completion.message,
        )
        debug_logger.debug("Langfuse generation ended.")

        debug_logger.debug("Updating Langfuse generation usage.")
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
        debug_logger.debug("Langfuse generation usage updated.")

        debug_logger.debug("Scoring Langfuse span as successful.")
        maybe_span.score(
            id=f"sc-{maybe_span.map(lambda span: span.id).unwrap()}",
            name="Success",
            value=1.0,
            comment="Choices generated successfully!",
        )
        debug_logger.debug("Langfuse span scored successfully.")


class SynapseCascade(msgspec.Struct, frozen=True):
    """If one synapse fails, the next one will be used. This class
    implements the same interface as Synapse, so you can use it
    like a normal Synapse object and also with union type hints like
    synapse: Synapse | SynapseCascade
    """

    synapses: Sequence[Synapse | SynapseCascade]
    """A sequence of Synapse or SynapseCascade objects"""

    shuffle: bool = False
    """Indicates whether the synapses should be shuffled before trying them"""

    @classmethod
    def of(
        cls, *synapses: Synapse | SynapseCascade, shuffle: bool = False
    ) -> SynapseCascade:
        return cls(synapses=synapses, shuffle=shuffle)

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
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        synapses = (
            self.synapses
            if not self.shuffle
            else cast(
                Sequence[Synapse | SynapseCascade],
                random.sample(self.synapses, len(self.synapses)),
            )
        )

        for synapse in synapses:
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
                debug_logger.warning(f"Synapse failed on complete: {e}")
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
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        synapses = (
            self.synapses
            if not self.shuffle
            else cast(
                Sequence[Synapse | SynapseCascade],
                random.sample(self.synapses, len(self.synapses)),
            )
        )

        for synapse in synapses:
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
                debug_logger.warning(f"Synapse failed on chat: {e}")
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
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        synapses = (
            self.synapses
            if not self.shuffle
            else cast(
                Sequence[Synapse | SynapseCascade],
                random.sample(self.synapses, len(self.synapses)),
            )
        )

        for synapse in synapses:
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
                debug_logger.warning(f"Synapse failed on complete_async: {e}")
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
        tools: Optional[Sequence[ToolInputType]] = None,
        general_web_search: Optional[bool] = None,
        language: Language = Language.ENGLISH,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        last_exception = None
        synapses = (
            self.synapses
            if not self.shuffle
            else cast(
                Sequence[Synapse | SynapseCascade],
                random.sample(self.synapses, len(self.synapses)),
            )
        )

        for synapse in synapses:
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
                debug_logger.warning(f"Synapse failed on chat_async: {e}")
                last_exception = e
                continue
        if last_exception:
            raise last_exception
        raise RuntimeError("All synapses failed for chat_async method.")


class TextTranscriptionSynapse(msgspec.Struct, frozen=True):
    """A synapse for audio transcriptions"""

    model: TranscriptionModelType
    api_key: Optional[str] = None
    langfuse: Maybe[Langfuse] = Maybe(None)

    @classmethod
    def of(
        cls,
        model: TranscriptionModelType,
        api_key: Optional[str] = None,
        langfuse: Optional[Langfuse] = None,
    ) -> TextTranscriptionSynapse:
        return cls(
            model=model,
            api_key=api_key,
            langfuse=Maybe(langfuse),
        )

    def transcribe(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> TextTranscriptionOutput:
        return run_sync(
            self.transcribe_async,
            audio=audio,
            temperature=temperature,
            language=language,
            prompt=prompt,
            trace_params=trace_params,
            max_retries=max_retries,
        )

    async def transcribe_async(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> TextTranscriptionOutput:
        debug_logger.debug("Entering transcribe_async method.")

        # Step 1: Initialize Trace Parameters
        trace_params = trace_params or {
            "name": "transcription",
            "user_id": "not_provided",
        }

        trace_params.setdefault("user_id", "not_provided")
        trace_params.setdefault("name", "transcription")

        debug_logger.debug(f"Trace parameters: {trace_params}")

        # Step 2: Generate a unique transcription ID
        transcription_id: uuid.UUID = uuid.uuid4()
        debug_logger.debug(f"Generated transcription ID: {transcription_id}")

        # Step 3: Initialize Langfuse Trace (if available)
        trace: Maybe[StatefulTraceClient] = self.langfuse.map(
            lambda lf: lf.trace(**trace_params)  # type: ignore
        )
        debug_logger.debug("Initialized Langfuse trace.")

        # Step 4: Create a Span for the transcription process
        span: Maybe[StatefulSpanClient] = trace.map(
            lambda t: t.span(  # type: ignore
                id=f"span-{transcription_id}",
                input={
                    "audio": "FileContent"
                },  # You can provide more detailed input if available
                name="Transcription Process",
            )
        )
        debug_logger.debug("Created Langfuse span for transcription.")

        # Step 5: Create a Transcription Model instance
        transcription_model: TranscriptionModel = TranscriptionModelFactory.create(
            model=self.model,
            params={
                "model_name": self.model.split("/")[2],
                "max_retries": max_retries,
                "api_key": self.api_key,
            },
        )
        debug_logger.debug(
            f"Created TranscriptionModel instance for model: {self.model}"
        )

        try:
            # Step 6: Perform the transcription asynchronously
            debug_logger.debug(
                "Calling transcribe_async method of the Transcription Model."
            )
            transcription_result: TextTranscriptionOutput = (
                await transcription_model.transcribe_async(
                    audio=audio,
                    temperature=temperature,
                    language=language,
                    prompt=prompt,
                )
            )
            debug_logger.debug("transcribe_async method call completed successfully.")

            # Step 7: Fire and forget the observability logic
            fire_and_forget(self.__end_observability_logic, span, transcription_result)
            debug_logger.debug("Observability logic triggered.")

            debug_logger.debug("Returning transcription result.")
            return transcription_result

        except Exception as e:
            debug_logger.error(
                f"An error occurred during transcription: {e}", exc_info=True
            )
            # Handle trace/span termination on error
            fire_and_forget(self.__handle_error_observability, span, e)
            raise e

    async def __end_observability_logic(
        self,
        span: Maybe[StatefulSpanClient],
        transcription_result: TextTranscriptionOutput,
    ) -> None:
        debug_logger.debug("Ending Langfuse span.")
        span.map(lambda s: s.end(output={"text": transcription_result.text}))  # type: ignore
        debug_logger.debug("Langfuse span ended.")

        debug_logger.debug("Updating Langfuse span usage.")
        span.map(
            lambda s: s.update(  # type: ignore
                usage=ModelUsage(
                    unit="SECONDS",
                    input=int(
                        transcription_result.duration
                    ),  # Assuming elapsed_time is in seconds
                    output=0,  # Transcription might not have a separate output metric
                    total=int(transcription_result.duration),
                    input_cost=transcription_result.cost,
                    output_cost=0.0,  # No output cost if not applicable
                    total_cost=transcription_result.cost,
                )
            )
        )
        debug_logger.debug("Langfuse span usage updated.")

        debug_logger.debug("Scoring Langfuse span as successful.")
        span.map(
            lambda s: s.score(  # type: ignore
                id=f"sc-{s.id}",
                name="Success",
                value=1.0,
                comment="Transcription completed successfully!",
            )
        )
        debug_logger.debug("Langfuse span scored as successful.")

    async def __handle_error_observability(
        self,
        span: Maybe[StatefulSpanClient],
        exception: Exception,
    ) -> None:
        debug_logger.debug("Handling error in observability logic.")

        debug_logger.debug("Ending Langfuse span due to error.")
        span.map(lambda s: s.end(output={"error": str(exception)}))  # type: ignore
        debug_logger.debug("Langfuse span ended with error.")

        debug_logger.debug("Updating Langfuse span status due to error.")
        span.map(
            lambda s: s.update(  # type: ignore
                status_message="Error in transcription",
                level="ERROR",
            )
        )
        debug_logger.debug("Langfuse span status updated to ERROR.")

        debug_logger.debug("Scoring Langfuse span as failed.")
        span.map(
            lambda s: s.score(  # type: ignore
                id=f"sc-{s.id}",
                name="Failure",
                value=0.0,
                comment=f"Error during transcription: {exception}",
            )
        )
        debug_logger.debug("Langfuse span scored as failed.")

        debug_logger.debug("Error observability logic completed.")
