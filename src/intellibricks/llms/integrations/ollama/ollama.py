import timeit
from typing import Optional, Sequence, TypeVar, overload, override, cast

import msgspec
from architecture.utils.decorators import ensure_module_installed
from intellibricks.llms.base import (
    LanguageModel,
)
from intellibricks.llms.util import get_parsed_response, ms_type_to_schema
from intellibricks.llms.types import (
    ChatCompletion,
    GeneratedAssistantMessage,
    Message,
    MessageChoice,
    Part,
    RawResponse,
    ToolInputType,
    Usage,
)
import httpx

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)


class OllamaLanguageModel(LanguageModel, frozen=True):
    model_name: str
    max_retries: int = 2
    base_url: str | httpx.URL | None = None

    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...

    @ensure_module_installed("ollama", "ollama")
    @override
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        from ollama._client import AsyncClient
        from ollama._types import ChatResponse as OllamaChatResponse

        n = n or 1

        past = timeit.default_timer()

        client = AsyncClient(
            host=str(self.base_url) if self.base_url else None, timeout=timeout
        )

        choices: list[MessageChoice[S]] = []
        for i in range(n):
            #  TODO(arthur): fix type errors.
            chat_response: OllamaChatResponse = await client.chat(  # type: ignore
                model=self.model_name,
                messages=[m.to_ollama_format() for m in messages],
                format=ms_type_to_schema(response_model) if response_model else None,
            )

            choice = MessageChoice(
                index=i,
                message=GeneratedAssistantMessage(
                    contents=[Part.from_text(chat_response.message.content or "")],
                    refusal=None,
                    parsed=get_parsed_response(
                        contents=chat_response.message.content,
                        response_model=response_model,
                    )
                    if response_model and chat_response.message.content
                    else cast(S, RawResponse()),
                ),
            )

            choices.append(choice)

        return ChatCompletion(
            elapsed_time=timeit.default_timer() - past,
            model=self.model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                input_cost=0.0,
            ),
        )
