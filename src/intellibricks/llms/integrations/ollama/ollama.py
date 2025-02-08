# import timeit
# from typing import Literal, Optional, Sequence, TypeVar, overload, override

# import msgspec
# from architecture.utils.decorators import ensure_module_installed
# from langfuse.client import os
# from intellibricks.llms.util import ms_type_to_schema
# from intellibricks.llms.base import (
#     LanguageModel,
# )
# from intellibricks.llms.constants import FinishReason
# from intellibricks.llms.types import (
#     AudioTranscription,
#     CalledFunction,
#     ChatCompletion,
#     CompletionTokensDetails,
#     Function,
#     GeneratedAssistantMessage,
#     Message,
#     MessageChoice,
#     OpenAIModelType,
#     Part,
#     PromptTokensDetails,
#     RawResponse,
#     SentenceSegment,
#     ToolCall,
#     ToolCallSequence,
#     ToolInputType,
#     TypeAlias,
#     Usage,
# )
# from ollama import AsyncClient

# S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)
# DeepSeekModels = Literal[
#     "deepseek-r1:1.5b",
#     "deepseek-r1:7b",
#     "deepseek-r1:8b",
#     "deepseek-r1:14b",
#     "deepseek-r1:32b",
# ]

# ChatModel = Literal[DeepSeekModels]


# class OllamaLanguageModel(LanguageModel, frozen=True):
#     model_name: ChatModel
#     max_retries: int = 2

#     @overload
#     async def chat_async(
#         self,
#         messages: Sequence[Message],
#         *,
#         response_model: None = None,
#         n: Optional[int] = None,
#         temperature: Optional[float] = None,
#         max_completion_tokens: Optional[int] = None,
#         top_p: Optional[float] = None,
#         top_k: Optional[int] = None,
#         stop_sequences: Optional[Sequence[str]] = None,
#         tools: Optional[Sequence[ToolInputType]] = None,
#         timeout: Optional[float] = None,
#     ) -> ChatCompletion[RawResponse]: ...
#     @overload
#     async def chat_async(
#         self,
#         messages: Sequence[Message],
#         *,
#         response_model: type[S],
#         n: Optional[int] = None,
#         temperature: Optional[float] = None,
#         max_completion_tokens: Optional[int] = None,
#         top_p: Optional[float] = None,
#         top_k: Optional[int] = None,
#         stop_sequences: Optional[Sequence[str]] = None,
#         tools: Optional[Sequence[ToolInputType]] = None,
#         timeout: Optional[float] = None,
#     ) -> ChatCompletion[S]: ...

#     @ensure_module_installed("ollama", "ollama")
#     @override
#     async def chat_async(
#         self,
#         messages: Sequence[Message],
#         *,
#         response_model: Optional[type[S]] = None,
#         n: Optional[int] = None,
#         temperature: Optional[float] = None,
#         max_completion_tokens: Optional[int] = None,
#         top_p: Optional[float] = None,
#         top_k: Optional[int] = None,
#         stop_sequences: Optional[Sequence[str]] = None,
#         tools: Optional[Sequence[ToolInputType]] = None,
#         timeout: Optional[float] = None,
#     ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
#         now = timeit.default_timer()
#         client = AsyncClient()
#         completion = await client.chat(
#             model=self.model_name,
#             messages=[m.to_ollama_message() for m in messages],
#             format=ms_type_to_schema(response_model, openai_like=True)
#             if response_model is not None
#             else None,
#         )
