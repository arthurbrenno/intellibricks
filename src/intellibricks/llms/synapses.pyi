"""
Module: intellibricks.llms.synapses

This module defines the concept of "Synapses" within the intellibricks framework, drawing inspiration from the biological synapse as a junction for signal transmission.
In this context, Synapses serve as the interface for interacting with Language Model Models (LLMs) and Transcription Models.
This stub file (.pyi) provides type hints and signatures for the classes and functions defined in the corresponding Python module (intellibricks.llms.synapses.py).

**Core Concepts:**

*   **Synapse as an Interface:** A Synapse acts as a unified entry point to perform operations with various LLMs and transcription services. It abstracts away the complexities of model selection, API interactions, and configuration.
*   **Abstraction and Flexibility:** Synapses allow developers to switch between different LLMs or transcription models easily by changing the `model` attribute of a `Synapse` instance, without modifying the core application logic.
*   **Simplified API Interaction:** Synapses provide high-level methods like `complete`, `chat`, and `transcribe` that encapsulate the lower-level API calls to LLM and transcription services, making interactions more intuitive and developer-friendly.
*   **Cascade for Resilience:** The module introduces `SynapseCascade` and `TextTranscriptionsSynapseCascade` classes, which enable fault tolerance by allowing a sequence of Synapses to be tried in order. If one Synapse fails, the next one in the cascade is automatically attempted, enhancing robustness.
*   **Observability with Langfuse:**  Synapses are designed to integrate with Langfuse for observability. They automatically trace and monitor interactions with LLMs and transcription models, providing valuable insights into performance, usage, and potential issues.

**Key Classes (Type Stubs):**

*   **`SynapseProtocol`**: Defines the interface protocol for Synapse-like classes, outlining the expected methods for text completion and chat interactions.
*   **`Synapse`**: The primary class for interacting with Language Models. It offers methods for text completion (`complete`) and chat-based interactions (`chat`).
*   **`SynapseCascade`**:  A class that encapsulates a sequence of `Synapse` or `SynapseCascade` objects, providing fault tolerance for LLM interactions.
*   **`TextTranscriptionSynapse`**:  A specialized synapse for audio transcription tasks, providing a `transcribe` method.
*   **`TextTranscriptionsSynapseCascade`**: Similar to `SynapseCascade`, but specifically designed for `TextTranscriptionSynapse` objects, offering fault tolerance for audio transcription.

**Usage:**

This stub file is primarily used for static type checking and code completion in IDEs. It provides type hints for the classes and methods, allowing developers to ensure type correctness when working with Synapses and related classes. For detailed usage examples and explanations, refer to the documentation of the `intellibricks.llms.synapses` Python module.
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
    AudioTranscription,
    ToolInputType,
    TraceParams,
)
from .types import AIModel, TranscriptionModelType

debug_logger = log.create_logger(__name__, level=logging.DEBUG)
error_logger = log.create_logger(__name__, level=logging.ERROR)

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)

@runtime_checkable
class SynapseProtocol(Protocol):
    """
    Protocol defining the interface for Synapse-like classes.

    This protocol outlines the methods that any class intending to act as a Synapse should implement.
    It ensures type compatibility and allows for interchangeable use of different Synapse implementations.

    **Methods (Abstract - Defined in Protocol):**
    """
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
    ) -> ChatCompletion[S]:
        """
        Synchronously performs a text completion operation with a specified response model.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's response structured by response_model.
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Synchronously performs a text completion operation with a raw response.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw response.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Synchronously conducts a chat-based interaction with a specified response model.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's chat response structured by response_model.
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Synchronously conducts a chat-based interaction with a raw response.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw chat response.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Asynchronously performs a text completion operation with a specified response model.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's response structured by response_model (async).
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Asynchronously performs a text completion operation with a raw response.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw response (async).
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Asynchronously conducts a chat-based interaction with a specified response model.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's chat response structured by response_model (async).
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Asynchronously conducts a chat-based interaction with a raw response.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw chat response (async).
        """
        ...

class Synapse(msgspec.Struct, frozen=True, omit_defaults=True):
    """
    The primary class for interacting with Language Models (LLMs).

    `Synapse` encapsulates the configuration and methods to communicate with various LLM APIs.
    It offers functionalities for both simple text completion and more complex chat-based interactions.
    """

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
        """
        Class method to create a Synapse instance.

        :param model: The Language Model to use.
        :param api_key: API key for the Language Model service, if required.
        :param langfuse: Optional Langfuse client instance for observability.
        :param web_searcher: Optional WebSearchable instance to enable web search functionality.
        :param cloud_project: Optional Cloud project ID for cloud-based LLM services.
        :param cloud_location: Optional Cloud location for cloud-based LLM services.
        :return: A new Synapse instance.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Synchronously performs a text completion operation with a specified response model.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's response structured by response_model.
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Synchronously performs a text completion operation with a raw response.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw response.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Synchronously conducts a chat-based interaction with a specified response model.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's chat response structured by response_model.
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Synchronously conducts a chat-based interaction with a raw response.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw chat response.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Asynchronously performs a text completion operation with a specified response model.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's response structured by response_model (async).
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Asynchronously performs a text completion operation with a raw response.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw response (async).
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Asynchronously conducts a chat-based interaction with a specified response model.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's chat response structured by response_model (async).
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Asynchronously conducts a chat-based interaction with a raw response.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw chat response (async).
        """
        ...

class SynapseCascade:
    """
    Encapsulates a sequence of Synapses to provide fault tolerance for LLM interactions.

    `SynapseCascade` implements the same interface as `Synapse`, allowing it to be used interchangeably.
    It's designed to automatically try a sequence of Synapses in order if one fails, providing resilience
    against temporary issues with specific LLM services or configurations.
    """

    synapses: Sequence[Synapse | SynapseCascade]
    """A sequence of Synapse or SynapseCascade objects"""

    shuffle: bool = False
    """Indicates whether the synapses should be shuffled before trying them"""

    @classmethod
    def of(
        cls, *synapses: Synapse | SynapseCascade, shuffle: bool = False
    ) -> SynapseCascade:
        """
        Class method to create a SynapseCascade instance.

        :param synapses: Variable number of Synapse or SynapseCascade objects to include in the cascade.
        :param shuffle: Whether to shuffle the order of synapses before each attempt.
        :return: A new SynapseCascade instance.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Synchronously attempts text completion using the Synapses in the cascade with a specified response model.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's response structured by response_model.
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Synchronously attempts text completion using the Synapses in the cascade with a raw response.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw response.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Synchronously conducts a chat-based interaction using the Synapses in the cascade with a specified response model.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's chat response structured by response_model.
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Synchronously conducts a chat-based interaction using the Synapses in the cascade with a raw response.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw chat response.
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Asynchronously attempts text completion using the Synapses in the cascade with a specified response model.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's response structured by response_model (async).
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Asynchronously attempts text completion using the Synapses in the cascade with a raw response.

        :param prompt: The user prompt for text completion.
        :param system_prompt: Optional system prompt to guide the LLM.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw response (async).
        """
        ...
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
    ) -> ChatCompletion[S]:
        """
        Asynchronously conducts a chat-based interaction using the Synapses in the cascade with a specified response model.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Response model class to structure the LLM's output.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's chat response structured by response_model (async).
        """
        ...
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
    ) -> ChatCompletion[RawResponse]:
        """
        Asynchronously conducts a chat-based interaction using the Synapses in the cascade with a raw response.

        :param messages: A sequence of Message objects representing the conversation history.
        :param response_model: Defaults to None, indicating raw response.
        :param n: Number of completion choices to generate.
        :param temperature: Sampling temperature for generation (0.0 to 1.0).
        :param max_tokens: Maximum number of tokens in the generated completion.
        :param max_retries: Maximum number of retries in case of API errors (1 to 5).
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param stop_sequences: Sequences of tokens at which to stop generation.
        :param cache_config: Configuration for caching responses.
        :param trace_params: Parameters for Langfuse tracing.
        :param tools: Tools (functions) that the LLM can use.
        :param general_web_search: Whether to enable general web search for the LLM.
        :param language: Language for the LLM interaction.
        :param timeout: Request timeout in seconds.
        :return: A ChatCompletion object containing the LLM's raw chat response (async).
        """
        ...

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
    ) -> TextTranscriptionSynapse:
        """
        Class method to create a TextTranscriptionSynapse instance.

        :param model: The Transcription Model to use.
        :param api_key: API key for the Transcription Model service, if required.
        :param langfuse: Optional Langfuse client instance for observability.
        :return: A new TextTranscriptionSynapse instance.
        """
        ...
    def transcribe(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> AudioTranscription:
        """
        Synchronously transcribes an audio file to text.

        :param audio: The audio file content to transcribe.
        :param temperature: Sampling temperature for transcription.
        :param language: Language of the audio for transcription.
        :param prompt: Optional prompt to guide the transcription.
        :param trace_params: Parameters for Langfuse tracing.
        :param max_retries: Maximum number of retries in case of errors.
        :return: An AudioTranscription object containing the transcribed text and related metadata.
        """
        ...
    async def transcribe_async(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> AudioTranscription:
        """
        Asynchronously transcribes an audio file to text.

        :param audio: The audio file content to transcribe.
        :param temperature: Sampling temperature for transcription.
        :param language: Language of the audio for transcription.
        :param prompt: Optional prompt to guide the transcription.
        :param trace_params: Parameters for Langfuse tracing.
        :param max_retries: Maximum number of retries in case of errors.
        :return: An AudioTranscription object containing the transcribed text and related metadata (async).
        """
        ...

class TextTranscriptionsSynapseCascade:
    """
    Provides fault tolerance for audio transcription by cascading through multiple `TextTranscriptionSynapse` objects.

    `TextTranscriptionsSynapseCascade` mirrors the functionality of `SynapseCascade` but is specifically designed for
    `TextTranscriptionSynapse` objects. It allows you to define a sequence of transcription synapses, and if one fails,
    it automatically attempts transcription with the next one in the cascade.
    """

    synapses: Sequence[TextTranscriptionSynapse | TextTranscriptionsSynapseCascade]
    shuffle: bool

    @classmethod
    def of(
        cls,
        *synapses: TextTranscriptionSynapse | TextTranscriptionsSynapseCascade,
        shuffle: bool = False,
    ) -> TextTranscriptionsSynapseCascade:
        """
        Class method to create a TextTranscriptionsSynapseCascade instance.

        :param synapses: Variable number of TextTranscriptionSynapse or TextTranscriptionsSynapseCascade objects.
        :param shuffle: Whether to shuffle the order of synapses before each transcription attempt.
        :return: A new TextTranscriptionsSynapseCascade instance.
        """
        ...
    def transcribe(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> AudioTranscription:
        """
        Synchronously attempts audio transcription using the transcription Synapses in the cascade.

        :param audio: The audio file content to transcribe.
        :param temperature: Sampling temperature for transcription.
        :param language: Language of the audio for transcription.
        :param prompt: Optional prompt to guide the transcription.
        :param trace_params: Parameters for Langfuse tracing.
        :param max_retries: Maximum number of retries in case of errors.
        :return: An AudioTranscription object containing the transcribed text and related metadata.
        """
        ...
    async def transcribe_async(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        language: Optional[TranscriptionsLanguage] = None,
        prompt: Optional[str] = None,
        trace_params: Optional[TraceParams] = None,
        max_retries: int = 1,
    ) -> AudioTranscription:
        """
        Asynchronously attempts audio transcription using the transcription Synapses in the cascade.

        :param audio: The audio file content to transcribe.
        :param temperature: Sampling temperature for transcription.
        :param language: Language of the audio for transcription.
        :param prompt: Optional prompt to guide the transcription.
        :param trace_params: Parameters for Langfuse tracing.
        :param max_retries: Maximum number of retries in case of errors.
        :return: An AudioTranscription object containing the transcribed text and related metadata (async).
        """
        ...
