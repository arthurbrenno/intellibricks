"""
Module: intellibricks.llms.synapses

This module defines the concept of "Synapses" within the intellibricks framework, drawing inspiration from the biological synapse as a junction for signal transmission.
In this context, Synapses serve as the interface for interacting with Language Model Models (LLMs) and Transcription Models.

**Core Concepts:**

*   **Synapse as an Interface:** A Synapse acts as a unified entry point to perform operations with various LLMs and transcription services. It abstracts away the complexities of model selection, API interactions, and configuration.
*   **Abstraction and Flexibility:** Synapses allow developers to switch between different LLMs or transcription models easily by changing the `model` attribute of a `Synapse` instance, without modifying the core application logic.
*   **Simplified API Interaction:** Synapses provide high-level methods like `complete`, `chat`, and `transcribe` that encapsulate the lower-level API calls to LLM and transcription services, making interactions more intuitive and developer-friendly.
*   **Cascade for Resilience:** The module introduces `SynapseCascade` and `TextTranscriptionsSynapseCascade` classes, which enable fault tolerance by allowing a sequence of Synapses to be tried in order. If one Synapse fails, the next one in the cascade is automatically attempted, enhancing robustness.
*   **Observability with Langfuse:**  Synapses are designed to integrate with Langfuse for observability. They automatically trace and monitor interactions with LLMs and transcription models, providing valuable insights into performance, usage, and potential issues.

**Key Classes:**

*   **`Synapse`**: The primary class for interacting with Language Models. It offers methods for text completion (`complete`) and chat-based interactions (`chat`). It supports various configuration options like model selection, API keys, temperature, token limits, and more.
*   **`SynapseCascade`**:  A class that encapsulates a sequence of `Synapse` or `SynapseCascade` objects. It implements the same interface as `Synapse`, allowing for seamless replacement. If a call to a synapse in the cascade fails, it automatically falls back to the next one in the sequence.
*   **`TextTranscriptionSynapse`**:  A specialized synapse for audio transcription tasks. It provides a `transcribe` method to convert audio files into text using configured transcription models.
*   **`TextTranscriptionsSynapseCascade`**: Similar to `SynapseCascade`, but specifically for `TextTranscriptionSynapse` objects. It provides fault tolerance for audio transcription by trying a sequence of transcription synapses.

**Usage:**

To use this module, you typically instantiate a `Synapse` or `TextTranscriptionSynapse` object, configuring it with the desired model and API keys. Then, you can use the `complete`, `chat`, or `transcribe` methods to interact with the chosen LLM or transcription service. For increased reliability, especially in production environments, consider using `SynapseCascade` or `TextTranscriptionsSynapseCascade` to create fallback mechanisms.

**Example (Synapse for Chat Completion):**

```python
from intellibricks.llms.synapses import Synapse

# Instantiate a Synapse for a specific LLM
my_synapse = Synapse.of("google/genai/gemini-1.5-pro")

# Perform a chat completion
response = my_synapse.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.message.content) # Output: Paris
```

**Example (SynapseCascade for Fallback):**

```python
from intellibricks.llms.synapses import Synapse, SynapseCascade

# Create multiple Synapses, potentially for different models or API keys
synapse1 = Synapse.of("google/genai/gemini-1.5-pro")
synapse2 = Synapse.of("openai/api/gpt-4o")

# Create a SynapseCascade with the synapses
cascade_synapse = SynapseCascade.of(synapse1, synapse2)

# Use the cascade synapse just like a regular Synapse
response = cascade_synapse.complete(prompt="Explain quantum physics in simple terms.")

print(response.message.content) # Will try synapse1, then synapse2 if synapse1 fails
```

This module provides a flexible, robust, and easy-to-use interface for leveraging the power of Language Models and Transcription services within your applications.
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
    overload,
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
    AudioTranscription,
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
    """
    Protocol defining the interface for Synapse-like classes.

    This protocol outlines the methods that any class intending to act as a Synapse should implement.
    It ensures type compatibility and allows for interchangeable use of different Synapse implementations.

    **Methods (Abstract):**

    *   `complete(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Synchronously performs a text completion operation given a prompt.

    *   `chat(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Synchronously conducts a chat-based interaction given a sequence of messages.

    *   `complete_async(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Asynchronously performs a text completion operation given a prompt.

    *   `chat_async(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Asynchronously conducts a chat-based interaction given a sequence of messages.

    **Type Parameters:**

    *   `S` (bound=msgspec.Struct, default=RawResponse):
        A type variable representing the expected response model. It must be a subclass of `msgspec.Struct` and defaults to `RawResponse` if no specific model is provided.

    **Usage:**

    Classes like `Synapse` and `SynapseCascade` implement this protocol, ensuring they provide the necessary methods for interacting with language models. You can use `isinstance(my_synapse, SynapseProtocol)` to check if an object conforms to this interface.

    **Example (Protocol Usage - Type Hinting):**

    ```python
    from intellibricks.llms.synapses import Synapse, SynapseProtocol

    def process_with_synapse(synapse: SynapseProtocol):
        # Function that accepts any object adhering to SynapseProtocol
        response = synapse.complete(prompt="Tell me a joke.")
        print(response.message.content)

    my_synapse = Synapse.of("google/genai/gemini-1.5-pro")
    process_with_synapse(my_synapse) # Works because Synapse implements SynapseProtocol
    ```
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
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]: ...


class Synapse(msgspec.Struct, frozen=True, omit_defaults=True):
    """
    The primary class for interacting with Language Models (LLMs).

    `Synapse` encapsulates the configuration and methods to communicate with various LLM APIs.
    It offers functionalities for both simple text completion and more complex chat-based interactions.

    **Key Features:**

    *   **Model Abstraction:** Supports various Language Models through the `model` attribute, allowing easy switching between different LLMs.
    *   **API Key Management:** Handles API key configuration, either directly or via environment variables (depending on the underlying LLM implementation).
    *   **Parameter Configuration:** Provides extensive parameters to control LLM behavior, such as temperature, token limits, stop sequences, and more.
    *   **Synchronous and Asynchronous APIs:** Offers both synchronous (`complete`, `chat`) and asynchronous (`complete_async`, `chat_async`) methods for flexibility in different application contexts.
    *   **Langfuse Integration:** Seamlessly integrates with Langfuse for observability, automatically tracing requests and responses, and providing performance metrics.
    *   **Web Search Integration:** Optionally integrates with a `WebSearchable` instance to enable LLMs to perform general web searches as part of their response generation.
    *   **Tool Support:** Supports the use of tools (functions) that the LLM can invoke, enhancing its capabilities for complex tasks.
    *   **Caching:**  Offers caching mechanisms via `CacheConfig` to optimize performance and reduce API costs.

    **Attributes:**

    *   `model` (AIModel):
        Specifies the Language Model to be used. Defaults to `"google/genai/gemini-2.0-flash-exp"`.

        **Example:**
        ```python
        from intellibricks.llms.synapses import Synapse

        synapse_gemini_pro = Synapse("google/genai/gemini-1.5-pro")
        synapse_gpt_3_5 = Synapse("openai/api/gpt-4o")
        ```

    *   `api_key` (Optional[str]):
        API key for accessing the Language Model service, if required.  The necessity and method of providing this depends on the chosen `AIModel`.

        **Example:**
        ```python
        synapse_with_api_key = Synapse("openai/api/gpt-4o", api_key="YOUR_API_KEY")
        ```

    *   `cloud_project` (Optional[str]):
        Cloud project ID, relevant for some cloud-based LLM services like Google Cloud's Vertex AI.

    *   `cloud_location` (Optional[str]):
        Cloud location, relevant for some cloud-based LLM services like Google Cloud's Vertex AI.

    *   `langfuse` (Maybe[Langfuse]):
        Optional Langfuse client instance for observability. If provided, Synapse will automatically trace interactions. Defaults to `Maybe(None)`.

        **Example:**
        ```python
        from intellibricks.llms.synapses import Synapse
        from langfuse import Langfuse

        langfuse_client = Langfuse(public_key="...", secret_key="...")
        synapse_with_langfuse = Synapse(model="google/genai/gemini-1.5-pro", langfuse=Maybe(langfuse_client))
        ```

    *   `web_searcher` (Optional[WebSearchable]):
        Optional `WebSearchable` instance to enable web search functionality for the LLM. Defaults to `None`.

    **Class Methods:**

    *   `of(...) -> Synapse`:
        A class method to create a `Synapse` instance with a more readable and explicit parameter passing.

        **Parameters:**
        *   `model` (AIModel): The Language Model to use.
        *   `api_key` (Optional[str]): API key.
        *   `langfuse` (Optional[Langfuse]): Langfuse client instance.
        *   `web_searcher` (Optional[WebSearchable]): Web searcher instance.
        *   `cloud_project` (Optional[str]): Cloud project ID.
        *   `cloud_location` (Optional[str]): Cloud location.

        **Returns:**
        *   `Synapse`: A new `Synapse` instance.

        **Example:**
        ```python
        synapse_instance = Synapse.of(
            model="openai/api/gpt-4o",
            api_key="YOUR_API_KEY",
            # ... other parameters
        )
        ```

    **Methods:**

    *   `complete(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Synchronously performs a text completion operation. It takes a prompt and optional system prompt,
        along with various parameters to control the generation process.

        **Parameters:**
        *   `prompt` (str | Prompt | PartType | Sequence[PartType]): The user prompt for text completion. Can be a string, `Prompt` object, `PartType`, or a sequence of `PartType`.
        *   `system_prompt` (Optional[str | Prompt | PartType | Sequence[PartType]]): Optional system prompt to guide the LLM's behavior.
        *   `response_model` (Optional[type[S]]): Optional response model class to structure the LLM's output.
        *   `n` (Optional[int]): Number of completion choices to generate.
        *   `temperature` (Optional[float]): Sampling temperature for generation (0.0 to 1.0, higher values are more random).
        *   `max_tokens` (Optional[int]): Maximum number of tokens in the generated completion.
        *   `max_retries` (Optional[Literal[1, 2, 3, 4, 5]]): Maximum number of retries in case of API errors.
        *   `top_p` (Optional[float]): Top-p sampling parameter.
        *   `top_k` (Optional[int]): Top-k sampling parameter.
        *   `stop_sequences` (Optional[Sequence[str]]): Sequences of tokens at which to stop generation.
        *   `cache_config` (Optional[CacheConfig]): Configuration for caching responses.
        *   `trace_params` (Optional[TraceParams]): Parameters for Langfuse tracing.
        *   `tools` (Optional[Sequence[ToolInputType]]): Tools (functions) that the LLM can use.
        *   `general_web_search` (Optional[bool]): Whether to enable general web search for the LLM.
        *   `language` (Language):  Language for the LLM interaction. Defaults to `Language.ENGLISH`.
        *   `timeout` (Optional[float]): Request timeout in seconds.

        **Returns:**
        *   `ChatCompletion[S] | ChatCompletion[RawResponse]`: A `ChatCompletion` object containing the LLM's response, potentially structured according to `response_model`.

        **Example:**
        ```python
        response = my_synapse.complete(prompt="Write a short poem about autumn.")
        print(response.message.content)
        ```

    *   `chat(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Synchronously conducts a chat-based interaction with the LLM. It takes a sequence of `Message` objects
        representing the conversation history and parameters to control the chat.

        **Parameters:**
        *   `messages` (Sequence[Message]): A sequence of `Message` objects representing the conversation history.
        *   `response_model` (Optional[type[S]]): Optional response model class.
        *   ... (other parameters are the same as in `complete` method)

        **Returns:**
        *   `ChatCompletion[S] | ChatCompletion[RawResponse]`: A `ChatCompletion` object containing the LLM's chat response.

        **Example:**
        ```python
        chat_response = my_synapse.chat(
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print(chat_response.message.content)
        ```

    *   `complete_async(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Asynchronous version of the `complete` method.

    *   `chat_async(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Asynchronous version of the `chat` method.
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
        return cls(
            model=model,
            langfuse=Maybe(langfuse),
            api_key=api_key,
            web_searcher=web_searcher,
            cloud_project=cloud_project,
            cloud_location=cloud_location,
        )

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
    """
    Encapsulates a sequence of Synapses to provide fault tolerance for LLM interactions.

    `SynapseCascade` implements the same interface as `Synapse`, allowing it to be used interchangeably.
    It's designed to automatically try a sequence of Synapses in order if one fails, providing resilience
    against temporary issues with specific LLM services or configurations.

    **Key Features:**

    *   **Fault Tolerance:** Automatically retries requests with subsequent Synapses in the cascade if the current one fails.
    *   **Synapse Sequencing:** Allows defining a specific order of Synapses to be tried.
    *   **Shuffle Option:** Provides an option to shuffle the order of Synapses before each attempt, useful for load balancing or randomized testing.
    *   **Same Interface as Synapse:** Implements the `SynapseProtocol`, ensuring seamless integration where a `Synapse` is expected.
    *   **Cascade of Cascades:** Supports nesting of `SynapseCascade` objects within each other, allowing for complex fallback strategies.

    **Attributes:**

    *   `synapses` (Sequence[Synapse | SynapseCascade]):
        A sequence of `Synapse` or `SynapseCascade` objects. These are the Synapses that will be tried in order.

        **Example:**
        ```python
        from intellibricks.llms.synapses import Synapse, SynapseCascade

        synapse1 = Synapse.of("google/genai/gemini-1.5-pro")
        synapse2 = Synapse.of("google/vertexai/gemini-1.5-pro")

        cascade = SynapseCascade(synapses=[synapse1, synapse2])
        print(cascade.synapses) # Output: [<Synapse...>, <Synapse...>]
        ```

    *   `shuffle` (bool):
        A boolean flag indicating whether the `synapses` sequence should be shuffled before each attempt. Defaults to `False`.

        **Example:**
        ```python
        shuffled_cascade = SynapseCascade(synapses=[synapse1, synapse2], shuffle=True)
        print(shuffled_cascade.shuffle) # Output: True
        ```

    **Class Methods:**

    *   `of(*synapses: Synapse | SynapseCascade, shuffle: bool = False) -> SynapseCascade`:
        A class method to create a `SynapseCascade` instance with a more convenient way to pass Synapses as arguments.

        **Parameters:**
        *   `*synapses` (Synapse | SynapseCascade): Variable number of `Synapse` or `SynapseCascade` objects to include in the cascade.
        *   `shuffle` (bool): Whether to shuffle the synapses.

        **Returns:**
        *   `SynapseCascade`: A new `SynapseCascade` instance.

        **Example:**
        ```python
        cascade_instance = SynapseCascade.of(synapse1, synapse2, shuffle=True)
        ```

    **Methods:**

    *   `complete(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Synchronously attempts text completion using the Synapses in the cascade. It iterates through the `synapses` sequence. If a `complete` call is successful on a Synapse, it returns the result immediately. If a call fails (raises an exception), it catches the exception and tries the next Synapse in the sequence. If all Synapses fail, it raises the last caught exception or a `RuntimeError` if no exception was caught.

        **Parameters:**
        *   ... (parameters are the same as in `Synapse.complete` method)

        **Returns:**
        *   `ChatCompletion[S] | ChatCompletion[RawResponse]`: The `ChatCompletion` result from the first successful Synapse.

        **Raises:**
        *   Exception: The last exception raised by a Synapse in the cascade if all attempts fail.
        *   RuntimeError: If all synapses fail and no specific exception was last caught.

        **Example:**
        ```python
        cascade_response = cascade_synapse.complete(prompt="Translate 'Hello' to Spanish.")
        print(cascade_response.message.content) # Will try synapse1, then synapse2 if synapse1 fails
        ```

    *   `chat(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Synchronous version of chat for cascade synapse.

    *   `complete_async(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Asynchronous version of complete for cascade synapse.

    *   `chat_async(...) -> ChatCompletion[S] | ChatCompletion[RawResponse]`:
        Asynchronous version of chat for cascade synapse.
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
        return cls(synapses=synapses, shuffle=shuffle)

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
                if response_model:
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

                return synapse.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_model=None,
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
                if response_model:
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

                return await synapse.complete_async(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_model=None,
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
    """
    Specialized Synapse for handling audio transcriptions.

    `TextTranscriptionSynapse` is designed to interact with audio transcription services.
    It simplifies the process of transcribing audio files to text, handling model selection,
    API interactions, and observability.

    **Key Features:**

    *   **Transcription Model Abstraction:** Supports various transcription models through the `model` attribute.
    *   **API Key Management:** Handles API key configuration for transcription services.
    *   **Simplified Transcription Method:** Provides a `transcribe` method to easily convert audio files to text.
    *   **Langfuse Integration:** Integrates with Langfuse to trace and monitor transcription requests and results.
    *   **Retry Mechanism:** Implements retry logic for robust transcription operations.

    **Attributes:**

    *   `model` (TranscriptionModelType):
        Specifies the transcription model to be used.

        **Example:**
        ```python
        from intellibricks.llms.synapses import TextTranscriptionSynapse

        transcription_synapse = TextTranscriptionSynapse("groq/api/distil-whisper-large-v3-en")
        ```

    *   `api_key` (Optional[str]):
        API key for the transcription service, if required by the chosen `TranscriptionModelType`.

        **Example:**
        ```python
        transcription_synapse_with_key = TextTranscriptionSynapse(model=TranscriptionModelType.WHISPER_API, api_key="YOUR_WHISPER_API_KEY")
        ```

    *   `langfuse` (Maybe[Langfuse]):
        Optional Langfuse client instance for observability of transcription operations. Defaults to `Maybe(None)`.

        **Example:**
        ```python
        from intellibricks.llms.synapses import TextTranscriptionSynapse
        from langfuse import Langfuse

        langfuse_client = Langfuse(public_key="...", secret_key="...")
        transcription_synapse = TextTranscriptionSynapse("groq/api/distil-whisper-large-v3-en", langfuse=Maybe(langfuse_client))
        ```

    **Class Methods:**

    *   `of(...) -> TextTranscriptionSynapse`:
        A class method to create a `TextTranscriptionSynapse` instance with a more readable parameter passing style.

        **Parameters:**
        *   `model` (TranscriptionModelType): The transcription model to use.
        *   `api_key` (Optional[str]): API key for the transcription service.
        *   `langfuse` (Optional[Langfuse]): Langfuse client instance.

        **Returns:**
        *   `TextTranscriptionSynapse`: A new `TextTranscriptionSynapse` instance.

        **Example:**
        ```python
        transcription_synapse_instance = TextTranscriptionSynapse.of(
            model=TranscriptionModelType.WHISPER_API,
            api_key="YOUR_WHISPER_API_KEY",
            # ... other parameters
        )
        ```

    **Methods:**

    *   `transcribe(...) -> AudioTranscription`:
        Synchronously transcribes an audio file to text.

        **Parameters:**
        *   `audio` (FileContent): The audio file content to transcribe.
        *   `temperature` (Optional[float]): Sampling temperature for transcription.
        *   `language` (Optional[TranscriptionsLanguage]): Language of the audio for transcription.
        *   `prompt` (Optional[str]): Optional prompt to guide the transcription.
        *   `trace_params` (Optional[TraceParams]): Parameters for Langfuse tracing.
        *   `max_retries` (int): Maximum number of retries in case of errors. Defaults to 1.

        **Returns:**
        *   `AudioTranscription`: An `AudioTranscription` object containing the transcribed text and related metadata.

        **Example:**
        ```python
        from intellibricks.llms.synapses import TextTranscriptionSynapse
        from intellibricks.llms.types import FileContent

        transcription_synapse = TextTranscriptionSynapse.of("groq/api/distil-whisper-large-v3-en", api_key="YOUR_WHISPER_API_KEY")
        audio_file_content = FileContent(data=b"...", mime_type="audio/mp3") # Replace b"..." with actual audio bytes

        transcription = transcription_synapse.transcribe(audio=audio_file_content)
        print(transcription.text) # Output: Transcribed text from the audio
        ```

    *   `transcribe_async(...) -> AudioTranscription`:
        Asynchronous version of the `transcribe` method.
    """

    model: TranscriptionModelType
    api_key: Optional[str] = None
    langfuse: Maybe[Langfuse] = msgspec.field(default_factory=lambda: Maybe(None))

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
                "language": language,
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
            transcription_result: AudioTranscription = (
                await transcription_model.transcribe_async(
                    audio=audio,
                    temperature=temperature,
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
        transcription_result: AudioTranscription,
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


class TextTranscriptionsSynapseCascade(msgspec.Struct, frozen=True):
    """
    Provides fault tolerance for audio transcription by cascading through multiple `TextTranscriptionSynapse` objects.

    `TextTranscriptionsSynapseCascade` mirrors the functionality of `SynapseCascade` but is specifically designed for
    `TextTranscriptionSynapse` objects. It allows you to define a sequence of transcription synapses, and if one fails,
    it automatically attempts transcription with the next one in the cascade.

    **Key Features:**

    *   **Transcription Fault Tolerance:** Ensures transcription operations are more resilient to failures by automatically retrying with backup synapses.
    *   **Synapse Sequencing for Transcription:** Defines a specific order in which transcription synapses are attempted.
    *   **Shuffle Option for Transcription:** Offers the option to shuffle the order of transcription synapses for each transcription request.
    *   **Same Interface as TextTranscriptionSynapse:** Implements the same `transcribe` and `transcribe_async` methods as `TextTranscriptionSynapse`, allowing for seamless replacement.
    *   **Cascade Nesting for Transcription:** Supports nesting `TextTranscriptionsSynapseCascade` objects within each other for complex transcription fallback strategies.

    **Attributes:**

    *   `synapses` (Sequence[TextTranscriptionSynapse | TextTranscriptionsSynapseCascade]):
        A sequence of `TextTranscriptionSynapse` or nested `TextTranscriptionsSynapseCascade` objects. These are the transcription synapses that will be tried in order.

        **Example:**
        ```python
        from intellibricks.llms.synapses import TextTranscriptionSynapse, TextTranscriptionsSynapseCascade

        transcription_synapse1 = TextTranscriptionSynapse.of("groq/api/distil-whisper-large-v3-en")
        transcription_synapse2 = TextTranscriptionSynapse.of("groq/api/distil-whisper-large-v3")

        cascade = TextTranscriptionsSynapseCascade(synapses=[transcription_synapse1, transcription_synapse2])
        print(cascade.synapses) # Output: [<TextTranscriptionSynapse...>, <TextTranscriptionSynapse...>]
        ```

    *   `shuffle` (bool):
        A boolean flag indicating whether the `synapses` sequence should be shuffled before each transcription attempt. Defaults to `False`.

        **Example:**
        ```python
        shuffled_transcription_cascade = TextTranscriptionsSynapseCascade(synapses=[transcription_synapse1, transcription_synapse2], shuffle=True)
        print(shuffled_transcription_cascade.shuffle) # Output: True
        ```

    **Class Methods:**

    *   `of(*synapses: TextTranscriptionSynapse | TextTranscriptionsSynapseCascade, shuffle: bool = False) -> TextTranscriptionsSynapseCascade`:
        A class method to create a `TextTranscriptionsSynapseCascade` instance with a convenient way to pass transcription Synapses.

        **Parameters:**
        *   `*synapses` (TextTranscriptionSynapse | TextTranscriptionsSynapseCascade): Variable number of `TextTranscriptionSynapse` or `TextTranscriptionsSynapseCascade` objects.
        *   `shuffle` (bool): Whether to shuffle the order of synapses.

        **Returns:**
        *   `TextTranscriptionsSynapseCascade`: A new `TextTranscriptionsSynapseCascade` instance.

        **Example:**
        ```python
        transcription_cascade_instance = TextTranscriptionsSynapseCascade.of(transcription_synapse1, transcription_synapse2, shuffle=False)
        ```

    **Methods:**

    *   `transcribe(...) -> AudioTranscription`:
        Synchronously attempts audio transcription using the transcription Synapses in the cascade. It iterates through the `synapses` sequence and calls the `transcribe` method of each. If a transcription is successful, it returns the result. If a synapse fails, it tries the next one. If all fail, it raises the last exception or a `RuntimeError`.

        **Parameters:**
        *   ... (parameters are the same as in `TextTranscriptionSynapse.transcribe` method)

        **Returns:**
        *   `AudioTranscription`: The `AudioTranscription` result from the first successful transcription Synapse.

        **Raises:**
        *   Exception: The last exception raised by a transcription Synapse in the cascade if all attempts fail.
        *   RuntimeError: If all transcription synapses fail and no specific exception was last caught.

        **Example:**
        ```python
        audio_file_content = FileContent(data=b"...", mime_type="audio/mp3") # Replace b"..." with actual audio bytes
        cascade_transcription = transcription_cascade_instance.transcribe(audio=audio_file_content)
        print(cascade_transcription.text) # Will try transcription_synapse1, then transcription_synapse2 if synapse1 fails
        ```

    *   `transcribe_async(...) -> AudioTranscription`:
        Asynchronous version of the `transcribe` method for the cascade.
    """

    synapses: Sequence[TextTranscriptionSynapse | TextTranscriptionsSynapseCascade]
    """Sequence of transcription synapses or cascades"""

    shuffle: bool = False
    """Whether to shuffle the synapse order before trying"""

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
        return cls(synapses=synapses, shuffle=shuffle)

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
        last_exception = None
        synapses = (
            self.synapses
            if not self.shuffle
            else random.sample(self.synapses, len(self.synapses))
        )

        for synapse in synapses:
            try:
                return synapse.transcribe(
                    audio=audio,
                    temperature=temperature,
                    language=language,
                    prompt=prompt,
                    trace_params=trace_params,
                    max_retries=max_retries,
                )
            except Exception as e:
                debug_logger.warning(f"Transcription synapse failed: {e}")
                last_exception = e
                continue

        if last_exception:
            raise last_exception
        raise RuntimeError("All transcription synapses failed")

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
        last_exception = None
        synapses = (
            self.synapses
            if not self.shuffle
            else random.sample(self.synapses, len(self.synapses))
        )

        for synapse in synapses:
            try:
                return await synapse.transcribe_async(
                    audio=audio,
                    temperature=temperature,
                    language=language,
                    prompt=prompt,
                    trace_params=trace_params,
                    max_retries=max_retries,
                )
            except Exception as e:
                debug_logger.warning(f"Transcription synapse failed: {e}")
                last_exception = e
                continue

        if last_exception:
            raise last_exception
        raise RuntimeError("All transcription synapses failed")
