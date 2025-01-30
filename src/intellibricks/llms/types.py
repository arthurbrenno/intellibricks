"""
LLM Schema Definitions and Data Models

This module provides a comprehensive set of schemas and data models for working with Large Language Models (LLMs).
It enables type-safe interactions with various LLM providers while supporting multimodal inputs, tool calling,
and response handling across different platforms.

Key Components:
1. Core Message Structures:
   - Parts: Fundamental content units (TextPart, ImageFilePart, AudioFilePart, ToolCallPart)
   - Messages: Conversation components (UserMessage, AssistantMessage, SystemMessage)
   - Sequences: Ordered collections (MessageSequence, PartSequence)

2. Configuration Models:
   - GenerationConfig: Controls generation behavior (temperature, max_tokens, tools)
   - CacheConfig: Manages response caching strategies
   - TraceParams: Enables request tracing and analytics

3. Provider Compatibility:
   - Conversion methods for OpenAI, Google GenAI, Anthropic, Groq, and Cerebras formats
   - Type aliases for different provider models (OpenAIModelType, GoogleModelType, etc.)

4. Advanced Features:
   - Tool calling infrastructure with function definitions and call handling
   - Multimodal content handling (images, audio, video, websites)
   - Structured output parsing with ChainOfThought and GeneratedAssistantMessage
   - Audio transcription models and processing

5. Utility Classes:
   - Usage statistics tracking
   - Cost calculation
   - Media description models for visual/audio content analysis

Key Features:
- Strong typing with msgspec for validation and serialization
- Cross-provider compatibility through format conversion methods
- Extensible factory patterns for content creation
- Support for complex LLM interactions including:
  * Multi-turn conversations
  * Function/tool calling
  * Mixed media inputs (text + images + audio)
  * Response caching and tracing
  * Structured output generation

Example Usage:
    >>> from intellibricks.llms.types import (
    >>>     UserMessage, GenerationConfig,
    >>>     TextPart, ImageFilePart
    >>> )

    >>> # Create multimodal message
    >>> msg = UserMessage(contents=[
    >>>     TextPart("Analyze this image:"),
    >>>     ImageFilePart.from_url("https://example.com/image.jpg")
    >>> ])

    >>> # Configure generation
    >>> config = GenerationConfig(
    >>>     temperature=0.7,
    >>>     max_tokens=500,
    >>>     tools=[search_web, calculate]
    >>> )

    >>> # Get completion
    >>> completion = model.chat([msg])
    >>> print(completion.choices[0].message.contents)

Provider Support Matrix:
- OpenAI: Full (messages, tools, images)
- Google GenAI: Full (messages, tools, multimodal)
- Anthropic: Messages, basic tools
- Groq: Messages, basic tools
- Cerebras: Messages, basic tools

Note: Requires installation of provider-specific SDKs for format conversions
"""

from __future__ import annotations

import base64

# import dataclasses
import datetime
import inspect
import logging
import re
import uuid
from io import BytesIO
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Never,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    get_type_hints,
    overload,
    override,
)

import msgspec
from architecture import dp, log
from architecture.data.files import find_extension, ext_to_mime, bytes_to_mime
from architecture.utils.decorators import ensure_module_installed

from intellibricks.llms.util import (
    get_parts_llm_described_text,
    get_parts_raw_text,
    segments_to_srt,
)

from .constants import FinishReason, Language

if TYPE_CHECKING:
    from anthropic.types.content_block_param import ContentBlockParam
    from cerebras.cloud.sdk.types.chat.completion_create_params import (
        Message as CerebrasMessage,
    )
    from cerebras.cloud.sdk.types.chat.completion_create_params import (
        MessageUserMessageRequestContentUnionMember1,
    )
    from cerebras.cloud.sdk.types.chat.completion_create_params import (
        ToolFunctionTyped as CerebrasFunctionDefinition,
    )
    from cerebras.cloud.sdk.types.chat.completion_create_params import (
        ToolTyped as CerebrasTool,
    )
    from google.genai.types import Content as GenaiContent
    from google.genai.types import FunctionDeclaration as GenAIFunctionDeclaration
    from google.genai.types import Part as GenAIPart
    from google.genai.types import Tool as GenAITool
    from groq.types.chat.chat_completion_content_part_param import (
        ChatCompletionContentPartParam as GroqChatCompletionContentPartParam,
    )
    from groq.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam as GroqChatCompletionMessageParam,
    )
    from groq.types.chat.chat_completion_message_tool_call_param import (
        Function as GroqCalledFunction,
    )
    from groq.types.chat.chat_completion_tool_param import (
        ChatCompletionToolParam as GroqTool,
    )
    from groq.types.shared_params.function_definition import (
        FunctionDefinition as GroqFunctionDefinition,
    )
    from openai.types.chat.chat_completion_content_part_param import (
        ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
    )
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from openai.types.chat.chat_completion_message_tool_call_param import (
        Function as OpenAICalledFunction,
    )
    from openai.types.chat.chat_completion_tool_param import (
        ChatCompletionToolParam as OpenAITool,
    )
    from openai.types.shared_params.function_definition import (
        FunctionDefinition as OpenAIFunctionDefinition,
    )
    from PIL.Image import Image


_P = TypeVar("_P", bound="Part")
M = TypeVar(
    "M", bound="Message"
)  # NOTE: if I used "type M = Message" the "type[M]" would not work in the function signature

T = TypeVar("T", default="RawResponse")
R = TypeVar("R", default=Any)
_T = TypeVar("_T", default=str)
_FP = TypeVar("_FP", bound="FilePart")

GenAIModelType: TypeAlias = Literal[
    "google/genai/gemini-1.5-flash",
    "google/genai/gemini-1.5-flash-8b",
    "google/genai/gemini-1.5-flash-001",
    "google/genai/gemini-1.5-flash-002",
    "google/genai/gemini-1.5-pro",
    "google/genai/gemini-1.5-pro-001",
    "google/genai/gemini-1.0-pro-002",
    "google/genai/gemini-1.5-pro-002",
    "google/genai/gemini-flash-experimental",
    "google/genai/gemini-pro-experimental",
    "google/genai/gemini-2.0-flash-exp",
]

VertexAIModelType: TypeAlias = Literal[
    "google/vertexai/gemini-2.0-flash-exp",
    "google/vertexai/gemini-1.5-flash",
    "google/vertexai/gemini-1.5-flash-8b",
    "google/vertexai/gemini-1.5-flash-001",
    "google/vertexai/gemini-1.5-flash-002",
    "google/vertexai/gemini-1.5-pro",
    "google/vertexai/gemini-1.5-pro-001",
    "google/vertexai/gemini-1.0-pro-002",
    "google/vertexai/gemini-1.5-pro-002",
    "google/vertexai/gemini-flash-experimental",
    "google/vertexai/gemini-pro-experimental",
]

GoogleModelType: TypeAlias = Literal[
    GenAIModelType,
    VertexAIModelType,
]

OpenAIModelType: TypeAlias = Literal[
    "openai/api/o1",
    "openai/api/o1-2024-12-17",
    "openai/api/o1-preview",
    "openai/api/o1-preview-2024-09-12",
    "openai/api/o1-mini",
    "openai/api/o1-mini-2024-09-12",
    "openai/api/gpt-4o",
    "openai/api/gpt-4o-2024-11-20",
    "openai/api/gpt-4o-2024-08-06",
    "openai/api/gpt-4o-2024-05-13",
    "openai/api/gpt-4o-audio-preview",
    "openai/api/gpt-4o-audio-preview-2024-10-01",
    "openai/api/gpt-4o-audio-preview-2024-12-17",
    "openai/api/gpt-4o-mini-audio-preview",
    "openai/api/gpt-4o-mini-audio-preview-2024-12-17",
    "openai/api/chatgpt-4o-latest",
    "openai/api/gpt-4o-mini",
    "openai/api/gpt-4o-mini-2024-07-18",
    "openai/api/gpt-4-turbo",
    "openai/api/gpt-4-turbo-2024-04-09",
    "openai/api/gpt-4-0125-preview",
    "openai/api/gpt-4-turbo-preview",
    "openai/api/gpt-4-1106-preview",
    "openai/api/gpt-4-vision-preview",
    "openai/api/gpt-4",
    "openai/api/gpt-4-0314",
    "openai/api/gpt-4-0613",
    "openai/api/gpt-4-32k",
    "openai/api/gpt-4-32k-0314",
    "openai/api/gpt-4-32k-0613",
    "openai/api/gpt-3.5-turbo",
    "openai/api/gpt-3.5-turbo-16k",
    "openai/api/gpt-3.5-turbo-0301",
    "openai/api/gpt-3.5-turbo-0613",
    "openai/api/gpt-3.5-turbo-1106",
    "openai/api/gpt-3.5-turbo-0125",
    "openai/api/gpt-3.5-turbo-16k-0613",
]

GroqModelType: TypeAlias = Literal[
    "groq/api/gemma2-9b-it",
    "groq/api/llama3-groq-70b-8192-tool-use-preview",
    "groq/api/llama3-groq-8b-8192-tool-use-preview",
    "groq/api/llama-3.1-70b-specdec",
    "groq/api/llama-3.2-1b-preview",
    "groq/api/llama-3.2-3b-preview",
    "groq/api/llama-3.2-11b-vision-preview",
    "groq/api/llama-3.2-90b-vision-preview",
    "groq/api/llama-3.3-70b-specdec",
    "groq/api/llama-3.3-70b-versatile",
    "groq/api/llama-3.1-8b-instant",
    "groq/api/llama-guard-3-8b",
    "groq/api/llama3-70b-8192",
    "groq/api/llama3-8b-8192",
    "groq/api/mixtral-8x7b-32768",
]

CerebrasModelType: TypeAlias = Literal[
    "cerebras/api/llama3.1-8b",
    "cerebras/api/llama3.1-70b",
    "cerebras/api/llama-3.3-70b",
]

DeepInfraModelType: TypeAlias = Literal[
    "deepinfra/api/meta-llama/Llama-3.3-70B-Instruct",
    "deepinfra/api/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepinfra/api/meta-llama/Meta-Llama-3.1-70B-Instruct",
    "deepinfra/api/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepinfra/api/meta-llama/Meta-Llama-3.1-405B-Instruct",
    "deepinfra/api/Qwen/QwQ-32B-Preview",
    "deepinfra/api/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "deepinfra/api/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "deepinfra/api/Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepinfra/api/nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "deepinfra/api/Qwen/Qwen2.5-72B-Instruct",
    "deepinfra/api/meta-llama/Llama-3.2-90B-Vision-Instruct",
    "deepinfra/api/meta-llama/Llama-3.2-11B-Vision-Instruct",
    "deepinfra/api/microsoft/WizardLM-2-8x22B",
    "deepinfra/api/01-ai/Yi-34B-Chat",
    "deepinfra/api/Austism/chronos-hermes-13b-v2",
    "deepinfra/api/Gryphe/MythoMax-L2-13b",
    "deepinfra/api/Gryphe/MythoMax-L2-13b-turbo",
    "deepinfra/api/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
    "deepinfra/api/NousResearch/Hermes-3-Llama-3.1-405B",
    "deepinfra/api/Phind/Phind-CodeLlama-34B-v2",
    "deepinfra/api/Qwen/QVQ-72B-Preview",
    "deepinfra/api/Qwen/Qwen2-72B-Instruct",
    "deepinfra/api/Qwen/Qwen2-7B-Instruct",
    "deepinfra/api/Qwen/Qwen2.5-7B-Instruct",
    "deepinfra/api/Qwen/Qwen2.5-Coder-7B",
    "deepinfra/api/Sao10K/L3-70B-Euryale-v2.1",
    "deepinfra/api/Sao10K/L3-8B-Lunaris-v1",
    "deepinfra/api/Sao10K/L3.1-70B-Euryale-v2.2",
    "deepinfra/api/bigcode/starcoder2-15b",
    "deepinfra/api/bigcode/starcoder2-15b-instruct-v0.1",
    "deepinfra/api/codellama/CodeLlama-34b-Instruct-hf",
    "deepinfra/api/codellama/CodeLlama-70b-Instruct-hf",
    "deepinfra/api/cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    "deepinfra/api/cognitivecomputations/dolphin-2.9.1-llama-3-70b",
    "deepinfra/api/databricks/dbrx-instruct",
    "deepinfra/api/airoboros-70b",
    "deepinfra/api/google/codegemma-7b-it",
    "deepinfra/api/google/gemma-1.1-7b-it",
    "deepinfra/api/google/gemma-2-27b-it",
    "deepinfra/api/google/gemma-2-9b-it",
    "deepinfra/api/lizpreciatior/lzlv_70b_fp16_hf",
    "deepinfra/api/mattshumer/Reflection-Llama-3.1-70B",
    "deepinfra/api/meta-llama/Llama-2-13b-chat-hf",
    "deepinfra/api/meta-llama/Llama-2-70b-chat-hf",
    "deepinfra/api/meta-llama/Llama-2-7b-chat-hf",
    "deepinfra/api/meta-llama/Llama-3.2-1B-Instruct",
    "deepinfra/api/meta-llama/Llama-3.2-3B-Instruct",
    "deepinfra/api/meta-llama/Meta-Llama-3-70B-Instruct",
    "deepinfra/api/meta-llama/Meta-Llama-3-8B-Instruct",
    "deepinfra/api/microsoft/Phi-3-medium-4k-instruct",
    "deepinfra/api/microsoft/WizardLM-2-7B",
    "deepinfra/api/mistralai/Mistral-7B-Instruct-v0.1",
    "deepinfra/api/mistralai/Mistral-7B-Instruct-v0.2",
    "deepinfra/api/mistralai/Mistral-7B-Instruct-v0.3",
    "deepinfra/api/mistralai/Mistral-Nemo-Instruct-2407",
    "deepinfra/api/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "deepinfra/api/mistralai/Mixtral-8x22B-v0.1",
    "deepinfra/api/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepinfra/api/nvidia/Nemotron-4-340B-Instruct",
    "deepinfra/api/openbmb/MiniCPM-Llama3-V-2_5",
    "deepinfra/api/openchat/openchat-3.6-8b",
    "deepinfra/api/openchat/openchat_3.5",
]

AIModel: TypeAlias = Literal[
    # -- Google --
    GoogleModelType,
    # -- Cerebras --
    CerebrasModelType,
    # -- OpenAI --
    OpenAIModelType,
    # -- Groq --
    GroqModelType,
    # -- DeepInfra --
    DeepInfraModelType,
]

GroqTranscriptionModelType: TypeAlias = Literal[
    "groq/api/whisper-large-v3-turbo",
    "groq/api/distil-whisper-large-v3-en",
    "groq/api/whisper-large-v3",
]

OpenAITranscriptionsModelType: TypeAlias = Literal["openai/api/whisper-1"]

TranscriptionModelType: TypeAlias = Literal[
    GroqTranscriptionModelType, OpenAITranscriptionsModelType
]


warning_logger = log.create_logger(__name__, level=logging.DEBUG)


class GenerationConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration parameters for controlling LLM generation behavior.

    Attributes:
        n: Number of completions to generate. Default 1 if not specified.
        temperature: Controls randomness (0.0=deterministic, 1.0=creative).
        max_tokens: Maximum number of tokens to generate in response.
        max_retries: Maximum retry attempts for failed generations (1-5).
        top_p: Nucleus sampling probability threshold.
        top_k: Limit sampling to top K probable tokens.
        stop_sequences: List of strings that stop generation when encountered.
        cache_config: Configuration for response caching.
        trace_params: Parameters for request tracing and analytics.
        tools: List of functions available for tool calling.
        general_web_search: Whether to enable web search augmentation.
        language: Output language for localization. Defaults to English.
        timeout: Maximum time in seconds to wait for completion.

    Example:
        >>> config = GenerationConfig(
        ...     temperature=0.7,
        ...     max_tokens=500,
        ...     tools=[search_web, calculate]
        ... )
    """

    n: Annotated[
        Optional[int],
        msgspec.Meta(
            title="Number generations",
            description=("Describes how many completions to generate."),
        ),
    ] = msgspec.field(default=None)
    """Describes how many completions to generate."""

    temperature: Annotated[
        Optional[float],
        msgspec.Meta(
            title="Temperature",
            description=(
                "Controls the randomness of the generated completions. Lower temperatures make the model more deterministic, "
                "while higher temperatures make the model more creative."
            ),
        ),
    ] = msgspec.field(default=None)

    max_tokens: Annotated[
        Optional[int],
        msgspec.Meta(
            title="Maximum tokens",
            description=(
                "The maximum number of tokens to generate in each completion. "
                "This can be used to control the length of the generated completions."
            ),
        ),
    ] = msgspec.field(default=None)

    max_retries: Annotated[
        Optional[Literal[1, 2, 3, 4, 5]],
        msgspec.Meta(
            title="Maximum retries",
            description=(
                "The maximum number of times to retry generating completions if the model returns an error. "
                "This can be used to handle transient errors."
            ),
        ),
    ] = msgspec.field(default=None)

    top_p: Annotated[
        Optional[float],
        msgspec.Meta(
            title="Top-p",
            description=(
                "A value between 0 and 1 that controls the diversity of the generated completions. "
                "A lower value will result in more common completions, while a higher value will result in more diverse completions."
            ),
        ),
    ] = msgspec.field(default=None)

    top_k: Annotated[
        Optional[int],
        msgspec.Meta(
            title="Top-k",
            description=(
                "An integer that controls the diversity of the generated completions. "
                "A lower value will result in more common completions, while a higher value will result in more diverse completions."
            ),
        ),
    ] = msgspec.field(default=None)

    stop_sequences: Annotated[
        Optional[Sequence[str]],
        msgspec.Meta(
            title="Stop sequences",
            description=(
                "A list of strings that the model will use to determine when to stop generating completions. "
                "This can be used to generate completions that end at specific points in the text."
            ),
        ),
    ] = msgspec.field(default=None)

    cache_config: Annotated[
        Optional[CacheConfig],
        msgspec.Meta(
            title="Cache configuration",
            description=(
                "Specifies the configuration for caching completions. "
                "This can be used to cache completions and avoid redundant generation requests."
            ),
        ),
    ] = msgspec.field(default=None)

    trace_params: Annotated[
        Optional[TraceParams],
        msgspec.Meta(
            title="Trace parameters",
            description=(
                "Specifies the parameters for tracing completions. "
                "This can be used to trace completions and analyze the model's behavior."
            ),
        ),
    ] = msgspec.field(default=None)

    tools: Annotated[
        Optional[Sequence[Callable[..., Any]]],
        msgspec.Meta(
            title="Tools",
            description=(
                "A list of functions that the model can call during completion generation. "
                "This can be used to provide additional context or functionality to the model."
            ),
        ),
    ] = msgspec.field(default=None)

    general_web_search: Annotated[
        Optional[bool],
        msgspec.Meta(
            title="General web search",
            description=(
                "Specifies whether to enable general web search during completion generation. "
                "This can be used to provide additional context to the model."
            ),
        ),
    ] = msgspec.field(default=None)

    language: Annotated[
        Language,
        msgspec.Meta(
            title="Language",
            description=(
                "Specifies the language of the generated completions. "
                "This can be used to control the language model used by the model."
            ),
        ),
    ] = msgspec.field(default=Language.ENGLISH)

    timeout: Annotated[
        Optional[float],
        msgspec.Meta(
            title="Timeout",
            description=(
                "The maximum time to wait for the model to generate completions. "
                "This can be used to control the time taken to generate completions."
            ),
        ),
    ] = msgspec.field(default=None)


class RawResponse(msgspec.Struct, frozen=True):
    """Null object for the response from the model."""

    def __bool__(self) -> Literal[False]:
        return False


class TraceParams(TypedDict, total=False):
    """Parameters for tracking and analyzing LLM interactions.

    Attributes:
        name: Unique identifier for the trace
        input: Input parameters for the traced operation
        output: Result of the traced operation
        user_id: ID of user initiating the request
        session_id: Grouping identifier for related traces
        version: Version of the trace. Can be used for tracking changes
        release: Deployment release identifier
        metadata: Custom JSON-serializable metadata
        tags: Categorization labels for filtering
        public: Visibility flag for trace data

    Example:
        >>> trace = TraceParams(
        ...     name="customer_support",
        ...     tags=["urgent", "billing"]
        ... )
    """

    name: Optional[str]
    input: Optional[Any]
    output: Optional[Any]
    user_id: Optional[str]
    session_id: Optional[str]
    version: Optional[str]
    release: Optional[str]
    metadata: Optional[Any]
    tags: Optional[list[str]]
    public: Optional[bool]


class CacheConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for response caching mechanisms.

    Attributes:
        ttl: Time-to-live duration for cached entries
        cache_key: Unique identifier for cache lookups

    Example:
        >>> cache = CacheConfig(
        ...     ttl=datetime.timedelta(minutes=30),
        ...     cache_key="user_preferences"
        ... )
    """

    ttl: Annotated[
        datetime.timedelta,
        msgspec.Meta(
            title="Time-To-Live (TTL)",
            description=(
                "Specifies the time-to-live for cache entries. This can be defined either as an "
                "integer representing seconds or as a `datetime.timedelta` object for finer granularity. "
                "The TTL determines how long a cached system prompt remains valid before it is refreshed or invalidated."
            ),
        ),
    ] = msgspec.field(default_factory=lambda: datetime.timedelta(seconds=0))
    """Specifies the time-to-live for cache entries.

    The TTL can be set as an integer (in seconds) or as a `datetime.timedelta` object for finer granularity.
    This value determines how long a cached system prompt remains valid before it needs to be refreshed or invalidated.

    **Example:**
        >>> cache_config = CacheConfig(ttl=60)
        >>> print(cache_config.ttl)
        60
    """

    cache_key: Annotated[
        str,
        msgspec.Meta(
            title="Cache Key",
            description=(
                "Defines the key used to identify cached messages. This key is essential for storing and retrieving "
                "cache entries consistently. It should be unique enough to prevent collisions but also meaningful "
                "to facilitate easy management of cached data."
            ),
        ),
    ] = msgspec.field(default_factory=lambda: "default")
    """Defines the key used to identify cached system prompts.

    The `cache_key` is crucial for storing and retrieving cache entries consistently. It should be unique
    enough to prevent collisions with other cached data but also meaningful to facilitate easy management
    of cached entries.

    **Example:**
        >>> cache_config = CacheConfig(cache_key='user_session_prompt')
        >>> print(cache_config.cache_key)
        'user_session_prompt'
    """


"""
########     ###    ########  ########  ######
##     ##   ## ##   ##     ##    ##    ##    ##
##     ##  ##   ##  ##     ##    ##    ##
########  ##     ## ########     ##     ######
##        ######### ##   ##      ##          ##
##        ##     ## ##    ##     ##    ##    ##
##        ##     ## ##     ##    ##     ######
"""


@dp.AbstractFactory
class Part(msgspec.Struct, tag_field="type", frozen=True):
    """Base class for multimodal message components.

    Provides factory methods for creating specific part types and conversion
    methods to various provider formats. Direct subclasses should implement
    format-specific conversion logic.

    Example:
        >>> Part.from_text("Hello world")
        TextPart(text='Hello world')
    """

    @classmethod
    def from_text(cls, text: str) -> TextPart:
        return TextPart(text=text)

    @classmethod
    def from_image(cls, image: Image) -> ImageFilePart:
        ensure_module_installed("PIL.Image", "pillow")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_str = base64.b64encode(buffered.getvalue()).decode(
            "utf-8", errors="replace"
        )

        image_bytes = image_str.encode("utf-8")
        return ImageFilePart(
            data=image_str.encode("utf-8"),
            mime_type=bytes_to_mime(image_bytes),
        )

    @classmethod
    def from_openai_part(
        cls, openai_part: OpenAIChatCompletionContentPartParam
    ) -> Part:
        from intellibricks.llms.util import is_url

        match openai_part["type"]:
            case "text":
                return TextPart(text=openai_part["text"])  # type: ignore
            case "image_url":
                url_or_base_64 = openai_part["image_url"]["url"]  # type: ignore
                if is_url(url_or_base_64):
                    return ImageFilePart.from_url(url_or_base_64)

                image_url_bytes = url_or_base_64.encode("utf-8")
                return ImageFilePart(
                    data=image_url_bytes,
                    mime_type=bytes_to_mime(image_url_bytes),
                )
            case "input_audio":
                input_audio_bytes = base64.b64decode(openai_part["input_audio"]["data"])  # type: ignore
                return AudioFilePart(
                    data=input_audio_bytes,  # type: ignore
                    mime_type=f"audio/{openai_part['input_audio']['format']}",  # type: ignore
                )

    @classmethod
    def from_google_part(cls, google_part: GenAIPart) -> PartType:
        from google.genai import types

        # Check if it's a text part
        if google_part.text is not None:
            return TextPart(text=google_part.text)

        # Check if it's a file-based part
        file_data: Optional[types.FileData] = google_part.file_data
        if file_data is not None:
            if file_data.mime_type is None:
                raise ValueError("MIME type is required for file parts.")

            mime_type = file_data.mime_type.lower()

            if not file_data.file_uri:
                raise ValueError(
                    "file_data provided with no file_uri. Can't create FilePart."
                )

            if mime_type.startswith("image/"):
                return ImageFilePart.from_url(file_data.file_uri)
            elif mime_type.startswith("audio/"):
                return AudioFilePart.from_url(file_data.file_uri)
            elif mime_type.startswith("video/"):
                return VideoFilePart.from_url(file_data.file_uri)
            else:
                warning_logger.warning(
                    f"Unknown file type: {mime_type}, falling back to ImageFilePart."
                )
                return ImageFilePart.from_url(file_data.file_uri)

        # Check inline data
        inline_data: Optional[types.Blob] = google_part.inline_data
        if inline_data is not None:
            if inline_data.mime_type is None:
                raise ValueError("MIME type is required for inline_data")
            mime_type = inline_data.mime_type.lower()
            data = inline_data.data
            if data is None:
                raise ValueError("Data is required for inline_data")

            if mime_type.startswith("image/"):
                return ImageFilePart(data=data, mime_type=mime_type)
            elif mime_type.startswith("audio/"):
                return AudioFilePart(data=data, mime_type=mime_type)
            elif mime_type.startswith("video/"):
                return VideoFilePart(data=data, mime_type=mime_type)
            else:
                return ImageFilePart(data=data, mime_type=mime_type)

        function_call = google_part.function_call
        # Check if it's a function call part
        if function_call is not None:
            function_name = function_call.name
            if function_name is None:
                raise ValueError(
                    "The name of the function is None. Google did not return the name of it."
                )

            function_arguments = function_call.args
            if function_arguments is None:
                raise ValueError("The arguments of the function are None.")

            return ToolCallPart(
                function_name=function_name, arguments=function_arguments
            )

        # Check if it's a function response part
        if google_part.function_response is not None:
            # Function responses are not currently implemented
            raise NotImplementedError(
                "Function responses from google part are not yet implemented by Intellibricks."
            )

        # Check if it's executable code part
        if google_part.executable_code is not None:
            # Executable code is not currently implemented
            raise NotImplementedError(
                "Executable code from google part are not yet implemented by Intellibricks."
            )

        # Check if it's code execution result part
        if google_part.code_execution_result is not None:
            # Code execution result is not currently implemented
            raise NotImplementedError(
                "Code execution result from google part are not yet implemented by Intellibricks."
            )

        # Check video metadata only part (without content)
        if google_part.video_metadata is not None:
            # Video metadata alone doesn't map to a known Part type.
            raise NotImplementedError(
                "Video metadata from google part are not yet supported."
            )

        # If we get here, we can't determine a known part type
        raise ValueError("Cannot determine the part type from the given GenAIPart.")

    @classmethod
    def from_anthropic_part(cls, part: dict[str, Any]) -> Part:
        ensure_module_installed("anthropic.types.text_block_param", "anthropic")
        part_type: Optional[
            Literal["text", "image", "tool_use", "tool_result", "document"]
        ] = part.get("type", None)

        if part_type is None:
            raise ValueError("Couldn't find the part type")

        match part_type:
            case "text":
                return TextPart(text=part["text"])
            case "image":
                return ImageFilePart(
                    data=part["source"]["data"],
                    mime_type=part["source"]["media_type"],
                )
            case _:
                raise ValueError("Not supported yet.")

    # @abstractmethod
    def to_anthropic_part(self) -> ContentBlockParam:
        raise NotImplementedError

    # @abstractmethod
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        raise NotImplementedError

    # @abstractmethod
    def to_google_part(self) -> GenAIPart:
        raise NotImplementedError

    # @abstractmethod
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        raise NotImplementedError

    # @abstractmethod
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        raise NotImplementedError

    # @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    # @abstractmethod
    def to_llm_described_text(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_dict(cls: Type[_P], _d: dict[str, Any], /) -> _P:
        return cls(**_d)


class WebsitePart(Part, frozen=True, tag="website"):
    """Represents a website part in a multi-content message.

    Attributes:
        url: The URL of the website

    Example:
        >>> WebsitePart(url="https://www.example.com")
    """

    url: str

    def __post_init__(self) -> None:
        from intellibricks.llms.util import is_url

        if not is_url(self.url):
            raise ValueError(f"Invalid URL ({self.url})")

    def get_md_contents(self, timeout: float = 5.0) -> str:
        """Get the contents of the website and convert HTML to Markdown."""
        import requests

        response = requests.get(self.url, timeout=timeout)
        response.raise_for_status()
        html_text: str = response.text
        return f"TODO: {html_text}"

    @override
    def to_llm_described_text(self) -> str:
        return (
            f"<|website_part|>\n"
            f"URL: {self.url}\n"
            f"Contents: {self.get_md_contents()}\n"
            f"<|end_website_part|>"
        )

    @ensure_module_installed("anthropic", "anthropic")
    @override
    def to_anthropic_part(self) -> ContentBlockParam:
        return TextPart(text=self.get_md_contents()).to_anthropic_part()

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        return TextPart(text=self.get_md_contents()).to_openai_part()

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        return TextPart(text=self.get_md_contents()).to_groq_part()

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        return TextPart(text=self.get_md_contents()).to_google_part()

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestContentUnionMember1Typed,
        )

        return MessageUserMessageRequestContentUnionMember1Typed(
            text=self.get_md_contents(), type="text"
        )

    @override
    def __str__(self) -> str:
        return self.get_md_contents()


class TextPart(Part, frozen=True, tag="text"):
    """Text content component for messages.

    Attributes:
        text: String content of the text part

    Example:
        >>> TextPart(text="Welcome to our service!")
    """

    text: str

    @ensure_module_installed("anthropic", "anthropic")
    @override
    def to_anthropic_part(self) -> ContentBlockParam:
        ensure_module_installed("anthropic.types.text_block_param", "anthropic")
        from anthropic.types.text_block_param import TextBlockParam

        return TextBlockParam(text=self.text, type="text")

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        from openai.types.chat.chat_completion_content_part_text_param import (
            ChatCompletionContentPartTextParam,
        )

        return ChatCompletionContentPartTextParam(text=self.text, type="text")

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        from groq.types.chat.chat_completion_content_part_text_param import (
            ChatCompletionContentPartTextParam,
        )

        return ChatCompletionContentPartTextParam(text=self.text, type="text")

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        from google.genai.types import Part as GenAIPart

        return GenAIPart.from_text(text=self.text)

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestContentUnionMember1Typed,
        )

        return MessageUserMessageRequestContentUnionMember1Typed(
            text=self.text, type="text"
        )

    @override
    def to_llm_described_text(self) -> str:
        return f"<|text_part|>\nText: {self.text}\n<|end_text_part|>"

    @override
    def __str__(self) -> str:
        return self.text


class ToolResponsePart(Part, frozen=True, tag="tool_response"):
    """Represents a tool part in a multi-content message.

    Attributes:
        tool_name: The name of the tool
        tool_call_id: The ID of the tool call
        tool_response: The response from the tool

    Example:
        >>> ToolResponsePart(tool_name="calculator", tool_call_id="123", tool_response="5")
    """

    tool_name: str
    """The name of the tool."""

    tool_call_id: str
    """The ID of the tool call."""

    tool_response: str
    """The tool response."""

    @override
    def to_anthropic_part(self) -> ContentBlockParam:
        raise NotImplementedError("Intellibricks didn't implement this yet.")

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        text = str(self)
        return TextPart(text).to_openai_part()

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        text = str(self)
        return TextPart(text).to_groq_part()

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        from google.genai.types import Part as GenAIPart

        return GenAIPart.from_function_response(
            name=self.tool_name, response={"output": self.tool_response}
        )

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestContentUnionMember1Typed,
        )

        return MessageUserMessageRequestContentUnionMember1Typed(
            text=self.to_llm_described_text(), type="text"
        )

    @override
    def to_llm_described_text(self) -> str:
        return (
            f"<|tool_response_part|>\n"
            f"Tool: {self.tool_name}\n"
            f"Response: {self.tool_response}\n"
            f"<|end_tool_response_part|>"
        )

    @override
    def __str__(self) -> str:
        return (
            f"<|tool_call_part|>\n"
            f"Tool: {self.tool_name}\n\n"
            f"Returned: {self.tool_response}\n"
            f"<|end_tool_call_part|>"
        )


@dp.AbstractFactory
class FilePart(Part, frozen=True, tag="file"):
    """Base class for file-based parts in multimodal messages.
    Cannot be instantiated directly.

    Attributes:
        data: bytes file data
        mime_type: The MIME type of the file
        metadata: Optional metadata for the file

    Example:
        >>> ImageFilePart(data=b"...", mime_type=MimeType.image_jpeg)
    """

    data: bytes
    """bytes file data."""

    mime_type: str
    """The MIME type of the file."""

    metadata: Optional[dict[str | Literal["url"], Any]] = None

    @classmethod
    def from_url(cls: type[_FP], url: str) -> _FP:
        """Create a FilePart from a URL by downloading the file and detecting its type."""
        import requests
        from requests.exceptions import RequestException

        # Determine the file extension from the URL
        try:
            extension = find_extension(url=url)
        except ValueError as e:
            raise ValueError(
                f"Could not determine file extension from URL: {url}"
            ) from e

        # Convert the extension to its corresponding MIME type
        try:
            mime_type_str = ext_to_mime(extension)
        except ValueError as e:
            raise ValueError(
                f"Could not determine MIME type for extension {extension}"
            ) from e

        # Validate that the MIME type matches the subclass type
        if cls is not FilePart:  # When called from concrete subclass
            expected_prefix = cls.__name__.replace("FilePart", "").lower() + "/"
            if not mime_type_str.startswith(expected_prefix):
                raise ValueError(
                    f"URL contains {mime_type_str} but called from {cls.__name__}. "
                    f"Use FilePart.from_url() for automatic type detection."
                )

        # Download the file content
        try:
            response = requests.get(url)
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"Failed to download file from URL: {url}") from e

        data = response.content

        # Instantiate the appropriate subclass
        return cls(data=data, mime_type=mime_type_str, metadata={"url": url})


class VideoFilePart(FilePart, frozen=True, tag="video"):
    """Video content component for messages.

    Attributes:
        data: bytes file data
        mime_type: The MIME type of the file
        metadata: Optional metadata for the file

    Example:
        >>> VideoFilePart(data=b"...", mime_type=MimeType.video_mp4)
    """

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        from google.genai.types import Part as GenAIPart

        return GenAIPart.from_bytes(data=self.data, mime_type=self.mime_type)

    @ensure_module_installed("anthropic", "anthropic")
    @override
    def to_anthropic_part(self) -> ContentBlockParam:
        raise NotImplementedError(
            "None of the Anthropic models supports video understanding at the moment."
        )

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        raise NotImplementedError(
            "None of the OpenAI models supports video understanding at the moment."
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        raise NotImplementedError(
            "None of the Groq models supports video understanding at the moment."
        )

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestContentUnionMember1Typed,
        )

        return MessageUserMessageRequestContentUnionMember1Typed(
            text=self.to_llm_described_text(), type="text"
        )

    @override
    def to_llm_described_text(self) -> str:
        return (
            f"<|video_part|>\n"
            f"MIME type: {self.mime_type}\n"
            f"Data length: {len(self.data) if self.data else 0}\n"
            f"<|end_video_part|>"
        )

    @override
    def __str__(self) -> str:
        return f"VideoFilePart({self.mime_type})"


class AudioFilePart(FilePart, frozen=True, tag="audio"):
    """Audio content component for messages.

    Attributes:
        data: bytes file data
        mime_type: The MIME type of the file
        metadata: Optional metadata for the file

    Example:
        >>> AudioFilePart(data=b"...", mime_type=MimeType.audio_mp3)
    """

    @override
    def to_anthropic_part(self) -> ContentBlockParam:
        raise NotImplementedError(
            "None of the Anthropic models supports audio understanding at the moment."
        )

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        """
        Returns the Input Audio part in the OpenAI format.
        In this case, OpenAI uses typed dicts, so it
        will just return a dict.
        """
        from openai.types.chat.chat_completion_content_part_input_audio_param import (
            ChatCompletionContentPartInputAudioParam,
            InputAudio,
        )

        if not self.data:
            raise ValueError("Audio data (bytes) is required.")

        return ChatCompletionContentPartInputAudioParam(
            input_audio=InputAudio(
                data=base64.b64encode(self.data).decode("utf-8", errors="replace"),
                format=cast(Literal["mp3"], self.mime_type.split("/")[1]),
            ),
            type="input_audio",
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        raise NotImplementedError(
            "None of the Groq models supports audio understanding at the moment."
        )

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        from google.genai.types import Part as GenAIPart

        return GenAIPart.from_bytes(data=self.data, mime_type=self.mime_type)

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestContentUnionMember1Typed,
        )

        return MessageUserMessageRequestContentUnionMember1Typed(
            text=self.to_llm_described_text(), type="text"
        )

    @override
    def to_llm_described_text(self) -> str:
        return (
            f"<|audio_part|>\n"
            f"MIME type: {self.mime_type}\n"
            f"Data length: {len(self.data) if self.data else 0}\n"
            f"<|end_audio_part|>"
        )

    @override
    def __str__(self) -> str:
        return f"AudioFilePart({self.mime_type})"


class ImageFilePart(FilePart, frozen=True, tag="image"):
    """Image content component for messages.

    Attributes:
        data: bytes file data
        mime_type: The MIME type of the file
        metadata: Optional metadata for the file

    Example:
        >>> ImageFilePart(data=b"...", mime_type=MimeType.image_jpeg)
    """

    def to_png(self) -> "ImageFilePart":
        """Convert the image to PNG format."""
        from PIL import Image

        image = Image.open(BytesIO(self.data))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return ImageFilePart(data=buffered.getvalue(), mime_type="image/png")

    def to_jpeg(self) -> "ImageFilePart":
        """Convert the image to JPEG format."""
        from PIL import Image

        image = Image.open(BytesIO(self.data))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return ImageFilePart(data=buffered.getvalue(), mime_type="image/jpeg")

    def is_png(self) -> bool:
        """Check if the image is in PNG format."""
        return self.mime_type == "image/png"

    def is_jpeg(self) -> bool:
        """Check if the image is in JPEG format."""
        return self.mime_type == "image/jpeg" or self.mime_type == "image/jpg"

    def is_gif(self) -> bool:
        """Check if the image is in GIF format."""
        return self.mime_type == "image/gif"

    def is_webp(self) -> bool:
        """Check if the image is in WebP format."""
        return self.mime_type == "image/webp"

    @ensure_module_installed("anthropic.types.image_block_param", "anthropic")
    @override
    def to_anthropic_part(self) -> ContentBlockParam:
        ensure_module_installed("anthropic.types.image_block_param", "anthropic")
        from anthropic.types.image_block_param import ImageBlockParam, Source

        is_supported = any(
            [self.is_png(), self.is_jpeg(), self.is_gif(), self.is_webp()]
        )

        if not self.data:
            raise ValueError("Image data (bytes) is required.")

        return ImageBlockParam(
            source=Source(
                data=base64.b64encode(
                    self.data if is_supported else self.to_png().data
                ).decode("utf-8", errors="replace"),
                media_type=cast(
                    Literal["image/jpeg", "image/png", "image/gif", "image/webp"],
                    self.mime_type,
                ),
                type="base64",
            ),
            type="image",
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        from groq.types.chat.chat_completion_content_part_image_param import (
            ChatCompletionContentPartImageParam,
            ImageURL,
        )

        is_supported = self.is_png()

        return ChatCompletionContentPartImageParam(
            image_url=ImageURL(
                url=base64.b64encode(
                    self.data if is_supported else self.to_png().data
                ).decode("utf-8", errors="replace")
            ),
            type="image_url",
        )

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        from openai.types.chat.chat_completion_content_part_image_param import (
            ChatCompletionContentPartImageParam,
            ImageURL,
        )

        is_supported = any([self.is_png(), self.is_jpeg()])

        data = self.data if is_supported else self.to_png().data

        return ChatCompletionContentPartImageParam(
            image_url=ImageURL(
                url=data.decode("utf-8", errors="replace"), detail="auto"
            ),
            type="image_url",
        )

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        from google.genai.types import Part as GenAIPart

        is_supported = any([self.is_png(), self.is_jpeg()])

        return GenAIPart.from_bytes(
            data=self.data if is_supported else self.to_png().data,
            mime_type=self.mime_type,
        )

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestContentUnionMember1Typed,
        )

        return MessageUserMessageRequestContentUnionMember1Typed(
            text=self.to_llm_described_text(), type="text"
        )

    @override
    def to_llm_described_text(self) -> str:
        return (
            f"<|image_part|>\n"
            f"MIME type: {self.mime_type}\n"
            f"Data length: {len(self.data) if self.data else 0}\n"
            f"<|end_image_part|>"
        )

    @override
    def __str__(self) -> str:
        return f"ImageFilePart({self.mime_type})"


class ToolCallPart(Part, frozen=True, tag="tool_call"):
    """Represents a function/method invocation request.

    Attributes:
        function_name: Name of the function to call
        arguments: Keyword arguments for the function call

    Example:
        >>> ToolCallPart(
        ...     function_name="get_weather",
        ...     arguments={"location": "Paris"}
        ... )
    """

    function_name: str
    arguments: dict[str, Any]

    @ensure_module_installed("anthropic", "anthropic")
    @override
    def to_anthropic_part(self) -> ContentBlockParam:
        return TextPart(str(self)).to_anthropic_part()

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        return TextPart(str(self)).to_openai_part()

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        return TextPart(str(self)).to_groq_part()

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        return TextPart(str(self)).to_google_part()

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_part(self) -> MessageUserMessageRequestContentUnionMember1:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestContentUnionMember1Typed,
        )

        return MessageUserMessageRequestContentUnionMember1Typed(
            text=self.to_llm_described_text(), type="text"
        )

    @override
    def to_llm_described_text(self) -> str:
        return (
            f"<|tool_call_part|>\n"
            f"Function Name: {self.function_name}\n"
            f"Arguments: {self.arguments}\n"
            f"<|end_tool_call_part|>"
        )

    @override
    def __str__(self) -> str:
        return f"<|function_call|>\nFunction: {self.function_name}\nArguments: {self.arguments}\n<|end_function_call|>"


PartType: TypeAlias = (
    AudioFilePart
    | VideoFilePart
    | TextPart
    | ImageFilePart
    | ToolCallPart
    | ToolResponsePart
    | WebsitePart
)


class PartFactory:
    """Factory class for creating Part instances from dictionaries.

    Example:
        >>> part_dict = {
        ...     "type": "text",
        ...     "text": "Hello world"
        ... }
        >>> PartFactory.create_from_dict(part_dict)
        TextPart(text='Hello world')
    """

    @staticmethod
    def create_from_dict(part: dict[str, Any]) -> PartType:
        part_type = part.get("type", None)
        if part_type is None:
            raise ValueError("Couldn't find the part type")

        match part_type:
            case "text":
                return TextPart(text=part["text"])

            case "image":
                return ImageFilePart(
                    data=part["source"]["data"],
                    mime_type=part["source"]["media_type"],
                )

            case "audio":
                return AudioFilePart(
                    data=part["source"]["data"],
                    mime_type=part["source"]["media_type"],
                )

            case "video":
                return VideoFilePart(
                    data=part["source"]["data"],
                    mime_type=part["source"]["media_type"],
                )

            case "website":
                return WebsitePart(url=part["url"])

            case "tool_use":
                return ToolCallPart(
                    function_name=part["function_name"],
                    arguments=part["arguments"],
                )

            case "tool_result":
                return ToolResponsePart(
                    tool_name=part["tool_name"],
                    tool_call_id=part["tool_call_id"],
                    tool_response=part["tool_response"],
                )

            case _:
                raise ValueError(f"Unknown part type: {part_type}")


class PartSequece(msgspec.Struct, frozen=True):
    """Represents a sequence of parts in a multimodal message.

    Attributes:
        parts: Sequence of parts

    Example:
        >>> PartSequence(parts=[TextPart(text="Hello"), ImageFilePart(data=b"...")])
    """

    parts: Sequence[PartType]


"""
########  ########   #######  ##     ## ########  ########
##     ## ##     ## ##     ## ###   ### ##     ##    ##
##     ## ##     ## ##     ## #### #### ##     ##    ##
########  ########  ##     ## ## ### ## ########     ##
##        ##   ##   ##     ## ##     ## ##           ##
##        ##    ##  ##     ## ##     ## ##           ##
##        ##     ##  #######  ##     ## ##           ##
"""


class Prompt(msgspec.Struct, frozen=True):
    """Represents a prompt for an Generative AI model.
    Provides a way to compile the prompt with replacement values.

    Attributes:
        content: The content of the prompt

    Example:
        >>> Prompt(content="Hello! How are you?")
        >>> Prompt(content="Hi, my name is {{name}}.")
        >>> Prompt(content="I need help on solving a Python problem.")
    """

    content: Annotated[
        str,
        msgspec.Meta(
            title="Content",
            description="The content of the prompt",
            examples=[
                "Hello! How are you?",
                "I need help on solving a Python problem.",
                "Hi, my name is {{name}}.",
            ],
        ),
    ]

    def compile(self, **replacements: Any) -> Prompt:
        """
        Replace placeholders in the content with provided replacement values.

        Placeholders are in the format {{key}}.

        Args:
            **replacements: Arbitrary keyword arguments corresponding to placeholder keys.

        Returns:
            A string with all placeholders replaced by their respective values.

        Raises:
            KeyError: If a placeholder in the content does not have a corresponding replacement.
        """
        # Regular expression to find all placeholders like {{key}}
        pattern = re.compile(r"\{\{(\w+)\}\}")

        def replace_match(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in replacements:
                return str(replacements[key])
            else:
                raise KeyError(f"Replacement for '{key}' not provided.")

        # Substitute all placeholders with their replacements
        compiled_content = pattern.sub(replace_match, self.content)
        return Prompt(compiled_content)

    def as_string(self) -> str:
        return self.content

    def __str__(self) -> str:
        return self.content


"""
.########..#######...#######..##........######.
....##....##.....##.##.....##.##.......##....##
....##....##.....##.##.....##.##.......##......
....##....##.....##.##.....##.##........######.
....##....##.....##.##.....##.##.............##
....##....##.....##.##.....##.##.......##....##
....##.....#######...#######..########..######.
"""


class Tool(msgspec.Struct, frozen=True):
    """Represents a tool that can be used in a multimodal message.

    Example:
        >>> Tool(name="calculator", description="A simple calculator tool")
    """

    def to_callable(self) -> Callable[..., Any]:
        raise NotImplementedError

    def to_google_tool(self) -> GenAITool:
        from google.genai.types import Tool as GenAITool

        return GenAITool(
            function_declarations=[
                Function.from_callable(self.to_callable()).to_genai_function()
            ]
        )

    @ensure_module_installed("openai", "openai")
    def to_openai_tool(self) -> OpenAITool:
        from openai.types.chat.chat_completion_tool_param import (
            ChatCompletionToolParam as OpenAITool,
        )

        return OpenAITool(
            function=Function.from_callable(self.to_callable()).to_openai_function(),
            type="function",
        )

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    def to_cerebras_tool(self) -> CerebrasTool:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            ToolTyped as CerebrasTool,
        )

        return CerebrasTool(
            function=Function.from_callable(self.to_callable()).to_cerebras_function(),
            type="function",
        )

    def to_groq_tool(self) -> GroqTool:
        from groq.types.chat.chat_completion_tool_param import (
            ChatCompletionToolParam as GroqTool,
        )

        return GroqTool(
            function=Function.from_callable(self.to_callable()).to_groq_function(),
            type="function",
        )

    def to_deepinfra_tool(self) -> OpenAITool:
        return self.to_openai_tool()


type ToolInputType = Tool | Callable[..., Any]


class ToolCall[R = Any](msgspec.Struct, kw_only=True, frozen=True):
    """Represents a call to a tool with arguments.

    Attributes:
        id: The unique identifier of the tool call
        called_function: The function to call with arguments

    Example:
        >>> ToolCall(id="123", called_function=Tool(name="calculator", description="A simple calculator tool"))
    """

    id: str = msgspec.field(default_factory=lambda: str(uuid.uuid4()))
    called_function: CalledFunction[R]

    @overload
    def call(self, force_string_output: Literal[False] = False) -> R: ...

    @overload
    def call(self, force_string_output: Literal[True]) -> str: ...

    def call(self, force_string_output: bool = False) -> R | str:
        """Calls the function with the provided arguments."""
        return (
            self.called_function.call()
            if not force_string_output
            else str(self.called_function.call())
        )

    def to_llm_described_text(self) -> str:
        return (
            f"<|tool_call|>\n"
            f"ToolCall ID: {self.id}\n"
            f"Function: {self.called_function.function.name}\n"
            f"Arguments: {self.called_function.arguments}\n"
            f"<|end_tool_call|>"
        )

    def to_tool_message(self) -> ToolMessage:
        return ToolMessage(
            tool_call_id=self.id,
            contents=[ToolResponsePart.from_text(self.call(force_string_output=True))],
            name=self.called_function.function.name,
        )

    def to_openai_tool_call(self) -> dict[str, Any]:
        from openai.types.chat.chat_completion_message_tool_call_param import (
            ChatCompletionMessageToolCallParam,
            Function,
        )

        arguments: dict[str, Any] = self.called_function.arguments

        return cast(
            dict[str, Any],
            ChatCompletionMessageToolCallParam(
                id=self.id,
                function=Function(
                    name=self.called_function.function.name,
                    arguments=str(arguments),
                ),
                type="function",
            ),
        )


class ToolCallSequence[R = Any](msgspec.Struct, frozen=True):
    """Represents a sequence of tool calls.

    Attributes:
        sequence: Sequence of tool calls

    Example:
        >>> ToolCallSequence(sequence=[ToolCall(called_function=Tool(name="calculator", description="A simple calculator tool"))])
    """

    sequence: Sequence[ToolCall[R]]

    def call_all(self) -> tuple[R, ...]:
        """
        Invoke each tool call in order and return all results
        as a typed tuple matching Ts.
        """

        return tuple(tool.call() for tool in self.sequence)

    def to_llm_described_text(self) -> str:
        return "\n".join(tool.to_llm_described_text() for tool in self.sequence)

    def to_tool_message_sequence(self) -> Sequence[ToolMessage]:
        return [tool.to_tool_message() for tool in self.sequence]

    @property
    def first(self) -> ToolCall[R]:
        """
        Return the first tool call in the sequence.
        """
        return self.sequence[0]

    @property
    def last(self) -> ToolCall[R]:
        """
        Return the last tool call in the sequence.
        """
        return self.sequence[-1]

    def __len__(self) -> int:
        return len(self.sequence)

    def __iter__(self) -> Iterator[ToolCall[R]]:
        return iter(self.sequence)

    def __bool__(self) -> bool:
        return bool(self.sequence)

    def __getitem__(self, index: int) -> ToolCall[R]:
        """
        Return a ToolCall[object] so we avoid Any.
        But we lose precise type knowledge about each index.
        """
        return self.sequence[index]


"""
##     ## ########  ######   ######     ###     ######   ########  ######
###   ### ##       ##    ## ##    ##   ## ##   ##    ##  ##       ##    ##
#### #### ##       ##       ##        ##   ##  ##        ##       ##
## ### ## ######    ######   ######  ##     ## ##   #### ######    ######
##     ## ##             ##       ## ######### ##    ##  ##             ##
##     ## ##       ##    ## ##    ## ##     ## ##    ##  ##       ##    ##
##     ## ########  ######   ######  ##     ##  ######   ########  ######
"""


class Message(msgspec.Struct, tag_field="role", frozen=True):
    """Represents a message in a conversation.

    Attributes:
        contents: The contents of the message

    Example:
        >>> Message(contents=[TextPart("Hello! How are you?")])

    """

    contents: Annotated[
        Sequence[PartType],
        msgspec.Meta(
            title="Message Content",
            description="The contents of the message",
            examples=[
                TextPart("Hello! How are you?"),
            ],
        ),
    ]

    # @abstractmethod
    def to_google_format(self) -> GenaiContent: ...

    # @abstractmethod
    def to_openai_format(self) -> ChatCompletionMessageParam:
        raise NotImplementedError

    # @abstractmethod
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        raise NotImplementedError

    # @abstractmethod
    def to_cerebras_format(self) -> CerebrasMessage:
        raise NotImplementedError

    def to_markdown_str_message(self) -> str:
        """
        Convert the message to a string in markdown format.

        Example:
        <developer_message>...</developer_message>
        """
        class_name = re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()
        return f"<{class_name}>{get_parts_llm_described_text(self.contents)}</{class_name}>"

    @classmethod
    def from_text(cls: type[M], text: str) -> M:
        return cls(contents=[TextPart(text)])

    @classmethod
    def from_part(cls: type[M], part: PartType) -> M:
        return cls(contents=[part])

    @classmethod
    def from_dict(cls: Type[M], _d: dict[str, Any], /) -> M:
        return cls(
            contents=[PartFactory.create_from_dict(part) for part in _d["contents"]]
        )


class DeveloperMessage(Message, frozen=True, tag="developer"):
    """Represents a message from a developer in a conversation.

    Attributes:
        name: An optional name for the participant. Provides the model information to differentiate between participants of the same role.

    Example:
        >>> DeveloperMessage(contents=[TextPart("You are a helpful assistant.")])
    """

    name: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Name",
            description="An optional name for the participant. Provides the"
            "model information to differentiate between participants"
            "of the same role.",
            examples=["Alice", "Bob", "Ana"],
        ),
    ] = None

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_format(self) -> ChatCompletionMessageParam:
        from openai.types.chat.chat_completion_content_part_text_param import (
            ChatCompletionContentPartTextParam,
        )
        from openai.types.chat.chat_completion_developer_message_param import (
            ChatCompletionDeveloperMessageParam,
        )

        content: Iterable[ChatCompletionContentPartTextParam] = [
            cast(ChatCompletionContentPartTextParam, part.to_openai_part())
            for part in self.contents
        ]

        message = ChatCompletionDeveloperMessageParam(
            content=content,
            role="developer",
        )

        if self.name:
            message.update({"name": self.name})

        return message

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        from groq.types.chat.chat_completion_system_message_param import (
            ChatCompletionSystemMessageParam,
        )

        content: str = get_parts_llm_described_text(self.contents)

        message = ChatCompletionSystemMessageParam(content=content, role="system")

        if self.name:
            message.update({"name": self.name})

        return message

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_format(self) -> GenaiContent:
        from google.genai.types import Content as GenaiContent

        name_part = [TextPart(f"{self.name}: ").to_google_part()] if self.name else []
        parts = name_part + [part.to_google_part() for part in self.contents]
        return GenaiContent(role="user", parts=parts)

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_format(self) -> CerebrasMessage:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageSystemMessageRequestTyped,
        )

        message = MessageSystemMessageRequestTyped(
            content=get_parts_llm_described_text(self.contents),
            role="system",
        )

        if self.name:
            message.update({"name": self.name})

        return message


class SystemMessage(DeveloperMessage, frozen=True, tag="system"):
    """Represents a system message in a conversation.

    Attributes:
        name: An optional name for the participant. Provides the model information to differentiate between participants of the same role.

    Example:
        >>> SystemMessage(contents=[TextPart("You are a helpful assistant.")])
    """

    pass


class UserMessage(Message, frozen=True, tag="user"):
    """User role message in a conversation.

    Attributes:
        contents: Multimedia components of the message
        name: Optional identifier for multi-user scenarios

    Example:
        >>> UserMessage(
        ...     contents=[TextPart("What's the weather today?")],
        ...     name="Alice"
        ... )
    """

    name: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Name",
            description="An optional name for the participant."
            "Provides the model information to differentiate"
            "between participants of the same role.",
            examples=["Alice", "Bob", "Ana"],
        ),
    ] = None

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_format(
        self,
    ) -> ChatCompletionMessageParam:
        from openai.types.chat.chat_completion_user_message_param import (
            ChatCompletionUserMessageParam,
        )

        message = ChatCompletionUserMessageParam(
            role="user",
            content=[part.to_openai_part() for part in self.contents],
        )

        if self.name:
            message.update({"name": self.name})

        return message

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        from groq.types.chat.chat_completion_user_message_param import (
            ChatCompletionUserMessageParam,
        )

        message = ChatCompletionUserMessageParam(
            role="user",
            content=[part.to_groq_part() for part in self.contents],
        )

        if self.name:
            message.update({"name": self.name})

        return message

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_format(self) -> GenaiContent:
        from google.genai.types import Content as GenaiContent

        name_part = (
            [Part.from_text(f"{self.name}: ").to_google_part()] if self.name else []
        )

        parts = name_part + [part.to_google_part() for part in self.contents]
        return GenaiContent(role="user", parts=parts)

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_format(self) -> CerebrasMessage:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageUserMessageRequestTyped,
        )

        return MessageUserMessageRequestTyped(
            content=[part.to_cerebras_part() for part in self.contents],
            role="user",
            name=self.name,
        )


class AssistantMessage[R = Any](Message, frozen=True, kw_only=True, tag="assistant"):
    """AI-generated message with optional tool calls.

    Attributes:
        contents: Generated response content
        refusal: Optional refusal rationale if request denied
        name: Optional model identifier for multi-model dialogs
        tool_calls: List of function/method invocations

    Example:
        >>> AssistantMessage(
        ...     contents=[TextPart("It's sunny in Paris!")],
        ...     tool_calls=[weather_tool_call]
        ... )
    """

    refusal: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Refusal",
            description="The refusal message by the assistant. If the message was refused",
            examples=["I cannot provide that information."],
        ),
    ] = msgspec.field(default=None)

    name: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Name",
            description="An optional name for the participant. Provides the model"
            "information to differentiate between participants of the same role.",
        ),
    ] = msgspec.field(default=None)

    tool_calls: Annotated[
        ToolCallSequence[R],
        msgspec.Meta(
            title="Tool Calls", description="The tools called by the assistant"
        ),
    ] = msgspec.field(default_factory=lambda: ToolCallSequence([]))

    @property
    def text(self) -> str:
        def no_text_found() -> Never:
            raise ValueError("No text parts found in the message.")

        return (
            "".join([part.text for part in self.contents if isinstance(part, TextPart)])
            or no_text_found()
        )

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_format(self) -> ChatCompletionMessageParam:
        from openai.types.chat.chat_completion_assistant_message_param import (
            ChatCompletionAssistantMessageParam,
        )
        from openai.types.chat.chat_completion_content_part_text_param import (
            ChatCompletionContentPartTextParam,
        )
        from openai.types.chat.chat_completion_message_tool_call_param import (
            ChatCompletionMessageToolCallParam,
        )

        tool_calls = [
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=tool_call.called_function.to_openai_called_function(),
                type="function",
            )
            for tool_call in self.tool_calls
        ]

        message = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=[
                cast(ChatCompletionContentPartTextParam, content.to_openai_part())
                for content in self.contents
            ],
        )

        if self.name:
            message.update({"name": self.name})

        if self.tool_calls:
            message.update({"tool_calls": tool_calls})

        return message

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        from groq.types.chat.chat_completion_assistant_message_param import (
            ChatCompletionAssistantMessageParam,
        )
        from groq.types.chat.chat_completion_message_tool_call_param import (
            ChatCompletionMessageToolCallParam,
        )

        message = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=get_parts_llm_described_text(self.contents),
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=tool_call.id,
                    function=tool_call.called_function.to_groq_called_function(),
                    type="function",
                )
                for tool_call in self.tool_calls
            ]
            if self.tool_calls
            else [],
        )

        if self.name:
            message.update({"name": self.name})

        return message

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_format(self) -> GenaiContent:
        from google.genai.types import Content as GenaiContent
        from google.genai.types import Part as GenAIPart

        tool_parts: list[GenAIPart] = []
        if self.tool_calls:
            tool_parts = [
                GenAIPart.from_function_call(
                    name=tool.called_function.function.name,
                    args=tool.called_function.arguments,
                )
                for tool in self.tool_calls
            ]

        name_part = (
            [Part.from_text(f"{self.name}: ").to_google_part()] if self.name else []
        )

        return GenaiContent(
            role="model",
            parts=name_part
            + [part.to_google_part() for part in self.contents]
            + tool_parts,
        )

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_format(self) -> CerebrasMessage:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageAssistantMessageRequestToolCallFunctionTyped,
            MessageAssistantMessageRequestToolCallTyped,
            MessageAssistantMessageRequestTyped,
        )

        return MessageAssistantMessageRequestTyped(
            role="assistant",
            content=get_parts_llm_described_text(self.contents),
            name=self.name,
            tool_calls=[
                MessageAssistantMessageRequestToolCallTyped(
                    id=tool_call.id,
                    function=MessageAssistantMessageRequestToolCallFunctionTyped(
                        arguments=msgspec.json.encode(
                            tool_call.called_function.arguments
                        ).decode("utf-8", errors="replace"),
                        name=tool_call.called_function.function.name,
                    ),
                    type="function",
                )
                for tool_call in self.tool_calls
            ],
        )


class GeneratedAssistantMessage[T = RawResponse, R = Any](
    AssistantMessage[R], frozen=True, kw_only=True, tag="generated_assistant"
):
    """AI-generated message with structured model output.

    Attributes:
        parsed: The structured model output of the message

    Example:
        >>> GeneratedAssistantMessage(
        ...     contents=[TextPart("{\"text\": \"Hello!\"}")],
        ...     parsed=UserDefinedResponse(text="Hello!")
        ... )
    """

    parsed: Annotated[
        T,
        msgspec.Meta(
            title="Structured Model",
            description="Structured model of the message",
        ),
    ] = msgspec.field(default=cast(T, RawResponse()))


class ToolMessage(Message, frozen=True, tag="tool"):
    """Represents a message from a tool in a conversation.

    Attributes:
        tool_call_id: The unique identifier of the tool call
        name: The name of the tool

    Example:
        >>> ToolMessage(
        ...     contents=[TextPart("The weather is sunny today.")],
        ...     tool_call_id="123",
        ...     name="weather_tool"
        ... )
    """

    tool_call_id: str
    name: str

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_format(self) -> ChatCompletionMessageParam:
        from openai.types.chat.chat_completion_user_message_param import (
            ChatCompletionUserMessageParam,
        )

        # TODO(arthur): check if this is really the best choice
        return ChatCompletionUserMessageParam(
            role="user",
            content=get_parts_llm_described_text(self.contents),
            name="external_tool",
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        from groq.types.chat.chat_completion_user_message_param import (
            ChatCompletionUserMessageParam,
        )

        return ChatCompletionUserMessageParam(
            role="user",
            content=get_parts_llm_described_text(self.contents),
            name="external_tool",
        )

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_format(self) -> GenaiContent:
        from google.genai.types import Content as GenaiContent

        print([part.to_google_part() for part in self.contents])

        return GenaiContent(
            role="model",
            parts=[part.to_google_part() for part in self.contents],
        )

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_format(self) -> CerebrasMessage:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageToolMessageRequestTyped,
        )

        return MessageToolMessageRequestTyped(
            content=get_parts_llm_described_text(self.contents),
            role="tool",
            tool_call_id=self.tool_call_id,
            name=self.name,
        )


class MessageSequence(msgspec.Struct, frozen=True):
    """Ordered collection of conversation messages.

    Attributes:
        messages: Sequence of Message objects

    Example:
        >>> history = MessageSequence([
        ...     UserMessage.from_text("Hello"),
        ...     AssistantMessage.from_text("Hi there!")
        ... ])
    """

    messages: Sequence[Message]

    @property
    def full_text(self) -> str:
        text_parts: list[str] = []
        for message in self.messages:
            for content in message.contents:
                if isinstance(content, TextPart):
                    text_parts.append(content.text)

        return "".join(text_parts)

    @property
    def full_llm_described_text(self) -> str:
        return "\n".join(message.to_markdown_str_message() for message in self.messages)

    def count_images(self) -> int:
        return sum(
            1
            for message in self.messages
            for part in message.contents
            if isinstance(part, ImageFilePart)
        )

    def count_audios(self) -> int:
        return sum(
            1
            for message in self.messages
            for part in message.contents
            if isinstance(part, AudioFilePart)
        )

    def count_videos(self) -> int:
        return sum(
            1
            for message in self.messages
            for part in message.contents
            if isinstance(part, VideoFilePart)
        )

    def count_characters(self) -> int:
        return len(self.full_text)

    def __add__(self, other: MessageSequence) -> MessageSequence:
        return MessageSequence(messages=list(chain(self.messages, other.messages)))

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)


MessageType: TypeAlias = (
    DeveloperMessage | UserMessage | ToolMessage | AssistantMessage[Any] | SystemMessage
)


class MessageFactory(msgspec.Struct, frozen=True):
    """Factory class for creating Message instances from dictionaries.

    Example:
        >>> message_dict = {
        ...     "role": "user",
        ...     "contents": [
        ...         {"type": "text", "text": "Hello! How are you?"}
        ...     ]
        ... }
        >>> MessageFactory.create_from_dict(message_dict)
        UserMessage(contents=[TextPart(text='Hello! How are you?')])
    """

    @staticmethod
    def create_from_dict(_dict: dict[str, Any], /) -> MessageType:
        if "role" not in _dict:
            raise ValueError("Role not found in the message.")

        role = _dict["role"]
        match role:
            case "developer":
                return DeveloperMessage.from_dict(_dict)
            case "system":
                _dict.update({"role": "developer"})
                return DeveloperMessage.from_dict(_dict)
            case "user":
                return UserMessage.from_dict(_dict)
            case "tool":
                return ToolMessage.from_dict(_dict)
            case "assistant":
                return AssistantMessage.from_dict(_dict)
            case _:
                raise ValueError(f"Invalid role: {role}")


"""
 ######  ##     ##  #######  ####  ######  ########  ######
##    ## ##     ## ##     ##  ##  ##    ## ##       ##    ##
##       ##     ## ##     ##  ##  ##       ##       ##
##       ######### ##     ##  ##  ##       ######    ######
##       ##     ## ##     ##  ##  ##       ##             ##
##    ## ##     ## ##     ##  ##  ##    ## ##       ##    ##
 ######  ##     ##  #######  ####  ######  ########  ######
"""


class LogProb(msgspec.Struct):
    """Log probability of a token generated by the model.

    Attributes:
        token: The token generated by the model
        logprob: The log probability of the token
        bytes: The byte representation of the token

    Example:
        >>> LogProb(token="Hello", logprob=-0.5, bytes=[0x48, 0x65, 0x6C, 0x6C, 0x6F])
    """

    token: str = msgspec.field(default_factory=str)
    logprob: float = msgspec.field(default_factory=float)
    bytes: list[int] = msgspec.field(default_factory=list)


class MessageChoice[T](msgspec.Struct, frozen=True, kw_only=True):
    """A choice generated by the model.

    Attributes:
        index: Index of the choice in the list of choices returned by the model
        message: The message content for this choice, including role and text
        logprobs: Log probability of the choice
        finish_reason: The reason why the model stopped generating tokens for this choice

    Example:
        >>> MessageChoice(
        ...     index=0,
        ...     message=GeneratedAssistantMessage(
        ...         contents=[TextPart(text="Hello there, how may I assist you today?")]
        ...     )
        ... )
    """

    index: Annotated[
        int,
        msgspec.Meta(
            title="Index",
            description="Index of the choice in the list of choices returned by the model.",
            examples=[0, 1, 2],
        ),
    ]

    message: Annotated[
        GeneratedAssistantMessage[T, Any],
        msgspec.Meta(
            title="Message",
            description="The message content for this choice, including role and text.",
            examples=[
                GeneratedAssistantMessage(
                    contents=[
                        TextPart(text="Hello there, how may I assist you today?")
                    ],
                    parsed=cast(
                        T,
                        msgspec.defstruct("Example", [("example", str)])("example"),
                    ),
                )
            ],
        ),
    ]

    logprobs: Annotated[
        Optional[list[list[LogProb]]],
        msgspec.Meta(
            title="Log Probability",
            description="Log probability of the choice. Currently always None, reserved for future use.",
            examples=[None],
        ),
    ] = None

    finish_reason: Annotated[
        FinishReason,
        msgspec.Meta(
            title="Finish Reason",
            description="The reason why the model stopped generating tokens for this choice.",
            examples=[
                FinishReason.STOP,
                FinishReason.LENGTH,
                FinishReason.CONTENT_FILTER,
                FinishReason.TOOL_CALLS,
                FinishReason.NONE,
            ],
        ),
    ] = FinishReason.NONE


class PromptTokensDetails(msgspec.Struct, frozen=True):
    """Breakdown of tokens used in the prompt

    Attributes:
        audio_tokens: The number of audio tokens used in the prompt
        cached_tokens: The number of cached tokens used in the prompt

    Example:
        >>> PromptTokensDetails(audio_tokens=9, cached_tokens=3)
    """

    audio_tokens: Annotated[
        int | None,
        msgspec.Meta(
            title="Audio Tokens",
            description="The number of audio tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    cached_tokens: Annotated[
        int | None,
        msgspec.Meta(
            title="Cached Tokens",
            description="The number of cached tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    def __add__(self, other: PromptTokensDetails) -> PromptTokensDetails:
        def safe_add_ints(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return a + b

        audio_tokens = safe_add_ints(self.audio_tokens, other.audio_tokens)
        cached_tokens = safe_add_ints(self.cached_tokens, other.cached_tokens)

        return PromptTokensDetails(
            audio_tokens=audio_tokens,
            cached_tokens=cached_tokens,
        )


class CompletionTokensDetails(msgspec.Struct, frozen=True):
    """Breakdown of tokens generated in completion

    Attributes:
        audio_tokens: The number of audio tokens used in the prompt
        reasoning_tokens: Tokens generated by the model for reasoning

    Example:
        >>> CompletionTokensDetails(audio_tokens=9, reasoning_tokens=3)
    """

    audio_tokens: Annotated[
        int | None,
        msgspec.Meta(
            title="Audio Tokens",
            description="The number of audio tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    reasoning_tokens: Annotated[
        int | None,
        msgspec.Meta(
            title="Reasoning Tokens",
            description="Tokens generated by the model for reasoning.",
        ),
    ]

    def __add__(self, other: CompletionTokensDetails) -> CompletionTokensDetails:
        def safe_add_ints(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return a + b

        audio_tokens = safe_add_ints(self.audio_tokens, other.audio_tokens)
        reasoning_tokens = safe_add_ints(self.reasoning_tokens, other.reasoning_tokens)

        return CompletionTokensDetails(
            audio_tokens=audio_tokens,
            reasoning_tokens=reasoning_tokens,
        )


"""
##     ##  ######     ###     ######   ########
##     ## ##    ##   ## ##   ##    ##  ##
##     ## ##        ##   ##  ##        ##
##     ##  ######  ##     ## ##   #### ######
##     ##       ## ######### ##    ##  ##
##     ## ##    ## ##     ## ##    ##  ##
 #######   ######  ##     ##  ######   ########
"""


class Usage(msgspec.Struct, frozen=True):
    """Usage statistics for a completion response.

    Attributes:
        prompt_tokens: The number of tokens consumed by the input prompt
        completion_tokens: The number of tokens generated in the completion response
        total_tokens: The total number of tokens consumed, including both prompt and completion

    Example:
        >>> Usage(prompt_tokens=9, completion_tokens=12, total_tokens=21)
    """

    prompt_tokens: Annotated[
        int | None,
        msgspec.Meta(
            title="Prompt Tokens",
            description="The number of tokens consumed by the input prompt.",
            examples=[9, 145, 3, 25],
        ),
    ] = None

    completion_tokens: Annotated[
        int | None,
        msgspec.Meta(
            title="Completion Tokens",
            description="The number of tokens generated in the completion response.",
            examples=[12, 102, 32],
        ),
    ] = None

    total_tokens: Annotated[
        int | None,
        msgspec.Meta(
            title="Total Tokens",
            description="The total number of tokens consumed, including both prompt and completion.",
            examples=[21, 324, 12],
        ),
    ] = None

    input_cost: Annotated[
        float | None, msgspec.Meta(title="Input Cost", description="input cost")
    ] = None

    output_cost: Annotated[
        float | None, msgspec.Meta(title="Output Cost", description="Output Cost")
    ] = None

    total_cost: Annotated[
        float | None, msgspec.Meta(title="Total Cost", description="Total Cost")
    ] = None

    prompt_tokens_details: Annotated[
        PromptTokensDetails | None,
        msgspec.Meta(
            title="Prompt Tokens Details",
            description="Breakdown of tokens used in the prompt.",
        ),
    ] = None

    completion_tokens_details: Annotated[
        CompletionTokensDetails | None,
        msgspec.Meta(
            title="Completion Tokens Details",
            description="Breakdown of tokens generated in completion.",
        ),
    ] = None

    def __add__(self, other: Usage) -> Usage:
        def safe_add_ints(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return a + b

        def safe_add_prompt_details(
            a: PromptTokensDetails | None, b: PromptTokensDetails | None
        ) -> PromptTokensDetails | None:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return a + b

        def safe_add_completion_details(
            a: CompletionTokensDetails | None, b: CompletionTokensDetails | None
        ) -> CompletionTokensDetails | None:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return a + b

        prompt_tokens = safe_add_ints(self.prompt_tokens, other.prompt_tokens)
        completion_tokens = safe_add_ints(
            self.completion_tokens, other.completion_tokens
        )
        total_tokens = safe_add_ints(self.total_tokens, other.total_tokens)

        prompt_tokens_details = safe_add_prompt_details(
            self.prompt_tokens_details, other.prompt_tokens_details
        )

        completion_tokens_details = safe_add_completion_details(
            self.completion_tokens_details, other.completion_tokens_details
        )

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
            completion_tokens_details=completion_tokens_details,
        )


"""
 ######   #######  ##     ## ########  ##       ######## ######## ####  #######  ##    ##
##    ## ##     ## ###   ### ##     ## ##       ##          ##     ##  ##     ## ###   ##
##       ##     ## #### #### ##     ## ##       ##          ##     ##  ##     ## ####  ##
##       ##     ## ## ### ## ########  ##       ######      ##     ##  ##     ## ## ## ##
##       ##     ## ##     ## ##        ##       ##          ##     ##  ##     ## ##  ####
##    ## ##     ## ##     ## ##        ##       ##          ##     ##  ##     ## ##   ###
 ######   #######  ##     ## ##        ######## ########    ##    ####  #######  ##    ##
"""


class ChatCompletion[T = RawResponse](msgspec.Struct, kw_only=True, frozen=True):
    """Structured response from a chat model invocation.

    Attributes:
        elapsed_time: Total processing time in seconds
        id: Unique response identifier
        model: Model used for generation
        choices: List of generated message options
        usage: Token consumption statistics

    Example:
        >>> completion = ChatCompletion(
        ...     model="gpt-4",
        ...     choices=[message_choice],
        ...     usage=Usage(prompt_tokens=100, completion_tokens=50)
        ... )
    """

    elapsed_time: Annotated[
        float,
        msgspec.Meta(
            title="Elapsed Time",
            description="The amount of time it took to generate the Completion.",
        ),
    ] = msgspec.field()

    id: Annotated[
        str,
        msgspec.Meta(
            title="ID",
            description="The unique identifier of the completion.",
            examples=[
                "f50ec0b7-f960-400d-91f0-c42a6d44e3d0",
                "16fd2706-8baf-433b-82eb-8c7fada847da",
            ],
        ),
    ] = msgspec.field(default_factory=lambda: uuid.uuid4().__str__())

    object: Annotated[
        Literal["chat.completion"],
        msgspec.Meta(
            title="Object Type",
            description="The object type. Always `chat.completion`.",
            examples=["chat.completion"],
        ),
    ] = "chat.completion"

    created: Annotated[
        int,
        msgspec.Meta(
            title="Created",
            description="The Unix timestamp when the completion was created. Defaults to the current time.",
            examples=[1677652288, 1634020001],
        ),
    ] = msgspec.field(default_factory=lambda: int(datetime.datetime.now().timestamp()))

    model: Annotated[
        AIModel,
        msgspec.Meta(
            title="Model",
            description="The AI model used to generate the completion.",
        ),
    ] = msgspec.field()

    system_fingerprint: Annotated[
        str,
        msgspec.Meta(
            title="System Fingerprint",
            description="""This fingerprint represents the backend configuration that the model runs with.
                       Can be used in conjunction with the seed request parameter to understand when
                       backend changes have been made that might impact determinism.""",
            examples=["fp_44709d6fcb"],
        ),
    ] = "fp_none"

    choices: Annotated[
        Sequence[MessageChoice[T]],
        msgspec.Meta(
            title="Choices",
            description="""The choices made by the language model. 
                       The length of this list can be greater than 1 if multiple choices were requested.""",
            examples=[],
        ),
    ]

    usage: Annotated[
        Usage,
        msgspec.Meta(
            title="Usage",
            description="Usage statistics for the completion request.",
            examples=[
                Usage(
                    prompt_tokens=9,
                    completion_tokens=12,
                    total_tokens=21,
                    prompt_tokens_details=PromptTokensDetails(
                        audio_tokens=9, cached_tokens=0
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        audio_tokens=12, reasoning_tokens=0
                    ),
                )
            ],
        ),
    ]

    @property
    def tool_calls(self) -> ToolCallSequence:
        if len(self.choices) > 1:
            raise ValueError(
                "Completion has multiple choices. Please use get_tool_calls(choice=...), instead."
            )
        return self.choices[0].message.tool_calls

    def get_tool_calls(self, choice: int = 0) -> ToolCallSequence[Any]:
        return self.choices[choice].message.tool_calls

    @property
    def text(self) -> str:
        if len(self.choices) > 1:
            raise ValueError(
                "Completion has multiple choices. Please use get_text(choice=...), instead."
            )

        return self.get_text(0)

    @property
    def message(self) -> GeneratedAssistantMessage[T]:
        if len(self.choices) > 1:
            raise ValueError(
                "Completion has multiple choices. Please use get_text(choice=...), instead."
            )
        return self.get_message()

    @property
    def parsed(self) -> T:
        if len(self.choices) > 1:
            raise ValueError(
                "Completion has multiple choices. Please use get_parsed(choice=...), instead."
            )

        return self.get_parsed()

    def get_text(self, choice: int) -> str:
        message_contents = self.choices[choice].message.contents
        return get_parts_raw_text(parts=message_contents)

    def get_message(self, choice: int = 0) -> GeneratedAssistantMessage[T]:
        selected_choice = self.choices[choice]
        return selected_choice.message

    def get_parsed(self, choice: int = 0) -> T:
        selected_choice: MessageChoice[T] = self.choices[choice]
        parsed = selected_choice.message.parsed
        if parsed is None:
            raise ValueError("Parsed content is None")

        return parsed

    def __add__(self, other: ChatCompletion[T]) -> ChatCompletion[T]:
        return ChatCompletion(
            elapsed_time=round(self.elapsed_time + other.elapsed_time, 2),
            id=uuid.uuid4().__str__(),
            model=self.model,
            system_fingerprint=self.system_fingerprint,
            choices=list(self.choices) + list(other.choices),
            usage=self.usage + other.usage,
        )


class Property(msgspec.Struct, frozen=True, kw_only=True):
    """Represents a property within a parameter.

    Attributes:
        name: The name of the property
        type: The JSON schema type of the property
        description: The description of the property
        enum: The enum values of the property
        properties: Nested properties
        items: Items for arrays

    Example:
        >>> Property(
        ...     name="name",
        ...     type="string",
        ...     description="The name of the user",
        ...     enum=["Alice", "Bob", "Ana"]
        ... )
    """

    type: str | Literal["object", "string", "number", "integer", "boolean", "array"]
    description: Optional[str] = None
    enum: Optional[list[Any]] = None
    properties: Optional[dict[str, Property]] = None  # For nested objects
    items: Optional[Property] = None  # For arrays


class Parameter(msgspec.Struct, frozen=True, kw_only=True):
    """Represents a parameter within a function.

    Attributes:
        name: The name of the parameter
        type: The JSON schema type of the parameter
        description: The description of the parameter
        enum: The enum values of the parameter
        properties: Nested properties
        items: Items for arrays
        required: Whether the parameter is required

    Example:
        >>> Parameter(
        ...     name="name",
        ...     type="string",
        ...     description="The name of the user",
        ...     enum=["Alice", "Bob", "Ana"]
        ... )
    """

    name: str
    type: str | Literal["object", "string", "number", "integer", "boolean", "array"]
    description: Optional[str] = None
    enum: Optional[list[Any]] = None
    properties: Optional[dict[str, Property]] = None  # For nested objects
    items: Optional[Property] = None  # For arrays
    required: bool = False

    def to_object(self) -> dict[str, Any]:
        return msgspec.json.decode(msgspec.json.encode(self), type=dict)


class Function[R = Any](msgspec.Struct, frozen=True, kw_only=True):
    """Callable tool definition for LLM function calling.

    Attributes:
        name: Function identifier
        parameters: Input schema definitions
        callable: Actual implementation function
        description: Natural language description

    Example:
        >>> def add(a: int, b: int) -> int: return a + b
        >>> Function.from_callable(add)
        Function(name='add', parameters=[...])
    """

    name: str
    parameters: Sequence[Parameter]
    callable: Callable[..., R] = msgspec.field(
        default_factory=lambda: lambda: cast(R, None)
    )
    description: Optional[str] = None

    def get_parameters_as_dict(self) -> list[dict[str, Any]]:
        return msgspec.json.decode(msgspec.json.encode(self.parameters), type=list)

    @classmethod
    def from_callable(cls, func: Callable[..., Any]) -> "Function":
        """
        Generate a Function schema from a callable.
        This single method includes:
        - Checking if func is actually inspectable (i.e., a user-defined function)
        - Parsing annotations to determine JSON types
        - Recursively handling nested callable annotations (if desired)
        - Avoiding calls to inspect.signature for non-inspectable objects
        """

        def is_inspectable_function(obj: Any) -> bool:
            """
            Return True if obj is a user-defined function or method
            that supports introspection (not a built-in type).
            """
            return (
                inspect.isfunction(obj) or inspect.ismethod(obj)
            ) and obj.__module__ not in ("builtins", "abc")

        def parse_annotation(annotation: Any) -> tuple[str, Optional[list[Any]]]:
            """
            Convert a Python annotation into a JSON schema type
            plus optional enum values.
            You can expand this logic as needed.
            """
            # Simple map of some Python types to JSON schema types
            python_to_json = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            # If annotation is directly in our map:
            if annotation in python_to_json:
                return python_to_json[annotation], None

            # Fallback (e.g. for Any, custom classes, or anything else)
            return "string", None

        def extract_param_description(
            func: Callable[..., Any], param_name: str
        ) -> Optional[str]:
            """
            Stub for docstring extraction or additional metadata.
            Returns None here, but you can implement your own logic.
            """
            return None

        # --- Main logic ---

        # Ensure we're dealing with an actual function/method
        if not is_inspectable_function(func):
            raise ValueError(f"Object {func!r} is not an inspectable function/method.")

        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        parameters: list[Parameter] = []

        for name, param in sig.parameters.items():
            annotation = type_hints.get(name, Any)
            default = param.default
            # Convert the annotation -> JSON type, enum
            param_type, enum_values = parse_annotation(annotation)

            # Check if the parameter is required (no default)
            is_required = default is inspect.Parameter.empty

            # If the annotation is another inspectable function, we recursively build
            if is_inspectable_function(annotation):
                try:
                    nested_function = cls.from_callable(annotation)
                    # Build a dict of nested Parameter -> Property
                    nested_props = {
                        nested_param.name: Property(
                            type=nested_param.type,
                            description=nested_param.description,
                            enum=nested_param.enum,
                            properties=nested_param.properties,
                            items=nested_param.items,
                        )
                        for nested_param in nested_function.parameters
                    }
                    parameter = Parameter(
                        name=name,
                        type="object",
                        description=nested_function.description,
                        properties=nested_props,
                        required=is_required,
                    )
                except ValueError:
                    # If inspection fails for the annotation, treat it as normal
                    parameter = Parameter(
                        name=name,
                        type=param_type,
                        description=extract_param_description(func, name),
                        enum=enum_values,
                        required=is_required,
                    )
            else:
                # Normal parameter handling
                parameter = Parameter(
                    name=name,
                    type=param_type,
                    description=extract_param_description(func, name),
                    enum=enum_values,
                    required=is_required,
                )

            parameters.append(parameter)

        # Build and return the Function
        return cls(
            name=func.__name__,
            description=inspect.getdoc(func),
            parameters=parameters,
            callable=func,
        )

    @ensure_module_installed("openai", "openai")
    def to_openai_function(
        self, strict: Optional[bool] = None
    ) -> OpenAIFunctionDefinition:
        """
        Convert the Function instance into an OpenAI-compatible function definition.

        Args:
            strict (bool): Whether to enforce strict schema validation.

        Returns:
            dict: The OpenAI-compatible function definition.
        """
        from openai.types.shared_params.function_definition import (
            FunctionDefinition as OpenAIFunctionDefinition,
        )

        def property_to_schema(prop: Property) -> dict[str, Any]:
            """Convert a Property instance into a schema dictionary."""
            schema: dict[str, Any] = {"type": prop.type}

            if prop.description:
                schema["description"] = prop.description

            if prop.enum:
                schema["enum"] = prop.enum

            if prop.type == "object" and prop.properties:
                schema["properties"] = {
                    name: property_to_schema(sub_prop)
                    for name, sub_prop in prop.properties.items()
                }

            if prop.type == "array" and prop.items:
                schema["items"] = property_to_schema(prop.items)

            return schema

        def parameter_to_schema(param: Parameter) -> dict[str, Any]:
            """Convert a Parameter instance into a schema dictionary."""
            schema: dict[str, Any] = {"type": param.type}

            if param.description:
                schema["description"] = param.description

            if param.enum:
                schema["enum"] = param.enum

            if param.type == "object" and param.properties:
                schema["properties"] = {
                    name: property_to_schema(prop)
                    for name, prop in param.properties.items()
                }

            if param.type == "array" and param.items:
                schema["items"] = property_to_schema(param.items)

            return schema

        # Convert all parameters to the appropriate schema
        properties: dict[str, dict[str, Any]] = {
            param.name: parameter_to_schema(param) for param in self.parameters
        }

        required_params = [param.name for param in self.parameters if param.required]

        return OpenAIFunctionDefinition(
            name=self.name,
            description=self.description or "No description provided.",
            parameters={
                "type": "object",
                "properties": properties,
                **({"required": required_params} if required_params else {}),
            },
            strict=strict,
        )

    @ensure_module_installed("groq", "groq")
    def to_groq_function(self) -> GroqFunctionDefinition:
        """
        Convert the Function instance into a Groq-compatible function declaration.

        Returns:
            GroqFunctionDeclaration: The Groq-compatible function declaration.
        """
        from groq.types.shared_params.function_definition import (
            FunctionDefinition as GroqFunctionDefinition,
        )

        def property_to_schema(prop: Property) -> dict[str, Any]:
            """Convert a Property instance into a schema dictionary."""
            schema: dict[str, Any] = {"type": prop.type}

            if prop.description:
                schema["description"] = prop.description

            if prop.enum:
                schema["enum"] = prop.enum

            if prop.type == "object" and prop.properties:
                schema["properties"] = {
                    name: property_to_schema(sub_prop)
                    for name, sub_prop in prop.properties.items()
                }

            if prop.type == "array" and prop.items:
                schema["items"] = property_to_schema(prop.items)

            return schema

        def parameter_to_schema(param: Parameter) -> dict[str, Any]:
            """Convert a Parameter instance into a schema dictionary."""
            schema: dict[str, Any] = {"type": param.type}

            if param.description:
                schema["description"] = param.description

            if param.enum:
                schema["enum"] = param.enum

            if param.type == "object" and param.properties:
                schema["properties"] = {
                    name: property_to_schema(prop)
                    for name, prop in param.properties.items()
                }

            if param.type == "array" and param.items:
                schema["items"] = property_to_schema(param.items)

            return schema

        # Convert all parameters to the appropriate schema
        properties: dict[str, dict[str, Any]] = {
            param.name: parameter_to_schema(param) for param in self.parameters
        }

        required_params = [param.name for param in self.parameters if param.required]

        return GroqFunctionDefinition(
            name=self.name,
            description=self.description or "No description provided.",
            parameters={
                "type": "object",
                "properties": properties,
                **({"required": required_params} if required_params else {}),
            },
        )

    @ensure_module_installed("google.genai", "google-genai")
    def to_genai_function(self) -> GenAIFunctionDeclaration:
        """
        Convert the Function instance into a GenAI-compatible function declaration.

        Returns:
            GenAIFunctionDeclaration (google.genai.types.FunctionDeclaration)
        """
        from google.genai.types import FunctionDeclaration, Schema

        openapi_to_genai_type = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }

        def property_to_schema(prop: Property) -> Schema:
            """Convert a Property instance into a google.genai.types.Schema object."""
            # Determine the correct GenAI `Type` from the string type
            # Fallback to 'TYPE_UNSPECIFIED' if the type is not recognized
            schema_type: Literal[
                "TYPE_UNSPECIFIED",
                "STRING",
                "NUMBER",
                "INTEGER",
                "BOOLEAN",
                "ARRAY",
                "OBJECT",
            ] = cast(
                Literal[
                    "TYPE_UNSPECIFIED",
                    "STRING",
                    "NUMBER",
                    "INTEGER",
                    "BOOLEAN",
                    "ARRAY",
                    "OBJECT",
                ],
                openapi_to_genai_type.get(prop.type, "TYPE_UNSPECIFIED"),
            )

            # Build the Schema
            schema = Schema(
                type=schema_type,
                description=prop.description,
                enum=prop.enum if prop.enum else None,
            )

            # If it's an object, recurse on its properties
            if prop.type == "object" and prop.properties:
                schema.properties = {
                    name: property_to_schema(sub_prop)
                    for name, sub_prop in prop.properties.items()
                }

            # If it's an array, recurse on its items
            if prop.type == "array" and prop.items:
                schema.items = property_to_schema(prop.items)

            return schema

        def parameter_to_schema(param: Parameter) -> Schema:
            """Convert a Parameter instance into a google.genai.types.Schema object."""
            schema_type = openapi_to_genai_type.get(param.type, "TYPE_UNSPECIFIED")

            schema = Schema(
                type=cast(
                    Literal[
                        "TYPE_UNSPECIFIED",
                        "STRING",
                        "NUMBER",
                        "INTEGER",
                        "BOOLEAN",
                        "ARRAY",
                        "OBJECT",
                    ],
                    schema_type,
                ),
                description=param.description,
                enum=param.enum if param.enum else None,
            )

            if param.type == "object" and param.properties:
                schema.properties = {
                    name: property_to_schema(prop)
                    for name, prop in param.properties.items()
                }

            if param.type == "array" and param.items:
                schema.items = property_to_schema(param.items)

            return schema

        # Convert all parameters to the appropriate Schema objects
        properties: dict[str, Schema] = {
            param.name: parameter_to_schema(param) for param in self.parameters
        }

        required_params = [param.name for param in self.parameters if param.required]

        # Construct the top-level schema for parameters
        parameters_schema = Schema(
            type="OBJECT",
            properties=properties if properties else {},
            required=required_params if required_params else None,
        )

        # Return the FunctionDeclaration (the GenAI function definition)
        return FunctionDeclaration(
            name=self.name,
            description=self.description or "No description provided.",
            parameters=parameters_schema,
            response=None,  # Adjust if you have a response schema
        )

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    def to_cerebras_function(self) -> CerebrasFunctionDefinition:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            ToolFunctionTyped as CerebrasFunctionDefinition,
        )

        def property_to_schema(prop: Property) -> dict[str, Any]:
            """Convert a Property instance into a schema dictionary."""
            schema: dict[str, Any] = {"type": prop.type}

            if prop.description:
                schema["description"] = prop.description

            if prop.enum:
                schema["enum"] = prop.enum

            if prop.type == "object" and prop.properties:
                schema["properties"] = {
                    name: property_to_schema(sub_prop)
                    for name, sub_prop in prop.properties.items()
                }

            if prop.type == "array" and prop.items:
                schema["items"] = property_to_schema(prop.items)

            return schema

        def parameter_to_schema(param: Parameter) -> dict[str, Any]:
            """Convert a Parameter instance into a schema dictionary."""
            schema: dict[str, Any] = {"type": param.type}

            if param.description:
                schema["description"] = param.description

            if param.enum:
                schema["enum"] = param.enum

            if param.type == "object" and param.properties:
                schema["properties"] = {
                    name: property_to_schema(prop)
                    for name, prop in param.properties.items()
                }

            if param.type == "array" and param.items:
                schema["items"] = property_to_schema(param.items)

            return schema

        # Convert all parameters to the appropriate schema
        properties: dict[str, dict[str, Any]] = {
            param.name: parameter_to_schema(param) for param in self.parameters
        }

        required_params = [param.name for param in self.parameters if param.required]

        return CerebrasFunctionDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": properties,
                **({"required": required_params} if required_params else {}),
            },
        )

    @staticmethod
    def _extract_param_description(
        func: Callable[..., Any], param_name: str
    ) -> Optional[str]:
        """
        Extract parameter description from the function's docstring.

        Args:
            func (Callable): The function from which to extract the description.
            param_name (str): The parameter name.

        Returns:
            Optional[str]: The description of the parameter if found.
        """
        doc = inspect.getdoc(func)
        if not doc:
            return None
        # Simple parsing: look for "param <name>: <description>"
        for line in doc.splitlines():
            line = line.strip()
            if line.startswith(f"param {param_name}:"):
                return line.partition(":")[2].strip()
        return None


class CalledFunction[R = Any](msgspec.Struct, frozen=True, kw_only=True):
    """Represents a function that was called with arguments by an AI.

    Attributes:
        function: The function that was called
        arguments: The arguments that were passed to the function

    Example:
        >>> def add(a: int, b: int) -> int: return a + b
        >>> called_function = CalledFunction(function=Function.from_callable(add), arguments={"a": 2, "b": 3})
    """

    function: Function[R]
    arguments: dict[str, Any] = msgspec.field(default_factory=dict)

    def to_openai_called_function(self) -> OpenAICalledFunction:
        from openai.types.chat.chat_completion_message_tool_call_param import (
            Function as OpenAICalledFunction,
        )

        return OpenAICalledFunction(
            name=self.function.name,
            arguments=str(self.arguments),
        )

    def to_groq_called_function(self) -> GroqCalledFunction:
        from groq.types.chat.chat_completion_message_tool_call_param import (
            Function as GroqCalledFunction,
        )

        return GroqCalledFunction(
            name=self.function.name,
            arguments=str(self.arguments),
        )

    def call(self) -> R:
        """
        Call the function with the provided arguments.

        Returns:
            Any: The return value of the function.
        """

        # Call the function with the provided arguments
        return self.function.callable(**self.arguments)


class SentenceSegment(msgspec.Struct, frozen=True):
    """A segment of a transcribed audio file.

    Attributes:
        id: The unique identifier of the segment
        sentence: The transcribed text of the segment
        start: The start time of the segment in seconds
        end: The end time of the segment in seconds
        no_speech_prob: The probability that there is no speech in the segment

    Example:
        >>> SentenceSegment(id=0, sentence="Hello, world!", start=0.0, end=1.5, no_speech_prob=0.0)
    """

    id: int
    """The unique identifier of the segment."""

    sentence: str
    """The transcribed text of the segment."""

    start: float
    """The start time of the segment in seconds."""

    end: float
    """The end time of the segment in seconds."""

    no_speech_prob: float
    """The probability that there is no speech in the segment."""


class AudioTranscription(msgspec.Struct, frozen=True, kw_only=True):
    """Result of audio-to-text transcription.

    Attributes:
        elapsed_time: Processing time in seconds
        text: Full transcribed text
        segments: Timed text segments with confidence
        cost: API cost in USD
        duration: Audio length in seconds

    Example:
        >>> transcription = AudioTranscription(
        ...     text="Hello world",
        ...     segments=[...],
        ...     cost=0.02
        ... )
    """

    elapsed_time: Annotated[
        float,
        msgspec.Meta(
            title="Elapsed Time",
            description="The amount of time it took to generate the transcriptions.",
        ),
    ]

    text: Annotated[
        str,
        msgspec.Meta(
            title="Text",
            description="The transcribed text.",
        ),
    ]

    segments: Annotated[
        Sequence[SentenceSegment],
        msgspec.Meta(
            title="Parts",
            description="The transcribed text broken down into segments.",
        ),
    ]

    cost: Annotated[
        float,
        msgspec.Meta(
            title="Cost",
            description="The cost incurred by the transcriptions request.",
        ),
    ]

    duration: Annotated[
        float,
        msgspec.Meta(
            title="Duration",
            description="The duration of the audio file in seconds. May not be precise.",
        ),
    ]

    srt: Annotated[
        str,
        msgspec.Meta(
            title="SRT",
            description="The transcribed text in SubRip format.",
        ),
    ]

    def merge(self, *audio_transcriptions: AudioTranscription) -> AudioTranscription:
        all_transcriptions = [self] + list(audio_transcriptions)
        merged_elapsed_time = sum(t.elapsed_time for t in all_transcriptions)
        merged_text = " ".join(t.text for t in all_transcriptions)
        merged_cost = sum(t.cost for t in all_transcriptions)
        merged_duration = sum(t.duration for t in all_transcriptions)
        merged_segments: list[SentenceSegment] = []
        for i, current_transcription in enumerate(all_transcriptions):
            current_offset = sum(t.duration for t in all_transcriptions[:i])
            for segment in current_transcription.segments:
                new_start = segment.start + current_offset
                new_end = segment.end + current_offset
                new_id = len(merged_segments)
                new_segment = SentenceSegment(
                    id=new_id,
                    sentence=segment.sentence,
                    start=new_start,
                    end=new_end,
                    no_speech_prob=segment.no_speech_prob,
                )
                merged_segments.append(new_segment)

        return AudioTranscription(
            elapsed_time=round(merged_elapsed_time, 2),
            text=merged_text,
            segments=merged_segments,
            cost=merged_cost,
            duration=merged_duration,
            srt=segments_to_srt(merged_segments),
        )


class ThoughtDetail(msgspec.Struct, frozen=True):
    """A detailed explanation of a specific aspect of a reasoning step.

    Attributes:
        detail: A granular explanation of a specific aspect of the reasoning step

    Example:
        >>> ThoughtDetail(detail="First, I added 2 + 3")
    """

    detail: Annotated[
        str,
        msgspec.Meta(
            description="A granular explanation of a specific aspect of the reasoning step.",
            examples=["First, I added 2 + 3", "Checked if the number is even or odd"],
        ),
    ]


class Step(msgspec.Struct, frozen=True):
    """A step in a chain of thought.

    Attributes:
        step_number: The position of this step in the overall chain of thought
        explanation: A concise description of what was done in this step
        details: A list of specific details for each step in the reasoning

    Example:
        >>> Step(
        ...     step_number=1,
        ...     explanation="Analyze the input statement",
        ...     details=[
        ...         ThoughtDetail(detail="Check initial values"),
        ...         ThoughtDetail(detail="Confirm there are no inconsistencies")
        ...     ]
        ... )
    """

    step_number: Annotated[
        int,
        msgspec.Meta(
            description="The position of this step in the overall chain of thought.",
            examples=[1, 2, 3],
        ),
    ]
    explanation: Annotated[
        str,
        msgspec.Meta(
            description="A concise description of what was done in this step.",
            examples=["Analyze the input statement", "Apply the quadratic formula"],
        ),
    ]
    details: Annotated[
        Sequence[ThoughtDetail],
        msgspec.Meta(
            description="A list of specific details for each step in the reasoning.",
            examples=[
                [
                    {"detail": "Check initial values"},
                    {"detail": "Confirm there are no inconsistencies"},
                ]
            ],
        ),
    ]


class ChainOfThought(msgspec.Struct, Generic[_T], frozen=True):
    """Structured reasoning process with final answer.

    Attributes:
        general_title: High-level description of reasoning goal
        steps: Logical steps in reasoning process
        final_answer: Conclusion of the reasoning chain

    Example:
        >>> ChainOfThought(
        ...     general_title="Math problem solution",
        ...     steps=[step1, step2],
        ...     final_answer=42
        ... )
    """

    general_title: Annotated[
        str,
        msgspec.Meta(
            description="A brief label or description that identifies the purpose of the reasoning.",
            examples=["Sum of two numbers", "Logical problem solving"],
        ),
    ]
    steps: Annotated[
        Sequence[Step],
        msgspec.Meta(
            description="The sequence of steps that make up the full reasoning process.",
            examples=[
                [
                    {
                        "step_number": 1,
                        "explanation": "Analyze input data",
                        "details": [
                            {"detail": "Data: 234 and 567"},
                            {"detail": "Check if they are integers"},
                        ],
                    },
                    {
                        "step_number": 2,
                        "explanation": "Perform the calculation",
                        "details": [
                            {"detail": "234 + 567 = 801"},
                        ],
                    },
                ]
            ],
        ),
    ]
    final_answer: Annotated[
        _T,
        msgspec.Meta(
            description="The conclusion or result after all the reasoning steps.",
        ),
    ]


class GraphicalElementDescription(msgspec.Struct, frozen=True):
    """A description of a visual or auditory element within a media file.

    Attributes:
        type: The general category of this visual element within the media
        details: A detailed description of the characteristics and properties of this element
        role: The purpose or function of this element within the context of the media
        relationships: How this element is related to other elements in the media

    Example:
        >>> GraphicalElementDescription(
        ...     type="Text string",
        ...     details="The number '3' in the top-left corner",
        ...     role="Represents the coefficient of x",
        ...     relationships=["Located above the main equation"]
        ... )
    """

    type: Annotated[
        str,
        msgspec.Meta(
            title="Element Type",
            description="The general category of this visual element within the media. This could be a recognizable object, a symbol, a graphical component, a section of text, or any other distinct visual or temporal component. Be descriptive but not necessarily tied to real-world objects if the media is abstract or symbolic. Examples: 'Equation term', 'Geometric shape', 'Timeline marker', 'Audio waveform segment', 'Brushstroke', 'Data point'.",
            examples=[
                "Text string",
                "Geometric shape",
                "Timeline marker",
                "Component of a machine",
                "Area of color",
                "Video transition",
            ],
        ),
    ]

    details: Annotated[
        str,
        msgspec.Meta(
            title="Element Details",
            description="A detailed description of the characteristics and properties of this element. Focus on what is visually or audibly apparent. For text, provide the content. For shapes, describe form, color, and features. For abstract elements, describe visual properties like color, texture, and form, or temporal properties like duration and transitions. Be specific and descriptive. Examples: 'The text string 'y = mx + c' in bold font', 'A red circle with a thick black outline', 'A sudden fade to black', 'A high-pitched tone'.",
            examples=[
                "The number '3' in the top-left corner",
                "A thin, dashed black line",
                "A vibrant green triangular shape",
                "A slow zoom-in effect",
                "A burst of static",
            ],
        ),
    ]

    role: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Element Role/Function",
            description="The purpose or function of this element within the context of the media. How does it contribute to the overall meaning, structure, or flow? For example, in a formula, describe its mathematical role. In a diagram, its function. In a video, its narrative or informational contribution. Examples: 'Represents a variable in the equation', 'Indicates the direction of flow', 'Marks a key event in the timeline', 'Signals a change in scene'.",
            examples=[
                "Represents the coefficient of x",
                "Connects two stages in the process",
                "Highlights a critical moment",
                "Provides context for the following scene",
            ],
        ),
    ] = msgspec.field(default=None)

    relationships: Annotated[
        Optional[Sequence[str]],
        msgspec.Meta(
            title="Element Relationships",
            description="Describe how this element is related to other elements in the media. "
            "Explain its position relative to others, whether it's connected, overlapping, "
            "near, or otherwise associated with them, considering spatial and temporal "
            "relationships. Be specific about the other elements involved. Examples: "
            "'The arrow points from this box to the next', 'This circle is enclosed"
            "within the square', 'This scene follows the previous one', 'The music"
            "swells during this visual element'.",
            examples=[
                "Located above the main equation",
                "Connected to the previous step by a line",
                "Part of a larger assembly",
                "Occurs immediately after the title card",
            ],
        ),
    ] = msgspec.field(default=None)

    extracted_text: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Extracted Text Content",
            description="For elements that contains text elements, the actual textual content "
            "extracted through OCR. Preserves line breaks and spatial relationships where "
            "possible.",
            examples=["'3.14'", "'Warning: Do not open'", "'y = mx + b'"],
        ),
    ] = msgspec.field(default=None)

    def md(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        md_str = f"{indent}**Element Type**: {self.type}\n"
        md_str += f"{indent}**Element Details**: {self.details}\n"
        if self.role:
            md_str += f"{indent}**Role/Function**: {self.role}\n"
        if self.relationships:
            md_str += f"{indent}**Relationships**:\n"
            for rel in self.relationships:
                md_str += f"{indent}  - {rel}\n"
        return md_str


class MediaStructure(msgspec.Struct, frozen=True):
    """A description of the overall structure and organization of a visual or auditory media file.

    Attributes:
        layout: A description of how the elements are arranged and organized within the media
        groupings: Significant groupings or clusters of elements that appear to function together
        focal_point: The primary focal point that draws attention

    Example:
        >>> MediaStructure(
        ...     layout="A step-by-step diagram",
        ...     groupings=["The main body of the text", "The elements forming the control panel"],
        ...     focal_point="The large heading at the top"
        ... )
    """

    layout: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Overall Layout and Organization",
            description="A description of how the elements are arranged and organized within the media. Describe the overall structure, flow, or pattern, considering both spatial and temporal aspects. Is it linear, grid-based, hierarchical, sequential, or something else? How are the different parts connected or separated? Examples: 'A top-down flowchart', 'A grid of data points', 'A chronological sequence of scenes', 'A central diagram with surrounding labels'.",
            examples=[
                "A step-by-step diagram",
                "A clustered arrangement of shapes",
                "A formula presented on a single line",
                "A narrative with distinct acts",
            ],
        ),
    ] = msgspec.field(default=None)
    groupings: Annotated[
        Optional[Sequence[str]],
        msgspec.Meta(
            title="Significant Groupings of Elements",
            description="Describe any notable groupings or clusters of elements that appear to function together or have a shared context, considering both visual and temporal coherence. Explain what binds these elements together visually, aurally, or conceptually. Examples: 'The terms on the left side of the equation', 'The interconnected components of the circuit diagram', 'A montage of related images', 'A musical theme associated with a character'.",
            examples=[
                "The main body of the text",
                "The elements forming the control panel",
                "The interconnected nodes of the network",
                "A series of shots depicting the same event",
            ],
        ),
    ] = msgspec.field(default=None)
    focal_point: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Primary Focal Point",
            description="Identify the most prominent or central element or area that draws attention, considering visual, auditory, and temporal emphasis. Explain why this element stands out (e.g., size, color, position, duration, sound intensity). If there isn't a clear focal point, describe the distribution of emphasis. Examples: 'The main title of the document', 'The central component of the machine', 'The climax of the scene', 'The loudest sound'.",
            examples=[
                "The large heading at the top",
                "The brightly colored area in the center",
                "The main subject of the drawing",
                "The key moment of impact",
            ],
        ),
    ] = msgspec.field(default=None)

    def md(self, indent_level: int = 1) -> str:
        indent = "  " * indent_level
        md_str = ""
        if self.layout:
            md_str += f"{indent}**Overall Layout and Organization**: {self.layout}\n"
        if self.groupings:
            md_str += f"{indent}**Significant Groupings of Elements**:\n"
            for group in self.groupings:
                md_str += f"{indent}  - {group}\n"
        if self.focal_point:
            md_str += f"{indent}**Primary Focal Point**: {self.focal_point}\n"
        return md_str


class VisualMediaDescription(msgspec.Struct, frozen=True):
    """Detailed description of visual content.

    Attributes:
        overall_description: Comprehensive content summary
        content_type: Category (diagram, photo, etc.)
        visual_elements: Individual components
        structure: Spatial organization
        dominant_features: Salient visual characteristics
        intended_purpose: Interpreted purpose

    Example:
        >>> desc = VisualMediaDescription(
        ...     content_type="infographic",
        ...     visual_elements=[...]
        ... )
    """

    overall_description: Annotated[
        str,
        msgspec.Meta(
            title="Overall Media Description",
            description="Provide a comprehensive and detailed narrative describing the entire visual media, focusing on its content, structure, and key elements. Imagine you are explaining it to someone who cannot see or hear it. Describe the overall purpose or what information it is conveying. Detail the main components and how they are organized, considering both spatial and temporal aspects. Use precise language to describe visual characteristics like shapes, colors, patterns, and relationships, as well as temporal characteristics like duration, transitions, and pacing. For abstract media, focus on describing the properties and composition. Think about the key aspects someone needs to understand to grasp the content and structure. Examples: 'The video presents a step-by-step tutorial on assembling a device. Text overlays accompany the visual demonstrations.', 'The animated graphic shows the flow of data through a network, with arrows indicating direction and color-coding representing different types of data.', 'The abstract animation features pulsating colors and evolving geometric shapes set to a rhythmic soundtrack.'",
            examples=[
                "A diagram illustrating the water cycle",
                "A complex algebraic equation",
                "An abstract painting with bold colors",
                "A short film depicting a historical event",
            ],
        ),
    ]

    content_type: Annotated[
        str,
        msgspec.Meta(
            title="Content Type",
            description="A general categorization of the media's content. This helps to broadly define what kind of information or representation is being presented. Examples: 'Mathematical formula', 'Technical diagram', 'Architectural drawing', 'Abstract art', 'Photograph', 'Video tutorial', 'Animated infographic', 'Data visualization', 'Screencast'.",
            examples=[
                "Photograph",
                "Diagram",
                "Formula",
                "Abstract Art",
                "Video",
                "Animation",
            ],
        ),
    ]

    visual_elements: Annotated[
        Optional[Sequence[GraphicalElementDescription]],
        msgspec.Meta(
            title="Detailed Element Descriptions",
            description="A list of individual elements identified within the media, each with its own detailed description. For each element, provide its type, specific details (visual, auditory, or temporal), its role or function, and its relationships to other elements. The goal is to break down the media into its fundamental components and describe them comprehensively. This applies to all types of visual media, from objects in photographs to symbols in diagrams, shots in a video, or transitions in an animation. Focus on discrete, meaningful components.",
        ),
    ] = msgspec.field(default=None)

    structure: Annotated[
        Optional[MediaStructure],
        msgspec.Meta(
            title="Media Structure and Organization",
            description="A description of the overall structure and organization of the elements within the media. This section focuses on how the different parts are arranged and related to each other, considering both spatial and temporal aspects. Describe the layout, any significant groupings of elements, and the primary focal point or area of emphasis. This helps to understand the higher-level organization of the content.",
        ),
    ] = msgspec.field(default=None)

    dominant_features: Annotated[
        Optional[Sequence[str]],
        msgspec.Meta(
            title="Dominant Features",
            description="A list of the most striking features of the media that contribute significantly to its overall appearance and impact. This could include dominant colors, recurring patterns, distinctive shapes, lines, textures, pacing, sound design, or any other salient characteristics. Be specific and descriptive. Examples: 'Bold, contrasting colors', 'Repetitive geometric patterns', 'Fast-paced editing', 'Melancholy musical score'.",
            examples=[
                "Bright contrasting colors",
                "Repeating circular patterns",
                "Dominant horizontal lines",
                "Rapid scene changes",
            ],
        ),
    ] = msgspec.field(default=None)

    intended_purpose: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Intended Purpose or Meaning",
            description="An interpretation of the intended purpose or meaning of the media, based on its content and structure. What is it trying to convey or communicate? For a formula, it might be to express a mathematical relationship. For a diagram, to illustrate a process. For abstract art, to evoke certain emotions or ideas. For a video, to inform, entertain, or persuade. This is an interpretive field, so focus on reasonable inferences based on the evidence. Examples: 'To explain the steps in a manufacturing process', 'To visually represent a complex data set', 'To explore themes of color and form', 'To document a historical event'.",
            examples=[
                "To illustrate a scientific concept",
                "To present architectural plans",
                "To evoke a sense of calm",
                "To tell a compelling story",
            ],
        ),
    ] = msgspec.field(default=None)

    @property
    def ocr_text(self) -> str:
        """Aggregates all OCR-extracted text from the media."""
        parts: list[str] = []
        # Add individual element texts
        if self.visual_elements:
            for element in self.visual_elements:
                if element.extracted_text:
                    parts.append(element.extracted_text)

        return "\n".join(parts).strip()

    @property
    def md(self) -> str:
        md_str = f"## Overall Media Description\n{self.overall_description}\n\n"
        md_str += f"## Content Type\n{self.content_type}\n\n"

        if self.visual_elements:
            md_str += "## Detailed Element Descriptions\n"
            for element in self.visual_elements:
                md_str += element.md(indent_level=0) + "\n"

        if self.structure:
            md_str += "## Media Structure and Organization\n"
            md_str += self.structure.md(indent_level=1) + "\n"

        if self.dominant_features:
            md_str += "## Dominant Features\n"
            for feature in self.dominant_features:
                md_str += f"- {feature}\n"
            md_str += "\n"

        if self.intended_purpose:
            md_str += f"## Intended Purpose or Meaning\n{self.intended_purpose}\n\n"

        return md_str


class AudioElementDescription(msgspec.Struct, frozen=True):
    """A description of an audio element within a media file.

    Attributes:
        type: The general category of this audio element within the media
        details: A detailed description of the auditory characteristics and properties of this element
        role: The purpose or function of this audio element within the context of the media
        relationships: How this audio element is related to other elements in the media

    Example:
        >>> AudioElementDescription(
        ...     type="Speech segment",
        ...     details="The word 'example' spoken with emphasis",
        ...     role="Introduces the main subject",
        ...     relationships=["Occurs after a period of silence"]
        ... )
    """

    type: Annotated[
        str,
        msgspec.Meta(
            title="Element Type",
            description="The general category of this audio element within the media. This could be a type of sound, a segment of speech, a musical phrase, or any other distinct auditory component. Examples: 'Speech segment', 'Musical note', 'Sound effect', 'Silence', 'Jingle'.",
            examples=[
                "Speech",
                "Melody",
                "Footsteps",
                "Silence",
            ],
        ),
    ]

    details: Annotated[
        str,
        msgspec.Meta(
            title="Element Details",
            description="A detailed description of the auditory characteristics and properties of this element. For speech, provide the content or a description of the speaker's tone and delivery. For music, describe the melody, harmony, rhythm, and instrumentation. For sound effects, describe the sound and its characteristics. Be specific and descriptive. Examples: 'The spoken phrase 'Hello world' in a clear voice', 'A high-pitched sustained note on a violin', 'The sound of a door slamming shut', 'A brief period of complete silence'.",
            examples=[
                "The word 'example' spoken with emphasis",
                "A low humming sound",
                "A sharp, percussive beat",
            ],
        ),
    ]

    role: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Element Role/Function",
            description="The purpose or function of this audio element within the context of the media. How does it contribute to the overall meaning, mood, or structure? For example, in a song, describe its role in the melody or harmony. In a spoken piece, explain its informational or emotional contribution. In a soundscape, its contribution to the atmosphere. Examples: 'Conveys information about the topic', 'Creates a sense of tension', 'Marks the beginning of a new section', 'Provides background ambience'.",
            examples=[
                "Introduces the main subject",
                "Builds suspense",
                "Signals a transition",
                "Establishes the setting",
            ],
        ),
    ] = msgspec.field(default=None)

    relationships: Annotated[
        Optional[Sequence[str]],
        msgspec.Meta(
            title="Element Relationships",
            description="Describe how this audio element is related to other elements in the media. Explain its temporal relationship to others, whether it occurs before, during, or after other sounds, or how it interacts with other auditory elements. Be specific about the other elements involved. Examples: 'This musical phrase follows the introductory melody', 'The sound effect occurs simultaneously with the visual impact', 'The speaker's voice overlaps with the background music'.",
            examples=[
                "Occurs after a period of silence",
                "Plays under the main narration",
                "A response to the previous sound",
            ],
        ),
    ] = msgspec.field(default=None)

    def md(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        md_str = f"{indent}**Element Type**: {self.type}\n"
        md_str += f"{indent}**Element Details**: {self.details}\n"
        if self.role:
            md_str += f"{indent}**Role/Function**: {self.role}\n"
        if self.relationships:
            md_str += f"{indent}**Relationships**:\n"
            for rel in self.relationships:
                md_str += f"{indent}  - {rel}\n"
        return md_str


class AudioStructure(msgspec.Struct, frozen=True):
    """A description of the overall structure and organization of an audio media file.

    Attributes:
        organization: A description of how the audio elements are arranged and organized within the media
        groupings: Significant groupings of elements that appear to function together
        focal_point: The primary focal point that draws attention

    Example:
        >>> AudioStructure(
        ...     organization="A narrative with a clear beginning, middle, and end",
        ...     groupings=["The introduction of the song", "The main argument of the speech"],
        ...     focal_point="The main theme of the music"
        ... )
    """

    organization: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Overall Organization",
            description="A description of how the audio elements are arranged and organized within the media. Describe the overall structure, flow, or pattern. Is it linear, cyclical, thematic, or something else? How are the different parts connected or separated? Examples: 'A song with verse-chorus structure', 'A chronological sequence of spoken events', 'A layered soundscape with overlapping elements'.",
            examples=[
                "A narrative with a clear beginning, middle, and end",
                "A repeating musical motif",
                "A conversation with alternating speakers",
            ],
        ),
    ] = msgspec.field(default=None)
    groupings: Annotated[
        Optional[Sequence[str]],
        msgspec.Meta(
            title="Significant Groupings of Elements",
            description="Describe any notable groupings or clusters of audio elements that appear to function together or have a shared context. Explain what binds these elements together aurally or conceptually. Examples: 'The instrumental section of the song', 'A dialogue between two characters', 'A series of related sound effects'.",
            examples=[
                "The introduction of the song",
                "The main argument of the speech",
                "The sounds of a busy street",
            ],
        ),
    ] = msgspec.field(default=None)
    focal_point: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Primary Focal Point",
            description="Identify the most prominent or central audio element or section that draws the listener's attention. Explain why this element stands out (e.g., volume, pitch, prominence of a voice or instrument). If there isn't a clear focal point, describe the distribution of auditory emphasis. Examples: 'The lead vocalist's melody', 'The loudest sound effect', 'The central argument of the speech'.",
            examples=[
                "The main theme of the music",
                "The key statement in the narration",
                "A sudden loud bang",
            ],
        ),
    ] = msgspec.field(default=None)

    def md(self, indent_level: int = 1) -> str:
        indent = "  " * indent_level
        md_str = ""
        if self.organization:
            md_str += f"{indent}**Overall Organization**: {self.organization}\n"
        if self.groupings:
            md_str += f"{indent}**Significant Groupings of Elements**:\n"
            for group in self.groupings:
                md_str += f"{indent}  - {group}\n"
        if self.focal_point:
            md_str += f"{indent}**Primary Focal Point**: {self.focal_point}\n"
        return md_str


class AudioDescription(msgspec.Struct, frozen=True):
    """Detailed description of audio content.

    Attributes:
        overall_description: Comprehensive content summary
        content_type: Category (podcast, music, etc.)
        audio_elements: Individual components
        structure: Spatial organization
        dominant_auditory_features: Salient auditory characteristics
        intended_purpose: Interpreted purpose

    Example:
        >>> desc = AudioDescription(
        ...     content_type="podcast",
        ...     audio_elements=[...]
        ... )
    """

    overall_description: Annotated[
        str,
        msgspec.Meta(
            title="Overall Audio Description",
            description="Provide a comprehensive and detailed narrative describing the entire audio media, focusing on its content, structure, and key auditory elements. Imagine you are explaining the audio to someone who cannot hear it. Describe the overall purpose or what information the audio is conveying or what experience it aims to create. Detail the main components and how they are organized. Use precise language to describe auditory characteristics like pitch, tone, rhythm, tempo, and instrumentation. For abstract audio, focus on describing the sonic properties and composition. Think about the key aspects someone needs to understand to grasp the content and structure of the audio. Examples: 'The audio presents a news report detailing recent events, featuring a clear and professional narration with background music.', 'The audio is a piece of ambient music featuring layered synthesizers and natural soundscapes, creating a calming atmosphere.', 'The audio recording captures a lively conversation between two individuals, with distinct voices and occasional laughter.'",
            examples=[
                "A podcast discussing current events",
                "A musical piece with a strong melody",
                "A recording of nature sounds",
            ],
        ),
    ]

    content_type: Annotated[
        str,
        msgspec.Meta(
            title="Content Type",
            description="A general categorization of the audio's content. This helps to broadly define what kind of auditory experience or information is being presented. Examples: 'Podcast', 'Song', 'Speech', 'Sound effects', 'Ambient music', 'Audiobook', 'Interview'.",
            examples=["Podcast", "Music", "Speech", "Sound Effects"],
        ),
    ]

    audio_elements: Annotated[
        Optional[Sequence[AudioElementDescription]],
        msgspec.Meta(
            title="Detailed Audio Element Descriptions",
            description="A list of individual audio elements identified within the media, each with its own detailed description. For each element, provide its type, specific auditory details, its role or function within the audio's context, and its relationships to other elements. The goal is to break down the audio into its fundamental auditory components and describe them comprehensively. This applies to all types of audio, from spoken words in a podcast to musical notes in a song or distinct sound effects.",
        ),
    ] = msgspec.field(default=None)

    structure: Annotated[
        Optional[AudioStructure],
        msgspec.Meta(
            title="Audio Structure and Organization",
            description="A description of the overall structure and organization of the audio elements within the media. This section focuses on how the different parts are arranged and related to each other. Describe the overall organization, any significant groupings of elements, and the primary focal point or area of emphasis. This helps to understand the higher-level organization of the audio's content.",
        ),
    ] = msgspec.field(default=None)

    dominant_auditory_features: Annotated[
        Optional[Sequence[str]],
        msgspec.Meta(
            title="Dominant Auditory Features",
            description="A list of the most striking auditory features of the audio that contribute significantly to its overall character and impact. This could include dominant melodies, rhythmic patterns, distinctive voices or timbres, recurring sound effects, or any other salient auditory characteristics. Be specific and descriptive. Examples: 'A strong, repetitive beat', 'A high-pitched, clear female voice', 'Frequent use of echo and reverb', 'A melancholic piano melody'.",
            examples=[
                "A fast tempo",
                "A deep bassline",
                "Clear and articulate speech",
            ],
        ),
    ] = msgspec.field(default=None)

    intended_purpose: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Intended Purpose or Meaning",
            description="An interpretation of the intended purpose or meaning of the audio, based on its content and structure. What is the audio trying to convey or communicate? For a song, it might be to express emotions. For a podcast, to inform or entertain. For sound effects, to create a specific atmosphere. This is an interpretive field, so focus on reasonable inferences based on the auditory evidence. Examples: 'To tell a story through sound', 'To provide information on a specific topic', 'To create a relaxing and immersive soundscape', 'To evoke feelings of joy and excitement'.",
            examples=[
                "To entertain the listener",
                "To educate on a particular subject",
                "To create a sense of atmosphere",
            ],
        ),
    ] = msgspec.field(default=None)

    @property
    def md(self) -> str:
        md_str = ""
        md_str += f"## Overall Audio Description\n{self.overall_description}\n\n"
        md_str += f"## Content Type\n{self.content_type}\n\n"

        if self.audio_elements:
            md_str += "## Detailed Audio Element Descriptions\n"
            for element in self.audio_elements:
                md_str += element.md(indent_level=0) + "\n"

        if self.structure:
            md_str += "## Audio Structure and Organization\n"
            md_str += self.structure.md(indent_level=1) + "\n"

        if self.dominant_auditory_features:
            md_str += "## Dominant Auditory Features\n"
            for feature in self.dominant_auditory_features:
                md_str += f"- {feature}\n"
            md_str += "\n"

        if self.intended_purpose:
            md_str += f"## Intended Purpose or Meaning\n{self.intended_purpose}\n\n"

        return md_str
