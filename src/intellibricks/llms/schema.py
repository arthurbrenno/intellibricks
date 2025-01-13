"""llms schemas models"""

from __future__ import annotations

import base64

# import dataclasses
import datetime
import inspect
import re
import uuid

# from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
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
from architecture.logging import LoggerFactory
from architecture.types import NOT_GIVEN, NotGiven
from architecture.utils.decorators import ensure_module_installed

from intellibricks.llms.util import (
    HTMLToMarkdownParser,
    get_parts_llm_described_text,
    get_parts_raw_text,
)
from intellibricks.util import (
    get_file_extension,
    is_file_url,
    is_url,
)

from .constants import FinishReason, Language
from .types import AIModel

if TYPE_CHECKING:
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

    from intellibricks.llms.types import FileExtension


_P = TypeVar("_P", bound="Part")
M = TypeVar(
    "M", bound="Message"
)  # NOTE: if I used "type M = Message" the "type[M]" would not work in the function signature

T = TypeVar("T", default="RawResponse")
R = TypeVar("R", default=Any)

logger = LoggerFactory.create(__name__)


class GenerationConfig(msgspec.Struct, frozen=True, kw_only=True):
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


class MimeType(str, Enum):
    image_jpeg = "image/jpeg"
    image_png = "image/png"
    audio_mp3 = "audio/mp3"
    audio_wav = "audio/wav"
    video_mp4 = "video/mp4"
    video_avi = "video/avi"
    video_mov = "video/mov"
    video_webm = "video/webm"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_extension(
        cls,
        extension: Literal[
            ".jpeg",
            ".jpg",
            ".png",
            ".gif",
            ".mp3",
            ".wav",
            ".mp4",
        ],
    ) -> MimeType:
        match extension:
            case ".jpeg":
                return cls.image_jpeg
            case ".jpg":
                return cls.image_jpeg
            case ".png":
                return cls.image_png
            case ".gif":
                return cls.image_png
            case ".mp3":
                return cls.audio_mp3
            case ".wav":
                return cls.audio_wav
            case ".mp4":
                return cls.video_mp4


class RawResponse(msgspec.Struct, frozen=True):
    """Null object for the response from the model."""

    def __bool__(self) -> Literal[False]:
        return False


class TraceParams(TypedDict, total=False):
    """
    Parameters for updating the current trace, including metadata and context information.

    This TypedDict is used to specify the parameters that can be updated for the current trace.
    Each field corresponds to an attribute of the trace that can be dynamically modified
    during execution. These parameters are useful for categorization, filtering, and analysis
    within the Langfuse UI.

    Attributes:
        name (Optional[str]):
            Identifier of the trace. Useful for sorting and filtering in the UI.

        input (Optional[Any]):
            The input parameters of the trace, providing context about the observed operation
            or function call.

        output (Optional[Any]):
            The output or result of the trace.

        user_id (Optional[str]):
            The ID of the user that triggered the execution. Used to provide user-level analytics.

        session_id (Optional[str]):
            Used to group multiple traces into a session in Langfuse. Typically your own session
            or thread identifier.

        version (Optional[str]):
            The version of the trace type. Helps in understanding how changes to the trace type
            affect metrics and is useful for debugging.

        release (Optional[str]):
            The release identifier of the current deployment. Helps in understanding how changes
            in different deployments affect metrics and is useful for debugging.

        metadata (Optional[Any]):
            Additional metadata for the trace. Can be any JSON-serializable object. Metadata is
            merged when updated via the API.

        tags (Optional[list[str]]):
            Tags used to categorize or label traces. Traces can be filtered by tags in the
            Langfuse UI and through the GET API.

        public (Optional[bool]):
            Indicates whether the trace is public. If set to `True`, the trace is accessible publicly;
            otherwise, it remains private.
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
##     ## ########  ##        ######
##     ## ##     ## ##       ##    ##
##     ## ##     ## ##       ##
##     ## ########  ##        ######
##     ## ##   ##   ##             ##
##     ## ##    ##  ##       ##    ##
 #######  ##     ## ########  ######
"""


class WebsiteUrl(msgspec.Struct, frozen=True):
    url: str

    def __post_init__(self) -> None:
        if not is_url(self.url):
            raise ValueError(f"Invalid URL ({self.url})")


class FileUrl(msgspec.Struct, frozen=True):
    url: str

    def get_extension(
        self,
    ) -> FileExtension:
        return get_file_extension(self.url)

    def __post_init__(self) -> None:
        if not is_url(self.url):
            raise ValueError(f"Invalid URL ({self.url})")

        if not is_file_url(self.url):
            raise ValueError(f"Invalid file URL ({self.url})")


"""
########     ###    ########  ########  ######
##     ##   ## ##   ##     ##    ##    ##    ##
##     ##  ##   ##  ##     ##    ##    ##
########  ##     ## ########     ##     ######
##        ######### ##   ##      ##          ##
##        ##     ## ##    ##     ##    ##    ##
##        ##     ## ##     ##    ##     ######
"""


class Part(msgspec.Struct, tag_field="type", frozen=True):
    """
    Represents a part of a multi-content message. The use-case is for
    multimodal completions.
    """

    @classmethod
    def from_text(cls, text: str) -> TextPart:
        return TextPart(text=text)

    @classmethod
    def from_image(cls, image: Image) -> ImageFilePart:
        ensure_module_installed("PIL.Image", "pillow")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return ImageFilePart(
            data=image_str.encode("utf-8"),
            mime_type=MimeType(
                {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                }.get(image.format or "PNG", "image/jpeg")
            ),
        )

    @overload
    @classmethod
    def from_url(cls, url: WebsiteUrl) -> WebsitePart: ...

    @overload
    @classmethod
    def from_url(cls, url: FileUrl) -> FilePart: ...

    @overload
    @classmethod
    def from_url(cls, url: str) -> WebsitePart | FilePart: ...

    @classmethod
    def from_url(cls, url: str | WebsiteUrl | FileUrl) -> WebsitePart | FilePart:
        match url:
            case str():
                if not is_url(url):
                    raise ValueError(f"Invalid URL ({url})")

                # Check if the URL is a website URL or a file URL
                if is_file_url(url):
                    return FilePart.from_extension(
                        url=url, extension=get_file_extension(url)
                    )

                return WebsitePart(url=url)

            case WebsiteUrl():
                return WebsitePart(url=url.url)
            case FileUrl():
                return FilePart.from_extension(
                    extension=url.get_extension(), url=url.url
                )

    @classmethod
    def from_input_audio(
        cls, data_or_url: str, mime_type: Literal["audio/mp3"]
    ) -> AudioFilePart:
        if is_url(data_or_url):
            return AudioFilePart(url=data_or_url, mime_type=MimeType(mime_type))

        return AudioFilePart(
            data=data_or_url.encode("utf-8"), mime_type=MimeType(mime_type)
        )

    @classmethod
    def from_openai_part(
        cls, openai_part: OpenAIChatCompletionContentPartParam
    ) -> Part:
        match openai_part["type"]:
            case "text":
                    return TextPart(text=openai_part["text"])
            case "image_url":
                url_or_base_64 = openai_part["image_url"]["url"]
                if is_url(url_or_base_64):
                    return ImageFilePart(
                        url=url_or_base_64, mime_type=MimeType("image/jpeg")
                    )

                return ImageFilePart(
                    data=url_or_base_64.encode("utf-8"),
                    mime_type=MimeType("image/jpeg"),
                )
            case "input_audio":
                return AudioFilePart(
                    data=base64.b64decode(openai_part["input_audio"]["data"]),
                    mime_type=MimeType(f"audio/{openai_part['input_audio']['format']}"),
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
                return ImageFilePart(
                    url=file_data.file_uri, mime_type=MimeType(mime_type)
                )
            elif mime_type.startswith("audio/"):
                return AudioFilePart(
                    url=file_data.file_uri, mime_type=MimeType(mime_type)
                )
            elif mime_type.startswith("video/"):
                return VideoFilePart(
                    url=file_data.file_uri, mime_type=MimeType(mime_type)
                )
            else:
                logger.warning(
                    f"Unknown file type: {mime_type}, falling back to ImageFilePart."
                )
                return ImageFilePart(
                    url=file_data.file_uri, mime_type=MimeType(mime_type)
                )

        # Check inline data
        inline_data: Optional[types.Blob] = google_part.inline_data
        if inline_data is not None:
            if inline_data.mime_type is None:
                raise ValueError("MIME type is required for inline_data")
            mime_type = inline_data.mime_type.lower()
            data = inline_data.data

            if mime_type.startswith("image/"):
                return ImageFilePart(
                    data=data or NOT_GIVEN, mime_type=MimeType(mime_type)
                )
            elif mime_type.startswith("audio/"):
                return AudioFilePart(
                    data=data or NOT_GIVEN, mime_type=MimeType(mime_type)
                )
            elif mime_type.startswith("video/"):
                return VideoFilePart(
                    data=data or NOT_GIVEN, mime_type=MimeType(mime_type)
                )
            else:
                return ImageFilePart(
                    data=data or NOT_GIVEN, mime_type=MimeType(mime_type)
                )

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
    def to_anthropic_part(self) -> dict[str, Any]:
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
    url: str

    def __post_init__(self) -> None:
        if not is_url(self.url):
            raise ValueError(f"Invalid URL ({self.url})")

    def get_md_contents(self, timeout: float = 5.0) -> str:
        """Get the contents of the website and convert HTML to Markdown."""
        import requests

        response = requests.get(self.url, timeout=timeout)
        response.raise_for_status()
        html_text: str = response.text

        parser = HTMLToMarkdownParser()
        parser.feed(html_text)
        return parser.get_markdown()

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
    def to_anthropic_part(self) -> dict[str, Any]:
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
    text: str

    @ensure_module_installed("anthropic", "anthropic")
    @override
    def to_anthropic_part(self) -> dict[str, Any]:
        ensure_module_installed("anthropic.types.text_block_param", "anthropic")
        from anthropic.types.text_block_param import TextBlockParam

        return cast(dict[str, Any], TextBlockParam(text=self.text, type="text"))

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
    """Represents a tool part in a multi-content message."""

    tool_name: str
    """The name of the tool."""

    tool_call_id: str
    """The ID of the tool call."""

    tool_response: str
    """The tool response."""

    @override
    def to_anthropic_part(self) -> dict[str, Any]:
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


class FilePart(Part, frozen=True, tag="file"):
    mime_type: MimeType

    """The MIME type of the file."""

    url: str | NotGiven = msgspec.field(default_factory=lambda: NOT_GIVEN)
    """The URL of the file. If located on the web."""

    data: bytes | NotGiven = msgspec.field(default_factory=lambda: NOT_GIVEN)
    """Base64 encoded file data."""

    @overload
    @classmethod
    def from_extension(
        cls,
        extension: Literal[".jpeg", ".jpg", ".png", ".gif"],
        url: Optional[str] = None,
        data: Optional[bytes] = None,
    ) -> ImageFilePart: ...

    @overload
    @classmethod
    def from_extension(
        cls,
        extension: Literal[".mp3", ".wav"],
        url: Optional[str] = None,
        data: Optional[bytes] = None,
    ) -> AudioFilePart: ...

    @overload
    @classmethod
    def from_extension(
        cls,
        extension: Literal[".mp4", ".avi", ".mov", ".webm"],
        url: Optional[str] = None,
        data: Optional[bytes] = None,
    ) -> VideoFilePart: ...

    @classmethod
    def from_extension(
        cls,
        extension: FileExtension,
        url: Optional[str] = None,
        data: Optional[bytes] = None,
    ) -> VideoFilePart | AudioFilePart | ImageFilePart:
        if url is None and data is None:
            raise ValueError("Either url or data must be provided.")

        if extension in {".jpeg", ".jpg", ".png", ".gif"}:
            return ImageFilePart(
                url=url or NOT_GIVEN,
                data=data or NOT_GIVEN,
                mime_type=MimeType(f"image/{extension[1:]}"),
            )

        if extension in {".mp3", ".wav"}:
            return AudioFilePart(
                url=url or NOT_GIVEN,
                data=data or NOT_GIVEN,
                mime_type=MimeType(f"audio/{extension[1:]}"),
            )

        if extension in {".mp4", ".avi", ".mov", ".webm"}:
            return VideoFilePart(
                url=url or NOT_GIVEN,
                data=data or NOT_GIVEN,
                mime_type=MimeType(f"video/{extension[1:]}"),
            )

        return ImageFilePart(
            url=url or NOT_GIVEN,
            data=data or NOT_GIVEN,
            mime_type=MimeType(f"image/{extension[1:]}"),
        )

    def __post_init__(self) -> None:
        if self.data is NotGiven and self.url is NotGiven:
            raise ValueError("Either data or url must be provided.")


class VideoFilePart(FilePart, frozen=True, tag="video"):
    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        from google.genai.types import Part as GenAIPart

        if self.data:
            return GenAIPart.from_bytes(data=self.data, mime_type=self.mime_type)

        return GenAIPart.from_uri(
            file_uri=cast(str, self.url), mime_type=self.mime_type
        )

    @ensure_module_installed("anthropic", "anthropic")
    @override
    def to_anthropic_part(self) -> dict[str, Any]:
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
            f"URL: {self.url if self.url else 'N/A'}\n"
            f"Data length: {len(self.data) if self.data else 0}\n"
            f"<|end_video_part|>"
        )

    @override
    def __str__(self) -> str:
        return f"VideoFilePart({self.mime_type})"


class AudioFilePart(FilePart, frozen=True, tag="audio"):
    @override
    def to_anthropic_part(self) -> dict[str, Any]:
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
                data=base64.b64encode(self.data).decode("utf-8"),
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

        if self.data:
            return GenAIPart.from_bytes(data=self.data, mime_type=self.mime_type)

        return GenAIPart.from_uri(
            file_uri=cast(str, self.url), mime_type=self.mime_type
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
            f"<|audio_part|>\n"
            f"MIME type: {self.mime_type}\n"
            f"URL: {self.url if self.url else 'N/A'}\n"
            f"Data length: {len(self.data) if self.data else 0}\n"
            f"<|end_audio_part|>"
        )

    @override
    def __str__(self) -> str:
        return f"AudioFilePart({self.mime_type})"


class ImageFilePart(FilePart, frozen=True, tag="image"):
    @ensure_module_installed("anthropic.types.image_block_param", "anthropic")
    @override
    def to_anthropic_part(self) -> dict[str, Any]:
        ensure_module_installed("anthropic.types.image_block_param", "anthropic")
        from anthropic.types.image_block_param import ImageBlockParam, Source

        if not self.data:
            raise ValueError("Image data (bytes) is required.")

        return cast(
            dict[str, Any],
            ImageBlockParam(
                source=Source(
                    data=base64.b64encode(self.data).decode("utf-8"),
                    media_type=cast(
                        Literal["image/jpeg", "image/png", "image/gif", "image/webp"],
                        self.mime_type,
                    ),
                    type="base64",
                ),
                type="image",
            ),
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_part(self) -> GroqChatCompletionContentPartParam:
        from groq.types.chat.chat_completion_content_part_image_param import (
            ChatCompletionContentPartImageParam,
            ImageURL,
        )

        if self.data:
            return ChatCompletionContentPartImageParam(
                image_url=ImageURL(url=base64.b64encode(self.data).decode("utf-8")),
                type="image_url",
            )

        return ChatCompletionContentPartImageParam(
            image_url=ImageURL(url=cast(str, self.url), detail="auto"),
            type="image_url",
        )

    @ensure_module_installed("openai", "openai")
    @override
    def to_openai_part(self) -> OpenAIChatCompletionContentPartParam:
        from openai.types.chat.chat_completion_content_part_image_param import (
            ChatCompletionContentPartImageParam,
            ImageURL,
        )

        if self.data:
            return ChatCompletionContentPartImageParam(
                image_url=ImageURL(url=self.data.decode("utf-8"), detail="auto"),
                type="image_url",
            )

        return ChatCompletionContentPartImageParam(
            image_url=ImageURL(url=cast(str, self.url), detail="auto"),
            type="image_url",
        )

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_part(self) -> GenAIPart:
        from google.genai.types import Part as GenAIPart

        if self.url:
            return GenAIPart.from_uri(self.url, mime_type=self.mime_type)

        return GenAIPart.from_bytes(
            data=cast(bytes, self.data), mime_type=self.mime_type
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
            f"URL: {self.url if self.url else 'N/A'}\n"
            f"Data length: {len(self.data) if self.data else 0}\n"
            f"<|end_image_part|>"
        )

    @override
    def __str__(self) -> str:
        return f"ImageFilePart({self.mime_type})"


class ToolCallPart(Part, frozen=True, tag="tool_call"):
    function_name: str
    arguments: dict[str, Any]

    @ensure_module_installed("anthropic", "anthropic")
    @override
    def to_anthropic_part(self) -> dict[str, Any]:
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
    """Represents a prompt"""

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

        return ChatCompletionDeveloperMessageParam(
            content=content,
            role="developer",
            name=cast(str, self.name),  # OpenAI doesnt support Optional[str], idk why
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        from groq.types.chat.chat_completion_system_message_param import (
            ChatCompletionSystemMessageParam,
        )

        content: str = get_parts_llm_described_text(self.contents)

        return ChatCompletionSystemMessageParam(content=content, role="system")

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_format(self) -> GenaiContent:
        from google.genai.types import Content as GenaiContent

        parts = [part.to_google_part() for part in self.contents]
        return GenaiContent(role="user", parts=parts)

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    def to_cerebras_format(self) -> CerebrasMessage:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            MessageSystemMessageRequestTyped,
        )

        return MessageSystemMessageRequestTyped(
            content=get_parts_llm_described_text(self.contents),
            role="system",
            name=self.name,
        )


class SystemMessage(DeveloperMessage, frozen=True, tag="system"):
    pass


class UserMessage(Message, frozen=True, tag="user"):
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

        return ChatCompletionUserMessageParam(
            role="user",
            content=[
                part.to_openai_part()
                for part in self.contents
            ],
            name=cast(str, self.name),
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        from groq.types.chat.chat_completion_user_message_param import (
            ChatCompletionUserMessageParam,
        )

        return ChatCompletionUserMessageParam(
            role="user",
            content=[part.to_groq_part() for part in self.contents],
            name=self.name or "",
        )

    @ensure_module_installed("google.genai", "google-genai")
    @override
    def to_google_format(self) -> GenaiContent:
        from google.genai.types import Content as GenaiContent

        parts = [part.to_google_part() for part in self.contents]
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


class ToolCallSequence[R = Any](msgspec.Struct, frozen=True):
    """
    Container for a heterogeneous list of ToolCall objects.
    Each item in *Ts can be a different type, e.g. [str, int, float].
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

    def __getitem__(self, index: int) -> ToolCall[R]:
        """
        Return a ToolCall[object] so we avoid Any.
        But we lose precise type knowledge about each index.
        """
        return self.sequence[index]


class AssistantMessage[R = Any](Message, frozen=True, kw_only=True, tag="assistant"):
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

        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=[
                cast(ChatCompletionContentPartTextParam, content.to_openai_part())
                for content in self.contents
            ],
            name=self.name or "",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=tool_call.id,
                    function=tool_call.called_function.to_openai_called_function(),
                    type="function",
                )
                for tool_call in self.tool_calls
            ]
            if self.tool_calls
            else [],
        )

    @ensure_module_installed("groq", "groq")
    @override
    def to_groq_format(self) -> GroqChatCompletionMessageParam:
        from groq.types.chat.chat_completion_assistant_message_param import (
            ChatCompletionAssistantMessageParam,
        )
        from groq.types.chat.chat_completion_message_tool_call_param import (
            ChatCompletionMessageToolCallParam,
        )

        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=get_parts_llm_described_text(self.contents),
            name=self.name or "",
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

        return GenaiContent(
            role="model",
            parts=[part.to_google_part() for part in self.contents] + tool_parts,
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
            tool_calls=[
                MessageAssistantMessageRequestToolCallTyped(
                    id=tool_call.id,
                    function=MessageAssistantMessageRequestToolCallFunctionTyped(
                        arguments=msgspec.json.encode(
                            tool_call.called_function.arguments
                        ).decode("utf-8"),
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
    parsed: Annotated[
        T,
        msgspec.Meta(
            title="Structured Model",
            description="Structured model of the message",
        ),
    ] = msgspec.field(default=cast(T, RawResponse()))


class ToolMessage(Message, frozen=True, tag="tool"):
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


MessageType: TypeAlias = (
    DeveloperMessage | UserMessage | ToolMessage | AssistantMessage[Any] | SystemMessage
)


class MessageFactory(msgspec.Struct, frozen=True):
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
    """LogProb of a token."""

    token: str = msgspec.field(default_factory=str)
    logprob: float = msgspec.field(default_factory=float)
    bytes: list[int] = msgspec.field(default_factory=list)


class MessageChoice[T](msgspec.Struct, frozen=True, kw_only=True):
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
    """Breakdown of tokens used in prompt"""

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
    """Breakdown of tokens generated in completion"""

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
    """Immutable, memory-efficient representation of a completion response from a chat model."""

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
            elapsed_time=self.elapsed_time + other.elapsed_time,
            id=uuid.uuid4().__str__(),
            model=self.model,
            system_fingerprint=self.system_fingerprint,
            choices=list(self.choices) + list(other.choices),
            usage=self.usage + other.usage,
        )


class Property(msgspec.Struct, frozen=True, kw_only=True):
    """Represents a property within a parameter."""

    type: str | Literal["object", "string", "number", "integer", "boolean", "array"]
    description: Optional[str] = None
    enum: Optional[list[Any]] = None
    properties: Optional[dict[str, Property]] = None  # For nested objects
    items: Optional[Property] = None  # For arrays


class Parameter(msgspec.Struct, frozen=True, kw_only=True):
    """Represents a single parameter."""

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
    """Represents a function with a name, description, and parameters."""

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
    def _extract_param_description(func: Callable[..., Any], param_name: str) -> Optional[str]:
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
    """Represents a function that was called with arguments by an AI."""

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
