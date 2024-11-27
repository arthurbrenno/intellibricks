"""llms schemas models"""

from __future__ import annotations

from abc import ABC
import dataclasses
import datetime
import re
from typing_extensions import NotRequired
import uuid
from typing import (
    Annotated,
    Any,
    Final,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
    TypedDict,
    Union,
    cast,
)

import validators
from architecture import BaseModel, Meta, field
from architecture.logging import logger
from bs4 import BeautifulSoup, NavigableString
from llama_index.core.base.llms.types import LogProb
from llama_index.core.base.llms.types import MessageRole as LlamaIndexMessageRole
from llama_index.core.llms import ChatMessage as LlamaIndexChatMessage
from tiktoken.core import Encoding

from intellibricks.util import deserialize_json

from .constants import (
    AIModel,
    FinishReason,
    MessageRole,
)
from .exceptions import MessageNotParsedError

T = TypeVar("T")
ReplacementText = str
ReplacementValue = str


class Tag(BaseModel):
    tag_name: str
    content: Optional[str] = field(default=None)
    attributes: dict[str, Optional[str]] = field(default_factory=dict)

    @classmethod
    def from_string(
        cls,
        string: str,
        *,
        tag_name: Optional[str] = None,
        attributes: Optional[dict[str, Optional[str]]] = None,
    ) -> Optional[Tag]:
        """
        Create a Tag instance from a string containing a tag.

        This method searches for a tag in the given string and creates a Tag instance
        if a matching tag is found. It can optionally filter by tag name or attributes.

        Args:
            string (str): The input string containing the tag.
            tag_name (Optional[str], optional): If provided, only match tags with this name.
            attributes (Optional[dict[str, str]], optional): If provided, only match tags with these attributes.

        Returns:
            Optional[Tag]: A Tag instance if a matching tag is found, None otherwise.
        """
        # logger.debug(f"Parsing tag from string: {string}")
        # logger.debug(f"Tag name: {tag_name}, Attributes: {attributes}")

        # Remove leading and trailing code block markers if present
        string = string.strip()
        if string.startswith("```"):
            # Remove the first line (e.g., ```xml)
            first_newline = string.find("\n")
            if first_newline != -1:
                string = string[first_newline + 1 :]
            # Remove the last triple backticks
            if string.endswith("```"):
                string = string[:-3]

        # Initialize code block placeholders
        code_blocks = {}

        # Function to replace code blocks with placeholders
        def replace_code_blocks(match: re.Match) -> str:
            code_block = match.group(0)
            placeholder = f"__CODE_BLOCK_{uuid.uuid4()}__"
            code_blocks[placeholder] = code_block
            return placeholder

        # Replace fenced code blocks (triple backticks)
        string_with_placeholders = re.sub(
            r"```[\s\S]*?```", replace_code_blocks, string
        )
        # Replace inline code blocks (single backticks)
        string_with_placeholders = re.sub(
            r"`[^`]*`", replace_code_blocks, string_with_placeholders
        )

        # Parse the string with BeautifulSoup
        soup = BeautifulSoup(string_with_placeholders, "html.parser")

        # Find the tag
        if tag_name:
            if attributes:
                elem = soup.find(tag_name, attrs=attributes)
            else:
                elem = soup.find(tag_name)
        else:
            elem = soup.find()

        if isinstance(elem, NavigableString):
            raise ValueError("Element cannot be instance of NavigableString")

        if elem is not None:
            elem_attributes: dict[str, Optional[str]] = dict(elem.attrs)
            # Get the inner HTML content of the tag
            content = "".join(str(child) for child in elem.contents).strip()

            # Replace placeholders with original code blocks in content
            for placeholder, code_block in code_blocks.items():
                content = content.replace(placeholder, code_block)

            return cls(
                tag_name=elem.name or "",
                content=content,
                attributes=elem_attributes,
            )

        logger.debug("No matching tag found.")
        return None

    def as_object(self) -> dict[str, Any]:
        """
        Extracts the content of the tag as a Python dictionary by parsing the JSON content.

        This method is extremely robust and can handle various nuances in the JSON content, such as:
        - JSON content wrapped in code blocks with backticks (e.g., ```json ... ```)
        - JSON content starting with '{'
        - JSON content with unescaped newlines within strings
        - JSON content with inner backticks in some values
        - Complex and nested JSON structures

        Returns:
            dict[str, Any]: The parsed JSON content as a Python dictionary.

        Raises:
            ValueError: If no valid JSON content is found in the tag or if the JSON content is not a dictionary.

        Examples:
            >>> tag = Tag(content='```json\\n{\\n  "key": "value"\\n}\\n```')
            >>> tag.as_object()
            {'key': 'value'}

            >>> tag = Tag(content='Some text before { "key": "value" } some text after')
            >>> tag.as_object()
            {'key': 'value'}

            >>> tag = Tag(content='{"key": "value with backticks ``` inside"}')
            >>> tag.as_object()
            {'key': 'value with backticks ``` inside'}

            >>> tag = Tag(content='[1, 2, 3]')
            Traceback (most recent call last):
                ...
            ValueError: JSON content is not a dictionary.

            >>> tag = Tag(content=None)
            Traceback (most recent call last):
                ...
            ValueError: Tag content is None.
        """
        if self.content is None:
            raise ValueError("Tag content is None.")

        content: str = self.content.strip()

        try:
            parsed_obj: dict[str, Any] = deserialize_json(content)
            return parsed_obj
        except ValueError:
            raise ValueError("No valid JSON content found in the tag.")

    @staticmethod
    def _parse_attributes(attributes_string: str) -> dict[str, str]:
        """Parse the attributes string into a dictionary."""
        return dict(re.findall(r'(\w+)="([^"]*)"', attributes_string))

    def as_string(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.content if self.content is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag_name": self.tag_name,
            "content": self.content,
            "attributes": self.attributes,
        }


# FACADE
@dataclasses.dataclass
class Part(ABC):
    type: Literal["image_url", "text", "input_audio"]

    @classmethod
    def from_string(cls: Type[Part], string: str) -> TextContentPart:
        return TextContentPart(text=string)

    @classmethod
    def from_uri(
        cls: Type[Part], url: str, detail: Optional[str] = None
    ) -> ImageUrlPart:
        return ImageUrlPart(image_url=ImageUrl(url=url, detail=detail))

    @classmethod
    def from_input_audio(cls: Type[Part], data: str, format: str) -> InputAudioPart:
        return InputAudioPart(input_audio=InputAudio(data=data, format=format))

    def dict(self):
        return {k: str(v) for k, v in dataclasses.asdict(self).items()}


class InputAudio(TypedDict):
    data: str
    format: str


class ImageUrl(TypedDict):
    url: str
    detail: NotRequired[Optional[str]]


@dataclasses.dataclass(kw_only=True)
class InputAudioPart(Part):
    type: Literal["image_url", "text", "input_audio"] = dataclasses.field(
        default=cast(Literal["image_url", "text", "input_audio"], lambda: "input_audio")
    )
    input_audio: InputAudio


@dataclasses.dataclass(kw_only=True)
class TextContentPart(Part):
    type: Literal["image_url", "text", "input_audio"] = dataclasses.field(
        default=cast(Literal["image_url", "text", "input_audio"], lambda: "text")
    )
    text: str


@dataclasses.dataclass(kw_only=True)
class ImageUrlPart(Part):
    type: Literal["image_url", "text", "input_audio"] = dataclasses.field(
        default=cast(Literal["image_url", "text", "input_audio"], lambda: "type")
    )
    image_url: ImageUrl

    def __post_init__(self) -> None:
        validators.url(self.image_url["url"])


class Prompt(BaseModel):
    """Represents a prompt"""

    content: Annotated[
        str,
        Meta(
            title="Content",
            description="The content of the prompt",
            examples=[
                "Hello! How are you?",
                "I need help on solving a Python problem.",
                "Hi, my name is {{name}}.",
            ],
        ),
    ]

    def compile(
        self, **replacements: dict[ReplacementText, ReplacementValue]
    ) -> Prompt:
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

        def replace_match(match: re.Match) -> str:
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


class PromptTokensDetails(BaseModel):
    """Breakdown of tokens used in prompt"""

    audio_tokens: Annotated[
        Optional[int],
        Meta(
            title="Audio Tokens",
            description="The number of audio tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    cached_tokens: Annotated[
        Optional[int],
        Meta(
            title="Cached Tokens",
            description="The number of cached tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]


class CompletionTokensDetails(BaseModel):
    """Breakdown of tokens generated in completion"""

    audio_tokens: Annotated[
        Optional[int],
        Meta(
            title="Audio Tokens",
            description="The number of audio tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    reasoning_tokens: Annotated[
        Optional[int],
        Meta(
            title="Reasoning Tokens",
            description="Tokens generated by the model for reasoning.",
        ),
    ]


class Usage(BaseModel):
    prompt_tokens: Annotated[
        Optional[int],
        Meta(
            title="Prompt Tokens",
            description="The number of tokens consumed by the input prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    completion_tokens: Annotated[
        Optional[int],
        Meta(
            title="Completion Tokens",
            description="The number of tokens generated in the completion response.",
            examples=[12, 102, 32],
        ),
    ]

    input_cost: Annotated[
        Optional[float],
        Meta(
            title="USD Cost",
            description="The cost of the input prompt in USD.",
            examples=[0.02, 0.1, 0.03],
        ),
    ]

    output_cost: Annotated[
        Optional[float],
        Meta(
            title="USD Cost",
            description="The cost of the output completion in USD.",
            examples=[0.01, 0.15, 0.07],
        ),
    ]

    total_cost: Annotated[
        Optional[float],
        Meta(
            title="USD Cost",
            description="The cost of the completion in USD.",
            examples=[0.03, 0.25, 0.1],
        ),
    ]

    total_tokens: Annotated[
        Optional[int],
        Meta(
            title="Total Tokens",
            description="The total number of tokens consumed, including both prompt and completion.",
            examples=[21, 324, 12],
        ),
    ]

    prompt_tokens_details: Annotated[
        PromptTokensDetails,
        Meta(
            title="Prompt Tokens Details",
            description="Breakdown of tokens used in the prompt.",
        ),
    ]

    completion_tokens_details: Annotated[
        CompletionTokensDetails,
        Meta(
            title="Completion Tokens Details",
            description="Breakdown of tokens generated in completion.",
        ),
    ]


class Message(BaseModel, kw_only=True):
    role: Annotated[
        MessageRole,
        Meta(
            title="Message Role",
            description="The role of the message sender",
            examples=["user", "system", "assistant"],
        ),
    ] = MessageRole.USER

    content: Annotated[
        Union[str, list[Part]],
        Meta(
            title="Message Content",
            description="The content of the message",
            examples=[
                "Hello! How are you?",
                "I need help on solving a Python problem.",
            ],
        ),
    ]

    name: Annotated[
        Optional[str],
        Meta(
            title="Name",
            description="An optional name for the participant. Provides the model information to differentiate between participants of the same role.",
            examples=["Alice", "Bob", "Ana"],
        ),
    ] = None

    def extract_tag(
        self,
        *,
        name: Optional[str] = None,
        attributes: Optional[dict[str, Optional[str]]] = None,
    ) -> Optional[Tag]:
        """
        Extracts a tag from the message content based on tag name and/or identifier.
        Uses regex, BeautifulSoup, and XML parsing for robust extraction.

        Args:
            tag_name (Optional[str]): The name of the tag to extract.
            attributes (Optional[dict[str, str]]): The attributes of the tag to extract.

        Returns:
            Optional[Tag]: The extracted tag, or None if not found.

        Raises:
            ValueError: If neither tag_name nor identifier is provided.
        """
        if self.content is None:
            return None

        return Tag.from_string(self.content, tag_name=name, attributes=attributes)

    def to_llama_index_chat_message(self) -> LlamaIndexChatMessage:
        return LlamaIndexChatMessage(
            role=LlamaIndexMessageRole(self.role),
            content=self.content
            if self.name is None
            else f"{self.name}: {self.content}",
        )

    @classmethod
    def from_llama_index_message(cls, message: LlamaIndexChatMessage) -> Message:
        return cls(role=MessageRole(message.role.value), content=message.content)

    def count_tokens(self, encoder: Encoding) -> int:
        return len(self.get_tokens(encoder=encoder))

    def get_tokens(self, encoder: Encoding) -> list[int]:
        if self.content is None:
            return []
        tokens: list[int] = encoder.encode(text=self.content)
        return tokens


class CompletionMessage(Message, Generic[T]):
    parsed: Annotated[
        T,
        Meta(
            title="Structured Model",
            description="Structured model of the message",
        ),
    ]


class MessageChoice(BaseModel, Generic[T], tag=True):  # type: ignore
    index: Annotated[
        int,
        Meta(
            title="Index",
            description="Index of the choice in the list of choices returned by the model.",
            examples=[0, 1, 2],
        ),
    ]

    message: Annotated[
        CompletionMessage[T],
        Meta(
            title="Message",
            description="The message content for this choice, including role and text.",
            examples=[
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Hello there, how may I assist you today?",
                )
            ],
        ),
    ]

    logprobs: Annotated[
        Optional[list[list[LogProb]]],
        Meta(
            title="Log Probability",
            description="Log probability of the choice. Currently always None, reserved for future use.",
            examples=[None],
        ),
    ] = None

    finish_reason: Annotated[
        FinishReason,
        Meta(
            title="Finish Reason",
            description="The reason why the model stopped generating tokens for this choice.",
            examples=[
                "stop",
                "length",
                "content_filter",
                "tool_calls",
                FinishReason.STOP,
                FinishReason.LENGTH,
                FinishReason.CONTENT_FILTER,
                FinishReason.TOOL_CALLS,
                FinishReason.NONE,
            ],
        ),
    ] = FinishReason.NONE

    def __post_init__(self) -> None:
        if isinstance(self.finish_reason, str):
            self.finish_reason = FinishReason(self.finish_reason)


# class Delta(CompletionMessage, Generic[T]):
#     """Stream message"""


# class StreamChoice(BaseModel, Generic[T], tag=True):  # type: ignore
#     index: Annotated[
#         int,
#         Meta(
#             title="Index",
#             description="Index of the choice",
#             examples=[0, 1, 2],
#         ),
#     ]

#     delta: Annotated[
#         Optional[Delta],
#         Meta(
#             title="Delta",
#             description="Partial contents (token) of the final message",
#             examples=[
#                 Delta(
#                     role=MessageRole.ASSISTANT,
#                     content="\n\nHello there, how may I assist you today?",
#                 )
#             ],
#         ),
#     ] = None

#     logprobs: Annotated[
#         Optional[list[list[LogProb]]],
#         Meta(
#             title="Log Probability",
#             description='log probability of the choice. For now, always "null"',
#             examples=[None],
#         ),
#     ] = None

#     finish_reason: Annotated[
#         FinishReason,
#         Meta(
#             title="Finish Reason",
#             description="The reason the model stopped generating tokens.",
#             examples=[
#                 "stop",
#                 "length",
#                 "content_filter",
#                 "tool_calls",
#                 FinishReason.STOP,
#                 FinishReason.LENGTH,
#                 FinishReason.CONTENT_FILTER,
#                 FinishReason.TOOL_CALLS,
#                 FinishReason.NONE,
#             ],
#         ),
#     ] = FinishReason.NONE

#     def __post_init__(self) -> None:
#         if isinstance(self.finish_reason, str):
#             self.finish_reason = FinishReason(self.finish_reason)


class CompletionOutput(BaseModel, Generic[T], kw_only=True):
    elapsed_time: Annotated[
        float,
        Meta(
            title="Elapsed Time",
            description="The amount of time it took to generate the Completion.",
        ),
    ] = field()

    id: Annotated[
        uuid.UUID,
        Meta(
            title="ID",
            description="The unique identifier of the completion.",
            examples=[
                "f50ec0b7-f960-400d-91f0-c42a6d44e3d0",
                "16fd2706-8baf-433b-82eb-8c7fada847da",
            ],
        ),
    ] = field(default_factory=lambda: uuid.uuid4())

    object: Annotated[
        Literal["chat.completion"],
        Meta(
            title="Object Type",
            description="The object type. Always `chat.completion`.",
            examples=["chat.completion"],
        ),
    ] = "chat.completion"

    created: Annotated[
        float,
        Meta(
            title="Created",
            description="The Unix timestamp when the completion was created. Defaults to the current time.",
            examples=[1677652288, 1634020001],
        ),
    ] = field(default_factory=lambda: int(datetime.datetime.now().timestamp()))

    model: Annotated[
        AIModel,
        Meta(
            title="Model",
            description="The AI model used to generate the completion.",
        ),
    ] = field(default_factory=lambda: AIModel.STUDIO_GEMINI_1P5_FLASH)

    system_fingerprint: Annotated[
        str,
        Meta(
            title="System Fingerprint",
            description="""This fingerprint represents the backend configuration that the model runs with.
                       Can be used in conjunction with the seed request parameter to understand when
                       backend changes have been made that might impact determinism.""",
            examples=["fp_44709d6fcb"],
        ),
    ] = "fp_none"

    choices: Annotated[
        list[MessageChoice[T]],
        Meta(
            title="Choices",
            description="""The choices made by the language model. 
                       The length of this list can be greater than 1 if multiple choices were requested.""",
            examples=[],
        ),
    ] = field(default_factory=list)

    usage: Annotated[
        Usage,
        Meta(
            title="Usage",
            description="Usage statistics for the completion request.",
            examples=[
                Usage(
                    prompt_tokens=9,
                    completion_tokens=12,
                    total_tokens=21,
                    input_cost=0.02,
                    output_cost=0.01,
                    total_cost=0.03,
                    prompt_tokens_details=PromptTokensDetails(
                        audio_tokens=9, cached_tokens=None
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        audio_tokens=12, reasoning_tokens=None
                    ),
                )
            ],
        ),
    ] = field(
        default_factory=lambda: Usage(
            prompt_tokens=0,
            completion_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            total_tokens=0,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None, cached_tokens=None
            ),
            completion_tokens_details=CompletionTokensDetails(
                audio_tokens=None, reasoning_tokens=None
            ),
        )
    )

    def get_message(self, choice: int = 0) -> Message:
        selected_choice: MessageChoice = self.choices[choice]

        return selected_choice.message

    def get_parsed(self, choice: int = 0) -> T:
        selected_choice = self.choices[choice]

        parsed: Optional[T] = selected_choice.message.parsed
        if parsed is None:
            raise MessageNotParsedError(
                "Message could not be parsed. Parsed content is None."
            )

        return parsed
