"""llms schemas models"""

from __future__ import annotations

import datetime
import re
import typing
import uuid

from bs4 import BeautifulSoup, NavigableString
from llama_index.core.base.llms.types import LogProb
from llama_index.core.base.llms.types import MessageRole as LlamaIndexMessageRole
from llama_index.core.llms import ChatMessage as LlamaIndexChatMessage
from tiktoken.core import Encoding
from architecture import BaseModel, Meta, field
from architecture.logging import logger

from intellibricks.util import deserialize_json

from .constants import (
    AIModel,
    FinishReason,
    MessageRole,
)
from .exceptions import MessageNotParsedError

T = typing.TypeVar("T")


class Tag(BaseModel):
    tag_name: str
    content: typing.Optional[str] = field(default=None)
    attributes: dict[str, typing.Optional[str]] = field(default_factory=dict)

    @classmethod
    def from_string(
        cls,
        string: str,
        *,
        tag_name: typing.Optional[str] = None,
        attributes: typing.Optional[dict[str, typing.Optional[str]]] = None,
    ) -> typing.Optional[Tag]:
        """
        Create a Tag instance from a string containing a tag.

        This method searches for a tag in the given string and creates a Tag instance
        if a matching tag is found. It can optionally filter by tag name or attributes.

        Args:
            string (str): The input string containing the tag.
            tag_name (typing.Optional[str], optional): If provided, only match tags with this name.
            attributes (typing.Optional[dict[str, str]], optional): If provided, only match tags with these attributes.

        Returns:
            typing.Optional[Tag]: A Tag instance if a matching tag is found, None otherwise.
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
            elem_attributes: dict[str, typing.Optional[str]] = dict(elem.attrs)
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

    def as_object(self) -> dict[str, typing.Any]:
        """
        Extracts the content of the tag as a Python dictionary by parsing the JSON content.

        This method is extremely robust and can handle various nuances in the JSON content, such as:
        - JSON content wrapped in code blocks with backticks (e.g., ```json ... ```)
        - JSON content starting with '{'
        - JSON content with unescaped newlines within strings
        - JSON content with inner backticks in some values
        - Complex and nested JSON structures

        Returns:
            dict[str, typing.Any]: The parsed JSON content as a Python dictionary.

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
            parsed_obj: dict[str, typing.Any] = deserialize_json(content)
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

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "tag_name": self.tag_name,
            "content": self.content,
            "attributes": self.attributes,
        }


class ImageFile(BaseModel):
    """Represents a type of the Message"""

    file_id: typing.Annotated[
        str,
        Meta(
            title="File ID",
            description="The File ID of the image in the message content.",
        ),
    ]

    detail: typing.Annotated[
        typing.Literal["low", "high", "auto"],
        Meta(
            title="Detail",
            description="The detail of the image in the message content.",
        ),
    ]


class ImageFilePart(BaseModel, kw_only=True):  # type: ignore
    """Represents a type of the Message"""

    type: typing.Annotated[
        typing.Literal["image_file"],
        Meta(
            title="Type",
            description="The type of the part.",
        ),
    ] = field(default_factory=lambda: "image_file")

    image_file: typing.Annotated[
        ImageFile,
        Meta(
            title="Image File",
            description="The image file in the message content.",
        ),
    ]


class ImageURL(BaseModel):
    """Represents a type of the Message"""

    url: typing.Annotated[
        str,
        Meta(
            title="URL",
            description="The external URL of the image, must be a supported image types: jpeg, jpg, png, gif, webp.",
        ),
    ]

    detail: typing.Annotated[
        typing.Literal["low", "high", "auto"],
        Meta(
            title="Detail",
            description="The detail of the image in the message content.",
        ),
    ]

    def __post_init__(self) -> None:
        # Validate if the mime type is valid
        supported_mime_types = ["jpeg", "jpg", "png", "gif", "webp"]
        mime: str = self.url.split(".")[-1]
        if mime not in supported_mime_types:
            raise ValueError(
                f"Unsupported image type: {mime}. Supported types are: {supported_mime_types}"
            )


class ImageURLPart(BaseModel, kw_only=True):  # type: ignore
    """Represents a type of the Message"""

    type: typing.Annotated[
        typing.Literal["image_url"],
        Meta(
            title="Type",
            description="The type of the part",
        ),
    ] = field(default_factory=lambda: "image_url")

    image_url: typing.Annotated[
        ImageURL,
        Meta(
            title="Image URL",
            description="Image URL details in the message content.",
        ),
    ]


class FileCitation(BaseModel):
    """Represents a type of the Message"""

    file_id: typing.Annotated[
        str,
        Meta(
            title="File ID",
            description="The File ID of the file in the message content.",
        ),
    ]


class FilePath(BaseModel):
    """Represents a type of the Message"""

    file_id: typing.Annotated[
        str,
        Meta(
            title="File Path",
            description="The File Path of the file in the message content.",
        ),
    ]


class FileCitationAnnotation(BaseModel, kw_only=True):  # type: ignore
    type: typing.Annotated[
        typing.Literal["file_citation"],
        Meta(
            title="Type",
            description="The type of the part",
        ),
    ] = field(default_factory=lambda: "file_citation")

    text: typing.Annotated[
        str,
        Meta(
            title="Text",
            description="The text in the message content that needs to be replaced.",
        ),
    ]

    file_citation: typing.Annotated[
        FileCitation,
        Meta(
            title="File Citation",
            description="The file citation in the message content.",
        ),
    ]

    start_index: typing.Annotated[
        int,
        Meta(
            title="Start Index",
            description="The start index of the text in the message content.",
        ),
    ]

    end_index: typing.Annotated[
        int,
        Meta(
            title="End Index",
            description="The end index of the text in the message content.",
        ),
    ]


class FilePathAnnotation(BaseModel, kw_only=True):  # type: ignore
    type: typing.Annotated[
        typing.Literal["file_path"],
        Meta(
            title="Type",
            description="The type of the part",
        ),
    ] = field(default_factory=lambda: "file_path")

    text: typing.Annotated[
        str,
        Meta(
            title="Text",
            description="The text in the message content that needs to be replaced.",
        ),
    ]

    file_path: typing.Annotated[
        FilePath,
        Meta(
            title="File Path",
            description="The file path in the message content.",
        ),
    ]

    start_index: typing.Annotated[
        int,
        Meta(
            title="Start Index",
            description="The start index of the text in the message content.",
        ),
    ]

    end_index: typing.Annotated[
        int,
        Meta(
            title="End Index",
            description="The end index of the text in the message content.",
        ),
    ]


class Text(BaseModel):
    """Represents a Text in the Message"""

    value: typing.Annotated[
        str,
        Meta(
            title="Value",
            description="The text value in the message content.",
        ),
    ]

    annotations: typing.Annotated[
        list[typing.Union[FileCitationAnnotation, FilePathAnnotation]],
        Meta(
            title="Annotations",
            description="The annotations in the text.",
        ),
    ]


class TextPart(BaseModel, kw_only=True):  # type: ignore
    """Represents a type of the Message"""

    type: typing.Annotated[
        typing.Literal["text"],
        Meta(
            title="Type",
            description="The type of the part",
        ),
    ] = field(default_factory=lambda: "text")

    text: typing.Annotated[
        Text,
        Meta(
            title="Text",
            description="The text in the message content.",
        ),
    ]


class RefusalPart(BaseModel, kw_only=True):  # type: ignore
    """Represents a type of the Message"""

    type: typing.Annotated[
        typing.Literal["refusal"],
        Meta(
            title="Type",
            description="The type of the part",
        ),
    ] = field(default_factory=lambda: "refusal")

    refusal: typing.Annotated[
        str,
        Meta(
            title="Refusal",
            description="The refusal in the message content.",
        ),
    ]


class Prompt(BaseModel):
    """Represents a prompt"""

    content: typing.Annotated[
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

    def compile(self, **replacements: typing.Any) -> Prompt:
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

    audio_tokens: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Audio Tokens",
            description="The number of audio tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    cached_tokens: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Cached Tokens",
            description="The number of cached tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]


class CompletionTokensDetails(BaseModel):
    """Breakdown of tokens generated in completion"""

    audio_tokens: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Audio Tokens",
            description="The number of audio tokens used in the prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    reasoning_tokens: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Reasoning Tokens",
            description="Tokens generated by the model for reasoning.",
        ),
    ]


class Usage(BaseModel):
    prompt_tokens: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Prompt Tokens",
            description="The number of tokens consumed by the input prompt.",
            examples=[9, 145, 3, 25],
        ),
    ]

    completion_tokens: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Completion Tokens",
            description="The number of tokens generated in the completion response.",
            examples=[12, 102, 32],
        ),
    ]

    input_cost: typing.Annotated[
        typing.Optional[float],
        Meta(
            title="USD Cost",
            description="The cost of the input prompt in USD.",
            examples=[0.02, 0.1, 0.03],
        ),
    ]

    output_cost: typing.Annotated[
        typing.Optional[float],
        Meta(
            title="USD Cost",
            description="The cost of the output completion in USD.",
            examples=[0.01, 0.15, 0.07],
        ),
    ]

    total_cost: typing.Annotated[
        typing.Optional[float],
        Meta(
            title="USD Cost",
            description="The cost of the completion in USD.",
            examples=[0.03, 0.25, 0.1],
        ),
    ]

    total_tokens: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Total Tokens",
            description="The total number of tokens consumed, including both prompt and completion.",
            examples=[21, 324, 12],
        ),
    ]

    prompt_tokens_details: typing.Annotated[
        PromptTokensDetails,
        Meta(
            title="Prompt Tokens Details",
            description="Breakdown of tokens used in the prompt.",
        ),
    ]

    completion_tokens_details: typing.Annotated[
        CompletionTokensDetails,
        Meta(
            title="Completion Tokens Details",
            description="Breakdown of tokens generated in completion.",
        ),
    ]


class VisionMessage(BaseModel):
    pass  # TODO


class Message(BaseModel, kw_only=True):
    role: typing.Annotated[
        MessageRole,
        Meta(
            title="Message Role",
            description="The role of the message sender",
            examples=["user", "system", "assistant"],
        ),
    ] = MessageRole.USER

    content: typing.Annotated[
        typing.Optional[str],
        Meta(
            title="Message Content",
            description="The content of the message",
            examples=[
                "Hello! How are you?",
                "I need help on solving a Python problem.",
            ],
        ),
    ] = field(default=None)

    name: typing.Annotated[
        typing.Optional[str],
        Meta(
            title="Name",
            description="An optional name for the participant. Provides the model information to differentiate between participants of the same role.",
            examples=["Alice", "Bob", "Ana"],
        ),
    ] = None

    def extract_tag(
        self,
        *,
        name: typing.Optional[str] = None,
        attributes: typing.Optional[dict[str, typing.Optional[str]]] = None,
    ) -> typing.Optional[Tag]:
        """
        Extracts a tag from the message content based on tag name and/or identifier.
        Uses regex, BeautifulSoup, and XML parsing for robust extraction.

        Args:
            tag_name (typing.Optional[str]): The name of the tag to extract.
            attributes (typing.Optional[dict[str, str]]): The attributes of the tag to extract.

        Returns:
            typing.Optional[Tag]: The extracted tag, or None if not found.

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


class CompletionMessage(Message, typing.Generic[T]):
    parsed: typing.Annotated[
        T,
        Meta(
            title="Structured Model",
            description="Structured model of the message",
        ),
    ]


class MessageChoice(BaseModel, typing.Generic[T], tag=True):  # type: ignore
    index: typing.Annotated[
        int,
        Meta(
            title="Index",
            description="Index of the choice in the list of choices returned by the model.",
            examples=[0, 1, 2],
        ),
    ]

    message: typing.Annotated[
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

    logprobs: typing.Annotated[
        typing.Optional[list[list[LogProb]]],
        Meta(
            title="Log Probability",
            description="Log probability of the choice. Currently always None, reserved for future use.",
            examples=[None],
        ),
    ] = None

    finish_reason: typing.Annotated[
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


# class Delta(CompletionMessage, typing.Generic[T]):
#     """Stream message"""


# class StreamChoice(BaseModel, typing.Generic[T], tag=True):  # type: ignore
#     index: typing.Annotated[
#         int,
#         Meta(
#             title="Index",
#             description="Index of the choice",
#             examples=[0, 1, 2],
#         ),
#     ]

#     delta: typing.Annotated[
#         typing.Optional[Delta],
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

#     logprobs: typing.Annotated[
#         typing.Optional[list[list[LogProb]]],
#         Meta(
#             title="Log Probability",
#             description='log probability of the choice. For now, always "null"',
#             examples=[None],
#         ),
#     ] = None

#     finish_reason: typing.Annotated[
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


class CompletionOutput(BaseModel, typing.Generic[T]):
    id: typing.Annotated[
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

    object: typing.Annotated[
        typing.Literal["chat.completion"],
        Meta(
            title="Object Type",
            description="The object type. Always `chat.completion`.",
            examples=["chat.completion"],
        ),
    ] = "chat.completion"

    created: typing.Annotated[
        float,
        Meta(
            title="Created",
            description="The Unix timestamp when the completion was created. Defaults to the current time.",
            examples=[1677652288, 1634020001],
        ),
    ] = field(default_factory=lambda: int(datetime.datetime.now().timestamp()))

    model: typing.Annotated[
        AIModel,
        Meta(
            title="Model",
            description="The AI model used to generate the completion.",
        ),
    ] = field(default_factory=lambda: AIModel.STUDIO_GEMINI_1P5_FLASH)

    system_fingerprint: typing.Annotated[
        str,
        Meta(
            title="System Fingerprint",
            description="""This fingerprint represents the backend configuration that the model runs with.
                       Can be used in conjunction with the seed request parameter to understand when
                       backend changes have been made that might impact determinism.""",
            examples=["fp_44709d6fcb"],
        ),
    ] = "fp_none"

    choices: typing.Annotated[
        list[MessageChoice[T]],
        Meta(
            title="Choices",
            description="""The choices made by the language model. 
                       The length of this list can be greater than 1 if multiple choices were requested.""",
            examples=[],
        ),
    ] = field(default_factory=list)

    usage: typing.Annotated[
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

        parsed: typing.Optional[T] = selected_choice.message.parsed
        if parsed is None:
            raise MessageNotParsedError(
                "Message could not be parsed. Parsed content is None."
            )

        return parsed
