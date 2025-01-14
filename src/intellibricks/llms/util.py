from __future__ import annotations

import json
import struct
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

import msgspec
from architecture.logging import LoggerFactory

from intellibricks.llms.base import FileContent
from intellibricks.util import fix_broken_json, flatten_msgspec_schema

if TYPE_CHECKING:
    from intellibricks.llms.constants import Language
    from intellibricks.llms.schema import (
        Function,
        Message,
        Part,
        PartType,
        TextPart,
        ToolInputType,
    )


logger = LoggerFactory.create(__name__)


def find_text_part(parts: Sequence[Part]) -> TextPart:
    from intellibricks.llms.schema import TextPart

    text_part: Optional[Part] = next(
        filter(lambda part: isinstance(part, TextPart), parts), None
    )

    if text_part is None:
        raise ValueError("Text part was not found in the provided parts list.")

    return cast(TextPart, text_part)


def get_parts_llm_described_text(parts: Sequence[PartType]) -> str:
    return "".join([part.to_llm_described_text() for part in parts])


def get_parts_raw_text(parts: Sequence[PartType]) -> str:
    return "".join([str(part) for part in parts])


def get_parsed_response[S](
    contents: Sequence[PartType] | str,
    response_model: type[S],
) -> S:
    """Gets the parsed response from the contents. of the message."""
    match contents:
        case str():
            text = contents
        case _:
            text = get_parts_llm_described_text(contents)

    encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    dict_decoder: msgspec.json.Decoder[dict[str, Any]] = msgspec.json.Decoder(
        type=dict[str, Any]
    )
    rm_decoder: msgspec.json.Decoder[S] = msgspec.json.Decoder(type=response_model)

    try:
        structured: dict[str, Any] = dict_decoder.decode(encoder.encode(text))
    except Exception:
        structured = fix_broken_json(text, decoder=dict_decoder)

    model: S = rm_decoder.decode(encoder.encode(structured))
    return model


def get_structured_prompt_instructions_by_language(
    language: Language, schema: dict[str, Any]
) -> str:
    from intellibricks.llms.constants import Language

    schema_str = json.dumps(schema)
    match language:
        case Language.ENGLISH:
            return f"Return only a valid json adhering to the following schema:\n{schema_str}"
        case Language.SPANISH:
            return f"Devuelve únicamente un json válido que cumpla con el siguiente esquema:\n{schema_str}"
        case Language.FRENCH:
            return f"Retourne uniquement un json valide conforme au schéma suivant :\n{schema_str}"
        case Language.GERMAN:
            return f"Gib ausschließlich ein gültiges json zurück, das dem folgenden Schema entspricht:\n{schema_str}"
        case Language.CHINESE:
            return f"仅返回符合以下 json 模式的有效 json：\n{schema_str}"
        case Language.JAPANESE:
            return f"次のスキーマに準拠した有効な json のみを返してください：\n{schema_str}"
        case Language.PORTUGUESE:
            return f"Retorne apenas um json válido que esteja de acordo com o seguinte esquema:\n{schema_str}"


def get_new_messages_with_response_format_instructions[S](
    *,
    messages: Sequence[Message],
    response_model: type[S],
    language: Optional[Language] = None,
) -> Sequence[Message]:
    """
    Return a new list of messages with additional instructions appended to an existing
    DeveloperMessage, if present. Otherwise, prepend a new DeveloperMessage with the instructions.
    """
    from intellibricks.llms.constants import Language
    from intellibricks.llms.schema import DeveloperMessage, TextPart

    if not messages:
        raise ValueError("Empty messages list")

    basemodel_schema = flatten_msgspec_schema(msgspec.json.schema(response_model))
    instructions = get_structured_prompt_instructions_by_language(
        language=language or Language.ENGLISH, schema=basemodel_schema
    )

    # Try to find the first DeveloperMessage, append instructions, and return immediately.
    for i, msg in enumerate(messages):
        if isinstance(msg, DeveloperMessage):
            new_system_msg = DeveloperMessage(
                contents=[*msg.contents, TextPart(text=instructions)]
            )
            return [*messages[:i], new_system_msg, *messages[i + 1 :]]

    # If no DeveloperMessage was found, prepend a brand new one.
    new_system_msg = DeveloperMessage(
        contents=[TextPart(text=f"You are a helpful assistant.{instructions}")]
    )
    return [new_system_msg, *messages]


def _get_function_name(func: Callable[..., Any]) -> str:
    """
    Returns the name of a callable as a string.
    If the callable doesn't have a __name__ attribute (e.g., lambdas),
    it returns 'anonymous_function'.

    Args:
        func (Callable): The callable whose name is to be retrieved.

    Returns:
        str: The name of the callable, or 'anonymous_function' if unnamed.
    """
    return getattr(func, "__name__", "anonymous_function")


def create_function_mapping_by_tools(tools: Sequence[ToolInputType]):
    """
    Maps the function name to it's function object.
    Useful in all Integration modules in this lib
    and should only be used internally.
    """
    functions: dict[str, Function] = {
        _get_function_name(
            function if callable(function) else function.to_callable()
        ): Function.from_callable(function)
        if callable(function)
        else Function.from_callable(function.to_callable())
        for function in tools or []
    }

    return functions


def get_audio_duration(file_content: FileContent) -> float | None:
    """
    Attempts to quickly determine the duration of an audio file (WAV or basic MP3)
    without using external audio libraries.

    Args:
        file_content: The audio file content (path, bytes, or file object).

    Returns:
        The duration in seconds, or None if the duration cannot be determined.
    """
    try:
        if isinstance(file_content, (str, PathLike)):
            with open(file_content, "rb") as f:
                header = f.read(100)  # Read enough for basic header info
        elif isinstance(file_content, bytes):
            header = file_content[:100]
        else:  # Assume it's a file object
            header = file_content.read(100)
            file_content.seek(0)  # Reset file pointer

        if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
            # WAV file
            try:
                fmt_start = header.find(b"fmt ")
                if fmt_start != -1:
                    fmt_chunk = header[fmt_start + 4 :]
                    num_channels = struct.unpack("<H", fmt_chunk[2:4])[0]
                    sample_rate = struct.unpack("<I", fmt_chunk[4:8])[0]
                    bits_per_sample = struct.unpack("<H", fmt_chunk[14:16])[0]

                    data_start = header.find(b"data")
                    if data_start != -1:
                        data_chunk_size = struct.unpack(
                            "<I", header[data_start + 4 : data_start + 8]
                        )[0]
                        bytes_per_second = (
                            sample_rate * num_channels * (bits_per_sample // 8)
                        )
                        if bytes_per_second > 0:
                            return data_chunk_size / bytes_per_second
            except struct.error:
                return None  # Could not unpack WAV header

        elif (
            header.startswith(b"\xff\xfb")
            or header.startswith(b"ID3")
            and b"Xing" in header
            or b"Info" in header
        ):
            # Basic attempt for MP3 (limited accuracy, assumes CBR or has Xing/Info tag)
            try:
                # Look for Xing or Info tag
                xing_index = header.find(b"Xing")
                info_index = header.find(b"Info")

                if xing_index != -1 or info_index != -1:
                    tag_start = xing_index if xing_index != -1 else info_index
                    frames_offset = tag_start + (
                        12 if header[tag_start : tag_start + 4] == b"Xing" else 8
                    )  # Offset varies slightly
                    num_frames_bytes = header[frames_offset : frames_offset + 4]
                    if len(num_frames_bytes) == 4:
                        num_frames = struct.unpack(">I", num_frames_bytes)[0]

                        # Try to find bitrate (often near Xing/Info) - this is a simplified guess
                        bitrate_loc = header.find(
                            b"\x00\x00", tag_start + 4
                        )  # Look for potential bitrate marker
                        if bitrate_loc != -1 and bitrate_loc + 1 < len(header):
                            bitrate_bytes = header[bitrate_loc - 1 : bitrate_loc + 1]
                            try:
                                bitrate_kbps = int(
                                    bitrate_bytes.hex(), 16
                                )  # Simplistic interpretation
                                if bitrate_kbps > 0:
                                    return (num_frames * 1152) / (
                                        bitrate_kbps * 1000
                                    )  # Rough calculation
                            except ValueError:
                                pass  # Couldn't parse bitrate

            except Exception:
                return None  # Error parsing basic MP3 info

    except Exception:
        return None  # Error reading or processing the file

    return None
