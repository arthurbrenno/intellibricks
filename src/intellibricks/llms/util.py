from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

import msgspec
from architecture.logging import LoggerFactory

from intellibricks.util import flatten_msgspec_schema, fix_broken_json

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
