"""
Module: intellibricks.llms.util

This module provides utility functions and helper classes for the `intellibricks.llms` package.
It includes tools for handling different data types, parsing responses from Language Model Models (LLMs),
managing JSON schemas, and performing common tasks related to file content and data manipulation.

**Key Functionalities:**

*   **Part Handling:** Functions for extracting text parts from sequences, and converting parts to LLM-described text or raw text.
*   **Response Parsing:** Utility to parse LLM responses, especially structured JSON responses, handling potential formatting issues and extracting data according to response models.
*   **Prompt Structuring:** Functions to generate structured prompt instructions based on language and schema, and to modify message sequences with response format instructions.
*   **Function and Tool Management:** Utilities for creating function mappings from tools, used for integrating custom functions with LLMs.
*   **Audio Duration Handling:** Function to determine the duration of audio files without external audio libraries.
*   **JSON Schema Conversion:** Tools to convert msgspec Struct types to JSON schemas, with options for handling nullable fields and OpenAI compatibility.
*   **Broken JSON Fixer:** Robust function to parse and fix common issues in JSON strings, enhancing resilience against imperfect LLM outputs.
*   **URL and File Handling:** Utilities to check if a string is a URL, if a URL is a file URL (based on extension), and to guess file extensions from content.
*   **File Output:** Function to write content to a file in a specified output directory, guessing the extension based on content.
*   **Subtitle Generation:** Utility to convert sentence segments into SRT (SubRip Text) subtitle format.

**Module Structure:**

*   **Functions:** A collection of standalone utility functions for various tasks.
*   **Dependencies:** Relies on `msgspec` for Struct handling and JSON processing, `architecture` for logging, `typing` for type hints, `json` for JSON operations, `os` for file system operations, `re` for regular expressions, `io` for input/output streams, `mimetypes` for MIME type guessing, and optional dependency `python-magic` for file type detection.

**Usage:**

This module is intended for internal use within the `intellibricks.llms` package and by developers extending or customizing LLM interactions. The functions provided here are building blocks for creating more complex LLM-based applications, handling data transformations, and ensuring robustness in communication with LLM APIs.

**Example:**

```python
from intellibricks.llms.util import get_parsed_response
from intellibricks.llms.types import RawResponse
from typing import Sequence

# Assume 'response_parts' is a Sequence[PartType] obtained from an LLM call
response_parts = ... # your LLM response parts here

# Parse the response into a RawResponse object
parsed_response: RawResponse = get_parsed_response(response_parts, RawResponse)

print(f"Parsed response text: {parsed_response.text}")
```

This module enhances the functionality of `intellibricks.llms` by providing a suite of utilities for data manipulation, error handling, and integration with external services, making it easier to build robust and versatile LLM-powered applications.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
from io import BytesIO
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
    BinaryIO,
)

import msgspec
from architecture import log

from intellibricks.llms.base import FileContent

if TYPE_CHECKING:
    from intellibricks.llms.constants import Language
    from intellibricks.llms.types import (
        Function,
        Message,
        Part,
        PartType,
        TextPart,
        SentenceSegment,
        ToolInputType,
    )


debug_logger = log.create_logger(__name__, level=logging.DEBUG)


def find_text_part(parts: Sequence[Part]) -> TextPart:
    """
    Finds and returns the first TextPart from a sequence of Part objects.

    This function iterates through a sequence of `Part` objects and returns the first instance that is of type `TextPart`.
    If no `TextPart` is found in the sequence, it raises a ValueError.

    Args:
        parts (Sequence[Part]): A sequence of Part objects to search within.

    Returns:
        TextPart: The first TextPart found in the sequence.

    Raises:
        ValueError: If no TextPart is found in the provided parts list.

    Example:
        >>> from intellibricks.llms.types import Part, TextPart, ImagePart
        >>> parts_list = [ImagePart(mime_type="image/png", data=b"...", name="image.png"), TextPart(text="Sample text")]
        >>> text_part = find_text_part(parts_list)
        >>> print(text_part.text) # Output: Sample text
    """
    from intellibricks.llms.types import TextPart

    text_part: Optional[Part] = next(
        filter(lambda part: isinstance(part, TextPart), parts), None
    )

    if text_part is None:
        raise ValueError("Text part was not found in the provided parts list.")

    return cast(TextPart, text_part)


def get_parts_llm_described_text(parts: Sequence[PartType]) -> str:
    """
    Concatenates the LLM-described text representation of a sequence of PartType objects.

    This function takes a sequence of `PartType` objects (which can be `Part` or its subclasses)
    and concatenates their LLM-described text representations into a single string.

    Args:
        parts (Sequence[PartType]): A sequence of PartType objects.

    Returns:
        str: A string containing the concatenated LLM-described text from all parts.

    Example:
        >>> from intellibricks.llms.types import TextPart, ImagePart
        >>> parts_example = [TextPart(text="Text content"), ImagePart(mime_type="image/png", data=b"...", name="image.png")]
        >>> llm_text = get_parts_llm_described_text(parts_example)
        >>> print(llm_text) # Output: Text content<image>
    """
    return "".join([part.to_llm_described_text() for part in parts])


def get_parts_raw_text(parts: Sequence[PartType]) -> str:
    """
    Concatenates the raw text representation of a sequence of PartType objects.

    This function takes a sequence of `PartType` objects and concatenates their raw text representations
    (obtained by calling `str(part)` on each part) into a single string.

    Args:
        parts (Sequence[PartType]): A sequence of PartType objects.

    Returns:
        str: A string containing the concatenated raw text from all parts.

    Example:
        >>> from intellibricks.llms.types import TextPart, ImagePart
        >>> parts_example = [TextPart(text="Text content"), ImagePart(mime_type="image/png", data=b"...", name="image.png")]
        >>> raw_text = get_parts_raw_text(parts_example)
        >>> print(raw_text) # Output: Text content
    """
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
    """
    Returns structured prompt instructions in different languages, guiding LLMs to respond in JSON format.

    This function generates instructions for Language Model Models (LLMs) to ensure they respond with JSON output
    that adheres to a given schema. The instructions are tailored to different languages to potentially improve
    the LLM's understanding and compliance based on the language context.

    Args:
        language (Language): The target language for the instructions.
        schema (dict[str, Any]): The JSON schema to which the LLM's response should adhere.

    Returns:
        str: A string containing prompt instructions in the specified language.

    Example:
        >>> from intellibricks.llms.constants import Language
        >>> json_schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
        >>> instructions_en = get_structured_prompt_instructions_by_language(Language.ENGLISH, json_schema)
        >>> print(instructions_en) # Output: Return only a valid json adhering to the following schema:\n{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
    """
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


def get_new_messages_with_response_format_instructions[S: msgspec.Struct](
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
    from intellibricks.llms.types import DeveloperMessage, TextPart

    if not messages:
        raise ValueError("Empty messages list")

    basemodel_schema = ms_type_to_schema(response_model)

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


def get_audio_duration(file_content: FileContent) -> float:
    """
    Attempts to determine the duration of an audio file (WAV or basic MP3)
    without using external audio libraries. Guarantees a float return.

    Args:
        file_content: The audio file content (path, bytes, or file object).

    Returns:
        The duration in seconds, or 0.0 if the duration cannot be determined.
    """
    from pydub import AudioSegment

    try:
        if isinstance(file_content, (str, os.PathLike)):
            path = os.fspath(file_content)
            audio = AudioSegment.from_file(path)
        elif isinstance(file_content, bytes):
            audio = AudioSegment.from_file(BytesIO(file_content))
        else:
            # Handle file-like objects
            try:
                if file_content.seekable():
                    file_content.seek(0)
            except AttributeError:
                pass

            # Type narrowing: cast to BinaryIO to match pydub's expected type
            audio = AudioSegment.from_file(cast(BinaryIO, file_content))

        return len(audio) / 1000.0
    except Exception:
        return 0.0


def get_struct_from_schema(
    json_schema: dict[str, Any],
    *,
    bases: Optional[tuple[type[msgspec.Struct], ...]] = None,
    name: Optional[str] = None,
    module: Optional[str] = None,
    namespace: Optional[dict[str, Any]] = None,
    tag_field: Optional[str] = None,
    tag: Union[None, bool, str, int, Callable[[str], str | int]] = None,
    rename: Optional[
        Literal["lower", "upper", "camel", "pascal", "kebab"]
        | Callable[[str], Optional[str]]
        | Mapping[str, str]
    ] = None,
    omit_defaults: bool = False,
    forbid_unknown_fields: bool = False,
    frozen: bool = False,
    eq: bool = True,
    order: bool = False,
    kw_only: bool = False,
    repr_omit_defaults: bool = False,
    array_like: bool = False,
    gc: bool = True,
    weakref: bool = False,
    dict_: bool = False,
    cache_hash: bool = False,
) -> type[msgspec.Struct]:
    """
    Create a msgspec.Struct type from a JSON schema at runtime.

    If the schema contains local references ($ref = "#/..."), we
    resolve them recursively. The top-level must be an object schema
    with a "properties" field. Each property is turned into a struct
    field, with its "type" mapped into Python types.

    Returns a new Struct subclass.
    """

    def resolve_refs(node: Any, root_schema: dict[str, Any]) -> Any:
        """
        Recursively resolve local $ref references within `node`,
        using `root_schema` as the top-level reference container.
        """
        if isinstance(node, dict):
            node_dict = cast(dict[str, Any], node)  # <-- The crucial fix (type cast)
            if "$ref" in node_dict:
                ref_val: Any = node_dict["$ref"]
                if not isinstance(ref_val, str):
                    raise TypeError(
                        f"Expected $ref to be a string, got {type(ref_val)!r}."
                    )
                if not ref_val.startswith("#/"):
                    raise ValueError(
                        f"Only local references of the form '#/...'' are supported, got: {ref_val}"
                    )
                ref_path = ref_val.lstrip("#/")
                parts = ref_path.split("/")
                current: Any = root_schema
                for part in parts:
                    if not isinstance(current, dict):
                        raise TypeError(
                            "Encountered a non-dict node while traversing $ref path. "
                            f"Invalid path or schema content: {ref_val!r}"
                        )
                    if part not in current:
                        raise ValueError(
                            f"Reference {ref_val!r} cannot be resolved; key '{part}' not found."
                        )
                    current = current[part]
                return resolve_refs(current, root_schema)
            else:
                # Recurse into child values
                for k, v in list(node_dict.items()):
                    node_dict[k] = resolve_refs(v, root_schema)
                return node_dict

        elif isinstance(node, list):
            new_list: list[Any] = []
            for item in node:
                resolved_item = resolve_refs(item, root_schema)
                new_list.append(resolved_item)
            return new_list
        else:
            return node

    # 1) Resolve references
    resolved_schema = resolve_refs(json_schema, json_schema)

    # 2) Ensure the top-level result is a dict
    if not isinstance(resolved_schema, dict):
        raise TypeError(
            f"After reference resolution, the top-level schema is not a dict. Got: {type(resolved_schema)!r}"
        )

    # 3) top-level "type" must be "object"
    if "type" in resolved_schema:
        raw_type: Any = resolved_schema["type"]
        if not isinstance(raw_type, str):
            raise TypeError(
                f"Top-level 'type' should be a string, got {type(raw_type)!r}"
            )
        top_type = raw_type
    else:
        # If no "type" key, let's treat it as None or error
        top_type = None

    if top_type != "object":
        raise ValueError("JSON schema must define a top-level 'object' type.")

    # 4) "properties" must be a dict
    if "properties" not in resolved_schema:
        raise ValueError("JSON schema must define a 'properties' key at the top level.")

    raw_properties: dict[str, Any] = resolved_schema["properties"]
    if not isinstance(raw_properties, dict):
        raise ValueError(
            "JSON schema must define a 'properties' dict at the top level."
        )

    # 5) Derive struct name
    if name is None:
        if "title" in resolved_schema:
            schema_title = resolved_schema["title"]
            if isinstance(schema_title, str) and schema_title:
                name = schema_title
            else:
                name = "DynamicStruct"
        else:
            name = "DynamicStruct"

    # Ensure the name is a valid Python identifier (coarse):
    name = re.sub(r"\W|^(?=\d)", "_", name)

    # 6) Basic type mapping
    basic_type_map: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
    }

    # 7) Gather required fields
    if "required" in resolved_schema:
        r_val = resolved_schema["required"]
        if not isinstance(r_val, list):
            raise TypeError("'required' must be a list if present.")
        required_list = r_val
    else:
        required_list = []

    required_fields: list[str] = []
    for elem in required_list:
        if not isinstance(elem, str):
            raise TypeError(f"Found a non-string item in 'required': {elem!r}")
        required_fields.append(elem)

    # 8) Build up the fields
    fields: list[tuple[str, Any, Any]] = []

    for prop_name, prop_schema_any in raw_properties.items():
        if not isinstance(prop_name, str):
            raise TypeError(f"Property name must be a string, got {prop_name!r}")

        if not isinstance(prop_schema_any, dict):
            raise TypeError(
                f"Each property schema must be a dict, got {type(cast(object, prop_schema_any))!r} for '{prop_name}'"
            )
        prop_schema: dict[str, Any] = prop_schema_any

        # get 'type' from prop_schema
        if "type" in prop_schema:
            maybe_type = prop_schema["type"]
        else:
            maybe_type = None

        field_type: Any
        if maybe_type is None:
            # If there's no type in the property schema, just treat it as Any
            field_type = Any

        # Prepare for the following lines of code
        elif isinstance(maybe_type, str):
            if maybe_type == "array":
                # array -> items
                items_type_val: Any = None
                if "items" in prop_schema:
                    items_schema = prop_schema["items"]
                    if isinstance(items_schema, dict):
                        if "type" in items_schema:
                            it_val = items_schema["type"]
                            if isinstance(it_val, str):
                                items_type_val = basic_type_map.get(it_val, Any)
                            elif isinstance(it_val, list):
                                sub_union: list[Any] = []
                                for sub_t in it_val:
                                    if isinstance(sub_t, str):
                                        sub_union.append(basic_type_map.get(sub_t, Any))
                                    else:
                                        sub_union.append(Any)
                                if len(sub_union) == 1:
                                    items_type_val = sub_union[0]
                                else:
                                    items_type_val = Union[tuple(sub_union)]
                            else:
                                items_type_val = Any
                        else:
                            items_type_val = Any
                    else:
                        items_type_val = Any
                field_type = list[items_type_val]
            else:
                if maybe_type in basic_type_map:
                    field_type = basic_type_map[maybe_type]
                elif maybe_type == "object":
                    field_type = dict[str, Any]
                else:
                    field_type = Any

        elif isinstance(maybe_type, list):
            # handle union of possible types
            union_members: list[Any] = []
            for t_ in maybe_type:
                if not isinstance(t_, str):
                    union_members.append(Any)
                    continue
                if t_ == "array":
                    arr_item_type: Any = Any
                    if "items" in prop_schema:
                        arr_items = prop_schema["items"]
                        if isinstance(arr_items, dict):
                            if "type" in arr_items:
                                arr_it_type = arr_items["type"]
                                if isinstance(arr_it_type, str):
                                    arr_item_type = basic_type_map.get(arr_it_type, Any)
                                elif isinstance(arr_it_type, list):
                                    sub_union2: list[Any] = []
                                    for st in arr_it_type:
                                        if isinstance(st, str):
                                            sub_union2.append(
                                                basic_type_map.get(st, Any)
                                            )
                                        else:
                                            sub_union2.append(Any)
                                    arr_item_type = Union[tuple(sub_union2)]
                    union_members.append(list[arr_item_type])
                elif t_ in basic_type_map:
                    union_members.append(basic_type_map[t_])
                elif t_ == "object":
                    union_members.append(dict[str, Any])
                else:
                    union_members.append(Any)

            if len(union_members) == 1:
                field_type = union_members[0]
            else:
                field_type = Union[tuple(union_members)]
        else:
            field_type = Any

        # default
        if prop_name in required_fields:
            default_val: Any = msgspec.NODEFAULT
        else:
            if "default" in prop_schema:
                default_val = prop_schema["default"]
            else:
                default_val = msgspec.NODEFAULT

        fields.append((prop_name, field_type, default_val))

    struct_type = msgspec.defstruct(
        name=name,
        fields=fields,
        bases=bases,
        module=module,
        namespace=namespace,
        tag=tag,
        tag_field=tag_field,
        rename=rename,
        omit_defaults=omit_defaults,
        forbid_unknown_fields=forbid_unknown_fields,
        frozen=frozen,
        eq=eq,
        order=order,
        kw_only=kw_only,
        repr_omit_defaults=repr_omit_defaults,
        array_like=array_like,
        gc=gc,
        weakref=weakref,
        dict=dict_,
        cache_hash=cache_hash,
    )

    return struct_type


def fix_broken_json(
    string: str, *, decoder: msgspec.json.Decoder[dict[str, Any]]
) -> dict[str, Any]:
    """
    Parses a python object (JSON) into an instantiated Python dictionary, applying automatic corrections for common formatting issues.

    This function attempts to extract JSON objects from a string containing JSON data possibly embedded within other text. It handles JSON strings that may be embedded within code block markers (e.g., Markdown-style ```json code blocks) and applies a series of fix-up functions to correct common JSON formatting issues such as unescaped characters, missing commas, and control characters that may prevent successful parsing.

    Parameters
    ----------
    string : str
        The string containing JSON string to deserialize. This may include code block markers, surrounding text, and may have minor formatting issues.

    Returns
    -------
    dict[str, Any]
        A Python dictionary representing the parsed JSON string.

    Raises
    ------
    ValueError
        If no JSON object could be found in the string, or if parsing fails after applying all fix functions.
    """

    # Remove code block markers if present
    string = re.sub(r"^```(?:json)?\n", "", string, flags=re.IGNORECASE | re.MULTILINE)
    string = re.sub(r"\n```$", "", string, flags=re.MULTILINE)

    # Helper function to find substrings with balanced braces
    def find_json_substrings(s: str) -> list[str]:
        substrings: list[str] = []
        stack: list[str] = []
        start: Optional[int] = None
        for i, c in enumerate(s):
            if c == "{":
                if not stack:
                    # Potential start of JSON object
                    start = i
                stack.append(c)
            elif c == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        # Potential end of JSON object
                        end = i + 1  # Include the closing brace
                        substrings.append(s[start:end])
                        start = None  # Reset start
        return substrings

    # Find all potential JSON substrings
    json_substrings: list[str] = find_json_substrings(string)

    if not json_substrings:
        raise ValueError("No JSON object could be found in the string.")

    # Initialize variables for parsing attempts
    parsed_obj: dict[str, Any]

    # Define fix functions as inner functions
    def _fix_unescaped_backslashes(input_string: str) -> str:
        """
        Fix unescaped backslashes by escaping them.

        Args:
            input_string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        return re.sub(r'(?<!\\)\\(?![\\"])', r"\\\\", input_string)

    def _escape_unescaped_newlines(input_string: str) -> str:
        """
        Escape unescaped newline and carriage return characters within JSON strings.

        Args:
            input_string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        # Pattern to find JSON strings
        string_pattern = r'"((?:\\.|[^"\\])*)"'

        def replace_newlines_in_string(match: re.Match[str]) -> str:
            content_inside_quotes = match.group(1)
            # Escape unescaped newlines and carriage returns
            content_inside_quotes = content_inside_quotes.replace("\n", "\\n").replace(
                "\r", "\\r"
            )
            return f'"{content_inside_quotes}"'

        fixed_content = re.sub(
            string_pattern, replace_newlines_in_string, input_string, flags=re.DOTALL
        )
        return fixed_content

    def _insert_missing_commas(input_string: str) -> str:
        """
        Insert missing commas between JSON objects in arrays.

        Args:
            input_string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        # Insert commas between closing and opening braces/brackets
        patterns = [
            (r"(\})(\s*\{)", r"\1,\2"),  # Between } and {
            (r"(\])(\s*\[)", r"\1,\2"),  # Between ] and [
            (r"(\])(\s*\{)", r"\1,\2"),  # Between ] and {
            (r"(\})(\s*\[)", r"\1,\2"),  # Between } and [
        ]
        fixed_content = input_string
        for pattern, replacement in patterns:
            fixed_content = re.sub(pattern, replacement, fixed_content)
        return fixed_content

    def _remove_control_characters(input_string: str) -> str:
        """
        Remove control characters that may interfere with JSON parsing.

        Args:
            input_string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        return "".join(c for c in input_string if c >= " " or c == "\n")

    def _remove_invalid_characters(input_string: str) -> str:
        """
        Remove any remaining invalid characters (non-printable ASCII characters).

        Args:
            input_string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        return re.sub(r"[^\x20-\x7E]+", "", input_string)

    # Define a list of fix functions
    fix_functions: list[Callable[[str], str]] = [
        lambda x: x,  # First attempt without any fixes
        _fix_unescaped_backslashes,
        _escape_unescaped_newlines,
        _insert_missing_commas,
        _remove_control_characters,
        _remove_invalid_characters,
    ]

    # Attempt parsing for each JSON substring, applying fixes sequentially
    for json_content in json_substrings:
        for fix_func in fix_functions:
            try:
                # Apply the fix function
                fixed_content: str = fix_func(json_content)
                # Try parsing the JSON string
                parsed_obj = decoder.decode(fixed_content)
                return parsed_obj
            except (msgspec.DecodeError, ValueError) as e:
                debug_logger.error(
                    f"Failed to parse JSON string after applying fix: {fix_func.__name__}"
                )
                debug_logger.error(f"Exception: {e}")
                continue  # Try next fix function
        # If parsing fails for this substring, continue to next
        continue

    # If all attempts fail, raise an error
    raise ValueError("Failed to parse JSON string after multiple attempts.")


def is_url(url: str) -> bool:
    """
    Check if a string is a valid URL.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    try:
        from urllib.parse import urlparse

        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def struct_to_dict(struct: msgspec.Struct) -> dict[str, Any]:
    """
    Converts a msgspec Struct object to a Python dictionary.

    Args:
        struct (msgspec.Struct): The msgspec Struct object to convert.

    Returns:
        dict[str, Any]: A Python dictionary representation of the Struct.
    """
    return msgspec.json.decode(msgspec.json.encode(struct), type=dict)


def dict_to_struct[S: msgspec.Struct](d: dict[str, Any], struct: type[S]) -> S:
    """
    Converts a Python dictionary to a msgspec Struct object of a specified type.

    Args:
        d (dict[str, Any]): The Python dictionary to convert.
        struct (type[S]): The msgspec Struct type to convert to.

    Returns:
        S: An instance of the specified msgspec Struct, populated with data from the dictionary.
    """
    return msgspec.json.decode(msgspec.json.encode(d), type=struct)


def is_file_url(url: str) -> bool:
    """
    Check if a URL is a file URL based on its extension.

    Parameters:
        url (str): The URL to check.

    Returns:
        bool: True if the URL ends with a known file extension, False otherwise.
    """
    from urllib.parse import urlparse

    # Parse the URL to extract the path
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Guess the MIME type based on the file extension
    mime_type, _ = mimetypes.guess_type(path)

    # If a MIME type is found, the URL likely points to a file
    return mime_type is not None


def ms_type_to_schema(
    struct: type[msgspec.Struct],
    remove_parameters: Optional[Sequence[str]] = None,
    openai_like: bool = False,
    ensure_str_enum: bool = False,
    nullable_style: Optional[
        Literal[
            "remove_null",
            "standard_nullable",
            "openapi_nullable",
            "custom_schema_nullable",
        ]
    ] = None,
) -> dict[str, Any]:
    """Generates a fully dereferenced JSON schema for a given msgspec Struct type,
    with enhanced handling of nullable fields for different providers.

    Args:
        struct: The msgspec Struct type to convert
        remove_parameters: Optional list of parameters to remove from the schema
        openai_like: Whether to add OpenAI-specific modifications
        ensure_str_enum: Whether to ensure enum values are strings
        nullable_style: How to handle nullable fields:
            - "remove_null": Remove null types entirely (default)
            - "standard_nullable": Add standard "nullable": true flag
            - "openapi_nullable": Add OpenAPI-style "x-nullable": true flag
            - "custom_schema_nullable": Add custom schema property "schema-nullable": true
            If None, defaults to "remove_null"
    """
    schemas, components = msgspec.json.schema_components([struct])
    main_schema = schemas[0]
    memo: dict[str, Any] = {}

    nullable_style = nullable_style or "standard_nullable"  # Set default if None

    def ensure_enum_string(schema: dict[str, Any]) -> dict[str, Any]:
        if not ensure_str_enum:
            return schema

        debug_logger.warning(
            "WARNING: ENSURING ENUMS ARE STRINGS FOR PROVIDER COMPATIBILITY!"
            "THE PROVIDER MAY NOT SUPPORT ENUMS WITH NON-STRING VALUES!"
            "IT WILL RETURN AN ENUM WITH STRING VALUES!"
        )
        if "enum" in schema:
            schema["type"] = "string"
            schema["enum"] = [str(value) for value in schema["enum"]]
        return schema

    def handle_nullable_type(schema: dict[str, Any]) -> dict[str, Any]:
        """Convert anyOf with null type to appropriate nullable format."""
        if "anyOf" not in schema:
            return schema

        # Check if this is a nullable type (has both null and non-null types)
        null_type = any(
            isinstance(t, dict) and t.get("type") == "null"  # type: ignore
            for t in schema["anyOf"]
        )
        non_null_types: list[dict[str, Any]] = [
            t
            for t in schema["anyOf"]
            if isinstance(t, dict) and t.get("type") != "null"  # type: ignore
        ]

        if null_type and len(non_null_types) == 1:
            base_type = non_null_types[0]

            if nullable_style == "remove_null":
                # Just use the non-null type
                return base_type

            # Start with the base type
            result = base_type.copy()

            # Add appropriate nullable flag
            if nullable_style == "standard_nullable":
                result["nullable"] = True
            elif nullable_style == "openapi_nullable":
                result["x-nullable"] = True
            elif nullable_style == "custom_schema_nullable":
                result["schema-nullable"] = True

            return result

        elif len(non_null_types) > 1:
            # If multiple non-null types, keep anyOf but handle according to style
            if nullable_style == "remove_null":
                schema["anyOf"] = non_null_types
            else:
                # Add nullable flag to the anyOf schema itself
                if nullable_style == "standard_nullable":
                    schema["nullable"] = True
                elif nullable_style == "openapi_nullable":
                    schema["x-nullable"] = True
                elif nullable_style == "custom_schema_nullable":
                    schema["schema-nullable"] = True

        return schema

    def dereference(schema: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in schema:
            ref_path = schema["$ref"]
            component_name = ref_path.split("/")[-1]
            if component_name in memo:
                return memo[component_name]
            elif component_name in components:
                memo[component_name] = {"$ref": ref_path}
                dereferenced = components[component_name]
                if isinstance(dereferenced, dict):
                    if (
                        openai_like
                        and "properties" in dereferenced
                        and "additionalProperties" not in dereferenced
                    ):
                        dereferenced["additionalProperties"] = False
                    dereferenced = _dereference_recursive(dereferenced)
                memo[component_name] = dereferenced
                return dereferenced
            else:
                raise ValueError(
                    f"Component '{component_name}' not found in schema components."
                )
        return _dereference_recursive(schema)

    def _dereference_recursive(data: Any) -> Any:
        if isinstance(data, dict):
            if "$ref" in data:
                return dereference(cast(dict[str, Any], data))

            new_data: dict[str, Any] = {}
            for key, value in data.items():
                if remove_parameters and key in remove_parameters:
                    debug_logger.warning(
                        f"WARNING: REMOVING PARAMETER: {key} BECAUSE THE PROVIDER DOES NOT SUPPORT IT IN JSON SCHEMAS!"
                    )
                    continue
                new_data[key] = _dereference_recursive(value)

            # Apply conversions
            new_data = ensure_enum_string(new_data)
            new_data = handle_nullable_type(new_data)

            # Update required fields
            if "properties" in new_data and "required" in new_data:
                properties = new_data["properties"]
                required = new_data["required"]
                new_required: list[str] = []
                for prop in required:
                    if prop in properties:
                        # Check if property is marked as nullable
                        prop_schema = properties[prop]
                        is_nullable = (
                            prop_schema.get("nullable")
                            or prop_schema.get("x-nullable")
                            or prop_schema.get("schema-nullable")
                        )
                        if not is_nullable:
                            new_required.append(prop)

                if new_required:
                    new_data["required"] = new_required
                else:
                    del new_data["required"]

            return new_data
        elif isinstance(data, list):
            return [_dereference_recursive(item) for item in data]
        return data

    dereferenced_schema = dereference(main_schema)
    return dereferenced_schema


def guess_extension(
    file_content: Union[IO[bytes], bytes, os.PathLike[str]],
) -> Union[
    Literal[
        "jpeg",
        "png",
        "gif",
        "webp",
        "bmp",
        "tiff",
        "svg",
        "pdf",
        "zip",
        "tar",
        "gzip",
        "bz2",
        "7z",
        "rar",
        "txt",
        "csv",
        "json",
        "html",
        "xml",
        "mp4",
        "mov",
        "avi",
        "mkv",
        "mp3",
        "wav",
        "ogg",
        "flac",
        "aac",
    ],
    str,
]:
    """
    Guesses the file extension based on the file content.

    Args:
        file_content: The file content as bytes, an IO stream, or a file path.

    Returns:
        The file extension as a string (e.g., "jpg", "png").

    Raises:
        ValueError: If the file type cannot be determined or no suitable extension is found.
    """
    import magic

    try:
        if isinstance(file_content, (str, os.PathLike)):
            mime_type = magic.from_file(str(file_content), mime=True)  # type: ignore
        elif isinstance(file_content, bytes):
            mime_type = magic.from_buffer(file_content, mime=True)
        elif hasattr(file_content, "read"):
            mime_type = magic.from_buffer(file_content.read(), mime=True)
        else:
            raise ValueError("Unsupported file content type")
    except magic.MagicException as e:
        raise ValueError(f"Error during file type detection: {e}") from e

    mime_to_ext = {
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/svg+xml": "svg",
        "application/pdf": "pdf",
        "application/zip": "zip",
        "application/x-tar": "tar",
        "application/gzip": "gzip",
        "application/x-bzip2": "bz2",
        "application/x-7z-compressed": "7z",
        "application/vnd.rar": "rar",
        "text/plain": "txt",
        "text/csv": "csv",
        "application/json": "json",
        "text/html": "html",
        "text/xml": "xml",
        "video/mp4": "mp4",
        "video/quicktime": "mov",
        "video/x-msvideo": "avi",
        "video/x-matroska": "mkv",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
        "audio/aac": "aac",
        "audio/x-wav": "wav",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/msword": "doc",
        "application/vnd.ms-excel": "xls",
        "application/vnd.ms-powerpoint": "ppt",
        "application/rtf": "rtf",
        "text/css": "css",
        "application/javascript": "js",
        "text/javascript": "js",
        "text/markdown": "md",
        "image/x-icon": "ico",
        "image/vnd.microsoft.icon": "ico",
        "image/apng": "apng",
        "video/mpeg": "mpeg",
        "video/webm": "webm",
        "video/x-flv": "flv",
        "audio/webm": "weba",
        "audio/x-m4a": "m4a",
        "font/ttf": "ttf",
        "font/otf": "otf",
        "font/woff": "woff",
        "font/woff2": "woff2",
        "application/x-xz": "xz",
        "application/zstd": "zst",
        "application/x-msdownload": "exe",
        "application/x-sh": "sh",
        "application/epub+zip": "epub",
        "application/ogg": "ogx",
        "application/x-csh": "csh",
        "image/heic": "heic",
        "image/heif": "heif",
        "image/vnd.adobe.photoshop": "psd",
        "application/xml": "xml",
        "audio/x-aiff": "aiff",
        "application/vnd.oasis.opendocument.text": "odt",
        "application/vnd.oasis.opendocument.spreadsheet": "ods",
        "application/vnd.oasis.opendocument.presentation": "odp",
        "application/x-executable": "bin",
        "application/x-dosexec": "exe",
        "application/xhtml+xml": "xhtml",
        "application/vnd.amazon.ebook": "azw",
        "video/x-m4v": "m4v",
    }
    extension = mime_to_ext.get(mime_type)

    if extension is None:
        raise ValueError(f"No suitable extension found for mime type: {mime_type}")

    return extension


def write_content_to_file(
    file_content: Union[IO[bytes], bytes, os.PathLike[str]],
    output_dir: str,
    mode: Optional[str] = None,
) -> str:
    """
    Writes file content to a file in the specified output directory, guessing the extension from content.

    This utility function takes file content, determines an appropriate file extension based on the content type,
    and writes the content to a file within the specified output directory. It supports various types of file content
    including bytes, IO streams, and file paths.

    Args:
        file_content (Union[IO[bytes], bytes, os.PathLike[str]]): The content to write to a file. Can be bytes,
            an IO stream of bytes, or a file path.
        output_dir (str): The directory where the file should be written.
        mode (Optional[str]): The mode in which the file should be opened (e.g., 'wb' for binary write, 'wt' for text write).
            Defaults to 'wb' (binary write) if not specified.

    Returns:
        str: The path to the file that was created.

    Example:
        >>> file_bytes = b"This is a sample file content."
        >>> output_directory = "output_files"
        >>> os.makedirs(output_directory, exist_ok=True)
        >>> file_path = write_content_to_file(file_bytes, output_directory)
        >>> print(file_path) # Output: output_files/file.txt
    """
    _mode = mode or "wb"
    extension = guess_extension(file_content)
    if isinstance(file_content, bytes):
        file_path: str = f"{output_dir}/file.{extension}"
        with open(file_path, _mode) as f:
            f.write(file_content)
    elif hasattr(file_content, "read"):
        file_path = f"{output_dir}/file.{extension}"
        with open(file_path, _mode) as f:
            f.write(file_content.read())  # type: ignore
    else:
        file_path = str(file_content)

    return file_path


def segments_to_srt(segments: Sequence[SentenceSegment]) -> str:
    """
    Converts a sequence of SentenceSegment objects into SRT (SubRip Text) subtitle format.

    This function takes a sequence of `SentenceSegment` objects, each representing a segment of transcribed speech with start and end times,
    and formats them into a SRT subtitle string. SRT is a widely used subtitle format that includes an index, time range, and subtitle text for each segment.

    Args:
        segments (Sequence[SentenceSegment]): A sequence of SentenceSegment objects, each containing `sentence`, `start`, and `end` attributes.

    Returns:
        str: A string containing the segments formatted as SRT subtitles.

    Example:
        >>> from intellibricks.llms.types import SentenceSegment
        >>> segments_example = [
        ...     SentenceSegment(sentence="Hello world.", start=0.0, end=2.5),
        ...     SentenceSegment(sentence="This is a test subtitle.", start=3.0, end=6.7)
        ... ]
        >>> srt_content = segments_to_srt(segments_example)
        >>> print(srt_content)
        1
        00:00:00,000 --> 00:00:02,500
        Hello world.

        2
        00:00:03,000 --> 00:00:06,700
        This is a test subtitle.
    """

    def format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        remaining %= 60
        seconds_int = int(remaining)
        milliseconds = int((remaining - seconds_int) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

    parts: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        start_time = format_time(segment.start)
        end_time = format_time(segment.end)
        part = f"{idx}\n{start_time} --> {end_time}\n{segment.sentence}"
        parts.append(part)

    return "\n\n".join(parts)
