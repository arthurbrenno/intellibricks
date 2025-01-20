from __future__ import annotations

import json
import logging
import mimetypes
import struct
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)
import re
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
        ToolInputType,
    )


debug_logger = log.create_logger(__name__, level=logging.DEBUG)


def find_text_part(parts: Sequence[Part]) -> TextPart:
    from intellibricks.llms.types import TextPart

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
    try:
        if isinstance(file_content, (str, PathLike)):
            with open(file_content, "rb") as f:
                file_data = f.read()
        elif isinstance(file_content, bytes):
            file_data = file_content
        else:  # Assume it's a file object
            file_data = file_content.read()
            try:
                file_content.seek(0)  # Reset file pointer
            except Exception:
                pass  # If seek fails, it's likely not a seekable stream

        header = file_data[:100]  # Read enough for basic header info

        if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
            # WAV file
            try:
                fmt_start = header.find(b"fmt ")
                if fmt_start != -1 and fmt_start + 16 <= len(header):
                    fmt_chunk = header[fmt_start + 4 :]
                    num_channels = struct.unpack("<H", fmt_chunk[2:4])[0]
                    sample_rate = struct.unpack("<I", fmt_chunk[4:8])[0]
                    bits_per_sample = struct.unpack("<H", fmt_chunk[14:16])[0]

                    data_start = header.find(b"data")
                    if data_start != -1 and data_start + 4 <= len(header):
                        data_chunk_size = struct.unpack(
                            "<I", header[data_start + 4 : data_start + 8]
                        )[0]
                        bytes_per_second = (
                            sample_rate * num_channels * (bits_per_sample // 8)
                        )
                        if bytes_per_second > 0:
                            return float(data_chunk_size / bytes_per_second)
            except struct.error:
                pass  # Could not unpack WAV header

        elif header.startswith(b"\xff\xfb"):
            # Attempt for CBR MP3 (more precise if CBR)
            try:
                bitrate_table = [
                    0,
                    32,
                    40,
                    48,
                    56,
                    64,
                    80,
                    96,
                    112,
                    128,
                    160,
                    192,
                    224,
                    256,
                    320,
                    0,
                ]
                sampling_rate_table = [44100, 48000, 32000, 0]
                sampling_rate = 0  # Initialize sampling_rate

                if len(header) >= 4:
                    header_bytes = header[:4]
                    if (
                        header_bytes[1] & 0xF0 == 0xF0
                        and (header_bytes[1] >> 1) & 0x03 != 0x00
                    ):
                        bitrate_index = (header_bytes[2] >> 4) & 0x0F
                        sampling_rate_index = (header_bytes[2] >> 2) & 0x03

                        if 0 < bitrate_index < len(
                            bitrate_table
                        ) and sampling_rate_index < len(sampling_rate_table):
                            bitrate_kbps = bitrate_table[bitrate_index]
                            sampling_rate = sampling_rate_table[sampling_rate_index]
                            if (
                                bitrate_kbps > 0
                                and sampling_rate > 0
                                and isinstance(file_content, (str, PathLike))
                            ):
                                import os

                                file_size = os.path.getsize(file_content)
                                if file_size > 0:
                                    return float(
                                        (file_size * 8) / (bitrate_kbps * 1000)
                                    )

            except (IndexError, struct.error):
                pass

            # Fallback to Xing/Info tag check (less precise)
            if b"Xing" in header or b"Info" in header:
                try:
                    xing_index = header.find(b"Xing")
                    info_index = header.find(b"Info")
                    tag_start = xing_index if xing_index != -1 else info_index

                    if tag_start != -1 and tag_start + 16 < len(header):
                        num_frames = struct.unpack(
                            ">I", header[tag_start + 4 : tag_start + 8]
                        )[0]
                        if num_frames > 0:
                            bitrate_loc = header.find(b"\x00\x00", tag_start + 8)
                            if bitrate_loc != -1 and bitrate_loc + 1 < len(header):
                                try:
                                    bitrate_bytes = header[
                                        bitrate_loc - 1 : bitrate_loc + 1
                                    ]
                                    bitrate_kbps = int(bitrate_bytes.hex(), 16)
                                    if bitrate_kbps > 0:
                                        default_sampling_rate = (
                                            44100  # Default if not determined earlier
                                        )
                                        return float(
                                            (num_frames * 1152) / default_sampling_rate
                                        )
                                except ValueError:
                                    pass
                except (IndexError, struct.error):
                    pass

    except Exception:
        pass  # Catch any unexpected errors during file processing

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

    Examples
    --------
    Extracting JSON from text with embedded JSON:

        >>> json_str = 'Sure! Here is your formatted json:\\n\\n```json\\n{"name": "Alice", "age": 30}\\n```'
        >>> fix_broken_json(json_str)
        {'name': 'Alice', 'age': 30}

        >>> json_str = '{ "name": "Bob", "age": 25 }'
        >>> fix_broken_json(json_str)
        {'name': 'Bob', 'age': 25}

        >>> json_str = 'Here is the json\\n\\n{ "name": "Charlie", "age": 28 }'
        >>> fix_broken_json(json_str)
        {'name': 'Charlie', 'age': 28}

        >>> json_str = '{ "name": "David", "age": 35 }\\n\\nI provided the json above'
        >>> fix_broken_json(json_str)
        {'name': 'David', 'age': 35}

    Basic usage:

        >>> json_str = '{"name": "Alice", "age": 30}'
        >>> fix_broken_json(json_str)
        {'name': 'Alice', 'age': 30}

    Handling code block markers:

        >>> json_str = '''
        ... ```json
        ... {
        ...     "name": "Bob",
        ...     "age": 25
        ... }
        ... ```
        ... '''
        >>> fix_broken_json(json_str)
        {'name': 'Bob', 'age': 25}

    Handling unescaped backslashes:

        >>> json_str = '{"path": "C:\\Users\\Bob"}'
        >>> deserialize_json(json_str)
        {'path': 'C:\\Users\\Bob'}

    Handling unescaped newlines within strings:

        >>> json_str = '{"text": "Line1\nLine2"}'
        >>> deserialize_json(json_str)
        {'text': 'Line1\\nLine2'}

    Handling missing commas between objects in an array:

        >>> json_str = '{"items": [{"id": 1} {"id": 2}]}'
        >>> deserialize_json(json_str)
        {'items': [{'id': 1}, {'id': 2}]}

    Removing control characters:

        >>> json_str = '{"text": "Hello\\x00World"}'
        >>> deserialize_json(json_str)
        {'text': 'HelloWorld'}

    Attempting to parse invalid JSON:

        >>> json_str = 'Not a JSON string'
        >>> deserialize_json(json_str)
        Traceback (most recent call last):
            ...
        ValueError: No JSON object could be found in the content.

    Parsing fails after all fixes:

        >>> json_str = '{"name": "David", "age": }'
        >>> deserialize_json(json_str)
        Traceback (most recent call last):
            ...
        ValueError: Failed to parse JSON content after multiple attempts.


    Notes
    -----
    The function applies a series of fix functions to correct common issues that may prevent JSON parsing. The fix functions applied are:

    - **No fix**: Attempts to parse the string as-is.
    - **Escaping unescaped backslashes**: Fixes unescaped backslashes in the string.
    - **Escaping unescaped newlines within strings**: Escapes unescaped newline and carriage return characters within JSON strings.
    - **Inserting missing commas between JSON objects in arrays**: Inserts missing commas between JSON objects in arrays.
    - **Removing control characters**: Removes control characters that may interfere with JSON parsing.
    - **Removing invalid characters**: Removes any remaining invalid characters (non-printable ASCII characters).

    If parsing fails after all fixes, a `ValueError` is raised.

    Dependencies
    ------------
    - **msgspec**: Used for JSON decoding. Install via `pip install msgspec`.
    - **re**: Used for regular expression operations.
    - **logging**: Used for logging errors during parsing attempts.

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
    return msgspec.json.decode(msgspec.json.encode(struct), type=dict)


def dict_to_struct[S: msgspec.Struct](d: dict[str, Any], struct: type[S]) -> S:
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


# def get_file_extension(url: str) -> FileExtension:
#     """
#     Get the file extension from a URL.

#     Parameters:
#         url (str): The URL to extract the file extension from.

#     Returns:
#         str: The file extension (e.g., '.txt', '.jpg') extracted from the URL.
#     """
#     from intellibricks.llms.types import FileExtension

#     extension = url[url.rfind(".") :]
#     if extension not in get_args(FileExtension):
#         raise ValueError(f"Unsupported file extension: {extension}")

#     return cast(FileExtension, extension)


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
