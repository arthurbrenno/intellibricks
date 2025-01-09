import base64
import inspect
import io
import logging
import mimetypes
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Union,
    Optional,
    TypeVar,
    cast,
    get_args,
    get_origin,
)
from urllib.parse import urlparse

import msgspec
from architecture.logging import LoggerFactory
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from typing_extensions import TypedDict

from intellibricks.llms.types import FileExtension

from .types import JsonType

logger = LoggerFactory.create(__name__)

S = TypeVar("S", bound=msgspec.Struct)


class CallerInfo(TypedDict):
    caller_class: Optional[str]
    caller_method: Optional[str]
    filename: Optional[str]
    line_number: Optional[int]
    caller_id: Optional[str]


def file_get_contents(filename: str) -> str:
    """
    Read the entire contents of a file and return it as a string.
    Supports various path scenarios and attempts to find the file
    even if only a partial path is provided.

    Args:
        filename (str): The path to the file to be read.

    Returns:
        str: The contents of the file as a string.

    Raises:
        FileNotFoundError: If the specified file cannot be found.
        IOError: If there's an error reading the file.
    """
    paths_to_try = [
        Path(filename),  # As provided
        Path(filename).resolve(),  # Absolute path
        Path(os.getcwd()) / filename,  # Relative to current working directory
        Path(os.path.dirname(inspect.stack()[1].filename))
        / filename,  # Relative to caller's directory
    ]

    for path in paths_to_try:
        try:
            return path.read_text()
        except FileNotFoundError:
            continue
        except IOError as e:
            raise IOError(f"Error reading file '{path}': {str(e)}")

    # If file not found, try to find it in the current directory structure
    current_dir = Path.cwd()
    filename_parts = Path(filename).parts

    for root, dirs, files in os.walk(current_dir):
        root_path = Path(root)
        if all(part in root_path.parts for part in filename_parts[:-1]):
            potential_file = root_path / filename_parts[-1]
            if potential_file.is_file():
                try:
                    return potential_file.read_text()
                except IOError as e:
                    raise IOError(f"Error reading file '{potential_file}': {str(e)}")

    raise FileNotFoundError(
        f"File '{filename}' not found in any of the attempted locations."
    )


def markdown_to_html(markdown_text: str) -> str:
    # Create a Markdown instance with basic features including inline code
    # TODO(arthur): Implement this feature with no extra dependencies
    raise NotImplementedError("This feature is not yet implemented.")


def format_code_blocks(text: str) -> str:
    pattern = r"```(\w+)\n(.*?)```"

    def replace_code_block(match: re.Match) -> str:
        language, code = match.groups()

        try:
            lexer = get_lexer_by_name(language, stripall=True)
        except ClassNotFound:
            lexer = get_lexer_by_name("text", stripall=True)

        formatter = HtmlFormatter(
            style="bw", noclasses=True, nowrap=True, cssclass="sourcecode"
        )
        highlighted_code = highlight(code.strip(), lexer, formatter)

        lines = highlighted_code.split("\n")
        line_html = "".join(
            f'<div style="display: flex;">'
            f'<span style="user-select: none; text-align: right; padding-right: 8px; color: #6e7781; min-width: 30px;">{i + 1}</span>'
            f'<span style="white-space: pre; flex: 1;">{line}</span>'
            f"</div>"
            for i, line in enumerate(lines)
        )

        container_styles = (
            "margin: 10px 0; "
            "overflow: hidden; "
            "font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;"
        )

        code_container_styles = "overflow-x: auto;"

        return f"""
            <div style="{container_styles}">
                <div style="{code_container_styles}">
                    {line_html}
                </div>
            </div>
        """

    return re.sub(pattern, replace_code_block, text, flags=re.DOTALL)


def replace_placeholders(
    s: str, case_sensitive: bool = True, **replacements: Any
) -> str:
    """
    Replace placeholders in the format `{{key}}` within the string `s` with their corresponding values from `replacements`.

    Parameters:
        s (str): The input string containing placeholders.
        case_sensitive (bool, optional): If False, perform case-insensitive replacements. Defaults to True.
        **replacements: Arbitrary keyword arguments where each key corresponds to a placeholder in the string.

    Returns:
        str: The modified string with placeholders replaced by their corresponding values.

    Examples:
        >>> replace_placeholders("Hello, {{name}}!", name="Alice")
        'Hello, Alice!'

        >>> replace_placeholders(
        ...     "Dear {{title}} {{lastname}}, your appointment is on {{date}}.",
        ...     title="Dr.",
        ...     lastname="Smith",
        ...     date="Monday"
        ... )
        'Dear Dr. Smith, your appointment is on Monday.'

        >>> replace_placeholders(
        ...     "Coordinates: {{latitude}}, {{longitude}}",
        ...     latitude="40.7128° N",
        ...     longitude="74.0060° W"
        ... )
        'Coordinates: 40.7128° N, 74.0060° W'
    """
    return str_replace(
        s, replace_placeholders=True, case_sensitive=case_sensitive, **replacements
    )


def str_replace(
    s: str,
    *,
    case_sensitive: bool = True,
    use_regex: bool = False,
    count: int = -1,
    replace_placeholders: bool = False,
    **replacements: Any,
) -> str:
    """
    Replace multiple substrings in a string using keyword arguments, with additional options to modify behavior.

    Parameters:
        s (str): The input string on which to perform replacements.
        case_sensitive (bool, optional): If False, perform case-insensitive replacements. Defaults to True.
        use_regex (bool, optional): If True, treat the keys in replacements as regular expressions. Defaults to False.
        count (int, optional): Maximum number of occurrences to replace per pattern. Defaults to -1 (replace all).
        replace_placeholders (bool, optional): If True, replaces placeholders like '{{key}}' with their corresponding values. Defaults to False.
        **replacements: Arbitrary keyword arguments where each key is a substring or pattern to be replaced,
                        and each value is the replacement string.

    Returns:
        str: The modified string after all replacements have been applied.

    Examples:
        >>> str_replace("Hello, World!", Hello="Hi", World="Earth")
        'Hi, Earth!'

        >>> str_replace("The quick brown fox", quick="slow", brown="red")
        'The slow red fox'

        >>> str_replace("a b c d", a="1", b="2", c="3", d="4")
        '1 2 3 4'

        >>> str_replace("No changes", x="y")
        'No changes'

        >>> str_replace("Replace multiple occurrences", e="E", c="C")
        'REplaCE multiplE oCCurrEnCEs'

        >>> str_replace("Case Insensitive", case="CASE", case_sensitive=False)
        'CASE Insensitive'

        >>> str_replace(
        ...     "Use Regex: 123-456-7890",
        ...     use_regex=True,
        ...     pattern=r"\\d{3}-\\d{3}-\\d{4}",
        ...     replacement="PHONE"
        ... )
        'Use Regex: PHONE'

        >>> str_replace("Hello, {{name}}!", replace_placeholders=True, name="Alice")
        'Hello, Alice!'
    """

    # Determine the flags for regex based on case sensitivity
    flags = 0 if case_sensitive else re.IGNORECASE

    # Replace placeholders like {{key}} with their corresponding values
    if replace_placeholders:
        placeholder_pattern = r"\{\{(.*?)\}\}"

        def replace_match(match: re.Match) -> str:
            key = match.group(1)
            if not case_sensitive:
                key_lookup = key.lower()
                replacements_keys = {k.lower(): k for k in replacements}
                if key_lookup in replacements_keys:
                    actual_key = replacements_keys[key_lookup]
                    value = replacements[actual_key]
                    return str(value)
                else:
                    string: str = match.group(0)
                    return string
            else:
                if key in replacements:
                    value = replacements[key]
                    return str(value)
                else:
                    string = match.group(0)
                    return string

        s = re.sub(placeholder_pattern, replace_match, s, flags=flags)

    # Now perform the standard replacements
    for old, new in replacements.items():
        if use_regex:
            s = re.sub(old, new, s, count=0 if count == -1 else count, flags=flags)
        else:
            if not case_sensitive:
                pattern = re.compile(re.escape(old), flags=flags)
                s = pattern.sub(new, s, count=0 if count == -1 else count)
            else:
                if count != -1:
                    s = s.replace(old, new, count)
                else:
                    s = s.replace(old, new)
    return s


def get_struct_from_schema(
    json_schema: dict[str, Any],
    bases: Optional[tuple[type[msgspec.Struct], ...]] = None,
    name: Optional[str] = None,
    module: Optional[str] = None,
    namespace: Optional[dict[str, Any]] = None,
    tag_field: Optional[str] = None,
    tag: Optional[bool | str | int | Callable[[str], str | int]] = None,
    rename: Optional[
        Literal["lower", "upper", "camel", "pascal", "kebab"]
        | Callable[[str], Optional[str]]
        | dict[str, str]
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

    Args:
        json_schema (dict[str, Any]): The JSON schema defining the structure.
        bases (Optional[Tuple[Type[msgspec.Struct], ...]]): Base classes for the new Struct.
        name (Optional[str]): Name for the new Struct. If not provided, it's derived from the schema title.
        module (Optional[str]): Module name for the new Struct.
        namespace (Optional[dict[str, Any]]): Additional namespace for the new Struct.
        tag_field (Optional[str]): Name of the field to use for tagging.
        tag (Union[None, bool, str, int, Callable]): Tag value or function to generate tag.
        rename (Union[None, str, Callable, dict[str, str]]): Field renaming strategy.
        omit_defaults (bool): Whether to omit fields with default values during serialization.
        forbid_unknown_fields (bool): Whether to raise an error for unknown fields during deserialization.
        frozen (bool): Whether the resulting struct should be immutable.
        eq (bool): Whether to add __eq__ method to the struct.
        order (bool): Whether to add ordering methods to the struct.
        kw_only (bool): Whether all fields should be keyword-only in the __init__ method.
        repr_omit_defaults (bool): Whether to omit fields with default values in __repr__.
        array_like (bool): Whether to make the struct behave like an array.
        gc (bool): Whether the struct should be tracked by the garbage collector.
        weakref (bool): Whether to add support for weak references to the struct.
        dict_ (bool): Whether to add a __dict__ to the struct.
        cache_hash (bool): Whether to cache the hash value of the struct.

    Returns:
        Type[msgspec.Struct]: A new msgspec.Struct type based on the provided JSON schema.

    Raises:
        ValueError: If the JSON schema is invalid or missing required information.
    """

    def resolve_refs(schema: Any, root: dict[str, Any]) -> Any:
        """
        Recursively resolve $ref in a JSON schema.

        Args:
            schema (Any): The current schema node to resolve.
            root (dict[str, Any]): The root schema containing definitions.

        Returns:
            Any: The schema with all $ref resolved.
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref = schema["$ref"]
                if not ref.startswith("#/"):
                    raise ValueError(
                        f"Only local $ref references are supported, got: {ref}"
                    )
                # Split the ref path, e.g., "#/$defs/Joke" -> ["$defs", "Joke"]
                parts = ref.lstrip("#/").split("/")
                ref_schema = root
                for part in parts:
                    if part not in ref_schema:
                        raise ValueError(f"Reference {ref} cannot be resolved.")
                    ref_schema = ref_schema[part]
                # Recursively resolve in case the referenced schema also has $ref
                return resolve_refs(ref_schema, root)
            else:
                # Recursively resolve all dictionary values
                return {k: resolve_refs(v, root) for k, v in schema.items()}
        elif isinstance(schema, list):
            # Recursively resolve all items in the list
            return [resolve_refs(item, root) for item in schema]
        else:
            # Base case: neither dict nor list, return as is
            return schema

    # Step 1: Resolve all $ref in the schema
    resolved_schema = resolve_refs(json_schema, json_schema)

    # Step 2: Validate the resolved schema
    if not isinstance(resolved_schema, dict):
        raise ValueError("Resolved JSON schema must be a dictionary-like object")

    if resolved_schema.get("type") != "object":
        raise ValueError("JSON schema must define an object type")

    if "properties" not in resolved_schema:
        raise ValueError("JSON schema must define properties")

    # Step 3: Determine the name of the Struct
    if name is None:
        name = resolved_schema.get("title", "DynamicStruct")

    nm = name or ""

    # Ensure the name is a valid Python identifier
    name = re.sub(pattern=r"\W|^(?=\d)", repl="_", string=nm)

    # Step 4: Define the type mapping within the function
    type_mapping: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
        "array": list,  # This is okay for runtime
        "object": dict,
    }

    # Step 5: Process each property
    fields: list[tuple[str, Any, Any]] = []

    required_fields = resolved_schema.get("required", [])

    for prop_name, prop_schema in resolved_schema["properties"].items():
        # Determine the field type based on the property schema
        if "type" not in prop_schema:
            field_type: Any = Any
        else:
            prop_type = prop_schema["type"]

            if isinstance(prop_type, list):
                # Handle union types
                union_types: tuple[Any, ...] = ()
                for pt in prop_type:
                    if pt in type_mapping:
                        union_types += (type_mapping[pt],)
                    else:
                        raise ValueError(f"Unsupported type in union: {pt}")
                field_type = Union[union_types]
            elif prop_type == "array":
                # Handle array types with items
                items_schema = prop_schema.get("items", {})
                if "type" in items_schema:
                    item_type_key = items_schema["type"]
                    if item_type_key in type_mapping:
                        item_type = type_mapping[item_type_key]
                    else:
                        raise ValueError(
                            f"Unsupported array item type: {item_type_key}"
                        )
                else:
                    item_type = Any
                field_type = list[item_type]  # type: ignore
            elif prop_type in type_mapping:
                field_type = type_mapping[prop_type]
            else:
                raise ValueError(f"Unsupported type: {prop_type}")

        # Determine the default value
        if prop_name in required_fields:
            default = msgspec.NODEFAULT
        else:
            default = prop_schema.get("default", msgspec.NODEFAULT)

        if default is not msgspec.NODEFAULT:
            fields.append((prop_name, field_type, default))
        else:
            fields.append((prop_name, field_type, msgspec.NODEFAULT))

    # Step 6: Create the Struct using msgspec.defstruct
    return cast(
        type[msgspec.Struct],
        msgspec.defstruct(
            name,
            fields,
            bases=bases,
            module=module,
            namespace=namespace,
            tag_field=tag_field,
            tag=tag,
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
        ),
    )


def jsonify(string: str) -> dict[str, Any]:
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
        >>> jsonify(json_str)
        {'name': 'Alice', 'age': 30}

        >>> json_str = '{ "name": "Bob", "age": 25 }'
        >>> jsonify(json_str)
        {'name': 'Bob', 'age': 25}

        >>> json_str = 'Here is the json\\n\\n{ "name": "Charlie", "age": 28 }'
        >>> jsonify(json_str)
        {'name': 'Charlie', 'age': 28}

        >>> json_str = '{ "name": "David", "age": 35 }\\n\\nI provided the json above'
        >>> jsonify(json_str)
        {'name': 'David', 'age': 35}

    Basic usage:

        >>> json_str = '{"name": "Alice", "age": 30}'
        >>> jsonify(json_str)
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
        >>> jsonify(json_str)
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

    logger = logging.getLogger(__name__)

    # Remove code block markers if present
    string = re.sub(r"^```(?:json)?\n", "", string, flags=re.IGNORECASE | re.MULTILINE)
    string = re.sub(r"\n```$", "", string, flags=re.MULTILINE)

    # Helper function to find substrings with balanced braces
    def find_json_substrings(s: str) -> list[str]:
        substrings = []
        stack: list[str] = []
        start = None
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
    json_substrings = find_json_substrings(string)

    if not json_substrings:
        raise ValueError("No JSON object could be found in the string.")

    # Initialize variables for parsing attempts
    parsed_obj: Optional[dict[str, Any]] = None

    # Define fix functions as inner functions
    def _fix_unescaped_backslashes(string: str) -> str:
        """
        Fix unescaped backslashes by escaping them.

        Args:
            string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        return re.sub(r'(?<!\\)\\(?![\\"])', r"\\\\", string)

    def _escape_unescaped_newlines(string: str) -> str:
        """
        Escape unescaped newline and carriage return characters within JSON strings.

        Args:
            string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        # Pattern to find JSON strings
        string_pattern = r'"((?:\\.|[^"\\])*)"'

        def replace_newlines_in_string(match: re.Match) -> str:
            content_inside_quotes = match.group(1)
            # Escape unescaped newlines and carriage returns
            content_inside_quotes = content_inside_quotes.replace("\n", "\\n").replace(
                "\r", "\\r"
            )
            return f'"{content_inside_quotes}"'

        fixed_content = re.sub(
            string_pattern, replace_newlines_in_string, string, flags=re.DOTALL
        )
        return fixed_content

    def _insert_missing_commas(string: str) -> str:
        """
        Insert missing commas between JSON objects in arrays.

        Args:
            string (str): The JSON string to fix.

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
        fixed_content = string
        for pattern, replacement in patterns:
            fixed_content = re.sub(pattern, replacement, fixed_content)
        return fixed_content

    def _remove_control_characters(string: str) -> str:
        """
        Remove control characters that may interfere with JSON parsing.

        Args:
            string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        return "".join(c for c in string if c >= " " or c == "\n")

    def _remove_invalid_characters(string: str) -> str:
        """
        Remove any remaining invalid characters (non-printable ASCII characters).

        Args:
            string (str): The JSON string to fix.

        Returns:
            str: The fixed JSON string.
        """
        return re.sub(r"[^\x20-\x7E]+", "", string)

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
                parsed_obj = msgspec.json.decode(fixed_content, type=dict)
                if parsed_obj is not None:
                    return parsed_obj
            except (msgspec.DecodeError, ValueError) as e:
                logger.error(
                    f"Failed to parse JSON string after applying fix: {fix_func.__name__}"
                )
                logger.error(f"Exception: {e}")
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
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def struct_to_dict(struct: msgspec.Struct) -> dict[str, Any]:
    return msgspec.json.decode(msgspec.json.encode(struct), type=dict)


def dict_to_struct(d: dict[str, Any], struct: type[S]) -> S:
    return msgspec.json.decode(msgspec.json.encode(d), type=struct)


def python_type_to_json_type(python_type: Any) -> JsonType:
    """Convert Python type to JSON schema type."""
    PYTHON_TO_JSON_TYPE = {
        str: "string",
        int: "integer",
        float: "float",
        bool: "bool",
        list: "array",
        dict: "object",
    }

    origin = get_origin(python_type) or python_type
    if origin in PYTHON_TO_JSON_TYPE:
        return cast(JsonType, PYTHON_TO_JSON_TYPE.get(origin, "object"))
    elif isinstance(python_type, type) and issubclass(
        python_type, (str, int, float, bool, list, dict)
    ):
        return cast(JsonType, PYTHON_TO_JSON_TYPE.get(python_type, "object"))
    return "object"  # Default fallback


def flatten_msgspec_schema(
    schema: dict[str, Any],
    remove_parameters: Optional[list[str]] = None,
    openai_like: bool = False,
) -> dict[str, Any]:
    """Flatten a msgspec-generated JSON schema by resolving all $ref references and removing $defs. Additionally, remove specified parameters from the final schema and, if openai_like=True, recursively enforce 'additionalProperties': false for all 'object' types. This produces a standalone JSON schema with no external references, suitable for APIs (such as certain LLM endpoints) that do not accept schemas containing $ref and $defs, and optionally suitable for OpenAI strict mode.

    vbnet
    Copy code
    Parameters
    ----------
    schema : dict
        The msgspec-generated JSON schema dictionary.
    remove_parameters : Optional[List[str]]
        A list of parameter names to remove from the final schema. If provided,
        any key in the schema matching an entry in this list will be removed.
    openai_like : bool
        If True, recursively enforces 'additionalProperties': false for all
        'object' types in the schema. This is often needed for strict JSON
        schema usage with OpenAI's structured-output API. Defaults to False.

    Returns
    -------
    dict
        A new dictionary representing a flattened JSON schema with:
            - No $ref, $defs
            - Specified parameters removed
            - If openai_like=True, each object node has 'additionalProperties': false

    Examples
    --------
    Basic Example with Parameter Removal:
    input_schema = {
        "$defs": {
            "MyType": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"}
                },
                "required": ["name", "title"],
                "title": "MyType Title"
            }
        },
        "$ref": "#/$defs/MyType",
        "title": "Root Title"
    }
    output_schema = flatten_msgspec_schema(input_schema, remove_parameters=["title"])
    # output_schema -> {
    #     "type": "object",
    #     "properties": {
    #         "name": {"type": "string"}
    #     },
    #     "required": ["name"]
    # }

    In this example, all occurrences of the "title" parameter are removed from the schema.

    More Complex Example with Nested Parameters Removal:
    input_schema = {
        "type": "object",
        "properties": {
            "user": {
                "$ref": "#/$defs/User"
            }
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "details": {"$ref": "#/$defs/Details"},
                    "title": {"type": "string"}
                },
                "required": ["id", "details", "title"],
                "title": "User Title"
            },
            "Details": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                    "default": {"type": "string"}
                },
                "required": ["age", "email", "default"],
                "default": "N/A"
            }
        },
        "default": "Root Default"
    }
    output_schema = flatten_msgspec_schema(
        input_schema,
        remove_parameters=["title", "default"],
        openai_like=True
    )
    # In this more complex scenario, both "title" and "default" parameters are
    # removed from all levels of the schema, and 'additionalProperties' is enforced
    # to be false in object definitions for strict usage with OpenAI.

    """
    # Make a copy so as not to mutate the original
    schema = deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve_references(node: Any) -> Any:
        """Recursively walk through the schema and resolve any $ref entries."""
        if isinstance(node, dict):
            # Resolve $ref
            if "$ref" in node:
                ref_path = node["$ref"]
                ref_name = ref_path.split("/")[-1]
                if ref_name not in defs:
                    raise ValueError(
                        f"Reference {ref_path} cannot be resolved - {ref_name} not in $defs."
                    )
                # Substitute the definition
                resolved_node = deepcopy(defs[ref_name])
                return resolve_references(resolved_node)
            else:
                # Remove unwanted parameters if specified
                if remove_parameters:
                    for param in remove_parameters:
                        if param in node:
                            del node[param]

                # Recursively resolve children
                for k, v in list(node.items()):
                    node[k] = resolve_references(v)

                return node

        elif isinstance(node, list):
            # Resolve each item in the list
            return [resolve_references(item) for item in node]

        else:
            # Base case: return the node as is
            return node

    def enforce_additional_properties_false(n: Any) -> None:
        """
        Recursively enforce 'additionalProperties': false for every
        object-type node in the schema.
        """
        if isinstance(n, dict):
            if n.get("type") == "object":
                n.setdefault("additionalProperties", False)
            for child_val in n.values():
                enforce_additional_properties_false(child_val)
        elif isinstance(n, list):
            for i in n:
                enforce_additional_properties_false(i)

    # 1. Resolve references
    schema = resolve_references(schema)

    # 2. If openai_like, recursively enforce additionalProperties=false
    if openai_like:
        enforce_additional_properties_false(schema)

    return schema


def is_file_url(url: str) -> bool:
    """
    Check if a URL is a file URL based on its extension.

    Parameters:
        url (str): The URL to check.

    Returns:
        bool: True if the URL ends with a known file extension, False otherwise.
    """
    # Parse the URL to extract the path
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Guess the MIME type based on the file extension
    mime_type, _ = mimetypes.guess_type(path)

    # If a MIME type is found, the URL likely points to a file
    return mime_type is not None


def get_file_extension(url: str) -> FileExtension:
    """
    Get the file extension from a URL.

    Parameters:
        url (str): The URL to extract the file extension from.

    Returns:
        str: The file extension (e.g., '.txt', '.jpg') extracted from the URL.
    """
    extension = url[url.rfind(".") :]
    if extension not in get_args(FileExtension):
        raise ValueError(f"Unsupported file extension: {extension}")

    return cast(FileExtension, extension)
