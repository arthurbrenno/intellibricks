import base64
import inspect
import io
import os
import re
import typing
from pathlib import Path

import imgkit
import markdown2
import matplotlib.pyplot as plt
import msgspec
from bs4 import BeautifulSoup
from graphviz import Source
from matplotlib import rcParams, use
from matplotlib.backend_bases import FigureCanvasBase
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from typing_extensions import TypedDict
from architecture.logging import LoggerFactory

logger = LoggerFactory.create(__name__)


class CallerInfo(TypedDict):
    caller_class: typing.Optional[str]
    caller_method: typing.Optional[str]
    filename: typing.Optional[str]
    line_number: typing.Optional[int]
    caller_id: typing.Optional[str]


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


def render_latex_to_base64(
    latex_string: str,
    transparent: bool = True,
    fontsize: float = 13.0,  # Slightly reduced font size
    dpi: int = 120,  # Slightly reduced DPI
) -> str:
    use("Agg")
    # Configure Matplotlib parameters to use LaTeX
    rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": fontsize,
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        }
    )

    try:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis("off")

        # Add LaTeX text
        text = ax.text(
            0,
            0,
            latex_string,
            fontsize=fontsize,
        )

        # Adjust figure size based on text bounding box
        canvas: FigureCanvasBase = fig.canvas

        # Thanks matplotlib for this wonderful implementation of stubs
        renderer = canvas.get_renderer()  # type: ignore
        bbox = text.get_window_extent(renderer=renderer)
        width, height = bbox.width / dpi, bbox.height / dpi
        fig.set_size_inches(width, height)

        # Save the figure to a buffer
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format="png",
            dpi=dpi,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(fig)

        # Encode the image in base64
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return base64_image

    except Exception as e:
        logger.error(f"Failed to render LaTeX: {e}")
        raise e


def convert_latex_tags_to_images(text: str) -> str:
    """
    Converts <inner_latex> tags into <img> tags with rendered LaTeX content.

    Args:
        text (str): Text containing <inner_latex> tags with LaTeX content.

    Returns:
        str: Text with <inner_latex> tags replaced by <img> tags.
    """

    def clean_latex(latex: str) -> str:
        """
        Cleans and corrects the LaTeX string to avoid rendering errors.

        Args:
            latex (str): The LaTeX string to clean.

        Returns:
            str: Cleaned LaTeX string.
        """
        # Replace common typos
        latex = latex.replace("\\rac", "\\frac")  # Correct \rac to \frac

        # Remove unwanted control characters
        control_chars = "".join(map(chr, range(0, 32)))  # ASCII control characters
        latex = "".join(ch for ch in latex if ch not in control_chars)

        # Replace double backslashes with single backslashes
        latex = latex.replace("\\\\", "\\")

        # Remove redundant spaces
        latex = latex.strip()

        return latex

    def strip_math_delimiters(latex: str, inline: bool) -> str:
        """
        Removes existing math delimiters to avoid double invocation.

        Args:
            latex (str): The LaTeX expression.
            inline (bool): Whether the expression is inline.

        Returns:
            str: LaTeX without delimiters.
        """
        if inline:
            if latex.startswith("$") and latex.endswith("$"):
                return latex[1:-1].strip()
        else:
            if latex.startswith("\\[") and latex.endswith("\\]"):
                return latex[2:-2].strip()
        return latex

    # Initialize BeautifulSoup to parse the text
    soup = BeautifulSoup(text, "html.parser")

    # Find all <inner_latex> tags
    latex_tags = soup.find_all("inner_latex")

    for tag in latex_tags:
        try:
            # Extract the 'inline' attribute
            inline_attr = tag.get("inline")
            inline = inline_attr is None or str(inline_attr).strip().lower() == "true"

            # Extract and clean the LaTeX content
            latex_content = tag.get_text(strip=True)
            latex_content = clean_latex(latex_content)
            latex_content = strip_math_delimiters(latex_content, inline)

            # Set parameters based on inline or display mode
            if not inline:
                # Display mode
                latex_wrapped = f"\\[ {latex_content} \\]"
                style = "display: block; margin: 10px auto; text-align: center;"
                fontsize = 15.0  # Slightly reduced font size for display formulas
                dpi = 120  # Match DPI with the rendering function
            else:
                # Inline mode
                latex_wrapped = f"${latex_content}$"
                style = "vertical-align: middle; display: inline;"
                fontsize = 13.0  # Slightly reduced font size for inline formulas
                dpi = 120

            # Render LaTeX to base64 image
            base64_image = render_latex_to_base64(
                latex_wrapped, fontsize=fontsize, dpi=dpi
            )

            # Create new <img> tag
            img_tag = soup.new_tag(
                "img",
                src=f"data:image/png;base64,{base64_image}",
                alt="LaTeX",
                style=style,
            )

            # Replace the <inner_latex> tag with the <img> tag
            tag.replace_with(img_tag)

        except Exception as e:
            logger.error(f"Error processing <inner_latex> tag: {e}")
            tag.replace_with("(Error rendering mathematical formula.)")

    # Return the modified text
    return str(soup)


def graphviz_to_base64(
    dot_string: str,
    transparent: bool = True,
) -> typing.Optional[str]:
    """
    Converts a Graphviz DOT string into a Base64-encoded PNG image.

    Parameters:
    ----------
    dot_string : str
        The DOT language string describing the graph to be rendered.

    transparent : bool, optional, default=True
        A parameter reserved for potential future use related to image transparency.
        Currently, the function always generates a PNG image with transparency by default.

    Returns:
    -------
    typing.Optional[str]
        A Base64-encoded string of the PNG image, or None if an error occurs during rendering.

    Examples:
    --------
    Convert a simple DOT graph into a Base64 string:

    >>> dot = '''
    ... digraph G {
    ...     A -> B
    ...     B -> C
    ...     C -> A
    ... }
    ... '''
    >>> base64_image = graphviz_to_base64(dot)
    >>> print(base64_image)  # Outputs the Base64 encoded PNG image as a string

    """
    logger.debug(f"Parsing DOT string: {dot_string}")

    def replace_literal_newlines(dot_str: str) -> str:
        """
        Replaces literal '\\n' sequences with actual newline characters.
        """
        return dot_str.replace("\\n", "\n")

    def quote_invalid_identifiers(dot_str: str) -> str:
        """
        Quotes any identifiers that contain invalid characters.
        """
        # Valid unquoted identifiers in DOT start with a letter or underscore
        # and can be followed by letters, digits, or underscores.
        # Identifiers with special characters need to be quoted.
        # We need to avoid quoting numbers and keywords.

        # Define DOT keywords to exclude
        keywords = {
            "graph",
            "digraph",
            "subgraph",
            "node",
            "edge",
            "strict",
            "label",
            "rankdir",
        }

        # Pattern to match unquoted identifiers that contain invalid characters
        # Exclude numbers and keywords
        pattern = r'(?<!")\b([a-zA-Z_][a-zA-Z0-9_\-]*[^\s;\{\}"]*)\b(?!")'

        def replace_identifier(match: re.Match) -> str:
            identifier: str = match.group(1)
            # If identifier is a keyword or number, leave it
            if identifier in keywords or re.match(r"^[0-9]+$", identifier):
                return identifier
            # If identifier contains only valid characters, leave it
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
                return identifier
            # Otherwise, quote it
            return f'"{identifier}"'

        fixed_dot_str = re.sub(pattern, replace_identifier, dot_str)
        return fixed_dot_str

    def remove_invalid_statements(dot_str: str) -> str:
        """
        Removes invalid statements like 'node ;' or empty statements.
        """
        # Remove lines that are just 'node ;', 'edge ;', etc.
        fixed_dot_str = re.sub(
            r"^\s*(node|edge|graph|digraph|subgraph)\s*;\s*$",
            "",
            dot_str,
            flags=re.MULTILINE,
        )
        return fixed_dot_str

    # Define a list of fix functions
    fix_functions: list[typing.Callable[[str], str]] = [
        lambda x: x,  # First attempt without any fixes
        replace_literal_newlines,
        quote_invalid_identifiers,
        remove_invalid_statements,
        lambda x: remove_invalid_statements(quote_invalid_identifiers(x)),
        lambda x: quote_invalid_identifiers(remove_invalid_statements(x)),
    ]

    # Attempt to render, applying fixes sequentially
    for fix_func in fix_functions:
        try:
            fixed_dot_string: str = fix_func(dot_string)
            src: Source = Source(fixed_dot_string)
            png_data: bytes = src.pipe(format="png")
            base64_data: str = base64.b64encode(png_data).decode("utf-8")
            return base64_data
        except Exception as e:
            logger.error(
                f"Failed to render Graphviz after applying fix: {fix_func.__name__}"
            )
            logger.error(f"Exception: {e}")
            continue  # Try next fix function

    # If all attempts fail, return None
    return None


def table_to_base64(table_string: str) -> str:
    """
    Convert an HTML table string to a base64-encoded PNG image using imgkit.

    Args:
        table_string (str): A string containing valid HTML table markup.

    Returns:
        str: A base64-encoded string representing the PNG image of the rendered table.

    Raises:
        OSError: If there are issues with imgkit or wkhtmltoimage.
        PIL.UnidentifiedImageError: If the generated image data cannot be processed.
    """
    # Prepare the full HTML content
    html_content = f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        {table_string}
    </body>
    </html>
    """

    # Convert HTML string to image
    img_data = imgkit.from_string(html_content, False)

    if isinstance(img_data, bool):
        raise ValueError("IMGDATA is True")  # TODO: Check the impacts of this

    # Convert image data to base64
    img_str = base64.b64encode(img_data).decode()

    return img_str


def markdown_to_html(
    markdown_text: str, extras: typing.Optional[list[str]] = None
) -> str:
    # Create a Markdown instance with basic features including inline code
    if extras is None:
        extras = [
            "tables",
            "break-on-newline",
            "metadata",
            "code-friendly",
        ]

    md = markdown2.Markdown(extras=extras)

    # Convert the Markdown to HTML
    html_content: str = md.convert(markdown_text)
    return html_content


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
    s: str, case_sensitive: bool = True, **replacements: typing.Any
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
    **replacements: typing.Any,
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
    json_schema: dict[str, typing.Any],
    bases: typing.Optional[typing.Tuple[typing.Type[msgspec.Struct], ...]] = None,
    name: typing.Optional[str] = None,
    module: typing.Optional[str] = None,
    namespace: typing.Optional[dict[str, typing.Any]] = None,
    tag_field: typing.Optional[str] = None,
    tag: typing.Union[
        None, bool, str, int, typing.Callable[[str], typing.Union[str, int]]
    ] = None,
    rename: typing.Union[
        None,
        typing.Literal["lower", "upper", "camel", "pascal", "kebab"],
        typing.Callable[[str], typing.Optional[str]],
        dict[str, str],
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
) -> typing.Type[msgspec.Struct]:
    """
    Create a msgspec.Struct type from a JSON schema at runtime.

    Args:
        json_schema (dict[str, typing.Any]): The JSON schema defining the structure.
        bases (typing.Optional[typing.Tuple[typing.Type[msgspec.Struct], ...]]): Base classes for the new Struct.
        name (typing.Optional[str]): Name for the new Struct. If not provided, it's derived from the schema title.
        module (typing.Optional[str]): Module name for the new Struct.
        namespace (typing.Optional[dict[str, typing.Any]]): Additional namespace for the new Struct.
        tag_field (typing.Optional[str]): Name of the field to use for tagging.
        tag (typing.Union[None, bool, str, int, Callable]): Tag value or function to generate tag.
        rename (typing.Union[None, str, Callable, dict[str, str]]): Field renaming strategy.
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
        typing.Type[msgspec.Struct]: A new msgspec.Struct type based on the provided JSON schema.

    Raises:
        ValueError: If the JSON schema is invalid or missing required information.
    """

    def resolve_refs(schema: typing.Any, root: dict[str, typing.Any]) -> typing.Any:
        """
        Recursively resolve $ref in a JSON schema.

        Args:
            schema (typing.Any): The current schema node to resolve.
            root (dict[str, typing.Any]): The root schema containing definitions.

        Returns:
            typing.Any: The schema with all $ref resolved.
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
    type_mapping: dict[str, typing.Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
        "array": list,  # This is okay for runtime
        "object": dict,
    }

    # Step 5: Process each property
    fields: list[tuple[str, typing.Any, typing.Any]] = []

    required_fields = resolved_schema.get("required", [])

    for prop_name, prop_schema in resolved_schema["properties"].items():
        # Determine the field type based on the property schema
        if "type" not in prop_schema:
            field_type: typing.Any = typing.Any
        else:
            prop_type = prop_schema["type"]

            if isinstance(prop_type, list):
                # Handle union types
                union_types: tuple[typing.Any, ...] = ()
                for pt in prop_type:
                    if pt in type_mapping:
                        union_types += (type_mapping[pt],)
                    else:
                        raise ValueError(f"Unsupported type in union: {pt}")
                field_type = typing.Union[union_types]
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
                    item_type = typing.Any
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
    return typing.cast(
        typing.Type[msgspec.Struct],
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


def deserialize_json(content: str) -> dict[str, typing.Any]:
    """
    Parses a JSON-formatted string into a Python dictionary, applying automatic corrections for common formatting issues.

    This function attempts to deserialize a string containing JSON data into a Python dictionary. It handles JSON content that may be embedded within code block markers (e.g., Markdown-style ```json code blocks) and applies a series of fix-up functions to correct common JSON formatting issues such as unescaped characters, missing commas, and control characters that may prevent successful parsing.

    Parameters
    ----------
    content : str
        The string containing JSON content to deserialize. This may include code block markers and may have minor formatting issues.

    Returns
    -------
    dict[str, typing.Any]
        A Python dictionary representing the parsed JSON content.

    Raises
    ------
    ValueError
        If no JSON object could be found in the content, or if parsing fails after applying all fix functions.

    Examples
    --------
    Basic usage:

        >>> json_str = '{"name": "Alice", "age": 30}'
        >>> deserialize_json(json_str)
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
        >>> deserialize_json(json_str)
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

    - **No fix**: Attempts to parse the content as-is.
    - **Escaping unescaped backslashes**: Fixes unescaped backslashes in the content.
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
    content = re.sub(
        r"^```(?:json)?\n", "", content, flags=re.IGNORECASE | re.MULTILINE
    )
    content = re.sub(r"\n```$", "", content, flags=re.MULTILINE)

    # Extract the JSON content between the first '{' and the last '}'
    json_start: int = content.find("{")
    json_end: int = content.rfind("}") + 1

    if json_start == -1 or json_end == 0:
        raise ValueError("No JSON object could be found in the content.")

    json_content: str = content[json_start:json_end]

    # Initialize variables for parsing attempts
    parsed_obj: typing.Optional[dict[str, typing.Any]] = None

    # Define fix functions as inner functions
    def _fix_unescaped_backslashes(content: str) -> str:
        """
        Fix unescaped backslashes by escaping them.

        Args:
            content (str): The JSON content to fix.

        Returns:
            str: The fixed JSON content.
        """
        return re.sub(r'(?<!\\)\\(?![\\"])', r"\\\\", content)

    def _escape_unescaped_newlines(content: str) -> str:
        """
        Escape unescaped newline and carriage return characters within JSON strings.

        Args:
            content (str): The JSON content to fix.

        Returns:
            str: The fixed JSON content.
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
            string_pattern, replace_newlines_in_string, content, flags=re.DOTALL
        )
        return fixed_content

    def _insert_missing_commas(content: str) -> str:
        """
        Insert missing commas between JSON objects in arrays.

        Args:
            content (str): The JSON content to fix.

        Returns:
            str: The fixed JSON content.
        """
        # Insert commas between closing and opening braces/brackets
        patterns = [
            (r"(\})(\s*\{)", r"\1,\2"),  # Between } and {
            (r"(\])(\s*\[)", r"\1,\2"),  # Between ] and [
            (r"(\])(\s*\{)", r"\1,\2"),  # Between ] and {
            (r"(\})(\s*\[)", r"\1,\2"),  # Between } and [
        ]
        fixed_content = content
        for pattern, replacement in patterns:
            fixed_content = re.sub(pattern, replacement, fixed_content)
        return fixed_content

    def _remove_control_characters(content: str) -> str:
        """
        Remove control characters that may interfere with JSON parsing.

        Args:
            content (str): The JSON content to fix.

        Returns:
            str: The fixed JSON content.
        """
        return "".join(c for c in content if c >= " " or c == "\n")

    def _remove_invalid_characters(content: str) -> str:
        """
        Remove any remaining invalid characters (non-printable ASCII characters).

        Args:
            content (str): The JSON content to fix.

        Returns:
            str: The fixed JSON content.
        """
        return re.sub(r"[^\x20-\x7E]+", "", content)

    # Define a list of fix functions
    fix_functions: list[typing.Callable[[str], str]] = [
        lambda x: x,  # First attempt without any fixes
        _fix_unescaped_backslashes,
        _escape_unescaped_newlines,
        _insert_missing_commas,
        _remove_control_characters,
        _remove_invalid_characters,
    ]

    # Attempt parsing, applying fixes sequentially
    for fix_func in fix_functions:
        try:
            # Apply the fix function
            fixed_content: str = fix_func(json_content)
            # Try parsing the JSON content
            parsed_obj = msgspec.json.decode(fixed_content, type=dict)
            if parsed_obj is None:
                raise ValueError("Failed to parse JSON content.")
            return parsed_obj
        except (msgspec.DecodeError, ValueError) as e:
            logger.error(
                f"Failed to parse JSON content after applying fix: {fix_func.__name__}"
            )
            logger.error(f"Exception: {e}")
            continue  # Try next fix function

    # If all attempts fail, raise an error
    raise ValueError("Failed to parse JSON content after multiple attempts.")
