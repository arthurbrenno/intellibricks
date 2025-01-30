"""
Module: parsed_files

This module defines schema objects used to represent the structured output of file parsing operations.
It provides a set of classes designed to encapsulate different types of content extracted from files,
such as images, text items, tables, and document sections. These schema objects serve as a standardized
data structure for representing parsed file content, facilitating further processing and analysis.

**Core Concepts:**

The module revolves around the concept of a `ParsedFile`, which is the top-level container for the extracted content.
A `ParsedFile` is composed of `SectionContent` objects, each representing a section or page of the original document.
Within each `SectionContent`, you can find:

*   **Textual Content:**  `text` and `md` attributes store the raw text and Markdown representations of the section's content.
*   **Images:** The `images` attribute holds a list of `Image` objects, each representing an image found in the section.
*   **Page Items:** The `items` attribute contains a list of `PageItem` objects, which are abstract representations of structured content within a section. Concrete implementations of `PageItem` include `TextPageItem`, `HeadingPageItem`, and `TablePageItem`.

**Class Hierarchy:**

*   **`Image`**: Represents an image extracted from a document, including its content, OCR text, dimensions, and metadata.
*   **`PageItem`**: Abstract base class for representing structured items within a document section.
    *   **`TextPageItem`**: Represents a simple text item within a document section.
    *   **`HeadingPageItem`**: Represents a heading within a document section, including its level.
    *   **`TablePageItem`**: Represents a table extracted from a document section, including its rows, CSV representation, and table properties.
*   **`SectionContent`**: Represents a section or page of a parsed document, containing text, images, and page items.
*   **`JobMetadata`**: Represents metadata associated with a parsing job, such as credit usage and cache hit status.
*   **`Schema`**: Represents a schema of entities and relations that can be extracted from a document, typically used for structured information extraction tasks.
*   **`ParsedFile`**: The top-level class representing a parsed file, containing a list of `SectionContent` objects and overall file metadata.

**Usage:**

These schema objects are typically created and populated by file parser classes (as defined in the `parsers.py` module).
After parsing a file, you will receive a `ParsedFile` object, which you can then use to access and manipulate the extracted content in a structured manner.

**Example:**

```python
from intellibricks.files import (
    ParsedFile,
    SectionContent,
    TextPageItem,
    Image,
    PageItem,
    RawFile,
    TxtFileParser
)

# Assume you have a RawFile object (e.g., from reading a text file)
raw_file = RawFile.from_bytes(b"Sample text content", "txt")
parser = TxtFileParser()
parsed_file = parser.parse(raw_file)

# Accessing parsed content
if isinstance(parsed_file, ParsedFile):
    print(f"Parsed File Name: {parsed_file.name}")
    for section in parsed_file.sections: # Iterate through sections (pages)
        if isinstance(section, SectionContent):
            print(f"Section Number: {section.number}")
            print(f"Section Text (first 50 chars): {section.text[:50]}...")
            for item in section.items: # Iterate through page items in the section
                if isinstance(item, TextPageItem):
                    print(f"  Text Item: {item.text[:30]}...")
            for image in section.images: # Iterate through images in the section
                if isinstance(image, Image):
                    print(f"  Image Name: {image.name}")

```

This module provides a clear and organized way to represent parsed file data, enabling developers to work with extracted content efficiently and effectively.
"""

from __future__ import annotations
from typing import Annotated, Any, Optional, Sequence, cast

import msgspec
from architecture.utils import run_sync

from intellibricks.llms import ChainOfThought, Synapse, TraceParams


class Image(msgspec.Struct, frozen=True):
    """
    Represents an image extracted from a document.

    This class encapsulates the content of an image file, along with optional metadata such as
    OCR-extracted text, dimensions (height and width), original name, and alt text.

    **Attributes:**

    *   `contents` (bytes):
        The raw byte content of the image file.

        **Example:**
        ```python
        image_content = b"\\x89PNG...image data..." # Example PNG image data
        image = Image(contents=image_content)
        print(type(image.contents)) # Output: <class 'bytes'>
        ```

    *   `ocr_text` (Optional[str]):
        Text extracted from the image using Optical Character Recognition (OCR), if performed.
        Defaults to `None` if OCR was not applied or no text was found.

        **Example:**
        ```python
        image_with_ocr = Image(contents=b"...", ocr_text="Extracted text from image")
        image_no_ocr = Image(contents=b"...")
        print(image_with_ocr.ocr_text) # Output: Extracted text from image
        print(image_no_ocr.ocr_text) # Output: None
        ```

    *   `height` (Optional[float]):
        The height of the image in pixels. Defaults to `None` if the height is not available.

        **Example:**
        ```python
        image_with_dimensions = Image(contents=b"...", height=100.0, width=200.0)
        print(image_with_dimensions.height) # Output: 100.0
        ```

    *   `width` (Optional[float]):
        The width of the image in pixels. Defaults to `None` if the width is not available.

        **Example:**
        ```python
        image_with_dimensions = Image(contents=b"...", height=100.0, width=200.0)
        print(image_with_dimensions.width) # Output: 200.0
        ```

    *   `name` (Optional[str]):
        The name of the image file as it was present in the original document, if available.
        Defaults to `None`.

        **Example:**
        ```python
        named_image = Image(contents=b"...", name="logo.png")
        unnamed_image = Image(contents=b"...")
        print(named_image.name) # Output: logo.png
        print(unnamed_image.name) # Output: None
        ```

    *   `alt` (Optional[str]):
        The alternative text (alt text) associated with the image, if provided in the original document.
        Defaults to `None`.

        **Example:**
        ```python
        image_with_alt_text = Image(contents=b"...", alt="Company logo")
        image_no_alt_text = Image(contents=b"...")
        print(image_with_alt_text.alt) # Output: Company logo
        print(image_no_alt_text.alt) # Output: None
        ```
    """

    contents: Annotated[
        bytes,
        msgspec.Meta(
            title="Contents",
            description="Contents of the image file.",
        ),
    ]

    ocr_text: Annotated[
        Optional[str],
        msgspec.Meta(
            title="OCR Text",
            description="Text extracted from the image using OCR.",
        ),
    ] = None

    height: Annotated[
        Optional[float],
        msgspec.Meta(
            title="Height",
            description="Height of the image in pixels.",
        ),
    ] = None

    width: Annotated[
        Optional[float],
        msgspec.Meta(
            title="Width",
            description="Width of the image in pixels.",
        ),
    ] = None

    name: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Name",
            description="The name of the image file present in the original document.",
        ),
    ] = msgspec.field(default=None)

    alt: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Alt Text",
            description="The alt text of the image.",
        ),
    ] = msgspec.field(default=None)


class PageItem(msgspec.Struct, tag_field="type", frozen=True):
    """
    Abstract base class for representing structured items within a document section.

    `PageItem` is an abstract class and should not be instantiated directly. It serves as a base
    for concrete item types like `TextPageItem`, `HeadingPageItem`, and `TablePageItem`.
    All `PageItem` subclasses must include a `md` attribute, which provides a Markdown representation
    of the item.

    **Abstract Subclasses:**

    *   `TextPageItem`: Represents a simple text item.
    *   `HeadingPageItem`: Represents a heading.
    *   `TablePageItem`: Represents a table.

    **Attributes (Common to all subclasses):**

    *   `md` (str):
        Markdown representation of the page item. This attribute is essential for rendering
        and displaying the item in a human-readable format.

        **Example (Illustrative - PageItem is abstract):**
        ```python
        # PageItem is abstract, so this is just illustrative.
        # In practice, you would work with subclasses like TextPageItem.
        class MyCustomItem(PageItem):
            custom_data: str

        my_item = MyCustomItem(md="# Custom Item in Markdown", custom_data="Some data")
        print(my_item.md) # Output: # Custom Item in Markdown
        ```
    """

    md: Annotated[
        str,
        msgspec.Meta(
            title="Markdown Representation",
            description="Markdown representation of the item",
        ),
    ]


class TextPageItem(PageItem, tag="text", frozen=True):
    """
    Represents a simple text item within a document section.

    This class is a concrete implementation of `PageItem` and is used to represent
    plain text content within a section of a parsed document.

    **Inheritance:**

    *   Inherits from `PageItem`.

    **Attributes:**

    *   `text` (str):
        The actual text content of the item.

        **Example:**
        ```python
        text_item = TextPageItem(md="This is text in markdown.", text="This is the plain text value.")
        print(text_item.md)   # Output: This is text in markdown.
        print(text_item.text) # Output: This is the plain text value.
        print(text_item.type) # Output: text
        ```

    *   `md` (str):
        Inherited from `PageItem`. Markdown representation of the text item.
    """

    text: Annotated[
        str,
        msgspec.Meta(
            title="Value",
            description="Value of the text item",
        ),
    ]


class HeadingPageItem(PageItem, tag="heading", frozen=True):
    """
    Represents a heading within a document section.

    This class is a concrete implementation of `PageItem` and is used to represent
    headings (like section titles or subtitles) within a parsed document. It includes
    the heading text and the heading level (e.g., H1, H2, H3).

    **Inheritance:**

    *   Inherits from `PageItem`.

    **Attributes:**

    *   `heading` (str):
        The text content of the heading.

        **Example:**
        ```python
        heading_item = HeadingPageItem(md="## Section Title", heading="Section Title", lvl=2)
        print(heading_item.md)      # Output: ## Section Title
        print(heading_item.heading) # Output: Section Title
        print(heading_item.lvl)     # Output: 2
        print(heading_item.type)    # Output: heading
        ```

    *   `lvl` (int):
        The heading level (e.g., 1 for H1, 2 for H2, etc.).

    *   `md` (str):
        Inherited from `PageItem`. Markdown representation of the heading.
    """

    heading: Annotated[
        str,
        msgspec.Meta(
            title="Value",
            description="Value of the heading",
        ),
    ]

    lvl: Annotated[
        int,
        msgspec.Meta(
            title="Level",
            description="Level of the heading",
        ),
    ]


class TablePageItem(PageItem, tag="table", frozen=True):
    """
    Represents a table extracted from a document section.

    This class is a concrete implementation of `PageItem` and is used to represent
    tables found within a parsed document. It includes the table data as a list of rows,
    a CSV representation of the table, and a flag indicating if the table is considered a "perfect" table
    (e.g., well-structured without irregularities).

    **Inheritance:**

    *   Inherits from `PageItem`.

    **Attributes:**

    *   `rows` (Sequence[Sequence[str]]):
        A sequence of rows, where each row is a sequence of strings representing the cells in that row.

        **Example:**
        ```python
        table_rows = [
            ["Header 1", "Header 2"],
            ["Data 1", "Data 2"],
            ["Data 3", "Data 4"]
        ]
        table_item = TablePageItem(md="| Header 1 | Header 2 |\n|---|---|\n| Data 1 | Data 2 |\n| Data 3 | Data 4 |", rows=table_rows, csv="Header 1,Header 2\\nData 1,Data 2\\nData 3,Data 4", is_perfect_table=True)
        print(table_item.rows) # Output: [['Header 1', 'Header 2'], ['Data 1', 'Data 2'], ['Data 3', 'Data 4']]
        ```

    *   `csv` (str):
        A string containing the CSV (Comma Separated Values) representation of the table data.

        **Example:**
        ```python
        print(table_item.csv) # Output: Header 1,Header 2\nData 1,Data 2\nData 3,Data 4
        ```

    *   `is_perfect_table` (bool):
        A boolean flag indicating whether the table is considered a "perfect table".
        This can be used to differentiate between well-structured tables and tables with potential irregularities.
        Defaults to `False`.

        **Example:**
        ```python
        print(table_item.is_perfect_table) # Output: True
        ```

    *   `md` (str):
        Inherited from `PageItem`. Markdown representation of the table.
    """

    rows: Annotated[
        Sequence[Sequence[str]],
        msgspec.Meta(
            title="Rows",
            description="Rows of the table.",
        ),
    ]

    csv: Annotated[
        str,
        msgspec.Meta(
            title="CSV Representation",
            description="CSV representation of the table",
        ),
    ]

    is_perfect_table: Annotated[
        bool,
        msgspec.Meta(
            title="Is Perfect Table",
            description="Whether the table is a perfect table",
        ),
    ] = False


class SectionContent(msgspec.Struct, frozen=True):
    """
    Represents a section or page of a parsed document.

    This class aggregates various types of content extracted from a section of a document,
    such as text, images, and structured page items. It is typically used to represent a page
    in a PDF or a section in a document format that doesn't have explicit pages (like DOCX).

    **Attributes:**

    *   `number` (int):
        The section number or page number. This is used to maintain the order of sections in the document.

        **Example:**
        ```python
        section1 = SectionContent(number=1, text="Section 1 Text", md="# Section 1", images=[], items=[])
        print(section1.number) # Output: 1
        ```

    *   `text` (str):
        The plain text content of the section. This is the raw text extracted from the section.

        **Example:**
        ```python
        print(section1.text) # Output: Section 1 Text
        ```

    *   `md` (Optional[str]):
        Optional Markdown representation of the section's content. This can be used for richer formatting
        and display of the section content. Defaults to `None`.

        **Example:**
        ```python
        print(section1.md) # Output: # Section 1
        ```

    *   `images` (Sequence[Image]):
        A sequence of `Image` objects representing the images found in this section. Defaults to an empty list.

        **Example:**
        ```python
        image1 = Image(contents=b"...", name="image1.png")
        section_with_image = SectionContent(number=1, text="Section with image", md="...", images=[image1], items=[])
        print(len(section_with_image.images)) # Output: 1
        ```

    *   `items` (Sequence[PageItem]):
        A sequence of `PageItem` objects representing structured items (like text items, headings, tables)
        found in this section. Defaults to an empty list.

        **Example:**
        ```python
        text_item1 = TextPageItem(md="Text Item 1", text="Value 1")
        heading_item1 = HeadingPageItem(md="## Heading 1", heading="Heading 1", lvl=2)
        section_with_items = SectionContent(number=1, text="Section with items", md="...", images=[], items=[text_item1, heading_item1])
        print(len(section_with_items.items)) # Output: 2
        ```

    **Methods:**

    *   `get_id() -> str`:
        Returns a unique identifier for the section, typically based on its number.

        **Returns:**
        *   `str`: Section ID (e.g., "page_1").

        **Example:**
        ```python
        print(section1.get_id()) # Output: page_1
        ```

    *   `__add__(other: SectionContent) -> SectionContent`:
        Overloads the addition operator (+) to merge two `SectionContent` objects.
        When two sections are added, their text, Markdown content, images, and items are concatenated.
        The section number of the first section is preserved in the merged section.

        **Parameters:**
        *   `other` (SectionContent): The other `SectionContent` object to merge with.

        **Returns:**
        *   `SectionContent`: A new `SectionContent` object that is the result of merging the two input sections.

        **Example:**
        ```python
        section2 = SectionContent(number=1, text="Section 2 Text", md="# Section 2", images=[], items=[])
        merged_section = section1 + section2
        print(merged_section.text)    # Output: Section 1 TextSection 2 Text
        print(merged_section.md)      # Output: # Section 1# Section 2
        print(merged_section.number)  # Output: 1 (number from section1 is preserved)
        ```
    """

    number: Annotated[
        int,
        msgspec.Meta(
            title="Number",
            description="Section number",
        ),
    ]

    text: Annotated[
        str,
        msgspec.Meta(
            title="Text",
            description="Text content's of the page",
        ),
    ]

    md: Annotated[
        Optional[str],
        msgspec.Meta(
            title="Markdown Representation",
            description="Markdown representation of the section.",
        ),
    ] = None

    images: Annotated[
        Sequence[Image],
        msgspec.Meta(
            title="Images",
            description="Images present in the section",
        ),
    ] = msgspec.field(default_factory=list)

    items: Annotated[
        Sequence[PageItem],
        msgspec.Meta(
            title="Items",
            description="Items present in the page",
        ),
    ] = msgspec.field(default_factory=list)

    def get_id(self) -> str:
        return f"page_{self.number}"

    def __add__(self, other: SectionContent) -> SectionContent:
        from itertools import chain

        return SectionContent(
            number=self.number,
            text=self.text + other.text,
            md=(self.md or "") + (other.md or ""),
            images=list(chain(self.images, other.images)),
            items=list(chain(self.items, other.items)),
        )


class JobMetadata(msgspec.Struct, frozen=True):
    """
    Represents metadata associated with a parsing job.

    This class is used to track various metrics related to a file parsing job, such as
    credits used, maximum allowed credits, credit usage for the job, number of pages processed,
    and whether the job was a cache hit.

    **Attributes:**

    *   `credits_used` (float):
        The total credits used for the parsing job. Must be greater than or equal to 0. Defaults to 0.0.

        **Example:**
        ```python
        metadata = JobMetadata(credits_used=0.5)
        print(metadata.credits_used) # Output: 0.5
        ```

    *   `credits_max` (int):
        The maximum credits allowed for the parsing job. Must be greater than or equal to 0. Defaults to 0.

        **Example:**
        ```python
        metadata = JobMetadata(credits_max=100)
        print(metadata.credits_max) # Output: 100
        ```

    *   `job_credits_usage` (int):
        Credits specifically used for this particular job. Must be greater than or equal to 0. Defaults to 0.

        **Example:**
        ```python
        metadata = JobMetadata(job_credits_usage=10)
        print(metadata.job_credits_usage) # Output: 10
        ```

    *   `job_pages` (int):
        The number of pages processed in the job. Must be greater than or equal to 0. Defaults to 0.

        **Example:**
        ```python
        metadata = JobMetadata(job_pages=5)
        print(metadata.job_pages) # Output: 5
        ```

    *   `job_is_cache_hit` (bool):
        A boolean flag indicating whether the parsing job was a cache hit. Defaults to `False`.
        If `True`, it means the result was retrieved from a cache instead of being re-processed.

        **Example:**
        ```python
        metadata_cache_hit = JobMetadata(job_is_cache_hit=True)
        metadata_cache_miss = JobMetadata(job_is_cache_hit=False)
        print(metadata_cache_hit.job_is_cache_hit) # Output: True
        print(metadata_cache_miss.job_is_cache_hit) # Output: False
        ```
    """

    credits_used: Annotated[
        float,
        msgspec.Meta(
            title="Credits Used",
            description="Credits used for the job",
            ge=0,
        ),
    ] = 0.0

    credits_max: Annotated[
        int,
        msgspec.Meta(
            title="Credits Max",
            description="Maximum credits allowed for the job",
            ge=0,
        ),
    ] = 0

    job_credits_usage: Annotated[
        int,
        msgspec.Meta(
            title="Job Credits Usage",
            description="Credits used for the job",
            ge=0,
        ),
    ] = 0

    job_pages: Annotated[
        int,
        msgspec.Meta(
            title="Job Pages",
            description="Number of pages processed",
            ge=0,
        ),
    ] = 0

    job_is_cache_hit: Annotated[
        bool,
        msgspec.Meta(
            title="Job Is Cache Hit",
            description="Whether the job is a cache hit",
        ),
    ] = False


class Schema(msgspec.Struct, frozen=True):
    """
    Represents a schema of entities and relations present in a document.

    The `Schema` class is used to define the structure for extracting structured information
    from documents. It specifies the types of entities and relationships that are relevant
    for a particular document processing task.

    **Attributes:**

    *   `entities` (Sequence[str]):
        A sequence of entity names that are expected to be found in the document.
        Must contain at least one entity name.

        **Example:**
        ```python
        schema = Schema(
            entities=["Person", "Organization", "Location"],
            relations=["works_at", "located_in"],
            validation_schema={}
        )
        print(schema.entities) # Output: ['Person', 'Organization', 'Location']
        ```

    *   `relations` (Sequence[str]):
        A sequence of relation names that define how entities can be related to each other.
        Must contain at least one relation name.

        **Example:**
        ```python
        print(schema.relations) # Output: ['works_at', 'located_in']
        ```

    *   `validation_schema` (dict[str, Sequence[str]]):
        A dictionary that defines valid relationships for each entity type.
        The keys are entity names, and the values are sequences of relation names that are valid for that entity.

        **Example:**
        ```python
        validation = {
            "Person": ["works_at", "lives_in"],
            "Organization": ["located_in"]
        }
        schema_with_validation = Schema(
            entities=["Person", "Organization", "Location"],
            relations=["works_at", "located_in", "lives_in"],
            validation_schema=validation
        )
        print(schema_with_validation.validation_schema)
        # Output: {'Person': ['works_at', 'lives_in'], 'Organization': ['located_in']}
        ```
    """

    entities: Annotated[
        Sequence[str],
        msgspec.Meta(
            title="Entities",
            description="A list of entity names present in the document.",
            min_length=1,
            examples=[["Person", "Organization", "Location"]],
        ),
    ]

    relations: Annotated[
        Sequence[str],
        msgspec.Meta(
            title="Relations",
            description="A list of relation names present in the document.",
            min_length=1,
            examples=[["works_at", "located_in", "employs"]],
        ),
    ]

    validation_schema: Annotated[
        dict[str, Sequence[str]],
        msgspec.Meta(
            title="Validation Schema",
            description="A dictionary mapping entities to lists of valid relations.",
            examples=[
                {
                    "Person": ["works_at", "lives_in"],
                    "Organization": ["employs"],
                    "Location": [],
                }
            ],
        ),
    ]


class ParsedFile(msgspec.Struct, frozen=True):
    """
    Represents a parsed file, containing structured content extracted from a document.

    This is the top-level class that encapsulates the result of parsing a file. It contains the file's
    name and a sequence of `SectionContent` objects, representing the sections or pages of the parsed document.

    **Attributes:**

    *   `name` (str):
        The name of the parsed file.

        **Example:**
        ```python
        parsed_doc = ParsedFile(name="document.pdf", sections=[])
        print(parsed_doc.name) # Output: document.pdf
        ```

    *   `sections` (Sequence[SectionContent]):
        A sequence of `SectionContent` objects, each representing a section or page of the parsed document.

        **Example:**
        ```python
        section1 = SectionContent(number=1, text="Section 1 Text", md="...", images=[], items=[])
        section2 = SectionContent(number=2, text="Section 2 Text", md="...", images=[], items=[])
        parsed_doc_with_sections = ParsedFile(name="document.pdf", sections=[section1, section2])
        print(len(parsed_doc_with_sections.sections)) # Output: 2
        ```

    **Properties:**

    *   `llm_described_text` (str):
        A property that returns a string representation of the parsed file's sections, formatted for use with Language Model Models (LLMs).
        Each section's Markdown content is wrapped in `<section_num>` tags, and the file name and sections are included in `<file>` tags.

        **Returns:**
        *   `str`: LLM-formatted text representation of the parsed file.

        **Example:**
        ```python
        print(parsed_doc_with_sections.llm_described_text)
        # Output will be a string like:
        # <file>
        #
        # **name:** document.pdf
        # **sections:** <section_0> ...Markdown of Section 1... </section_0> <section_1> ...Markdown of Section 2... </section_1>
        #
        # </file>
        ```

    *   `md` (str):
        A property that concatenates the Markdown content of all sections in the parsed file into a single string.

        **Returns:**
        *   `str`: Combined Markdown content of all sections.

        **Example:**
        ```python
        print(parsed_doc_with_sections.md)
        # Output will be:
        # ...Markdown of Section 1...
        # ...Markdown of Section 2...
        ```

    **Methods:**

    *   `merge_all(others: Sequence[ParsedFile]) -> ParsedFile`:
        Merges the sections of multiple `ParsedFile` objects into a single `ParsedFile`.
        The sections from all input `ParsedFile` objects are concatenated into the `sections` attribute of the returned `ParsedFile`.
        The name of the new `ParsedFile` is taken from the first `ParsedFile` in the sequence.

        **Parameters:**
        *   `others` (Sequence[ParsedFile]): A sequence of other `ParsedFile` objects to merge with.

        **Returns:**
        *   `ParsedFile`: A new `ParsedFile` object containing the merged sections.

        **Example:**
        ```python
        parsed_doc2 = ParsedFile(name="document2.pdf", sections=[section3])
        merged_files = parsed_doc_with_sections.merge_all([parsed_doc2])
        print(len(merged_files.sections)) # Output: 3 (sections from parsed_doc_with_sections and parsed_doc2 combined)
        print(merged_files.name)       # Output: document.pdf (name from parsed_doc_with_sections)
        ```

    *   `from_sections(name: str, sections: Sequence[SectionContent]) -> ParsedFile`:
        Class method to create a `ParsedFile` object directly from a sequence of `SectionContent` objects and a file name.

        **Parameters:**
        *   `name` (str): The name of the file.
        *   `sections` (Sequence[SectionContent]): A sequence of `SectionContent` objects.

        **Returns:**
        *   `ParsedFile`: A new `ParsedFile` object.

        **Example:**
        ```python
        new_parsed_file = ParsedFile.from_sections("report.docx", [section1, section2])
        print(new_parsed_file.name)       # Output: report.docx
        print(len(new_parsed_file.sections)) # Output: 2
        ```

    *   `from_parsed_files(files: Sequence[ParsedFile]) -> ParsedFile`:
        Class method to create a merged `ParsedFile` from a sequence of existing `ParsedFile` objects.
        This is similar to `merge_all` but is a class method and creates a new `ParsedFile` named "MergedFile".

        **Parameters:**
        *   `files` (Sequence[ParsedFile]): A sequence of `ParsedFile` objects to merge.

        **Returns:**
        *   `ParsedFile`: A new merged `ParsedFile` object named "MergedFile".

        **Example:**
        ```python
        merged_from_class = ParsedFile.from_parsed_files([parsed_doc_with_sections, parsed_doc2])
        print(merged_from_class.name)       # Output: MergedFile
        print(len(merged_from_class.sections)) # Output: 3
        ```

    *   `get_schema(synapse: Synapse) -> Schema`:
        Synchronously retrieves a `Schema` object for the parsed file using a given `Synapse`.
        This method internally calls `get_schema_async` and runs it synchronously.

        **Parameters:**
        *   `synapse` (Synapse): A `Synapse` object used for communication with a language model or other backend service to infer the schema.

        **Returns:**
        *   `Schema`: A `Schema` object representing the extracted schema from the parsed file.

    *   `get_schema_async(synapse: Synapse, trace_params: Optional[TraceParams] = None) -> Schema`:
        Asynchronously retrieves a `Schema` object for the parsed file using a given `Synapse`.
        It utilizes the text content of the parsed file sections to prompt a language model (via the `Synapse`)
        to extract and define a schema of entities and relations relevant to the document.

        **Parameters:**
        *   `synapse` (Synapse): A `Synapse` object used for communication with a language model or other backend service to infer the schema.
        *   `trace_params` (Optional[TraceParams]): Optional tracing parameters for the operation.

        **Returns:**
        *   `Schema`: A `Schema` object representing the extracted schema from the parsed file.

        **Example (Illustrative - Requires a Synapse setup):**
        ```python
        # Assume you have a Synapse instance 'my_synapse'
        # schema_obj = parsed_doc_with_sections.get_schema(my_synapse)
        # print(schema_obj) # Output: Schema object based on document content
        ```
    """

    name: Annotated[
        str,
        msgspec.Meta(
            title="Name",
            description="Name of the file",
        ),
    ]

    sections: Annotated[
        Sequence[SectionContent],
        msgspec.Meta(
            title="Pages",
            description="Pages of the document",
        ),
    ]

    @property
    def llm_described_text(self) -> str:
        sections = " ".join(
            [
                f"<section_{num}> {section.md} </section_{num}>"
                for num, section in enumerate(self.sections)
            ]
        )
        return f"<file>\n\n**name:** {self.name} \n**sections:** {sections}\n\n</file>"

    def merge_all(self, others: Sequence[ParsedFile]) -> ParsedFile:
        from itertools import chain

        return ParsedFile(
            name=self.name,
            sections=list(chain(self.sections, *[other.sections for other in others])),
        )

    @classmethod
    def from_sections(cls, name: str, sections: Sequence[SectionContent]) -> ParsedFile:
        return cls(name=name, sections=sections)

    @classmethod
    def from_parsed_files(cls, files: Sequence[ParsedFile]) -> ParsedFile:
        from itertools import chain

        return ParsedFile(
            name="MergedFile",
            sections=list(chain(*[file.sections for file in files])),
        )

    @property
    def md(self) -> str:
        return "\n".join([sec.md or "" for sec in self.sections])

    def get_schema(self, synapse: Synapse) -> Schema:
        return run_sync(self.get_schema_async, synapse)

    async def get_schema_async(
        self,
        synapse: Synapse,
        *,
        trace_params: Optional[TraceParams] = None,
        temperature: int = 1,
    ) -> Schema:
        _trace_params = {
            "name": "NLP: Internal Entity Extraction",
            "user_id": "file_parser",
        }
        _trace_params.update(cast(dict[str, Any], trace_params) or {})

        output = await synapse.complete_async(
            prompt=f"<document> {[sec.text for sec in self.sections]} </document>",
            system_prompt="You are an AI assistant who is an expert in natural"
            "language processing and especially named entity recognition.",
            response_model=ChainOfThought[Schema],
            temperature=temperature,
            trace_params=cast(TraceParams, _trace_params),
        )

        return output.parsed.final_answer
