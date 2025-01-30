"""
Package: intellibricks.files

This module provides classes and utilities for handling files within the IntelliBricks framework.
It focuses on representing files as `RawFile` objects and parsing various file types to extract structured content.

**Core Functionality:**

*   **RawFile Representation:** The `RawFile` class is used to represent files in a structured manner, encapsulating file content (as bytes), name, and extension. It provides methods for loading files from paths, bytes, and saving to disk.
*   **File Parsing Infrastructure:** Integrates with the `intellibricks.parsers` module to offer a flexible and extensible system for parsing different file formats (e.g., text, PDF, Office documents, images, audio, video, archives).
*   **File Extension Handling:** Provides utilities for determining and managing file extensions, crucial for routing files to the correct parsers.
*   **Content Extraction:**  Facilitates the extraction of structured content from files, making it easier to process and analyze file data within IntelliBricks agents and applications.

**Key Classes Exported:**

*   **`RawFile`**: Represents a raw file with its content, name, and extension.

**Getting Started:**

To work with files in IntelliBricks, you would typically start by creating a `RawFile` object from a file path, bytes, or an existing file-like object. You can then use this `RawFile` with file parsers (from `intellibricks.parsers`) to extract content or process the file data as needed within your agents or applications.

**Example (Creating a RawFile from a file path):**

```python
from intellibricks.files import RawFile

# Assuming you have a file named 'document.txt' in the same directory
file_path = "document.txt"
raw_file = RawFile.from_file_path(file_path)

print(f"File name: {raw_file.name}")
print(f"File extension: {raw_file.extension}")
# raw_file.contents now contains the file content as bytes
```

Explore the `RawFile` class and the `intellibricks.parsers` module to learn more about file handling and parsing capabilities within IntelliBricks.
"""

from .parsed_files import ParsedFile
from .parsers import (
    FileParser,
    IntellibricksFileParser,
    PDFFileParser,
    ExcelFileParser,
    CompressedFileParser,
    XMLFileParser,
    DWGFileParser,
    PKTFileParser,
    DocxFileParser,
    PptxFileParser,
    StaticImageFileParser,
    AnimatedImageFileParser,
    TxtFileParser,
)
from .raw_file import RawFile

__all__: list[str] = [
    "ParsedFile",
    "FileParser",
    "RawFile",
    "IntellibricksFileParser",
    "PDFFileParser",
    "ExcelFileParser",
    "CompressedFileParser",
    "XMLFileParser",
    "DWGFileParser",
    "PKTFileParser",
    "DocxFileParser",
    "PptxFileParser",
    "StaticImageFileParser",
    "AnimatedImageFileParser",
    "TxtFileParser",
]
