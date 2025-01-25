Files Module: Intelligent File Handling
=======================================

The ``intellibricks.files`` module is designed to provide a robust and Pythonic way to handle files within your AI applications. It focuses on representing files as ``RawFile`` objects and provides a foundation for parsing and extracting content from various file types.

Core Concepts of the Files Module
---------------------------------

* **RawFile Abstraction:** The central class is ``RawFile``, which represents a file in a structured manner. It encapsulates:
    * ``contents``: The raw byte content of the file.
    * ``name``: The name of the file (without path).
    * ``extension``: The file extension (e.g., "pdf", "docx", "txt").

* **File Loading:** ``RawFile`` provides convenient class methods for loading files from:
    * File paths (``RawFile.from_file_path``)
    * Bytes data (``RawFile.from_bytes``)
    * In-memory file-like objects (``RawFile.from_file_obj``)

* **File Saving:** You can save the contents of a ``RawFile`` to disk using ``RawFile.to_file_path``.

* **File Parsing Infrastructure (WIP):** While parsing functionalities are still under development, the module lays the groundwork for integrating file parsers. The goal is to enable extraction of structured information (text, images, tables) from various file types.

* **File Extension Management:** The module helps in managing and determining file extensions, which is crucial for routing files to appropriate parsers in the future.

Working with RawFile Objects
-----------------------------

Let's explore how to create and use ``RawFile`` objects.

**Creating RawFile from a File Path**

Assume you have a file named ``document.pdf`` in your project directory. You can create a ``RawFile`` object like this:

.. code-block:: python

   from intellibricks.files import RawFile

   file_path = "document.pdf" # Or any path to your file
   raw_file = RawFile.from_file_path(file_path)

   print(f"File Name: {raw_file.name}")
   print(f"File Extension: {raw_file.extension}")
   # raw_file.contents now holds the raw bytes of the PDF file

**Creating RawFile from Bytes Data**

If you have file content in bytes format (e.g., read from a network stream or generated programmatically), you can create a ``RawFile`` using ``from_bytes``:

.. code-block:: python

   file_bytes = b"%PDF-1.5... (PDF file content bytes) ..." # Example PDF bytes
   raw_file_from_bytes = RawFile.from_bytes(file_bytes, "report.pdf")

   print(f"File Name: {raw_file_from_bytes.name}") # Output: report.pdf
   print(f"File Extension: {raw_file_from_bytes.extension}") # Output: pdf
   # raw_file_from_bytes.contents holds the provided bytes

**Creating RawFile from a File-Like Object**

You can also create a ``RawFile`` from an in-memory file-like object (e.g., from ``io.BytesIO`` or when you receive a file object from a web request):

.. code-block:: python

   import io

   # Simulate an in-memory file object
   file_content_str = "This is the content of my text file."
   file_obj = io.BytesIO(file_content_str.encode('utf-8'))

   raw_file_from_obj = RawFile.from_file_obj(file_obj, "sample.txt")

   print(f"File Name: {raw_file_from_obj.name}") # Output: sample.txt
   print(f"File Extension: {raw_file_from_obj.extension}") # Output: txt
   # raw_file_from_obj.contents holds the bytes read from file_obj

**Saving RawFile Contents to Disk**

To save the content of a ``RawFile`` to a new file path:

.. code-block:: python

   output_path = "output_documents/saved_document.pdf" # Define where to save
   raw_file.to_file_path(output_path)

   print(f"File saved to: {output_path}")

File Parsing (Work in Progress)
-------------------------------

IntelliBricks is actively developing file parsing capabilities within the ``files`` module and integrating it with the ``intellibricks.parsers`` module. The goal is to provide a flexible and extensible system for:

* **Text Extraction:** Extracting plain text content from various document formats (PDF, DOCX, TXT, etc.).
* **Image Extraction:** Identifying and extracting images embedded in files.
* **Table Data Extraction:** Recognizing and extracting tabular data from documents.
* **Structured Content Representation:** Representing parsed content in structured schema objects (e.g., headings, paragraphs, tables, images with metadata).

While full parsing functionalities are still evolving, the ``files`` module already provides the foundational ``RawFile`` abstraction and is being extended to handle parsed content.

Future Directions for File Parsing

The roadmap for file parsing in IntelliBricks includes:

* **Parser Implementations:** Developing parsers for common file formats (PDF, DOCX, PPTX, Markdown, HTML, images, audio, video, archives, etc.).
* **Parsing Strategies:** Implementing different parsing strategies (e.g., fast, medium, high accuracy) to balance speed and quality of content extraction.
* **Integration with Agents:** Seamlessly integrating parsed file content into Agents and LLM workflows, allowing Agents to process and understand file data intelligently.
* **Extensibility:** Designing the parsing system to be easily extensible, allowing developers to add custom parsers for specialized file types.

Summary

The ``intellibricks.files`` module provides the essential tools for representing and handling files in AI applications. The ``RawFile`` abstraction simplifies file loading, saving, and management. As file parsing capabilities are further developed, this module will become a cornerstone for building intelligent applications that can understand and process information from a wide range of file formats.

Stay tuned for updates and enhancements to file parsing in IntelliBricks!
