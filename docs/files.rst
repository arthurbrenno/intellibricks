.. _files_module:

Files Module: Intelligent File Handling
=======================================

The ``intellibricks.files`` module is designed to provide a robust and Pythonic way to handle files within your AI applications. It focuses on representing files as ``RawFile`` objects and provides functionalities for parsing and extracting content from various file types.

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

* **File Parsing Infrastructure:** The module provides a comprehensive system for file parsing, integrating various file parsers to enable extraction of structured information (text, images, tables) from different file types.

* **File Extension Management:** The module helps in managing and determining file extensions, which is crucial for routing files to appropriate parsers.

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

Using File Parsers
--------------------

IntelliBricks provides a set of file parsers within the ``intellibricks.files.parsers`` module to extract structured content from different file formats. Here's how you can use them:

**Available Parsers**

Currently, IntelliBricks offers the following file parsers:

* ``TxtFileParser``: For parsing plain text files (.txt).
* ``PdfFileParser``: For parsing PDF documents (.pdf).

You can import these parsers from ``intellibricks.files.parsers``.

**Basic Usage Example**

Let's demonstrate how to parse a PDF file to extract its content:

.. code-block:: python

   from intellibricks.files import RawFile
   from intellibricks.files.parsers import PdfFileParser, TxtFileParser

   # 1. Load a RawFile (e.g., from a file path)
   raw_pdf_file = RawFile.from_file_path("document.pdf") # Replace with your PDF file
   raw_txt_file = RawFile.from_file_path("document.txt") # Replace with your TXT file

   # 2. Instantiate the appropriate parser
   pdf_parser = PdfFileParser()
   txt_parser = TxtFileParser()

   # 3. Extract content using the parser
   parsed_pdf_document = pdf_parser.parse(raw_pdf_file)
   parsed_txt_document = txt_parser.parse(raw_txt_file)

   # 4. Access parsed content (ParsedFile object)
   print(f"Parsed document name: {parsed_pdf_document.name}")
   for section in parsed_pdf_document.sections:
      print(f"\nSection {section.number}:")
      print(f"  Text (first 100 chars): {section.text[:100]}...")
      # ... access other parsed content like images, items, etc.

   print(f"Parsed document name: {parsed_txt_document.name}")
   for section in parsed_txt_document.sections:
      print(f"\nSection {section.number}:")
      print(f"  Text (first 100 chars): {section.text[:100]}...")


**Handling Different File Types**

To parse different file types, you would:

1. Create a ``RawFile`` object for your file.
2. Instantiate the appropriate parser class based on the file type (e.g., ``PdfFileParser`` for PDFs, ``DocxFileParser`` for DOCX files - when available).
3. Call the ``parse()`` method of the parser, passing the ``RawFile`` object as input.
4. Work with the returned ``ParsedFile`` object to access the extracted structured content.

**Parsed File Structure**

The output of file parsing is a ``ParsedFile`` object, which contains:

* **Sections**: The document is divided into ``SectionContent`` objects, representing pages or logical sections. Each ``SectionContent`` contains:
    * **Text**: The extracted text content of the section.
    * **Markdown**: A Markdown representation of the section content, including headings, lists, and basic formatting.
    * **Images**: A list of ``Image`` objects found in the section, including image data and metadata.
    * **Items**: A list of structured ``PageItem`` objects, representing elements like paragraphs, headings, and tables.

**Example of Parsed Content (Illustrative)**

.. code-block:: python

   from intellibricks.files import RawFile, ParsedFile, PdfFileParser

   # Assume you have a RawFile object loaded from a PDF
   raw_pdf_file = RawFile.from_file_path("document.pdf")
   pdf_parser = PdfFileParser()

   parsed_document = pdf_parser.parse(raw_pdf_file)

   print(f"Parsed document: {parsed_document.name}")
   for section in parsed_document.sections:
      print(f"\nSection {section.number}:")
      print(f"  Text (first 100 chars): {section.text[:100]}...")
      if section.images:
         print(f"  Images found: {len(section.images)}")
      if section.items:
         print(f"  Items found: {len(section.items)}")
         # Iterate through items (TextPageItem, HeadingPageItem, TablePageItem, etc.)


API Reference
-------------

.. automodule:: intellibricks.files
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: intellibricks.files.parsed_files
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: intellibricks.files.parsers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: intellibricks.files.types
   :members:
   :undoc-members:
   :show-inheritance: