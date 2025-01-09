from __future__ import annotations

import abc
from io import BytesIO
from typing import TYPE_CHECKING, Protocol

from architecture.data.files import RawFile
from architecture.utils.decorators import ensure_module_installed
from typing_extensions import override

from .constants import ParsingMethod
from .parsed_document import (
    HeadingPageItem,
    Image,
    PageContent,
    ParsedDocument,
    TablePageItem,
    TextPageItem,
)
from .parsed_document import (
    PageItem as ArtifactPageItem,
)

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class AsyncFileExtractor(Protocol):
    """
    Abstract class for extracting content from files.
    This should be used as a base class for specific file extractors.
    """

    @abc.abstractmethod
    async def extract_contents_async(
        self,
        file: RawFile,
        parsing_method: ParsingMethod = ParsingMethod.DEFAULT,
        use_gpu: bool = False,
    ) -> ParsedDocument:
        """Extracts content from the file."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class DoclingFileExtractor(AsyncFileExtractor):
    @ensure_module_installed("docling", "intellibricks[files]")
    @override
    async def extract_contents_async(
        self,
        file: RawFile,
        parsing_method: ParsingMethod = ParsingMethod.DEFAULT,
        use_gpu: bool = False,
    ) -> ParsedDocument:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import ConversionResult
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.document_converter import (
            DocumentConverter,
            FormatOption,
            PdfFormatOption,
        )
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.io import DocumentStream

        match parsing_method:
            case ParsingMethod.DEFAULT:
                pdf_pipeline_options = PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                    ocr_options=EasyOcrOptions(
                        lang=["fr", "de", "es", "en", "pt"], use_gpu=use_gpu
                    ),
                )
                pdf_pipeline_options.table_structure_options.mode = TableFormerMode.FAST

            case ParsingMethod.FAST:
                pdf_pipeline_options = PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                    ocr_options=EasyOcrOptions(
                        lang=["fr", "de", "es", "en", "pt"], use_gpu=use_gpu
                    ),
                )
                pdf_pipeline_options.table_structure_options.mode = TableFormerMode.FAST

            case ParsingMethod.MEDIUM:
                pdf_pipeline_options = PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=True,
                    ocr_options=EasyOcrOptions(
                        lang=["fr", "de", "es", "en", "pt"], use_gpu=use_gpu
                    ),
                )
                pdf_pipeline_options.table_structure_options.mode = (
                    TableFormerMode.ACCURATE
                )

            case ParsingMethod.HIGH:
                pdf_pipeline_options = PdfPipelineOptions(
                    do_ocr=True,
                    do_table_structure=True,
                    ocr_options=EasyOcrOptions(
                        lang=["fr", "de", "es", "en", "pt"], use_gpu=use_gpu
                    ),
                )
                pdf_pipeline_options.table_structure_options.mode = (
                    TableFormerMode.ACCURATE
                )

        format_options: dict[InputFormat, FormatOption] = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }

        converter = DocumentConverter(format_options=format_options)
        conv: ConversionResult = converter.convert(
            DocumentStream(name=file.name, stream=BytesIO(file.contents))
        )
        document: DoclingDocument = conv.document

        return self.__docling_to_parsed_document(document)

    def __docling_to_parsed_document(self, document: DoclingDocument) -> ParsedDocument:
        """
        Private method to convert a DoclingDocument into a ParsedDocument.
        """
        import io
        from typing import List, Optional

        from docling_core.types.doc.document import (
            DocItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
        )
        from docling_core.types.doc.labels import DocItemLabel

        file_path = (
            document.origin.filename
            if document.origin and document.origin.filename
            else None
        )
        pages_list: List[PageContent] = []

        # Get sorted page numbers
        page_nos = sorted(document.pages.keys())

        for page_no in page_nos:
            # Extract items for this page
            items_for_page = list(
                document.iterate_items(
                    root=document.body, page_no=page_no, with_groups=True
                )
            )

            # page_items can now contain different PageItem subclasses
            page_items: List[ArtifactPageItem] = []
            page_images: List[Image] = []

            # Process items on this page
            for item, level in items_for_page:
                if not isinstance(item, DocItem):
                    # Skip groups
                    continue

                label = item.label

                # Handle images (PictureItem)
                if label == DocItemLabel.PICTURE and isinstance(item, PictureItem):
                    pil_img = item.get_image(document)
                    if pil_img is not None:
                        with io.BytesIO() as buf:
                            pil_img.save(buf, format="PNG")
                            img_bytes = buf.getvalue()

                        alt_text = item.caption_text(document) or None
                        page_images.append(
                            Image(
                                contents=img_bytes,
                                width=pil_img.width,
                                height=pil_img.height,
                                name=None,
                                alt=alt_text,
                            )
                        )
                    continue

                # Handle tables (TableItem)
                if (
                    label == DocItemLabel.TABLE or label == DocItemLabel.DOCUMENT_INDEX
                ) and isinstance(item, TableItem):
                    df = item.export_to_dataframe()
                    rows = df.values.tolist() if not df.empty else []
                    csv_str = df.to_csv(index=False) if not df.empty else ""
                    md_str = item.export_to_markdown()
                    page_items.append(
                        TablePageItem(
                            md=md_str, rows=rows, csv=csv_str, is_perfect_table=False
                        )
                    )
                    continue

                # Handle headings (SectionHeaderItem)
                if label == DocItemLabel.SECTION_HEADER and isinstance(
                    item, SectionHeaderItem
                ):
                    heading_md = f"{'#' * item.level} {item.text}"
                    page_items.append(
                        HeadingPageItem(
                            md=heading_md,
                            value=item.text,
                            lvl=item.level,
                        )
                    )
                    continue

                # Handle text items (TextItem)
                if isinstance(item, TextItem):
                    text_value = item.text
                    if label == DocItemLabel.LIST_ITEM:
                        item_md = f"- {text_value}"
                    elif label == DocItemLabel.CODE:
                        item_md = f"```\n{text_value}\n```"
                    else:
                        item_md = text_value

                    page_items.append(
                        TextPageItem(
                            md=item_md,
                            value=text_value,
                        )
                    )

            # Now build page_text and page_md from the items
            # page_text: Just all text from TextItems (including headings, lists)
            page_text = " ".join(
                p.value
                for p in page_items
                if isinstance(p, (TextPageItem, HeadingPageItem))
            ).strip()

            # page_md: Join markdown representations of all items
            page_md_list = [p.md for p in page_items]
            page_md_combined = "\n".join(page_md_list).strip() if page_md_list else ""

            # page_md should be Optional[str], None if empty
            page_md: Optional[str] = page_md_combined if page_md_combined else None

            page_content = PageContent(
                page=page_no,
                text=page_text,
                md=page_md,
                images=page_images,
                items=tuple(page_items),
            )
            pages_list.append(page_content)

        artifact = ParsedDocument(
            pages=pages_list,
            name=file_path or "",
        )

        return artifact


# class MarkitdownFileExtractor(AsyncFileExtractor):
#     @ensure_module_installed("markitdown", "intellibricks[markitdown]")
#     async def extract_contents_async(
#         self,
#         file: RawFile,
#         parsing_method: ParsingMethod = ParsingMethod.DEFAULT,
#         use_gpu: bool = False,
#     ) -> ParsedDocument:
#         from markitdown import MarkItDown
#         from markitdown._markitdown import DocumentConverterResult

#         with tempfile.NamedTemporaryFile(delete=True) as temp_file:
#             temp_file.write(file.contents)
#             temp_file.seek(0)
#             converter = MarkItDown()
#             result: DocumentConverterResult = converter.convert(temp_file.name)
#             markdown: str = result.text_content

#             # return a Document with one page only
#             page_content = PageContent(
#                 page=1,
#                 text=markdown,
#                 md=markdown,
#                 images=[],
#                 items=[],
#             )

#             return ParsedDocument(
#                 name=file.name,
#                 pages=[page_content],
#             )
