# WIP
import datetime
from typing import Sequence, Optional, Tuple, Union, Any
from typing_extensions import Annotated
import msgspec


class PDFMetadata(msgspec.Struct, frozen=True):
    """Metadata associated with a PDF document.

    Attributes:
        title: The title of the document
        author: Primary author/creator
        subject: Document subject description
        keywords: Search keywords
        creation_date: Initial creation timestamp
        modification_date: Last modification timestamp
    """

    title: Annotated[
        str,
        msgspec.Meta(
            description="The primary title of the PDF document",
            examples=["Annual Report 2023", "Technical Specifications"],
        ),
    ]
    author: Annotated[
        str,
        msgspec.Meta(
            description="The principal author or creator of the document",
            examples=["John Doe", "Acme Corporation"],
        ),
    ]
    subject: Annotated[
        str,
        msgspec.Meta(
            description="Descriptive subject of the document's content",
            examples=["Financial Report", "Product Manual"],
        ),
    ]

    keywords: Annotated[
        Sequence[str],
        msgspec.Meta(
            description="Array of keywords for search and categorization",
            examples=[["finance", "report"], ["technical", "manual"]],
        ),
    ]

    creation_date: Annotated[
        datetime.datetime,
        msgspec.Meta(
            description="Exact date and time of document creation",
            examples=[datetime.datetime(2023, 1, 1, 9, 0)],
        ),
    ]

    modification_date: Annotated[
        Optional[datetime.datetime],
        msgspec.Meta(
            description="Most recent modification timestamp if available",
            examples=[datetime.datetime(2023, 6, 15, 14, 30)],
        ),
    ] = None


class PDFPageElement(msgspec.Struct, frozen=True):
    """Base class for all PDF page content elements"""

    coordinates: Annotated[
        Tuple[float, float, float, float],
        msgspec.Meta(
            description="Bounding box coordinates (x1, y1, x2, y2) in points",
            examples=[[72.0, 700.0, 500.0, 720.0]],
        ),
    ]


class PDFTextBlock(PDFPageElement, frozen=True):
    """Formatted text content within a PDF

    Attributes:
        text: Raw text content
        font: Font family name
        font_size: Size in points
        color: Hex color code
    """

    text: Annotated[
        str,
        msgspec.Meta(
            description="The actual text content",
            examples=["Lorem ipsum dolor sit amet"],
        ),
    ]
    font: Annotated[
        str,
        msgspec.Meta(
            description="Font family used for rendering",
            examples=["Helvetica", "Times New Roman"],
        ),
    ]
    font_size: Annotated[
        float,
        msgspec.Meta(
            description="Font size in typographic points", examples=[12.0, 10.5]
        ),
    ]
    color: Annotated[
        str,
        msgspec.Meta(
            description="Text color in hex format", examples=["#000000", "#FF0000"]
        ),
    ] = "#000000"


class PDFImage(PDFPageElement, frozen=True):
    """Embedded image within a PDF

    Attributes:
        data: Binary image data
        format: Image format type
        resolution: DPI resolution
    """

    data: Annotated[
        bytes,
        msgspec.Meta(
            description="Raw binary image data",
            examples=[b"iVBORw0KGgoAAAANSUhEUgAA..."],
        ),
    ]
    format: Annotated[
        str,
        msgspec.Meta(
            description="Image format specification", examples=["JPEG", "PNG", "TIFF"]
        ),
    ]
    resolution: Annotated[
        Tuple[int, int],
        msgspec.Meta(
            description="Horizontal and vertical resolution in DPI",
            examples=[[300, 300], [72, 72]],
        ),
    ] = (72, 72)


class PDFPage(msgspec.Struct, frozen=True):
    """Individual page within a PDF document

    Attributes:
        number: Page number (1-based index)
        size: Page dimensions in points
        rotation: Clockwise rotation angle
        content: Ordered page elements
    """

    number: Annotated[
        int,
        msgspec.Meta(description="1-based page number in sequence", examples=[1, 5]),
    ]
    size: Annotated[
        Tuple[float, float],
        msgspec.Meta(
            description="Width and height in points (1/72 inch)",
            examples=[[612.0, 792.0], [595.0, 842.0]],
        ),
    ]

    content: Annotated[
        Sequence[Union[PDFTextBlock, PDFImage]],
        msgspec.Meta(
            description="Ordered list of page content elements",
            examples=[
                [
                    {"text": "Introduction", "font": "Helvetica", "font_size": 12},
                    {"data": b"...", "format": "PNG"},
                ]
            ],
        ),
    ]

    rotation: Annotated[
        int,
        msgspec.Meta(
            description="Clockwise rotation angle in degrees", examples=[0, 90, 180]
        ),
    ] = 0


class PDFDocument(msgspec.Struct, frozen=True):
    """Complete PDF document structure

    Attributes:
        metadata: Document metadata
        pages: Ordered page sequence
        outline: Table of contents hierarchy
        attachments: Embedded files
    """

    metadata: Annotated[
        PDFMetadata, msgspec.Meta(description="Document metadata properties")
    ]
    pages: Annotated[
        Sequence[PDFPage],
        msgspec.Meta(
            description="Ordered collection of document pages",
            examples=[[{"number": 1, "size": (612, 792), "content": []}]],
        ),
    ]
    outline: Annotated[
        Sequence[dict[str, Any]],
        msgspec.Meta(
            description="Hierarchical document outline structure",
            examples=[[{"title": "Chapter 1", "page": 1}]],
        ),
    ] = ()
    attachments: Annotated[
        Sequence[dict[str, bytes]],
        msgspec.Meta(
            description="Embedded files with metadata",
            examples=[[{"name": "data.csv", "content": b"..."}]],
        ),
    ] = ()


class PPTXSlideElement(msgspec.Struct, frozen=True):
    """Base class for all PowerPoint slide elements"""

    position: Annotated[
        Tuple[float, float],
        msgspec.Meta(
            description="X/Y coordinates in slide points", examples=[[100.0, 150.0]]
        ),
    ]
    dimensions: Annotated[
        Tuple[float, float],
        msgspec.Meta(description="Width/height in points", examples=[[300.0, 200.0]]),
    ]


class PPTXTextElement(PPTXSlideElement, frozen=True):
    """Formatted text content in a slide

    Attributes:
        content: Text content with formatting
        style: Named text style
        font: Font properties
    """

    content: Annotated[
        str,
        msgspec.Meta(
            description="Actual text content with formatting",
            examples=["Main Title", "Bullet Point 1"],
        ),
    ]
    style: Annotated[
        str,
        msgspec.Meta(
            description="Named style from template", examples=["Title", "Body"]
        ),
    ]
    font: Annotated[
        dict[str, Any],
        msgspec.Meta(
            description="Detailed font properties",
            examples=[{"size": 44, "color": "#000000", "bold": True}],
        ),
    ] = {}


class PPTXMediaElement(PPTXSlideElement, frozen=True):
    """Embedded media in a slide

    Attributes:
        type: Media type classification
        data: Binary media content
        preview: Thumbnail preview
    """

    type: Annotated[
        str,
        msgspec.Meta(
            description="Media type identifier", examples=["image", "video", "audio"]
        ),
    ]
    data: Annotated[
        bytes,
        msgspec.Meta(
            description="Raw binary media content",
            examples=[b"iVBORw0KGgoAAAANSUhEUgAA..."],
        ),
    ]
    preview: Annotated[
        Optional[bytes],
        msgspec.Meta(
            description="Preview thumbnail image",
            examples=[b"iVBORw0KGgoAAAANSUhEUgAA..."],
        ),
    ] = None


class PPTXSlide(msgspec.Struct, frozen=True):
    """Individual slide in a presentation

    Attributes:
        number: Slide position in deck
        layout: Slide layout template
        background: Background properties
        elements: Slide content elements
    """

    number: Annotated[
        int, msgspec.Meta(description="1-based slide sequence number", examples=[1, 5])
    ]
    layout: Annotated[
        str,
        msgspec.Meta(
            description="Layout template identifier",
            examples=["Title Slide", "Content with Caption"],
        ),
    ]

    elements: Annotated[
        Sequence[Union[PPTXTextElement, PPTXMediaElement]],
        msgspec.Meta(
            description="Ordered collection of slide elements",
            examples=[
                [
                    {"content": "Main Title", "style": "Title"},
                    {"type": "image", "data": b"..."},
                ]
            ],
        ),
    ]

    background: Annotated[
        dict[str, Any],
        msgspec.Meta(
            description="Background styling properties",
            examples=[{"color": "#FFFFFF", "image": None}],
        ),
    ] = msgspec.field(default_factory=dict)


class PPTXDocument(msgspec.Struct, frozen=True):
    """Complete PowerPoint presentation structure

    Attributes:
        metadata: Presentation metadata
        slides: Ordered slide sequence
        template: Base template information
        masters: Slide master layouts
    """

    metadata: Annotated[
        PDFMetadata,  # Reuse PDF metadata structure as it's similar
        msgspec.Meta(description="Document metadata properties"),
    ]
    slides: Annotated[
        Sequence[PPTXSlide],
        msgspec.Meta(
            description="Ordered collection of presentation slides",
            examples=[[{"number": 1, "layout": "Title Slide", "elements": []}]],
        ),
    ]
    template: Annotated[
        dict[str, Any],
        msgspec.Meta(
            description="Template information and styles",
            examples=[{"name": "Corporate", "author": "Design Team"}],
        ),
    ] = {}
    masters: Annotated[
        Sequence[dict[str, Any]],
        msgspec.Meta(
            description="Slide master layouts and themes",
            examples=[[{"name": "Office Theme", "layouts": []}]],
        ),
    ] = ()
