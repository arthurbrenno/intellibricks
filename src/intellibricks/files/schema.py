"""Schema objects used to file extraction"""

from __future__ import annotations

import hashlib
import typing
import uuid

import msgspec
from langchain_core.documents import Document as LangchainDocument
from langchain_core.documents.transformers import BaseDocumentTransformer
from llama_index.core.schema import Document as LlamaIndexDocument
from architecture import BaseModel, Meta, field

from intellibricks.llms import AIModel, CompletionEngineProtocol, CompletionOutput


class Image(BaseModel):
    name: typing.Annotated[
        typing.Optional[str],
        Meta(
            title="Name",
            description="The name of the image file present in the pdf.",
        ),
    ] = ""

    height: typing.Annotated[
        float,
        Meta(
            title="Height",
            description="Height of the image in pixels.",
        ),
    ] = 0

    width: typing.Annotated[
        float,
        Meta(
            title="Width",
            description="Width of the image in pixels.",
        ),
    ] = 0


class PageItem(BaseModel):
    type: typing.Annotated[
        typing.Literal["text", "heading", "table"],
        Meta(
            title="Type",
            description="Type of the item",
        ),
    ] = "text"

    rows: typing.Annotated[
        typing.Optional[typing.Sequence[typing.Sequence[str]]],
        Meta(
            title="Rows",
            description="Rows of the table, if the item is a table.",
        ),
    ] = []

    is_perfect_table: typing.Annotated[
        typing.Optional[bool],
        Meta(
            title="Is Perfect Table",
            description="Whether the table is a perfect table",
        ),
    ] = False

    value: typing.Annotated[
        typing.Optional[str],
        Meta(
            title="Value",
            description="Value of the item",
        ),
    ] = ""

    md: typing.Annotated[
        typing.Optional[str],
        Meta(
            title="Markdown Representation",
            description="Markdown representation of the item",
        ),
    ] = ""

    lvl: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Level",
            description="Level of the heading",
        ),
    ] = None

    csv: typing.Annotated[
        typing.Optional[str],
        Meta(
            title="CSV Representation",
            description="CSV representation of the table",
        ),
    ] = ""


class PageContent(BaseModel):
    id: typing.Annotated[
        str,
        Meta(
            title="ID",
            description="Unique identifier for this specific page."
            "Pages may be converted to Langchain and LlamaIndex"
            "Documents later.",
        ),
    ] = field(default_factory=lambda: uuid.uuid4().__str__())

    page: typing.Annotated[
        typing.Optional[int],
        Meta(
            title="Page",
            description="Page number",
        ),
    ] = 0

    text: typing.Annotated[
        str,
        Meta(
            title="Text",
            description="Text content's of the page",
        ),
    ] = ""

    md: typing.Annotated[
        str,
        Meta(
            title="Markdown Representation",
            description="Markdown representation of the page.",
        ),
    ] = ""

    images: typing.Annotated[
        list[typing.Optional[Image]],
        Meta(
            title="Images",
            description="Images present in the page",
        ),
    ] = []

    items: typing.Annotated[
        list[PageItem],
        Meta(
            title="Items",
            description="Items present in the page",
        ),
    ] = []


class JobMetadata(BaseModel, frozen=True):  # type: ignore[call-arg, misc]
    credits_used: typing.Annotated[
        float,
        Meta(
            title="Credits Used",
            description="Credits used for the job",
            ge=0,
        ),
    ] = 0.0

    credits_max: typing.Annotated[
        int,
        Meta(
            title="Credits Max",
            description="Maximum credits allowed for the job",
            ge=0,
        ),
    ] = 0

    job_credits_usage: typing.Annotated[
        int,
        Meta(
            title="Job Credits Usage",
            description="Credits used for the job",
            ge=0,
        ),
    ] = 0

    job_pages: typing.Annotated[
        int,
        Meta(
            title="Job Pages",
            description="Number of pages processed",
            ge=0,
        ),
    ] = 0

    job_is_cache_hit: typing.Annotated[
        bool,
        Meta(
            title="Job Is Cache Hit",
            description="Whether the job is a cache hit",
        ),
    ] = False


class Schema(BaseModel, frozen=True):  # type: ignore[call-arg, misc]
    """
    A class representing the schema of entities and relations present in a document.

    The `Schema` class encapsulates three primary attributes:
    - `entities`: A list of entity names present in the document.
    - `relations`: A list of relation names that define how entities are connected.
    - `validation_schema`: A dictionary mapping entities to lists of valid relations.

    Each attribute is annotated with metadata that includes title, description, constraints,
    and examples to ensure data integrity and provide clarity.

    Attributes:
        entities (list[str]): A list of entity names.
            - Must contain at least one entity.
            - Each entity name should be a non-empty string.
            - Examples: `['Person', 'Organization', 'Location']`

        relations (list[str]): A list of relation names.
            - Must contain at least one relation.
            - Each relation name should be a non-empty string.
            - Examples: `['works_at', 'located_in', 'employs']`

        validation_schema (dict[str, list[str]]): A dictionary mapping entities to lists of valid relations.
            - Defines which entities can have which relationships.
            - Keys are entity names; values are lists of valid relations.
            - Examples:
                ```python
                {
                    'Person': ['works_at', 'lives_in'],
                    'Organization': ['employs'],
                    'Location': []
                }
                ```

    Examples:
        >>> schema = Schema(
        ...     entities=['Person', 'Organization', 'Location'],
        ...     relations=['works_at', 'located_in', 'employs'],
        ...     validation_schema={
        ...         'Person': ['works_at', 'lives_in'],
        ...         'Organization': ['employs'],
        ...         'Location': []
        ...     }
        ... )
        >>> print(schema.entities)
        ['Person', 'Organization', 'Location']
        >>> print(schema.relations)
        ['works_at', 'located_in', 'employs']
        >>> print(schema.validation_schema)
        {'Person': ['works_at', 'lives_in'], 'Organization': ['employs'], 'Location': []}

        >>> # Accessing valid relations for an entity
        >>> schema.validation_schema['Person']
        ['works_at', 'lives_in']

        >>> # Checking if 'Person' can 'works_at' an 'Organization'
        >>> 'works_at' in schema.validation_schema['Person']
        True

    """

    entities: typing.Annotated[
        list[str],
        Meta(
            title="Entities",
            description="A list of entity names present in the document.",
            min_length=1,
            examples=[["Person", "Organization", "Location"]],
        ),
    ]

    relations: typing.Annotated[
        list[str],
        Meta(
            title="Relations",
            description="A list of relation names present in the document.",
            min_length=1,
            examples=[["works_at", "located_in", "employs"]],
        ),
    ]

    validation_schema: typing.Annotated[
        dict[str, list[str]],
        Meta(
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


class DocumentArtifact(BaseModel):
    pages: typing.Annotated[
        list[PageContent],
        Meta(
            title="Pages",
            description="Pages of the document",
        ),
    ]

    job_metadata: typing.Annotated[
        typing.Optional[JobMetadata],
        Meta(
            title="Job Metadata",
            description="Metadata of the job",
        ),
    ] = None

    job_id: typing.Annotated[
        str,
        Meta(
            title="Job ID",
            description="ID of the job",
        ),
    ] = field(default_factory=lambda: str(uuid.uuid4()))

    file_path: typing.Annotated[
        typing.Optional[str],
        Meta(
            title="File Path",
            description="Path of the file",
        ),
    ] = field(default=None)

    uid: typing.Annotated[
        str,
        Meta(
            title="UID",
            description="An unique identifier for the DocumentArtifact. The same document"
            "would return the same uid so we can keep track of",
        ),
    ] = field(default_factory=str)

    def __post_init__(self) -> None:
        # Serialize relevant parts of the document to JSON
        content_dict = {
            "pages": [page.as_dict() for page in self.pages],
            "job_metadata": self.job_metadata.as_dict() if self.job_metadata else {},
            "file_path": self.file_path,
        }
        # Convert the dictionary to a JSON string with sorted keys to ensure consistency
        content_json = msgspec.json.encode(content_dict, order="sorted")
        # Compute the SHA-256 hash of the JSON string
        self.uid = hashlib.sha256(content_json).hexdigest()

    async def get_schema_async(
        self, completion_engine: CompletionEngineProtocol
    ) -> Schema:
        output: CompletionOutput[Schema] = await completion_engine.complete_async(
            system_prompt="TODO",
            prompt=f"<documento> {[page.text for page in self.pages]} </documento>",
            response_format=Schema,
            model=AIModel.VERTEX_GEMINI_1P5_FLASH_002,
            temperature=1,
            trace_params={
                "name": "NLP: Internal Entity Extraction",
                "user_id": "cortex_content_extractor",
            },
        )
        possible_schema: typing.Optional[Schema] = output.get_parsed()
        if possible_schema is None:
            raise ValueError(
                "The entities and relationships could not be extracted from this document."
            )

        return possible_schema

    def as_llamaindex_documents(self) -> list[LlamaIndexDocument]:
        adapted_docs: list[LlamaIndexDocument] = []

        filename: typing.Optional[str] = self.file_path
        for page in self.pages:
            page_number: int = page.page or 0
            images: list[typing.Optional[Image]] = page.images

            metadata = {
                "page_number": page_number,
                "images": [image.as_dict() for image in images if image is not None]
                or [],
                "source": filename,
            }

            content: str = page.md
            adapted_docs.append(LlamaIndexDocument(text=content, metadata=metadata))  # type: ignore[call-arg]

        return adapted_docs

    def as_langchain_documents(
        self, transformations: list[BaseDocumentTransformer]
    ) -> list[LangchainDocument]:
        """Converts itself representation to a List of Langchain Document"""
        filename: typing.Optional[str] = self.file_path

        # May contain a whole page as document.page_content.
        # If text splitters are provided, this problem
        # will be gone.
        raw_docs: list[LangchainDocument] = [
            LangchainDocument(
                page_content=page.md,
                id=page.id,
                metadata={
                    "filename": filename,
                },
            )
            for page in self.pages
        ]

        transformed_documents: list[LangchainDocument] = []

        # TODO

        return transformed_documents or raw_docs
