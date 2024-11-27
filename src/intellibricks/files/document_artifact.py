"""Schema objects used to file extraction"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, Optional, Sequence

import msgspec
from architecture import Meta, field
from architecture.utils.structs import dictify
from langchain_core.documents import Document as LangchainDocument
from langchain_core.documents.transformers import BaseDocumentTransformer
from llama_index.core.schema import Document as LlamaIndexDocument

from intellibricks.llms import AIModel, CompletionEngineProtocol


class Image(msgspec.Struct, frozen=True):
    name: Annotated[
        Optional[str],
        Meta(
            title="Name",
            description="The name of the image file present in the original document.",
        ),
    ] = field(default=None)

    height: Annotated[
        float,
        Meta(
            title="Height",
            description="Height of the image in pixels.",
        ),
    ] = 0

    width: Annotated[
        float,
        Meta(
            title="Width",
            description="Width of the image in pixels.",
        ),
    ] = 0


class PageItem(msgspec.Struct, frozen=True):
    type: Annotated[
        Literal["text", "heading", "table"],
        Meta(
            title="Type",
            description="Type of the item",
        ),
    ] = field(default_factory=lambda: "text")

    rows: Annotated[
        Optional[Sequence[Sequence[str]]],
        Meta(
            title="Rows",
            description="Rows of the table, if the item is a table.",
        ),
    ] = field(default_factory=list)

    is_perfect_table: Annotated[
        Optional[bool],
        Meta(
            title="Is Perfect Table",
            description="Whether the table is a perfect table",
        ),
    ] = False

    value: Annotated[
        Optional[str],
        Meta(
            title="Value",
            description="Value of the item",
        ),
    ] = None

    md: Annotated[
        Optional[str],
        Meta(
            title="Markdown Representation",
            description="Markdown representation of the item",
        ),
    ] = None

    lvl: Annotated[
        Optional[int],
        Meta(
            title="Level",
            description="Level of the heading",
        ),
    ] = None

    csv: Annotated[
        Optional[str],
        Meta(
            title="CSV Representation",
            description="CSV representation of the table",
        ),
    ] = None


class PageContent(msgspec.Struct, frozen=True):
    page: Annotated[
        int,
        Meta(
            title="Page",
            description="Page number",
        ),
    ]

    text: Annotated[
        Optional[str],
        Meta(
            title="Text",
            description="Text content's of the page",
        ),
    ] = None

    md: Annotated[
        Optional[str],
        Meta(
            title="Markdown Representation",
            description="Markdown representation of the page.",
        ),
    ] = None

    images: Annotated[
        list[Optional[Image]],
        Meta(
            title="Images",
            description="Images present in the page",
        ),
    ] = field(default_factory=list)

    items: Annotated[
        list[PageItem],
        Meta(
            title="Items",
            description="Items present in the page",
        ),
    ] = field(default_factory=list)

    def get_id(self) -> str:
        return f"page_{self.page}"


class JobMetadata(msgspec.Struct, frozen=True):
    credits_used: Annotated[
        float,
        Meta(
            title="Credits Used",
            description="Credits used for the job",
            ge=0,
        ),
    ] = 0.0

    credits_max: Annotated[
        int,
        Meta(
            title="Credits Max",
            description="Maximum credits allowed for the job",
            ge=0,
        ),
    ] = 0

    job_credits_usage: Annotated[
        int,
        Meta(
            title="Job Credits Usage",
            description="Credits used for the job",
            ge=0,
        ),
    ] = 0

    job_pages: Annotated[
        int,
        Meta(
            title="Job Pages",
            description="Number of pages processed",
            ge=0,
        ),
    ] = 0

    job_is_cache_hit: Annotated[
        bool,
        Meta(
            title="Job Is Cache Hit",
            description="Whether the job is a cache hit",
        ),
    ] = False


class Schema(msgspec.Struct, frozen=True):
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

    entities: Annotated[
        list[str],
        Meta(
            title="Entities",
            description="A list of entity names present in the document.",
            min_length=1,
            examples=[["Person", "Organization", "Location"]],
        ),
    ]

    relations: Annotated[
        list[str],
        Meta(
            title="Relations",
            description="A list of relation names present in the document.",
            min_length=1,
            examples=[["works_at", "located_in", "employs"]],
        ),
    ]

    validation_schema: Annotated[
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


class DocumentArtifact(msgspec.Struct, frozen=True):
    pages: Annotated[
        list[PageContent],
        Meta(
            title="Pages",
            description="Pages of the document",
        ),
    ]

    job_metadata: Annotated[
        Optional[JobMetadata],
        Meta(
            title="Job Metadata",
            description="Metadata of the job",
        ),
    ] = None

    job_id: Annotated[
        str,
        Meta(
            title="Job ID",
            description="ID of the job",
        ),
    ] = field(default_factory=lambda: str(uuid.uuid4()))

    file_path: Annotated[
        Optional[str],
        Meta(
            title="File Path",
            description="Path of the file",
        ),
    ] = field(default=None)

    async def get_schema_async(
        self, completion_engine: CompletionEngineProtocol
    ) -> Schema:
        output = await completion_engine.complete_async(
            system_prompt="You are an AI assistant who is an expert in natural language processing and especially name entity recognition.",
            prompt=f"<document> {[page.text for page in self.pages]} </document>",
            response_model=Schema,
            model=AIModel.VERTEX_GEMINI_1P5_FLASH_002,
            temperature=1,
            trace_params={
                "name": "NLP: Internal Entity Extraction",
                "user_id": "cortex_content_extractor",
            },
        )

        possible_schema = output.get_parsed()

        if possible_schema is None:
            raise ValueError(
                "The entities and relationships could not be extracted from this document."
            )

        return possible_schema

    def as_llamaindex_documents(self) -> list[LlamaIndexDocument]:
        adapted_docs: list[LlamaIndexDocument] = []

        filename: Optional[str] = self.file_path
        for page in self.pages:
            page_number: int = page.page or 0
            images: list[Optional[Image]] = page.images

            metadata = {
                "page_number": page_number,
                "images": [dictify(image) for image in images if image is not None]
                or [],
                "source": filename,
            }

            content: str = page.md or ""
            adapted_docs.append(LlamaIndexDocument(text=content, metadata=metadata))  # type: ignore[call-arg]

        return adapted_docs

    def as_langchain_documents(
        self, transformations: list[BaseDocumentTransformer]
    ) -> list[LangchainDocument]:
        """Converts itself representation to a List of Langchain Document"""
        filename: Optional[str] = self.file_path

        # May contain a whole page as document.page_content.
        # If text splitters are provided, this problem
        # will be gone.
        raw_docs: list[LangchainDocument] = [
            LangchainDocument(
                page_content=page.md or "",
                id=page.id,
                metadata={
                    "filename": filename,
                },
            )
            for page in self.pages
        ]

        transformed_documents: list[LangchainDocument] = []

        # TODO (arthur): Implement transformations

        return transformed_documents or raw_docs
