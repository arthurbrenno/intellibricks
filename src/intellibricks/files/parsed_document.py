"""Schema objects used to file extraction"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Optional, Sequence

import msgspec
from architecture.utils import run_sync
from architecture.utils.decorators import ensure_module_installed
from architecture.utils.structs import dictify

from intellibricks.llms.synapses import Synapse

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangchainDocument
    from llama_index.core.schema import Document as LlamaIndexDocument
    from intellibricks.rag.transformations import DocumentTransformer


class Image(msgspec.Struct, frozen=True):
    contents: Annotated[
        bytes,
        msgspec.Meta(
            title="Contents",
            description="Contents of the image file.",
        ),
    ]

    height: Annotated[
        float,
        msgspec.Meta(
            title="Height",
            description="Height of the image in pixels.",
        ),
    ]

    width: Annotated[
        float,
        msgspec.Meta(
            title="Width",
            description="Width of the image in pixels.",
        ),
    ]

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


@dataclass(frozen=True)
class PageItem(ABC):
    md: Annotated[
        str,
        msgspec.Meta(
            title="Markdown Representation",
            description="Markdown representation of the item",
        ),
    ]


@dataclass(frozen=True)
class TextPageItem(PageItem):
    value: Annotated[
        str,
        msgspec.Meta(
            title="Value",
            description="Value of the text item",
        ),
    ]


@dataclass(frozen=True)
class HeadingPageItem(PageItem):
    value: Annotated[
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


@dataclass(frozen=True)
class TablePageItem(PageItem):
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


class PageContent(msgspec.Struct, frozen=True):
    page: Annotated[
        int,
        msgspec.Meta(
            title="Page",
            description="Page number",
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
            description="Markdown representation of the page.",
        ),
    ] = None

    images: Annotated[
        list[Image],
        msgspec.Meta(
            title="Images",
            description="Images present in the page",
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
        return f"page_{self.page}"


class JobMetadata(msgspec.Struct, frozen=True):
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
        msgspec.Meta(
            title="Entities",
            description="A list of entity names present in the document.",
            min_length=1,
            examples=[["Person", "Organization", "Location"]],
        ),
    ]

    relations: Annotated[
        list[str],
        msgspec.Meta(
            title="Relations",
            description="A list of relation names present in the document.",
            min_length=1,
            examples=[["works_at", "located_in", "employs"]],
        ),
    ]

    validation_schema: Annotated[
        dict[str, list[str]],
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


class ParsedDocument(msgspec.Struct, frozen=True):
    name: Annotated[
        str,
        msgspec.Meta(
            title="Name",
            description="Name of the file",
        ),
    ]

    pages: Annotated[
        list[PageContent],
        msgspec.Meta(
            title="Pages",
            description="Pages of the document",
        ),
    ]

    @property
    def md(self) -> str:
        return "\n".join([page.md or "" for page in self.pages])

    def get_schema(self, synapse: Synapse) -> Schema:
        return run_sync(self.get_schema_async, synapse)

    async def get_schema_async(self, synapse: Synapse) -> Schema:
        output = await synapse.complete_async(
            prompt=f"<document> {[page.text for page in self.pages]} </document>",
            system_prompt="You are an AI assistant who is an expert in natural language processing and especially name entity recognition.",
            response_model=Schema,
            temperature=1,
            trace_params={
                "name": "NLP: Internal Entity Extraction",
                "user_id": "cortex_content_extractor",
            },
        )

        return output.parsed

    @ensure_module_installed("llama_index.core.schema", "llama-index")
    def as_llamaindex_documents(self) -> list[LlamaIndexDocument]:
        from llama_index.core.schema import Document as LlamaIndexDocument

        adapted_docs: list[LlamaIndexDocument] = []

        filename: str = self.name
        for page in self.pages:
            page_number: int = page.page or 0
            images: list[Image] = page.images

            metadata = {
                "page_number": page_number,
                "images": [dictify(image) for image in images if image is not None]
                or [],
                "source": filename,
            }

            content: str = page.md or ""
            adapted_docs.append(LlamaIndexDocument(text=content, metadata=metadata))  # type: ignore[call-arg]

        return adapted_docs

    @ensure_module_installed("langchain_core", "langchain")
    def as_langchain_documents(
        self, transformations: Optional[list[DocumentTransformer]] = None
    ) -> list[LangchainDocument]:
        """Converts itself representation to a List of Langchain Document"""
        from langchain_core.documents import Document as LangchainDocument

        # Each page, initially, will be a document.
        documents: list[LangchainDocument] = [
            LangchainDocument(
                page_content=page.md or "",
                metadata={
                    "source": self.name,
                },
            )
            for page in self.pages
        ]

        return documents
