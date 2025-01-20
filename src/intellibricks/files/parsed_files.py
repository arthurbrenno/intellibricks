"""Schema objects used to file extraction"""

from __future__ import annotations
from typing import Annotated, Any, Optional, Sequence, cast

import msgspec
from architecture.utils import run_sync

from intellibricks import ChainOfThought, Synapse, TraceParams


class Image(msgspec.Struct, frozen=True):
    contents: Annotated[
        bytes,
        msgspec.Meta(
            title="Contents",
            description="Contents of the image file.",
        ),
    ]

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
    md: Annotated[
        str,
        msgspec.Meta(
            title="Markdown Representation",
            description="Markdown representation of the item",
        ),
    ]


class TextPageItem(PageItem, tag="text", frozen=True):
    text: Annotated[
        str,
        msgspec.Meta(
            title="Value",
            description="Value of the text item",
        ),
    ]


class HeadingPageItem(PageItem, tag="heading", frozen=True):
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
        sections = ' '.join(
            [
                f"<section_{num}> {section.md} </section_{num}>"
                for num, section in enumerate(self.sections)
            ]
        )
        return (
            f"<file>\n\n"
            f"**name:** {self.name} \n"
            f"**sections:** {sections}\n\n"
            f"</file>"
        )

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
        self, synapse: Synapse, trace_params: Optional[TraceParams] = None
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
            temperature=1,
            trace_params=cast(TraceParams, _trace_params),
        )

        return output.parsed.final_answer
