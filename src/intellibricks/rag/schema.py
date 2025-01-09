from __future__ import annotations

from typing import Annotated, Sequence

import msgspec


class IngestionInfo(msgspec.Struct, frozen=True):
    document_ids: list[str]


class Query(msgspec.Struct, frozen=True):
    text: str

    @classmethod
    def from_text(cls, text: str) -> Query:
        return cls(text=text)


class Source(msgspec.Struct, frozen=True):
    name: str


class ContextPart(msgspec.Struct, frozen=True):
    raw_text: Annotated[
        str,
        msgspec.Meta(
            title="Text",
            description="Text of the part",
        ),
    ]

    score: Annotated[
        float,
        msgspec.Meta(
            title="Score",
            description="Relevance score of the part",
            ge=0.0,
            le=1.0,
        ),
    ]

    source: Annotated[
        Source, msgspec.Meta(title="Source", description="Source of the part")
    ]


class Context(msgspec.Struct, frozen=True):
    parts: Sequence[ContextPart]

    @property
    def raw_text(self) -> str:
        return " ".join([part.raw_text for part in self.parts])


class ContextSourceSequence(msgspec.Struct, frozen=True):
    contexts: Sequence[Context]

    @property
    def full_text(self) -> str:
        if len(self.contexts) == 0:
            return ""

        return (
            "<|context|>"
            + " ".join([context.raw_text for context in self.contexts])
            + "<|end_context|>"
        )
