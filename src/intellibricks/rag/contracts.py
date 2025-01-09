from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from intellibricks.files.parsed_document import ParsedDocument
from intellibricks.rag.schema import (
    Context,
    IngestionInfo,
    Query,
)

from .transformations import DocumentTransformer


@runtime_checkable
class SupportsAsyncIngestion(Protocol):
    async def ingest_async(
        self,
        document: ParsedDocument,
        transformations: Optional[list[DocumentTransformer]] = None,
    ) -> IngestionInfo: ...


@runtime_checkable
class SupportsAsyncDeletion(Protocol):
    async def delete_async(self, document_id: str) -> None: ...

    async def delete_all_async(self, document_ids: list[str]) -> None:
        for document_id in document_ids:
            await self.delete_async(document_id)


@runtime_checkable
class SupportsContextRetrieval(Protocol):
    async def retrieve_context_async(self, query: Query) -> Context: ...
