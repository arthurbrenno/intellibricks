from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

from intellibricks.rag.types import (
    Context,
    IngestionInfo,
    Query,
)

if TYPE_CHECKING:
    from intellibricks.files.parsed_files import ParsedFile


@runtime_checkable
class SupportsAsyncIngestion(Protocol):
    async def ingest_async(
        self,
        document: ParsedFile,
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
