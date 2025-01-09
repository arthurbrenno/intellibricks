from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

from architecture.utils.decorators import ensure_module_installed
from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

from intellibricks.files import ParsedDocument
from intellibricks.rag.contracts import (
    SupportsAsyncIngestion,
    SupportsContextRetrieval,
)
from intellibricks.rag.schema import IngestionInfo
from intellibricks.rag.transformations import DocumentTransformer


@dataclass
class LangchainVectorDatabaseContextSource(
    abc.ABC, SupportsAsyncIngestion, SupportsContextRetrieval
):
    embeddings: Embeddings

    @override
    async def ingest_async(
        self,
        document: ParsedDocument,
        transformations: Optional[list[DocumentTransformer[LangchainDocument]]] = None,
    ) -> IngestionInfo:
        """Stores the document in the database and returns the document ids."""
        vector_store: VectorStore = self._get_vector_store()

        documents: list[LangchainDocument] = document.as_langchain_documents(
            transformations=transformations
        )

        ingested_documents_ids: list[str] = await vector_store.aadd_documents(
            documents=documents, ids=[document.id for document in documents]
        )
        return IngestionInfo(document_ids=ingested_documents_ids)

    @abc.abstractmethod
    def _get_vector_store(  # Hook method
        self,
    ) -> VectorStore: ...


@dataclass
class LangchainMilvusVectorDatabaseContextSource(LangchainVectorDatabaseContextSource):
    uri: str
    collection: str

    @ensure_module_installed("langchain_milvus", "langchain_milvus")
    @override
    def _get_vector_store(self) -> VectorStore:
        from langchain_milvus import Milvus

        return Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": self.uri},
            collection_name=self.collection,
        )


@dataclass
class LangchainClickhouseVectorDatabaseContextSource(
    LangchainVectorDatabaseContextSource
):
    table: str
    host: str = field(default_factory=lambda: "localhost")
    port: int = field(default_factory=lambda: 8123)
    username: Optional[str] = None
    password: Optional[str] = None
    secure: bool = False
    database: str = field(default_factory=lambda: "default")
    metric: str = field(default_factory=lambda: "angular")

    @ensure_module_installed("langchain-community", "langchain_community")
    @override
    def _get_vector_store(self) -> VectorStore:
        from langchain_community.vectorstores import Clickhouse, ClickhouseSettings

        return Clickhouse(
            embedding=self.embeddings, config=ClickhouseSettings(table=self.table)
        )
