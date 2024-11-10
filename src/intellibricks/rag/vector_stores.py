import abc
from typing import Optional

from langchain_community.vectorstores import Clickhouse, ClickhouseSettings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TextSplitter

from intellibricks.files import DocumentArtifact

from .contracts import RAGQueriable


class RAGDocumentRepository(abc.ABC, RAGQueriable):
    embeddings: Embeddings
    collection_name: str

    def __init__(
        self, embeddings: Embeddings, collection_name: Optional[str] = None
    ) -> None:
        self.embeddings = embeddings
        self.collection_name = collection_name or "default"

    async def ingest_async(
        self,
        document: DocumentArtifact,
        text_splitters: Optional[list[TextSplitter]] = None,
    ) -> list[str]:
        """Stores the document in the database and returns the document ids."""
        vector_store: VectorStore = self._get_vector_store()

        documents: list[Document] = document.as_langchain_documents(
            text_splitters=text_splitters
            or [SemanticChunker(embeddings=self.embeddings)]
        )

        ingested_documents_ids: list[str] = await vector_store.aadd_documents(
            documents=documents, ids=[document.id for document in documents]
        )
        return ingested_documents_ids

    async def similarity_search_async(self, query: str) -> None:
        vector_store: VectorStore = self._get_vector_store()

    @abc.abstractmethod
    def _get_vector_store(
        self,
    ) -> VectorStore: ...


class ClickHouseDataStore(RAGDocumentRepository):
    def _get_vector_store(self, collection_name: Optional[str] = None) -> VectorStore:
        return Clickhouse(embedding=self.embeddings, config=ClickhouseSettings())
