import abc
import typing

from langchain_community.vectorstores import Clickhouse, ClickhouseSettings
from langchain_core.documents import Document as LangchainDocument
from langchain_core.documents.transformers import BaseDocumentTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker

from intellibricks.files import DocumentArtifact

from .contracts import RAGQueriable
from .results import QueryResult


class RAGDocumentRepository(abc.ABC):
    embeddings: Embeddings
    collection_name: str

    def __init__(
        self, embeddings: Embeddings, collection_name: typing.Optional[str] = None
    ) -> None:
        self.embeddings = embeddings
        self.collection_name = collection_name or "default"

    async def ingest_async(
        self,
        document: DocumentArtifact,
        transformations: typing.Optional[list[BaseDocumentTransformer]] = None,
    ) -> list[str]:
        """Stores the document in the database and returns the document ids."""
        vector_store: VectorStore = self._get_vector_store()

        documents: list[LangchainDocument] = document.as_langchain_documents(
            transformations=transformations
            or [SemanticChunker(embeddings=self.embeddings)]
        )

        ingested_documents_ids: list[str] = await vector_store.aadd_documents(
            documents=documents, ids=[document.id for document in documents]
        )
        return ingested_documents_ids

    async def similarity_search_async(
        self, query: str, k: int = 4
    ) -> list[LangchainDocument]:
        # TODO: make more robust implementation
        vector_store: VectorStore = self._get_vector_store()
        return await vector_store.asimilarity_search(query, k=k)

    @abc.abstractmethod
    def _get_vector_store(
        self,
    ) -> VectorStore: ...


class ClickHouseDataStore(RAGDocumentRepository, RAGQueriable):
    def query(self, query: str) -> QueryResult:
        return QueryResult()

    def _get_vector_store(
        self, collection_name: typing.Optional[str] = None
    ) -> VectorStore:
        return Clickhouse(embedding=self.embeddings, config=ClickhouseSettings())
