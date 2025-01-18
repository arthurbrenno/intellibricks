from __future__ import annotations

# TODO(arthur): implement
import abc
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from intellibricks.rag.contracts import (
    SupportsAsyncIngestion,
    SupportsContextRetrieval,
)


class LangchainVectorDatabaseContextSource(
    abc.ABC, SupportsAsyncIngestion, SupportsContextRetrieval
):
    embeddings: Embeddings

    @abc.abstractmethod
    def _get_vector_store(  # Hook method
        self,
    ) -> VectorStore: ...


@dataclass
class LangchainMilvusVectorDatabaseContextSource(LangchainVectorDatabaseContextSource):
    uri: str
    collection: str


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
