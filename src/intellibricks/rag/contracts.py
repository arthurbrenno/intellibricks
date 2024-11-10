from typing import Protocol, runtime_checkable
import abc
from .results import QueryResult


@runtime_checkable
class RAGQueriable(Protocol):
    @abc.abstractmethod
    async def query_async(self, query: str) -> QueryResult: ...

    @abc.abstractmethod
    def query(self, query: str) -> QueryResult: ...
