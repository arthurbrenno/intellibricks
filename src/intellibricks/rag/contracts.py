import abc
import typing

from .results import QueryResult


@typing.runtime_checkable
class RAGQueriable(typing.Protocol):
    @abc.abstractmethod
    async def query_async(self, query: str) -> QueryResult: ...

    @abc.abstractmethod
    def query(self, query: str) -> QueryResult: ...
