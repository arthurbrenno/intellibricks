from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

import msgspec

from intellibricks.files.parsed_document import ParsedDocument

T = TypeVar("T")
D = TypeVar("D", bound="DocumentTransformer")


class DocumentTransformer[T](msgspec.Struct, frozen=True):
    document: ParsedDocument

    def __post_init__(self) -> None:
        if self.__class__ is DocumentTransformer:
            raise TypeError(
                "DocumentTransformer is an abstract class and cannot be instantiated."
            )

    @classmethod
    def of(cls: type[D], document: ParsedDocument) -> D:
        return cls(document)

    @abstractmethod
    def transform(self) -> T: ...
