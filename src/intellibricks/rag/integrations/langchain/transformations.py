from __future__ import annotations

from typing import TYPE_CHECKING

from intellibricks.rag.transformations import DocumentTransformer

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangchainDocument


class RecursiveCharacterTextSplitterTransformer(
    DocumentTransformer[LangchainDocument], frozen=True
):
    def transform(self) -> LangchainDocument:
        raise NotImplementedError
