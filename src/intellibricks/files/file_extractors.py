from __future__ import annotations

import abc
from typing import Literal, Optional, Protocol, Union, overload
from typing_extensions import Never

from architecture.data.files import RawFile

from .constants import ParsingMethod
from .schema import DocumentArtifact


class FileExtractorBase(Protocol):
    """
    Abstract class for extracting content from files.
    This should be used as a base class for specific file extractors.
    """

    @abc.abstractmethod
    async def extract_contents(
        self, file: RawFile, parsing_method: ParsingMethod, use_gpu: bool = False
    ) -> DocumentArtifact:
        """Extracts content from the file."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class MinerUFileExtractor(FileExtractorBase):
    async def extract_contents(
        self, file: RawFile, parsing_method: ParsingMethod, use_gpu: bool = False
    ) -> DocumentArtifact:
        """Extracts content from the file."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class FileExtractorFactory:
    @overload
    def create_extractor(
        self,
        extension: Literal["pdf", "jpg", "png", "html", "txt", "md", "json", "pptx"],
        *,
        auto: bool = False,
    ) -> FileExtractorBase: ...

    @overload
    def create_extractor(
        self,
        *,
        auto: bool = False,
    ) -> FileExtractorBase: ...

    def create_extractor(
        self,
        extension: Optional[
            Literal["pdf", "jpg", "png", "html", "txt", "md", "json", "pptx"]
        ] = None,
        *,
        auto: bool = False,
    ) -> FileExtractorBase:
        if extension:
            extractor = {
                "pdf": MinerUFileExtractor,
                "pptx": MinerUFileExtractor,
                # Add other extensions as needed
            }.get(extension)
            if extractor is None:
                raise ValueError("The provided extension is not valid")
            return extractor()

        return MinerUFileExtractor()  # TODO


factory = FileExtractorFactory()

factory.create_extractor("pdf")
