from __future__ import annotations

import abc

from architecture.data.files import RawFile

from .schema import DocumentArtifact


class FileExtractorProtocol(abc.ABC):
    """
    Abstract class for extracting content from files.
    This should be used as a base class for specific file extractors.
    """

    @abc.abstractmethod
    async def extract_contents(self, file: RawFile) -> DocumentArtifact:
        """Extracts content from the file."""
        raise NotImplementedError("This method should be implemented by subclasses.")


"""This module contains the implementation of the FileExtractorProtocol class."""


class DoclingFileExtractor(FileExtractorProtocol):
    pass
