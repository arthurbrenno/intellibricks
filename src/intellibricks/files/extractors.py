from __future__ import annotations

import abc

from architecture.data.files import RawFile

from .constants import ParsingMethod

from .schema import DocumentArtifact


class FileExtractorBase(abc.ABC):
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


class FileExtractor(FileExtractorBase):
    pass
