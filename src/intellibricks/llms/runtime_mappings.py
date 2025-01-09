"""Common constants of the LLMs module."""

from typing import Final, Optional

import msgspec


class RuntimeMappingsContainer(msgspec.Struct, frozen=True, kw_only=True):
    CACHE_KEY_TO_ID: Final[dict[str, Optional[str]]] = {}
    """
    Stores the developer-customized key to the ID generated
    by google cloud cache mechanism. Useful to store the
    cache keys so the context (system prompt) gets
    cached and the costs are saved by a 90%
    margin. The keys are stored at runtime.
    """


# Guarantee that CACHE_KEY_TO_ID reference will not be mutated, but only the dict it holds.
mappings = RuntimeMappingsContainer()
