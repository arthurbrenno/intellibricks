"""Common constants of the LLMs module."""

from typing import Optional


CACHE_KEY_TO_ID: dict[str, Optional[str]] = {}
"""
Stores the developer-customized key to the ID generated
by google cloud cache mechanism. Useful to store the
cache keys so the context (system prompt) gets
cached and the costs are saved by a 90%
margin. The keys are stored at runtime.
"""
