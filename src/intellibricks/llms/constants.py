"""Common constants used in the llm module"""

from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    PORTUGUESE = "pt"


class FinishReason(str, Enum):
    """Represents the reason the model stopped generating tokens."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    NONE = None
