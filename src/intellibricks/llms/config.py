"""Configuration module for caching system prompts in AI providers.

This module defines the `CacheConfig` class, which encapsulates the settings required
to configure caching behavior for system prompts used by AI providers. Proper caching
of system prompts can enhance performance by reducing redundant computations and API calls.

Classes:
    CacheConfig: Configuration settings for caching mechanisms.

Examples:
    Creating a `CacheConfig` instance with default settings:
        >>> cache_config = CacheConfig()
        >>> print(cache_config)
        CacheConfig(enabled=False, ttl=datetime.timedelta(seconds=30), cache_key='default_prompt')

    Enabling caching with a custom TTL and cache key:
        >>> custom_cache = CacheConfig(enabled=True, ttl=60, cache_key='user_session_prompt')
        >>> print(custom_cache)
        CacheConfig(enabled=True, ttl=60, cache_key='user_session_prompt')
"""

import datetime
from typing import Annotated

from architecture import BaseModel, Meta, field


class CacheConfig(BaseModel):
    """Configuration settings for caching system prompts in AI providers.

    The `CacheConfig` class encapsulates the settings required to configure caching behavior
    for system prompts used by AI providers. It allows developers to enable or disable caching,
    set the time-to-live (TTL) for cache entries, and define a cache key to identify cached data.

    Attributes:
        enabled (bool):
            Indicates whether caching is enabled for system prompts.

            **Default:** `False`

            **Example:**
                >>> cache_config = CacheConfig(enabled=True)
                >>> print(cache_config.enabled)
                True

        ttl (Union[int, datetime.timedelta]):
            Specifies the time-to-live for cache entries. It can be defined either as an integer
            representing seconds or as a `datetime.timedelta` object for more precise control.

            **Default:** `datetime.timedelta(seconds=30)`

            **Example:**
                >>> cache_config = CacheConfig(ttl=60)
                >>> print(cache_config.ttl)
                60

        cache_key (str):
            Defines the key used to identify cached system prompts. This key is essential for
            retrieving and storing cache entries consistently.

            **Default:** `'default_prompt'`

            **Example:**
                >>> cache_config = CacheConfig(cache_key='user_session_prompt')
                >>> print(cache_config.cache_key)
                'user_session_prompt'
    """

    enabled: Annotated[
        bool,
        Meta(
            title="Caching Enabled",
            description=(
                "Indicates whether caching is enabled for system prompts. "
                "When set to `True`, the system prompts will be cached to improve "
                "performance by reducing redundant computations and API calls."
            ),
        ),
    ] = field(default=False)
    """Indicates whether caching is enabled for system prompts.

    When set to `True`, the system prompts will be cached to improve performance by
    reducing redundant computations and API calls.
    """

    ttl: Annotated[
        datetime.timedelta,
        Meta(
            title="Time-To-Live (TTL)",
            description=(
                "Specifies the time-to-live for cache entries. This can be defined either as an "
                "integer representing seconds or as a `datetime.timedelta` object for finer granularity. "
                "The TTL determines how long a cached system prompt remains valid before it is refreshed or invalidated."
            ),
        ),
    ] = field(default_factory=lambda: datetime.timedelta(seconds=0))
    """Specifies the time-to-live for cache entries.

    The TTL can be set as an integer (in seconds) or as a `datetime.timedelta` object for finer granularity.
    This value determines how long a cached system prompt remains valid before it needs to be refreshed or invalidated.

    **Example:**
        >>> cache_config = CacheConfig(ttl=60)
        >>> print(cache_config.ttl)
        60
    """

    cache_key: Annotated[
        str,
        Meta(
            title="Cache Key",
            description=(
                "Defines the key used to identify cached system prompts. This key is essential for storing and retrieving "
                "cache entries consistently. It should be unique enough to prevent collisions but also meaningful "
                "to facilitate easy management of cached data."
            ),
        ),
    ] = field(default="default")
    """Defines the key used to identify cached system prompts.

    The `cache_key` is crucial for storing and retrieving cache entries consistently. It should be unique
    enough to prevent collisions with other cached data but also meaningful to facilitate easy management
    of cached entries.

    **Example:**
        >>> cache_config = CacheConfig(cache_key='user_session_prompt')
        >>> print(cache_config.cache_key)
        'user_session_prompt'
    """

    def __hash__(self) -> int:
        """Generate a hash based on the instance's attributes."""
        return hash((self.enabled, self.ttl.total_seconds(), self.cache_key))

    def __eq__(self, other: object) -> bool:
        """Equality check to complement hash."""
        if not isinstance(other, CacheConfig):
            return NotImplemented
        return (
            self.enabled == other.enabled
            and self.ttl == other.ttl
            and self.cache_key == other.cache_key
        )
