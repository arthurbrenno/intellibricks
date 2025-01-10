from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Protocol, cast

from architecture.utils.decorators import ensure_module_installed
from msgspec import Meta, Struct, field
from msgspec.json import decode, encode


class SearchEngine(str, Enum):
    GOOGLE = "google"
    BRAVE = "brave"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"


class Category(str, Enum):
    GENERAL = "general"
    IMAGES = "images"
    NEWS = "news"
    VIDEOS = "videos"
    SHOPPING = "shopping"


class Language(str, Enum):
    ENGLISH = "en"
    PORTUGUESE = "pt"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"


class SortOrder(str, Enum):
    RELEVANCE = "relevance"
    DATE = "date"
    POPULARITY = "popularity"


class DateRange(str, Enum):
    PAST_DAY = "past_day"
    PAST_WEEK = "past_week"
    PAST_MONTH = "past_month"
    PAST_YEAR = "past_year"


class SearchResult(Struct):
    category: Annotated[
        Category,
        Meta(title="category", description="category", examples=["general"]),
    ]
    link: str
    snippet: str
    engines: list[Literal["google", "brave"]]
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchParams(Struct, kw_only=True):
    host: Annotated[
        Optional[str],
        Meta(
            title="Host",
            description="Possible host of the search engine (e.g searxng, API endpoint etc)",
        ),
    ] = field(default=None)

    engines: Annotated[
        list[SearchEngine],
        Meta(
            title="engines",
            description="List of search engines to use.",
            examples=[["google", "brave"]],
        ),
    ] = field(default_factory=lambda: [SearchEngine.GOOGLE])

    category: Annotated[
        Optional[Category],
        Meta(
            title="category",
            description="Category of the search.",
            examples=["general", "images"],
        ),
    ] = field(default=Category.GENERAL)

    language: Annotated[
        Language,
        Meta(
            title="language",
            description="Language of the search results.",
            examples=["en", "pt"],
        ),
    ] = field(default=Language.ENGLISH)

    region: Annotated[
        Optional[str],
        Meta(
            title="region",
            description="Geographical region code (e.g., 'US', 'BR').",
            examples=["US", "BR"],
        ),
    ] = field(default=None)

    date_range: Annotated[
        Optional[DateRange],
        Meta(
            title="date_range",
            description="Timeframe for the search results.",
            examples=["past_week", "past_month"],
        ),
    ] = field(default=None)

    num_results: Annotated[
        Optional[int],
        Meta(
            title="num_results",
            description="Number of search results to return.",
            examples=[10, 20],
        ),
    ] = field(default=10)

    sort_order: Annotated[
        Optional[SortOrder],
        Meta(
            title="sort_order",
            description="Order to sort the search results.",
            examples=["relevance", "date"],
        ),
    ] = field(default=SortOrder.RELEVANCE)

    filters: Annotated[
        dict[str, Any],
        Meta(
            title="filters",
            description="Additional filters for the search.",
            examples=[{"safe_search": "on"}, {"image_size": "large"}],
        ),
    ] = field(default_factory=dict)


class WebSearchable(Protocol):
    @abstractmethod
    def search(
        self, query: str, search_params: Optional[SearchParams] = None
    ) -> list[SearchResult]: ...


@dataclass
class SearxngWebSearchEngine(WebSearchable):
    host: str

    @ensure_module_installed("langchain_community", "langchain-community")
    def search(
        self, query: str, search_params: Optional[SearchParams] = None
    ) -> list[SearchResult]:
        from langchain_community.utilities import SearxSearchWrapper

        s = SearxSearchWrapper(searx_host=self.host)
        results: list[dict[str, Any]] = cast(
            list[dict[str, Any]],
            s.results("presidente estados unidos 2024", num_results=2),
        )

        return [decode(encode(result), type=SearchResult) for result in results]
