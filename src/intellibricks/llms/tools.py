from typing import Literal
from dataclasses import field

from intellibricks.llms.types import Tool


class DuckDuckGoTool(Tool, frozen=True):
    query: str
    n: int = 1
    safe_search: bool = False
    region: Literal[
        "us-en",
        "uk-en",
        "de-de",
        "es-es",
        "fr-fr",
        "it-it",
        "nl-nl",
        "pl-pl",
        "pt-br",
        "tr-tr",
    ] = field(default_factory=lambda: "us-en")
