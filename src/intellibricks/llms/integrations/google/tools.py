from typing import TYPE_CHECKING, Any, Callable
from intellibricks.llms.types import Tool

if TYPE_CHECKING:
    from google.genai.types import Tool as GenAITool


class VertexAISearchTool(Tool, frozen=True):
    datastore: str

    def to_callable(self) -> Callable[..., Any]:
        raise NotADirectoryError(
            "It is possible to implement that, but I didn't have time to do it."
        )

    def to_google_tool(self) -> GenAITool:
        from google.genai import types

        return types.Tool(
            retrieval=types.Retrieval(
                vertex_ai_search=types.VertexAISearch(datastore=self.datastore)
            )
        )
