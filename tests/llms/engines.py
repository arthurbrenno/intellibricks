from typing import Annotated

from dotenv import load_dotenv
from msgspec import Meta, Struct

from intellibricks import CompletionEngine
from intellibricks.llms.schema import CompletionOutput

load_dotenv(override=True)


# Step #1: Define your response structure
class President(Struct):
    name: str
    age: Annotated[int, Meta(ge=40, le=107)]


class PresidentsResponse(Struct):
    presidents: list[President]

# Call the CompletionEngine
engine = CompletionEngine()
response: CompletionOutput[PresidentsResponse] = engine.complete(
    prompt="What were the presidents of the USA until your knowledge?",
    response_format=PresidentsResponse,
)

# Manipulate the response as you want.
presidents_response: PresidentsResponse = response.get_parsed()
print(f"First president name is {presidents_response.presidents[0].name}")
print(f"First president age is {presidents_response.presidents[0].age}")
