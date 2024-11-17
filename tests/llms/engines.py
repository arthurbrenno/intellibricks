import asyncio
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


async def main():
    # Call the CompletionEngine
    engine = CompletionEngine()
    response: CompletionOutput[PresidentsResponse] = await engine.complete_async(
        prompt="What were the presidents of the USA until your knowledge?",
        response_format=PresidentsResponse,
    )

    # Manipulate the response as you want.
    presidents_response: PresidentsResponse = response.get_parsed()
    print(f"First president name is {presidents_response.presidents[0].name}")
    print(f"First president age is {presidents_response.presidents[0].age}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(main())
        else:
            loop.run_until_complete(main())
