import asyncio
import getpass
import os
import pprint
from typing import Annotated

from msgspec import Meta, Struct

from intellibricks import CompletionEngine
# Your Google AI Studio API key (free Gemini)
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

# Define your response structure
class President(Struct):
    name: str
    age: Annotated[int, Meta(ge=40, le=107)]


class PresidentsResponse(Struct):
    presidents: list[President]


# Instantiate the CompletionEngine (defaults to Google's free Gemini model)
async def main():
    response = await CompletionEngine().complete_async(
        prompt="What were the presidents of the USA until your knowledge?",
        response_format=PresidentsResponse,
    )
    pprint.pprint(response.get_parsed()) # PresidentsResponse


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(main())
        else:
            loop.run_until_complete(main())
