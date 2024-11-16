import asyncio
import getpass
import os
import pprint

from msgspec import Struct

from intellibricks import CompletionEngine, CompletionOutput

# Your Google AI Studio API key (free Gemini)
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

# Define your response structure
class President(Struct):
    name: str
    age: int

class PresidentsResponse(Struct):
    presidents: list[President]

# Instantiate the CompletionEngine (defaults to Google's free Gemini model)
async def main():
    engine = CompletionEngine()
    # Generate and parse the response
    response: CompletionOutput[PresidentsResponse] = await engine.complete_async(
        prompt="What were the presidents of the USA until your knowledge?",
        response_format=PresidentsResponse,
    )

    structured_response = response.get_parsed()
    pprint.pprint(structured_response)

def run_async_main():
    try:
        asyncio.run(main())  # Works in most cases if no loop is already running
    except RuntimeError:  # Handle running loop scenario
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Run the coroutine in the existing event loop
            return asyncio.ensure_future(main())
        else:
            # Start a new event loop
            loop.run_until_complete(main())

if __name__ == "__main__":
    run_async_main()

# TODO use pytest here
