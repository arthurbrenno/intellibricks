from dotenv import load_dotenv
from msgspec import Struct
from intellibricks import CompletionEngine

load_dotenv(override=True)


class Joke(Struct):
    joke: str


# One line to rule them all 🔥
output = CompletionEngine().complete(prompt="Tell me a joke", response_format=Joke)

print(output.get_parsed())  # Joke obj
