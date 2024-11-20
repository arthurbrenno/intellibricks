from dotenv import load_dotenv
from msgspec import Struct

from intellibricks import CompletionEngine

load_dotenv(override=True)


class Joke(Struct):
    joke: str


output = CompletionEngine().complete(
    prompt="Tell me a joke",
)  # Evaluates to CompletionOutput[Joke]

output.choices
funny_joke = output.get_parsed()  # Evaluates to Joke
print(funny_joke)
