from intellibricks import Synapse
from msgspec import Struct


def get_weather(city: str) -> str:
    text = f"O clima de {city} está ensolarado!"
    print(text)
    return text


class Joke(Struct):
    joke: str


# Step #1: Define a synapse to a model
synapse = Synapse.of("google/genai/gemini-2.0-flash-exp")

# Step #2: Perform a chat completion
completion = synapse.complete(
    "What is the weather in Uberaba today?", tools=[get_weather]
)

print(completion)
message = completion.message
result: str = message.tool_calls.first.call()
