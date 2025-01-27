import msgspec
import uvicorn

from intellibricks.agents import Agent
from intellibricks.llms import Synapse


def get_weather(city: str) -> str:
    return f"The weather in {city} will be rainy with a high of 25°C and a low of 15°C. Updated."


class WeatherInfo(msgspec.Struct):
    city: str
    high: int
    low: int
    description: str


groq_synapse = Synapse.of("groq/api/llama-3.1-8b-instant")
google_synapse = Synapse.of("google/genai/gemini-2.0-flash-exp")
openai_synapse = Synapse.of("openai/api/gpt-4o")

agent = Agent(
    task="Weather Analysis",
    instructions=["Chat the user answering his weather questions."],
    synapse=google_synapse,
    metadata={"name": "Scarlet", "description": "A weather chat agent."},
    response_model=WeatherInfo,
    tools=[get_weather],
)

# response = agent.run(
#     "What will be the weather in Uberaba today? What is the high and low?"
# )

# print(response)

uvicorn.run(agent.fastapi_app, port=5000)
