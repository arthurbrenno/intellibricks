from intellibricks import (
    Synapse,
    Agent,
)
import uvicorn

agent = Agent(
    task="Chat With the User",
    instructions=[
        "Chat with the user",
    ],
    metadata={"name": "Bob", "description": "A simple chat agent."},
    synapse=Synapse.of("google/genai/gemini-2.0-flash-exp"),
)

uvicorn.run(agent.to_litestar_async_app())
