from dotenv import load_dotenv

from intellibricks.llms import (
    DeveloperMessage,
    Synapse,
    UserMessage,
)

load_dotenv()

synapse = Synapse.of("ollama/api/gemma3:1b")

messages = (
    DeveloperMessage.from_text("You are a helpful assistant."),
    UserMessage.from_text("Hello, how are you?"),
)

completion = synapse.chat(messages)

print(completion)
