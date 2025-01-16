from typing import Annotated
from intellibricks import (
    Synapse,
    UserMessage,
    AssistantMessage,
    DeveloperMessage,
    ChainOfThought,
)
import msgspec

synapse = Synapse.of(
    "google/genai/gemini-2.0-flash-exp",
)

messages = (
    DeveloperMessage.from_text("You are a helpful assistant."),
    UserMessage.from_text("Hello, how are you?"),
    AssistantMessage.from_text("I am fine, thank you."),
    UserMessage.from_text("What is your name? And who created you?"),
)

print(messages)


class CreatorInfo(msgspec.Struct):
    name: Annotated[str, msgspec.Meta(description="Here you can enter your name.")]

    is_human: Annotated[
        bool,
        msgspec.Meta(
            description="Here you can specify whether the creator is a human or not.",
        ),
    ]


class ModelInfo(msgspec.Struct):
    name: Annotated[str, msgspec.Meta(description="Here you can enter your name.")]

    creator: Annotated[
        CreatorInfo,
        msgspec.Meta(description="Here you can enter the creator's name."),
    ]


completion = synapse.chat(messages, response_model=ChainOfThought[ModelInfo])

print(completion.parsed)
