from typing import Annotated, Literal

import msgspec
from dotenv import load_dotenv

from intellibricks.llms import (
    AssistantMessage,
    ChainOfThought,
    DeveloperMessage,
    Synapse,
    UserMessage,
)

load_dotenv()

synapse = Synapse.of(
    "cerebras/api/llama-3.3-70b",
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

    random_number: Annotated[
        Literal["1", "2", "3", "4", "5"],
        msgspec.Meta(description="Here you can specify a random number."),
    ]

    creator: Annotated[
        CreatorInfo,
        msgspec.Meta(description="Here you can enter the creator's name."),
    ]


completion = synapse.chat(messages, response_model=ChainOfThought[ModelInfo])

print(completion)
