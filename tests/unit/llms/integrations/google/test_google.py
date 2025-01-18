import msgspec
import pytest

from intellibricks.llms.integrations.google import GoogleLanguageModel
from intellibricks.llms.types import DeveloperMessage, UserMessage


@pytest.fixture
def vertex_model() -> GoogleLanguageModel:
    return GoogleLanguageModel(model_name="gemini-2.0-flash-exp", vertexai=True)


@pytest.fixture
def genai_model() -> GoogleLanguageModel:
    return GoogleLanguageModel(model_name="gemini-2.0-flash-exp", vertexai=False)


@pytest.mark.asyncio
async def test_structured_output_genai(genai_model: GoogleLanguageModel) -> None:
    class Mood(msgspec.Struct):
        mood: str

    completion = await genai_model.chat_async(
        messages=[
            DeveloperMessage.from_text("You are a helpful assistant."),
            UserMessage.from_text("Hello! How are you?"),
        ],
        response_model=Mood,
    )

    assert isinstance(completion.choices[0].message.parsed, Mood)
    assert isinstance(completion.choices[0].message.parsed.mood, str)
    assert len(completion.choices[0].message.parsed.mood) > 0
