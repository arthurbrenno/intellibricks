import msgspec
import pytest
from intellibricks.llms.integrations.groq import GroqLanguageModel
from intellibricks.llms.types import UserMessage


@pytest.fixture
def groq_model() -> GroqLanguageModel:
    return GroqLanguageModel(
        model_name="llama3-8b-8192",
    )


@pytest.mark.asyncio
async def test_groq_model(groq_model: GroqLanguageModel) -> None:
    class Mood(msgspec.Struct):
        mood: str

    completion = await groq_model.chat_async(
        messages=[UserMessage.from_text("How are you?")],
        response_model=Mood,
    )

    assert isinstance(completion.choices[0].message.parsed, Mood)
    assert isinstance(completion.choices[0].message.parsed.mood, str)
    assert len(completion.choices[0].message.parsed.mood) > 0
