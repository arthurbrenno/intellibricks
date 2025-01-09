from intellibricks.llms.integrations.openai import OpenAILanguageModel
import pytest


@pytest.fixture
def openai_model() -> OpenAILanguageModel:
    return OpenAILanguageModel(
        model_name="gpt-4o-mini",
    )
