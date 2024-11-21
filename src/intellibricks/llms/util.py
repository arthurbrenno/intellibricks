import tiktoken
from vertexai.generative_models import GenerativeModel
from architecture.utils.decorators import pure

from .constants import AIModel


@pure
def count_tokens(model: AIModel, text: str) -> int:
    """
    Counts the number of tokens in the given text using the specified AI model.

    Args:
        model (AIModel): The AI model to use for tokenization.
        text (str): The text to tokenize.

    Returns:
        int: The total number of tokens.

    Raises:
        ValueError: If an unsupported model is provided.
    """
    match model:
        case AIModel.VERTEX_GEMINI_1P5_FLASH_002 | AIModel.VERTEX_GEMINI_1P5_PRO_002:
            generative_model = GenerativeModel(model.value)
            token_info = generative_model.count_tokens(text)
            total_tokens: int = token_info.total_tokens
            return total_tokens
        case AIModel.STUDIO_GEMINI_1P5_FLASH:
            temp_encoding: tiktoken.core.Encoding = tiktoken.encoding_for_model(
                "gpt-4o"
            )
            total_tokens = len(temp_encoding.encode(text=text))
            return total_tokens
            model_name = model.value.replace("models/", "")
            generative_model = GenerativeModel(model_name)
            token_info = generative_model.count_tokens(text)
            total_tokens = token_info.total_tokens
            return total_tokens
        case AIModel.GPT_4O:
            encoding: tiktoken.core.Encoding = tiktoken.encoding_for_model("gpt-4o")
            total_tokens = len(encoding.encode(text=text))
            return total_tokens

        case AIModel.GPT_3P5_TURBO_0125:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            total_tokens = len(encoding.encode(text=text))
            return total_tokens
        case _:
            encoding = tiktoken.encoding_for_model("gpt-4o")
            total_tokens = len(encoding.encode(text=text))
            return total_tokens
