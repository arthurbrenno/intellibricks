"""Common constants used in the llm module"""

from __future__ import annotations

from enum import Enum
from typing import Tuple, Type

from llama_index.core.llms import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

from intellibricks.llms.custom_llms import EnhancedVertexAI


class MessageRole(str, Enum):
    """Represents the role of a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


class AIModel(str, Enum):
    """Allowed models to use in this endpoint"""

    #   ██████╗  ██████╗  ██████╗  ██████╗ ██╗     ███████╗
    #  ██╔════╝ ██╔═══██╗██╔═══██╗██╔════╝ ██║     ██╔════╝
    #  ██║  ███╗██║   ██║██║   ██║██║  ███╗██║     █████╗
    #  ██║   ██║██║   ██║██║   ██║██║   ██║██║     ██╔══╝
    #  ╚██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝███████╗███████╗
    #   ╚═════╝  ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚══════╝
    STUDIO_GEMINI_1P5_FLASH = "models/gemini-1.5-flash"
    VERTEX_GEMINI_1P5_FLASH_002 = "gemini-1.5-flash-002"
    VERTEX_GEMINI_1P5_PRO_002 = "gemini-1.5-pro-002"

    #   ██████╗ ██████╗ ███████╗███╗   ██╗ █████╗ ██╗
    #  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██║
    #  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║
    #  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║
    #  ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║
    #   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_3P5_TURBO_0125 = "gpt-3.5-turbo-0125"

    #   █████╗ ███╗   ██╗████████╗██╗  ██╗██████╗  ██████╗ ██████╗ ██╗ ██████╗
    #  ██╔══██╗████╗  ██║╚══██╔══╝██║  ██║██╔══██╗██╔═══██╗██╔══██╗██║██╔════╝
    #  ███████║██╔██╗ ██║   ██║   ███████║██████╔╝██║   ██║██████╔╝██║██║
    #  ██╔══██║██║╚██╗██║   ██║   ██╔══██║██╔══██╗██║   ██║██╔═══╝ ██║██║
    #  ██║  ██║██║ ╚████║   ██║   ██║  ██║██║  ██║╚██████╔╝██║     ██║╚██████╗
    #  ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20240620"

    #   ██████╗ ██████╗  ██████╗  ██████╗
    #  ██╔════╝ ██╔══██╗██╔═══██╗██╔═══██╗
    #  ██║  ███╗██████╔╝██║   ██║██║   ██║
    #  ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║
    #  ╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝
    #   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝
    GROQ_GEMMA_7B_IT = "gemma-7b-it"
    GROQ_GEMMA2_9B_IT = "gemma2-9b-it"
    GROQ_LLAMA3_70B_VERSATILE_128K = "llama3.1-70b-versatile-128k"
    GROQ_LLAMA3_1_8B_INSTANT_128K = "llama3.1-8b-instant-128k"
    GROQ_LLAMA3_70B_8K = "llama3-70b-8k"
    GROQ_LLAMA3_8B_8K = "llama3-8b-8k"
    GROQ_LLAMA3_8B_8192 = "llama3-8b-8192"
    GROQ_LLAMA3_70b_8192 = "llama3-70b-8192"
    GROQ_GEMMA_7B_8K_INSTRUCT = "gemma-7b-8k-instruct"
    GROQ_GEMMA2_9B_8K = "gemma2-9b-8k"
    GROQ_LLAMA3_70B_TOOL_USE_PREVIEW_8K = "llama3-groq-70b-tool-use-preview-8k"
    GROQ_LLAMA3_8B_TOOL_USE_PREVIEW_8K = "llama3-groq-8b-tool-use-preview-8k"
    GROQ_LLAMA_GUARD_3_8B_8K = "llama-guard-3-8b-8k"
    GROQ_LLAMA_3_2_3_B_PREVIEW = "llama-3.2-3b-preview"

    def __new__(cls, value: str) -> "AIModel":
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def provider_name(self) -> str:
        """DEPRECATED."""
        return {
            AIModel.STUDIO_GEMINI_1P5_FLASH: "google",
            AIModel.GPT_4O: "openai",
            AIModel.CLAUDE_35_SONNET: "anthropic",
            AIModel.GROQ_GEMMA_7B_IT: "groq",
            AIModel.GROQ_GEMMA2_9B_IT: "groq",
            AIModel.GROQ_LLAMA3_70B_8K: "groq",
            AIModel.GROQ_LLAMA3_8B_8K: "groq",
            AIModel.GPT_4O_MINI: "openai",
        }[self]

    def ppm(self) -> Tuple[float, float]:
        """
        Returns a tuple containing the input and output prices per token for the AI model.

        Returns:
            Tuple[float, float]: (input_price, output_price)
        """
        return {
            # Google
            AIModel.STUDIO_GEMINI_1P5_FLASH: (0, 0),
            AIModel.VERTEX_GEMINI_1P5_FLASH_002: (0.075, 0.30),
            AIModel.VERTEX_GEMINI_1P5_PRO_002: (1.25, 5.00),
            # OpenAI
            AIModel.GPT_4O: (2.50, 10.00),
            AIModel.GPT_4O_MINI: (0.150, 0.600),
            # Anthropic
            AIModel.CLAUDE_35_SONNET: (0.02, 0.021),
            # Groq
            AIModel.GROQ_GEMMA_7B_IT: (0.07, 0.07),
            AIModel.GROQ_GEMMA2_9B_IT: (0.20, 0.20),
            AIModel.GROQ_LLAMA3_70B_VERSATILE_128K: (0.59, 0.79),
            AIModel.GROQ_LLAMA3_1_8B_INSTANT_128K: (0.05, 0.08),
            AIModel.GROQ_LLAMA3_70B_8K: (0.59, 0.79),
            AIModel.GROQ_LLAMA3_8B_8K: (0.05, 0.08),
            AIModel.GROQ_GEMMA_7B_8K_INSTRUCT: (0.07, 0.07),
            AIModel.GROQ_GEMMA2_9B_8K: (0.20, 0.20),
            AIModel.GROQ_LLAMA3_70B_TOOL_USE_PREVIEW_8K: (0.89, 0.89),
            AIModel.GROQ_LLAMA3_8B_TOOL_USE_PREVIEW_8K: (0.19, 0.19),
            AIModel.GROQ_LLAMA_GUARD_3_8B_8K: (0.20, 0.20),
            AIModel.GROQ_LLAMA_3_2_3_B_PREVIEW: (0.06, 0.06),
        }.get(self, (0.0, 0.0))

    @classmethod
    def get_llama_index_model_cls(cls, model: "AIModel") -> Type["LLM"]:
        """
        Returns the corresponding LLM class for the given AI model.

        This method serves as a registry that maps AIModel enums to their
        respective LLM class implementations.

        Args:
            model (AIModel): The AI model enum for which to retrieve the LLM class.

        Returns:
            Type[LLM]: The LLM class corresponding to the given AI model.

        Raises:
            ValueError: If no matching LLM class is found for the given model.
        """
        registry: dict["AIModel", Type["LLM"]] = {
            # cls.STUDIO_GEMINI_1P5_FLASH: Gemini,
            cls.GPT_4O: OpenAI,
            cls.CLAUDE_35_SONNET: Anthropic,
            cls.GROQ_GEMMA_7B_IT: Groq,
            cls.GROQ_GEMMA2_9B_IT: Groq,
            cls.GROQ_LLAMA3_70b_8192: Groq,
            cls.GROQ_LLAMA3_8B_8192: Groq,
            cls.GPT_4O_MINI: OpenAI,
            cls.VERTEX_GEMINI_1P5_FLASH_002: EnhancedVertexAI,
            cls.VERTEX_GEMINI_1P5_PRO_002: EnhancedVertexAI,
            cls.STUDIO_GEMINI_1P5_FLASH: Gemini,  # DEPRECATED UNTIL llamaindex is updated.
            cls.GPT_3P5_TURBO_0125: OpenAI,
            cls.GROQ_LLAMA_3_2_3_B_PREVIEW: Groq,
        }

        llm_class = registry.get(model)
        if llm_class is None:
            raise ValueError(f"No matching LLM class found for model: {model}")

        return llm_class


class FinishReason(str, Enum):
    """Represents the reason the model stopped generating tokens."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    NONE = None
