import logging
from typing import Any, Optional

from architecture import log
from architecture.utils.creators import DynamicInstanceCreator

from intellibricks.llms.base import TranscriptionModel
from intellibricks.llms.base.contracts import LanguageModel
from intellibricks.llms.integrations.cerebras.cerebras import CerebrasLanguageModel
from intellibricks.llms.integrations.deepinfra import DeepInfraLanguageModel
from intellibricks.llms.integrations.google import GoogleLanguageModel
from intellibricks.llms.integrations.groq import (
    GroqLanguageModel,
    GroqTranscriptionModel,
)
from intellibricks.llms.integrations.openai import (
    OpenAILanguageModel,
    OpenAITranscriptionModel,
)
from intellibricks.llms.types import AIModel, TranscriptionModelType

debug_logger = log.create_logger(__name__, level=logging.DEBUG)


class LanguageModelFactory:
    @classmethod
    def create(
        cls, model: AIModel, params: Optional[dict[str, Any]] = None
    ) -> LanguageModel:
        debug_logger.debug(f"Creating model: {model}")

        model_to_model_class: dict[AIModel, type[LanguageModel]] = {
            "google/genai/gemini-2.0-flash-exp": GoogleLanguageModel,
            "google/genai/gemini-1.5-flash": GoogleLanguageModel,
            "google/genai/gemini-1.5-flash-8b": GoogleLanguageModel,
            "google/genai/gemini-1.5-flash-001": GoogleLanguageModel,
            "google/genai/gemini-1.5-flash-002": GoogleLanguageModel,
            "google/genai/gemini-1.5-pro": GoogleLanguageModel,
            "google/genai/gemini-1.5-pro-001": GoogleLanguageModel,
            "google/genai/gemini-1.0-pro-002": GoogleLanguageModel,
            "google/genai/gemini-1.5-pro-002": GoogleLanguageModel,
            "google/genai/gemini-flash-experimental": GoogleLanguageModel,
            "google/genai/gemini-pro-experimental": GoogleLanguageModel,
            "google/vertexai/gemini-2.0-flash-exp": GoogleLanguageModel,
            "google/vertexai/gemini-1.5-flash": GoogleLanguageModel,
            "google/vertexai/gemini-1.5-flash-8b": GoogleLanguageModel,
            "google/vertexai/gemini-1.5-flash-001": GoogleLanguageModel,
            "google/vertexai/gemini-1.5-flash-002": GoogleLanguageModel,
            "google/vertexai/gemini-1.5-pro": GoogleLanguageModel,
            "google/vertexai/gemini-1.5-pro-001": GoogleLanguageModel,
            "google/vertexai/gemini-1.0-pro-002": GoogleLanguageModel,
            "google/vertexai/gemini-1.5-pro-002": GoogleLanguageModel,
            "google/vertexai/gemini-flash-experimental": GoogleLanguageModel,
            "google/vertexai/gemini-pro-experimental": GoogleLanguageModel,
            "groq/api/gemma2-9b-it": GroqLanguageModel,
            "groq/api/llama3-groq-70b-8192-tool-use-preview": GroqLanguageModel,
            "groq/api/llama3-groq-8b-8192-tool-use-preview": GroqLanguageModel,
            "groq/api/llama-3.1-70b-specdec": GroqLanguageModel,
            "groq/api/llama-3.2-1b-preview": GroqLanguageModel,
            "groq/api/llama-3.2-3b-preview": GroqLanguageModel,
            "groq/api/llama-3.2-11b-vision-preview": GroqLanguageModel,
            "groq/api/llama-3.2-90b-vision-preview": GroqLanguageModel,
            "groq/api/llama-3.3-70b-specdec": GroqLanguageModel,
            "groq/api/llama-3.3-70b-versatile": GroqLanguageModel,
            "groq/api/llama-3.1-8b-instant": GroqLanguageModel,
            "groq/api/llama-guard-3-8b": GroqLanguageModel,
            "groq/api/llama3-70b-8192": GroqLanguageModel,
            "groq/api/llama3-8b-8192": GroqLanguageModel,
            "groq/api/mixtral-8x7b-32768": GroqLanguageModel,
            "openai/api/o1": OpenAILanguageModel,
            "openai/api/o1-2024-12-17": OpenAILanguageModel,
            "openai/api/o1-preview": OpenAILanguageModel,
            "openai/api/o1-preview-2024-09-12": OpenAILanguageModel,
            "openai/api/o1-mini": OpenAILanguageModel,
            "openai/api/o1-mini-2024-09-12": OpenAILanguageModel,
            "openai/api/gpt-4o": OpenAILanguageModel,
            "openai/api/gpt-4o-2024-11-20": OpenAILanguageModel,
            "openai/api/gpt-4o-2024-08-06": OpenAILanguageModel,
            "openai/api/gpt-4o-2024-05-13": OpenAILanguageModel,
            "openai/api/gpt-4o-audio-preview": OpenAILanguageModel,
            "openai/api/gpt-4o-audio-preview-2024-10-01": OpenAILanguageModel,
            "openai/api/gpt-4o-audio-preview-2024-12-17": OpenAILanguageModel,
            "openai/api/gpt-4o-mini-audio-preview": OpenAILanguageModel,
            "openai/api/gpt-4o-mini-audio-preview-2024-12-17": OpenAILanguageModel,
            "openai/api/chatgpt-4o-latest": OpenAILanguageModel,
            "openai/api/gpt-4o-mini": OpenAILanguageModel,
            "openai/api/gpt-4o-mini-2024-07-18": OpenAILanguageModel,
            "openai/api/gpt-4-turbo": OpenAILanguageModel,
            "openai/api/gpt-4-turbo-2024-04-09": OpenAILanguageModel,
            "openai/api/gpt-4-0125-preview": OpenAILanguageModel,
            "openai/api/gpt-4-turbo-preview": OpenAILanguageModel,
            "openai/api/gpt-4-1106-preview": OpenAILanguageModel,
            "openai/api/gpt-4-vision-preview": OpenAILanguageModel,
            "openai/api/gpt-4": OpenAILanguageModel,
            "openai/api/gpt-4-0314": OpenAILanguageModel,
            "openai/api/gpt-4-0613": OpenAILanguageModel,
            "openai/api/gpt-4-32k": OpenAILanguageModel,
            "openai/api/gpt-4-32k-0314": OpenAILanguageModel,
            "openai/api/gpt-4-32k-0613": OpenAILanguageModel,
            "openai/api/gpt-3.5-turbo": OpenAILanguageModel,
            "openai/api/gpt-3.5-turbo-16k": OpenAILanguageModel,
            "openai/api/gpt-3.5-turbo-0301": OpenAILanguageModel,
            "openai/api/gpt-3.5-turbo-0613": OpenAILanguageModel,
            "openai/api/gpt-3.5-turbo-1106": OpenAILanguageModel,
            "openai/api/gpt-3.5-turbo-0125": OpenAILanguageModel,
            "openai/api/gpt-3.5-turbo-16k-0613": OpenAILanguageModel,
            "deepinfra/api/meta-llama/Llama-3.3-70B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-3.3-70B-Instruct-Turbo": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Meta-Llama-3.1-70B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Meta-Llama-3.1-8B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Meta-Llama-3.1-405B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/QwQ-32B-Preview": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/Qwen2.5-Coder-32B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/nvidia/Llama-3.1-Nemotron-70B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/Qwen2.5-72B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-3.2-90B-Vision-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-3.2-11B-Vision-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/microsoft/WizardLM-2-8x22B": DeepInfraLanguageModel,
            "deepinfra/api/01-ai/Yi-34B-Chat": DeepInfraLanguageModel,
            "deepinfra/api/Austism/chronos-hermes-13b-v2": DeepInfraLanguageModel,
            "deepinfra/api/Gryphe/MythoMax-L2-13b": DeepInfraLanguageModel,
            "deepinfra/api/Gryphe/MythoMax-L2-13b-turbo": DeepInfraLanguageModel,
            "deepinfra/api/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1": DeepInfraLanguageModel,
            "deepinfra/api/NousResearch/Hermes-3-Llama-3.1-405B": DeepInfraLanguageModel,
            "deepinfra/api/Phind/Phind-CodeLlama-34B-v2": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/QVQ-72B-Preview": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/Qwen2-72B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/Qwen2-7B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/Qwen2.5-7B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/Qwen/Qwen2.5-Coder-7B": DeepInfraLanguageModel,
            "deepinfra/api/Sao10K/L3-70B-Euryale-v2.1": DeepInfraLanguageModel,
            "deepinfra/api/Sao10K/L3-8B-Lunaris-v1": DeepInfraLanguageModel,
            "deepinfra/api/Sao10K/L3.1-70B-Euryale-v2.2": DeepInfraLanguageModel,
            "deepinfra/api/bigcode/starcoder2-15b": DeepInfraLanguageModel,
            "deepinfra/api/bigcode/starcoder2-15b-instruct-v0.1": DeepInfraLanguageModel,
            "deepinfra/api/codellama/CodeLlama-34b-Instruct-hf": DeepInfraLanguageModel,
            "deepinfra/api/codellama/CodeLlama-70b-Instruct-hf": DeepInfraLanguageModel,
            "deepinfra/api/cognitivecomputations/dolphin-2.6-mixtral-8x7b": DeepInfraLanguageModel,
            "deepinfra/api/cognitivecomputations/dolphin-2.9.1-llama-3-70b": DeepInfraLanguageModel,
            "deepinfra/api/databricks/dbrx-instruct": DeepInfraLanguageModel,
            "deepinfra/api/airoboros-70b": DeepInfraLanguageModel,
            "deepinfra/api/google/codegemma-7b-it": DeepInfraLanguageModel,
            "deepinfra/api/google/gemma-1.1-7b-it": DeepInfraLanguageModel,
            "deepinfra/api/google/gemma-2-27b-it": DeepInfraLanguageModel,
            "deepinfra/api/google/gemma-2-9b-it": DeepInfraLanguageModel,
            "deepinfra/api/lizpreciatior/lzlv_70b_fp16_hf": DeepInfraLanguageModel,
            "deepinfra/api/mattshumer/Reflection-Llama-3.1-70B": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-2-13b-chat-hf": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-2-70b-chat-hf": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-2-7b-chat-hf": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-3.2-1B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Llama-3.2-3B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Meta-Llama-3-70B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/meta-llama/Meta-Llama-3-8B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/microsoft/Phi-3-medium-4k-instruct": DeepInfraLanguageModel,
            "deepinfra/api/microsoft/WizardLM-2-7B": DeepInfraLanguageModel,
            "deepinfra/api/mistralai/Mistral-7B-Instruct-v0.1": DeepInfraLanguageModel,
            "deepinfra/api/mistralai/Mistral-7B-Instruct-v0.2": DeepInfraLanguageModel,
            "deepinfra/api/mistralai/Mistral-7B-Instruct-v0.3": DeepInfraLanguageModel,
            "deepinfra/api/mistralai/Mistral-Nemo-Instruct-2407": DeepInfraLanguageModel,
            "deepinfra/api/mistralai/Mixtral-8x22B-Instruct-v0.1": DeepInfraLanguageModel,
            "deepinfra/api/mistralai/Mixtral-8x22B-v0.1": DeepInfraLanguageModel,
            "deepinfra/api/mistralai/Mixtral-8x7B-Instruct-v0.1": DeepInfraLanguageModel,
            "deepinfra/api/nvidia/Nemotron-4-340B-Instruct": DeepInfraLanguageModel,
            "deepinfra/api/openbmb/MiniCPM-Llama3-V-2_5": DeepInfraLanguageModel,
            "deepinfra/api/openchat/openchat-3.6-8b": DeepInfraLanguageModel,
            "deepinfra/api/openchat/openchat_3.5": DeepInfraLanguageModel,
            "cerebras/api/llama3.1-8b": CerebrasLanguageModel,
            "cerebras/api/llama3.1-70b": CerebrasLanguageModel,
            "cerebras/api/llama-3.3-70b": CerebrasLanguageModel,
        }

        async_chat_model_cls: type[LanguageModel] = model_to_model_class[model]

        model_extra_params: dict[AIModel, dict[str, Any]] = {
            "google/genai/gemini-2.0-flash-exp": {"vertexai": False},
            "google/genai/gemini-1.5-flash": {"vertexai": False},
            "google/genai/gemini-1.5-flash-8b": {"vertexai": False},
            "google/genai/gemini-1.5-flash-001": {"vertexai": False},
            "google/genai/gemini-1.5-flash-002": {"vertexai": False},
            "google/genai/gemini-1.5-pro": {"vertexai": False},
            "google/genai/gemini-1.5-pro-001": {"vertexai": False},
            "google/genai/gemini-1.0-pro-002": {"vertexai": False},
            "google/genai/gemini-1.5-pro-002": {"vertexai": False},
            "google/genai/gemini-flash-experimental": {"vertexai": False},
            "google/genai/gemini-pro-experimental": {"vertexai": False},
            "google/vertexai/gemini-2.0-flash-exp": {"vertexai": True},
            "google/vertexai/gemini-1.5-flash": {"vertexai": True},
            "google/vertexai/gemini-1.5-flash-8b": {"vertexai": True},
            "google/vertexai/gemini-1.5-flash-001": {"vertexai": True},
            "google/vertexai/gemini-1.5-flash-002": {"vertexai": True},
            "google/vertexai/gemini-1.5-pro": {"vertexai": True},
            "google/vertexai/gemini-1.5-pro-001": {"vertexai": True},
            "google/vertexai/gemini-1.0-pro-002": {"vertexai": True},
            "google/vertexai/gemini-1.5-pro-002": {"vertexai": True},
            "google/vertexai/gemini-flash-experimental": {"vertexai": True},
            "google/vertexai/gemini-pro-experimental": {"vertexai": True},
        }

        params = params or {}
        params.update(model_extra_params.get(model, {}))

        instance = DynamicInstanceCreator(cls=async_chat_model_cls).create_instance(
            **params
        )
        return instance


class TranscriptionModelFactory:
    @classmethod
    def create(
        cls,
        model: TranscriptionModelType,
        params: Optional[dict[str, Any]] = None,
    ) -> TranscriptionModel:
        debug_logger.info(f"Creating transcription model: {model}")

        transcription_model_type_to_transcription_model_class: dict[
            TranscriptionModelType, type[TranscriptionModel]
        ] = {
            "groq/api/whisper-large-v3-turbo": GroqTranscriptionModel,
            "groq/api/distil-whisper-large-v3-en": GroqTranscriptionModel,
            "groq/api/whisper-large-v3": GroqTranscriptionModel,
            "openai/api/whisper-1": OpenAITranscriptionModel,
        }

        async_chat_model_cls: type[TranscriptionModel] = (
            transcription_model_type_to_transcription_model_class[model]
        )

        instance = DynamicInstanceCreator(cls=async_chat_model_cls).create_instance(
            **(params or {})
        )
        return instance
