import logging
from typing import Any, Mapping, Sequence, ClassVar

import msgspec
from architecture import log
from architecture.utils.creators import DynamicInstanceCreator

from intellibricks.llms.base import TranscriptionModel, TtsModel
from intellibricks.llms.base.contracts import LanguageModel
from intellibricks.llms.integrations.cerebras.cerebras import CerebrasLanguageModel
from intellibricks.llms.integrations.deepinfra import DeepInfraLanguageModel
from intellibricks.llms.integrations.google import GoogleLanguageModel
from intellibricks.llms.integrations.groq import (
    GroqLanguageModel,
    GroqTranscriptionModel,
)
from intellibricks.llms.integrations.ollama import OllamaLanguageModel
from intellibricks.llms.integrations.openai import (
    OpenAILikeLanguageModel,
    OpenAITranscriptionModel,
    OpenAITtsModel,
)
from intellibricks.llms.types import AIModel, TranscriptionModelType, TtsModelType

debug_logger = log.create_logger(__name__, level=logging.DEBUG)


class ModelRegistry(msgspec.Struct, frozen=True):
    """
    Registry for managing model mappings and provider-specific settings.

    Provides pattern-based registration of models to their respective
    implementation classes, eliminating the need for large dictionaries.
    """

    _language_model_patterns: ClassVar[
        Sequence[tuple[str, type[LanguageModel], dict[str, Any]]]
    ] = [
        # Pattern tuple format: (prefix, model_class, extra_params_dict)
        ("google/genai/", GoogleLanguageModel, {"vertexai": False}),
        ("google/vertexai/", GoogleLanguageModel, {"vertexai": True}),
        ("groq/api/", GroqLanguageModel, {}),
        ("openai/api/", OpenAILikeLanguageModel, {}),
        ("deepinfra/api/", DeepInfraLanguageModel, {}),
        ("cerebras/api/", CerebrasLanguageModel, {}),
        ("ollama/api/", OllamaLanguageModel, {}),  # new ollama models
    ]

    _transcription_model_patterns: ClassVar[
        Sequence[tuple[str, type[TranscriptionModel]]]
    ] = [
        ("groq/api/whisper", GroqTranscriptionModel),
        ("groq/api/distil-whisper", GroqTranscriptionModel),
        ("openai/api/whisper", OpenAITranscriptionModel),
    ]

    _tts_model_patterns: ClassVar[Sequence[tuple[str, type[TtsModel]]]] = [
        ("openai/api/tts", OpenAITtsModel),
    ]

    @classmethod
    def get_language_model_class(
        cls, model_id: str
    ) -> tuple[type[LanguageModel], Mapping[str, Any]]:
        """
        Find the appropriate language model class based on model ID pattern matching.

        Args:
            model_id: The model identifier string

        Returns:
            tuple: (model_class, extra_params)

        Raises:
            ValueError: If no matching pattern is found
        """
        for prefix, model_class, extra_params in cls._language_model_patterns:
            if model_id.startswith(prefix):
                return model_class, extra_params

        raise ValueError(f"No model class found for model ID: {model_id}")

    @classmethod
    def get_transcription_model_class(cls, model_id: str) -> type[TranscriptionModel]:
        """
        Find the appropriate transcription model class based on model ID pattern matching.

        Args:
            model_id: The model identifier string

        Returns:
            TranscriptionModel class

        Raises:
            ValueError: If no matching pattern is found
        """
        for prefix, model_class in cls._transcription_model_patterns:
            if model_id.startswith(prefix):
                return model_class

        raise ValueError(f"No transcription model class found for model ID: {model_id}")

    @classmethod
    def get_tts_model_class(cls, model_id: str) -> type[TtsModel]:
        """
        Find the appropriate TTS model class based on model ID pattern matching.

        Args:
            model_id: The model identifier string

        Returns:
            TtsModel class

        Raises:
            ValueError: If no matching pattern is found
        """
        for prefix, model_class in cls._tts_model_patterns:
            if model_id.startswith(prefix):
                return model_class

        raise ValueError(f"No TTS model class found for model ID: {model_id}")


class LanguageModelFactory:
    """
    Factory for creating language model instances based on model identifiers.

    Uses pattern matching from ModelRegistry to determine the appropriate
    model class and configuration.
    """

    @classmethod
    def create(
        cls, model: AIModel | str, params: dict[str, Any] | None = None
    ) -> LanguageModel:
        """
        Create a language model instance based on the model identifier.

        Args:
            model: The model identifier
            params: Optional parameters to pass to the model constructor

        Returns:
            LanguageModel: An instance of the appropriate language model
        """
        debug_logger.debug(f"Creating model: {model}")

        model_class, model_extra_params = ModelRegistry.get_language_model_class(model)

        # Merge params, with passed params taking precedence
        effective_params: dict[str, Any] = {}
        if model_extra_params:
            effective_params.update(model_extra_params)
        if params:
            effective_params.update(params)

        return DynamicInstanceCreator(cls=model_class).create_instance(
            **effective_params
        )


class TranscriptionModelFactory:
    """
    Factory for creating transcription model instances based on model identifiers.

    Uses pattern matching from ModelRegistry to determine the appropriate model class.
    """

    @classmethod
    def create(
        cls,
        model: TranscriptionModelType,
        params: dict[str, Any] | None = None,
    ) -> TranscriptionModel:
        """
        Create a transcription model instance based on the model identifier.

        Args:
            model: The transcription model identifier
            params: Optional parameters to pass to the model constructor

        Returns:
            TranscriptionModel: An instance of the appropriate transcription model
        """
        debug_logger.info(f"Creating transcription model: {model}")

        model_class = ModelRegistry.get_transcription_model_class(str(model))
        return DynamicInstanceCreator(cls=model_class).create_instance(**(params or {}))


class TtsModelFactory:
    """
    Factory for creating text-to-speech model instances based on model identifiers.

    Uses pattern matching from ModelRegistry to determine the appropriate model class.
    """

    @classmethod
    def create(
        cls, model: TtsModelType, params: dict[str, Any] | None = None
    ) -> TtsModel:
        """
        Create a text-to-speech model instance based on the model identifier.

        Args:
            model: The TTS model identifier
            params: Optional parameters to pass to the model constructor

        Returns:
            TtsModel: An instance of the appropriate TTS model
        """
        debug_logger.info(f"Creating TTS model: {model}")

        model_class = ModelRegistry.get_tts_model_class(str(model))
        return DynamicInstanceCreator(cls=model_class).create_instance(**(params or {}))
