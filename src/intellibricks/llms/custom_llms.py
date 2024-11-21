"""
Adapters module. Provides a set of adapters for various AI providers. Like
Vertex AI context caching with Google Cloud AI Platform.
"""

from __future__ import annotations

import datetime
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.vertex import Vertex
from vertexai.generative_models import ChatSession, Content, GenerationResponse, Part
from vertexai.generative_models._generative_models import (
    SafetySettingsType,
    _GenerativeModel,
)
from vertexai.preview import caching
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel
from architecture.logging import LoggerFactory

from .config import CacheConfig
from .runtime_mappings import CACHE_KEY_TO_ID

logger = LoggerFactory.create(__name__)


class EnhancedVertexAI(Vertex):
    """Vertex AI Adapter for LLM."""

    cache_config: CacheConfig = Field(
        description="The cache configuration for the model.", default=None
    )

    safety_settings: Optional[dict] = Field(
        description="Safety settings for the model.", default=None
    )

    def __init__(
        self,
        model: str = "text-bison",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[Any] = None,
        examples: Optional[Sequence[ChatMessage]] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 10,
        iscode: bool = False,
        safety_settings: Optional[SafetySettingsType] = None,
        additional_kwargs: Optional[dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__(
            model=model,
            project=project,
            location=location,
            credentials=credentials,
            examples=examples,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            iscode=iscode,
            safety_settings=safety_settings,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        cfg = cache_config or CacheConfig.from_defaults()

        self.cache_config = cfg

    @llm_chat_callback()  # type: ignore[misc]
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        system_instruction: str = next(
            (
                message.content or ""
                for message in messages
                if message.role == MessageRole.SYSTEM
            ),
            "",
        )

        cache_is_enabled: bool = self.cache_config.enabled
        if cache_is_enabled:
            try:
                model: _GenerativeModel = self._get_cached_model(
                    system_instruction=system_instruction,
                    ttl=self.cache_config.ttl,
                )
            except Exception as e:
                logger.error(f"Failed to load GenerativeModel from cache. Error: {e}")
                model = GenerativeModel(
                    model_name=self.model,
                    safety_settings=self.safety_settings,
                    system_instruction=system_instruction,
                    generation_config=GenerationConfig(
                        temperature=self.temperature, max_output_tokens=self.max_tokens
                    ),
                )
        else:
            model = GenerativeModel(
                model_name=self.model,
                safety_settings=self.safety_settings,
                system_instruction=system_instruction,
                generation_config=GenerationConfig(
                    temperature=self.temperature, max_output_tokens=self.max_tokens
                ),
            )

        last_message = messages[-1]

        chat_history: Sequence[Content] = [
            Content(
                role=message.role.value, parts=[Part.from_text(str(message.content))]
            )
            for message in filter(
                lambda message: message.role != MessageRole.SYSTEM, messages[:-1]
            )
        ]

        chat_session: ChatSession = model.start_chat(
            history=chat_history,
            response_validation=False,
        )

        generation: GenerationResponse = await chat_session.send_message_async(
            content=Part.from_text(str(last_message.content))
        )

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=generation.text)
        )

    def _get_cached_model(
        self, system_instruction: str, ttl: datetime.timedelta
    ) -> _GenerativeModel:
        cache_key = self.cache_config.cache_key
        logger.debug(f"Attempting to retrieve cache ID for cache_key: {cache_key}")

        system_instruction_content = Content(
            role="system", parts=[Part.from_text(system_instruction)]
        )

        google_cache_id = CACHE_KEY_TO_ID.get(cache_key, None)
        logger.debug(f"Current CACHE_KEY_TO_ID state: {cache_key} -> {google_cache_id}")

        if google_cache_id is None:
            logger.debug(
                f"Cache key '{cache_key}' not found in CACHE_KEY_TO_ID. Creating new CachedContent.",
            )

            cached_content: caching.CachedContent = caching.CachedContent.create(
                model_name=self.model,
                system_instruction=system_instruction_content,
                contents=[Content(parts=[Part.from_text(" ")])],
                ttl=ttl,
                display_name=cache_key,
            )

            logger.debug(f"Created new CachedContent with name: {cached_content.name}")

            # Store the cached content name in the container
            CACHE_KEY_TO_ID[cache_key] = cached_content.name
            logger.debug(
                f"Updated CACHE_KEY_TO_ID with {cache_key} -> {cached_content.name}"
            )

        try:
            logger.debug(
                f"Attempting to retrieve CachedContent using cache_key: {cache_key}"
            )

            cached_content_name = CACHE_KEY_TO_ID[cache_key]

            if cached_content_name is None:
                raise ValueError("Cached content name is None.")

            cached_content = caching.CachedContent(
                cached_content_name=cached_content_name,
            )

            # Reset expire_time to now + ttl
            new_expire_time = datetime.datetime.utcnow() + ttl

            logger.debug(f"Current expire_time: {cached_content.expire_time}")
            logger.debug(f"Updating expire_time to {new_expire_time}")
            # cached_content.update(expire_time=new_expire_time)

            logger.debug(f"Successfully retrieved CachedContent: {cached_content.name}")

        except Exception as e:
            logger.exception(e)
            logger.debug(
                f"Failed to retrieve CachedContent for cache_key: {cache_key}. Error: {e}"
            )

            # Cache ID is stored in the container, but probably failed because it expired
            CACHE_KEY_TO_ID[cache_key] = None
            logger.debug(
                f"Set CACHE_KEY_TO_ID[{cache_key}] to None due to retrieval failure"
            )

            # Create a new cache ID
            logger.debug(f"Creating new CachedContent for cache_key: {cache_key}")

            cached_content = caching.CachedContent.create(
                model_name=self.model,
                system_instruction=system_instruction_content,
                contents=[Content(parts=[Part.from_text(" ")])],
                ttl=ttl,
                display_name=cache_key,
            )

            logger.debug(f"Created new CachedContent with name: {cached_content.name}")

            # Store the new cached content name in the container
            CACHE_KEY_TO_ID[cache_key] = cached_content.name
            logger.debug(
                f"Updated CACHE_KEY_TO_ID with new cache ID: {cache_key} -> {cached_content.name}"
            )

        logger.debug(
            f"Loading GenerativeModel from CachedContent: {cached_content.name}"
        )

        model: _GenerativeModel = GenerativeModel.from_cached_content(
            cached_content=cached_content, safety_settings=self.safety_settings
        )

        logger.debug(
            f"Successfully loaded GenerativeModel from cache. Model details: {model}"
        )

        return model
