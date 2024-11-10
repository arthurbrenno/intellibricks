"""LLM engines module"""

from __future__ import annotations

import abc
import asyncio
import typing
import uuid

import aiocache
import msgspec
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from google.oauth2 import service_account
from langfuse import Langfuse
from langfuse.client import (
    StatefulClient,
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import LLM
from weavearc import BaseModel, DynamicDict, Meta
from weavearc.logging import logger
from weavearc.utils.creators import DynamicInstanceCreator

from intellibricks import util
from intellibricks.rag.contracts import RAGQueriable

from .config import CacheConfig
from .constants import (
    AIModel,
    FinishReason,
    MessageRole,
)
from .exceptions import MaxRetriesReachedException
from .schema import (
    CompletionMessage,
    CompletionOutput,
    CompletionTokensDetails,
    Message,
    MessageChoice,
    Prompt,
    PromptTokensDetails,
    Tag,
    Usage,
)
from .types import TraceParams
from .util import count_tokens


@typing.runtime_checkable
class CompletionEngineProtocol(typing.Protocol):
    """
    Interface for AI Completion Engines.

    This interface defines the contract for AI model completions with fallback capability
    and retry management.
    """

    @abc.abstractmethod
    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Optional[
            typing.Union[dict[str, typing.Any], typing.Type[BaseModel]]
        ] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        stream: typing.Optional[bool] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[typing.Any]: ...

    @abc.abstractmethod
    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[
            typing.Union[dict[str, typing.Any], typing.Type[BaseModel]]
        ] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        stream: typing.Optional[bool] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[typing.Any]: ...


class ObservableCompletionEngine(CompletionEngineProtocol):
    """
    Provider for AI model completions with fallback capability retry management.

    This class manages the process of generating completions using various AI models,
    handling API keys, and providing fallback options if the primary model fails.
    """

    vertex_credentials: typing.Annotated[
        typing.Optional[
            service_account.Credentials
        ],  # Nota: Ou voce tira esse trem da interface, o que nao faz nem sentido, ou refatora essa classe pra receber algum outro tipo de objeto que faz mais sentido em receber as credenciais por construtor. Alem disso, deixei essas credenciais opcionais agora
        Meta(
            title="Google Cloud Credentials",
            description="The Google Cloud credentials to use for models that use Google Cloud services",
        ),
    ]

    langfuse: typing.Annotated[
        Langfuse,
        Meta(
            title="Langfuse",
            description="The Langfuse instance to use for generating observable completions",
        ),
    ]

    json_encoder: typing.Annotated[
        msgspec.json.Encoder,
        Meta(
            title="JSON Encoder",
            description="The JSON encoder to use for serializing structured responses",
        ),
    ]

    json_decoder: typing.Annotated[
        msgspec.json.Decoder,
        Meta(
            title="JSON Decoder",
            description="The JSON decoder to use for deserializing structured responses",
        ),
    ]

    def __init__(
        self,
        *,
        langfuse: Langfuse,
        json_encoder: msgspec.json.Encoder,
        json_decoder: msgspec.json.Decoder,
        vertex_credentials: typing.Optional[service_account.Credentials] = None,
    ) -> None:
        """
        Initialize the ObservableCompletionEngine.

        Args:
            langfuse: The Langfuse instance for generating observable completions
            json_encoder: JSON encoder for serializing structured responses
            json_decoder: JSON decoder for deserializing structured responses
            vertex_credentials: Optional Google Cloud credentials for models using Google Cloud services
        """
        self.langfuse = langfuse
        self.json_encoder = json_encoder
        self.json_decoder = json_decoder
        self.vertex_credentials = vertex_credentials

    async def complete_async(
        self,
        *,
        prompt: typing.Union[str, Prompt],
        system_prompt: typing.Optional[typing.Union[str, Prompt]] = None,
        response_format: typing.Optional[
            typing.Union[dict[str, typing.Any], typing.Type[BaseModel]]
        ] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        stream: typing.Optional[bool] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[typing.Any]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer in the same language the user asked."

        if isinstance(prompt, Prompt):
            prompt = prompt.content

        if isinstance(system_prompt, Prompt):
            system_prompt = system_prompt.content

        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=prompt),
        ]

        output: CompletionOutput[typing.Any] = await self.chat_async(
            messages=messages,
            response_format=response_format,
            model=model,
            fallback_models=fallback_models,
            n=n,
            temperature=temperature,
            stream=stream,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache_config=cache_config,
            trace_params=trace_params,
            postergate_token_counting=postergate_token_counting,
        )

        return output

    async def chat_async(
        self,
        *,
        messages: list[Message],
        response_format: typing.Optional[
            typing.Union[dict[str, typing.Any], typing.Type[BaseModel]]
        ] = None,
        model: typing.Optional[AIModel] = None,
        fallback_models: typing.Optional[list[AIModel]] = None,
        n: typing.Optional[int] = None,
        temperature: typing.Optional[float] = None,
        stream: typing.Optional[bool] = None,
        max_tokens: typing.Optional[int] = None,
        max_retries: typing.Optional[typing.Literal[1, 2, 3, 4, 5]] = None,
        cache_config: typing.Optional[CacheConfig] = None,
        trace_params: typing.Optional[TraceParams] = None,
        postergate_token_counting: bool = True,
        tools: typing.Optional[list[typing.Callable]] = None,
        data_stores: typing.Optional[typing.Sequence[RAGQueriable]] = None,
        web_search: typing.Optional[bool] = None,
    ) -> CompletionOutput[typing.Any]:
        """
        Asynchronously generate a chat completion using the configured models.

        This method attempts to use the main model first, then falls back to other models
        if necessary. It iterates through available API keys for each model until a
        successful completion is generated or all options are exhausted.

        Args:
            messages (list[Message]): The input messages for the chat completion.
            response_format (typing.Optional[typing.Union[json_schema_t, typing.Type[BaseModel]]]): The desired format for the response.

        Returns:
            CompletionOutput: The generated completion output.

        Raises:
            OutOfAPIKeysException: If all API keys for all models have been exhausted without success.
        """
        if trace_params is None:
            trace_params = {}

        if cache_config is None:
            cache_config = CacheConfig.from_defaults()

        # An unique ID for the completion
        completion_id: uuid.UUID = uuid.uuid4()

        trace: StatefulTraceClient = self.langfuse.trace(
            id=completion_id.__str__(), input=messages, **trace_params
        )

        choices: list[MessageChoice] = []

        if model is None:
            model = AIModel.VERTEX_GEMINI_1P5_FLASH_002

        if fallback_models is None:
            fallback_models = []

        if n is None:
            n = 1

        if temperature is None:
            temperature = 0.7

        if stream is None:
            stream = False

        if max_tokens is None:
            max_tokens = 5000

        if max_retries is None:
            max_retries = 1

        models: list[AIModel] = [model] + fallback_models

        logger.info(
            f"Starting chat completion. Main model: {model}, Fallback models: {fallback_models}"
        )

        for model in models:
            for retry in range(max_retries):
                try:
                    span_id: str = f"sp-{completion_id}-{retry}"
                    span: StatefulSpanClient = trace.span(
                        id=span_id,
                        input=messages,
                        name="Geração de Resposta",
                    )
                    choices, usage = await self._agen_choices(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        fallback_models=fallback_models,
                        n=n,
                        temperature=temperature,
                        stream=stream,
                        max_tokens=max_tokens,
                        cache_config=cache_config,
                        trace=trace,
                        span=span,
                        postergate_token_counting=postergate_token_counting,
                    )

                    logger.info(
                        f"Successfully generated completion with model {model} in retry {retry}"
                    )

                    output: CompletionOutput[typing.Any] = CompletionOutput(
                        id=completion_id,
                        model=model,
                        choices=choices,
                        usage=usage,
                    )

                    span.end(output=output.choices)

                    span.score(
                        id=f"sc-{span}",
                        name="Sucesso",
                        value=1.0,
                        comment="Escolhas geradas com sucesso!",
                    )

                    trace.update(output=output.choices)
                    return output
                except Exception as e:
                    # Log the error in span and continue to the next one
                    span.end(output={})
                    span.update(status_message="Erro na geração.", level="ERROR")
                    span.score(
                        id=f"sc-{span}",
                        name="Sucesso",
                        value=0.0,
                        comment=f"Erro ao gerar escolhas: {e}",
                    )
                    logger.error(
                        "An error ocurred in retry {retry}: {e}. CONTINUING...",
                        retry=retry,
                        e=e,
                    )
                    continue

        raise MaxRetriesReachedException()

    async def _agen_choices(
        self,
        *,
        model: AIModel,
        messages: list[Message],
        fallback_models: list[AIModel],
        n: int,
        temperature: float,
        stream: bool,
        max_tokens: int,
        trace: StatefulTraceClient,
        span: StatefulSpanClient,
        cache_config: CacheConfig,
        postergate_token_counting: bool,
        response_format: typing.Optional[typing.Union[dict, typing.Type[BaseModel]]],
    ) -> typing.Tuple[list[MessageChoice[typing.Any]], Usage]:
        """
        Internal method to generate chat completions for a specific model.

        Args:
            model (AIModel): The AI model to use for completion.
            messages (list[Message]): The input messages for the chat completion.
            fallback_models (list[AIModel]): A list of fallback models.
            n (int): The number of completions to generate.
            temperature (float): Sampling temperature.
            stream (bool): Whether to stream the output.
            max_tokens (int): Maximum number of tokens to generate.
            trace (StatefulTraceClient): The trace object for observability.
            span (StatefulSpanClient): The span object for observability.
            cache_config (CacheConfig): Configuration for caching.
            postergate_token_counting (bool): If True, postpone token counting to a background task.
            response_format (typing.Optional[typing.Union[dict, typing.Type[BaseModel]]]): The desired format for the response.

        Returns:
            typing.Tuple[list[MessageChoice[typing.Any]], Usage]: A tuple containing a list of generated message choices and a Usage object with token usage statistics.
        """
        # Initialize variables
        choices: list[MessageChoice] = []
        model_input_cost, model_output_cost = model.ppm()
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0
        total_input_cost: float = 0.0
        total_output_cost: float = 0.0

        # Assume llm is obtained as before
        llm: LLM = await self._get_cached_llm(
            model=model,
            max_tokens=max_tokens,
            cache_config=cache_config,
        )

        for i in range(n):
            current_messages = messages.copy()

            if response_format is not None:
                current_messages = self._append_response_format_to_prompt(
                    messages=current_messages,
                    response_format=response_format,
                )

            generation: StatefulGenerationClient = span.generation(
                id=f"gen-{uuid.uuid4()}-{i}",
                model=model.value,
                input=current_messages,
                model_parameters={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )

            chat_response: ChatResponse = await llm.achat(
                messages=[
                    message.to_llama_index_chat_message()
                    for message in current_messages
                ]
            )

            logger.debug(
                f"Received AI response from model {model.value}: {chat_response.message.content}"
            )

            generation.end(
                output=chat_response.message.content,
            )

            # Always call _calculate_token_usage
            usage_future = self._calculate_token_usage(
                model=model,
                messages=current_messages,
                chat_response=chat_response,
                generation=generation,
                span=span,
                index=i,
                model_input_cost=model_input_cost,
                model_output_cost=model_output_cost,
            )

            if postergate_token_counting:
                # Schedule background task for token counting and cost calculation
                asyncio.create_task(usage_future)
            else:
                # Perform token counting and cost calculation immediately
                usage = await usage_future

                # Accumulate total usage
                total_prompt_tokens += usage.prompt_tokens or 0
                total_completion_tokens += usage.completion_tokens or 0
                total_input_cost += usage.input_cost or 0.0
                total_output_cost += usage.output_cost or 0.0

            completion_message: CompletionMessage = CompletionMessage(
                role=MessageRole(chat_response.message.role.value),
                content=chat_response.message.content,
                parsed=self._get_parsed(
                    response_format,
                    chat_response.message.content,
                    trace=trace,
                    span=span,
                ),
            )

            choices.append(
                MessageChoice(
                    index=i,
                    message=completion_message,
                    logprobs=chat_response.logprobs,
                    finish_reason=FinishReason.NONE,
                )
            )
            logger.info(f"Successfully generated choice {i+1} for model {model.value}")

        # After the loop, set the usage
        if postergate_token_counting:
            # Since token counting is postponed, usage variables remain zero
            usage = Usage(
                prompt_tokens=None,
                completion_tokens=None,
                input_cost=None,
                output_cost=None,
                total_cost=None,
                total_tokens=None,
                prompt_tokens_details=PromptTokensDetails(
                    audio_tokens=None, cached_tokens=None
                ),
                completion_tokens_details=CompletionTokensDetails(
                    audio_tokens=None, reasoning_tokens=None
                ),
            )
        else:
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                input_cost=total_input_cost,
                output_cost=total_output_cost,
                total_cost=total_input_cost + total_output_cost,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    audio_tokens=None, cached_tokens=None
                ),
                completion_tokens_details=CompletionTokensDetails(
                    audio_tokens=None, reasoning_tokens=None
                ),
            )

        return choices, usage

    async def _calculate_token_usage(
        self,
        *,
        model: AIModel,
        messages: list[Message],
        chat_response: ChatResponse,
        generation: StatefulGenerationClient,
        span: StatefulSpanClient,
        index: int,
        model_input_cost: float,
        model_output_cost: float,
    ) -> Usage:
        """
        Perform token counting and cost calculation.

        This method can be called synchronously or scheduled as a background task.
        It calculates the number of tokens used in the prompt and completion,
        computes the associated costs, and updates the observability data.

        Args:
            model (AIModel): The AI model used for completion.
            messages (list[Message]): The input messages for the chat completion.
            chat_response (ChatResponse): The response from the LLM.
            generation (StatefulGenerationClient): The generation object for observability.
            span (StatefulSpanClient): The parent span for observability.
            index (int): The index of the current choice in the loop.
            model_input_cost (float): The per-token input cost of the model.
            model_output_cost (float): The per-token output cost of the model.

        Returns:
            Usage: An object containing token usage statistics.
        """
        # Perform token counting
        prompt_counting_span: StatefulSpanClient = span.span(
            id=f"sp-prompt-{span.id}-{index}",
            name="Contagem de Tokens",
            input={
                "mensagens": [
                    message.as_dict(encoder=self.json_encoder) for message in messages
                ]
                + [chat_response.model_dump()]
            },
        )

        prompt_tokens = sum(
            count_tokens(model=model, text=msg.content or "") for msg in messages
        )

        completion_tokens = count_tokens(
            model=model, text=chat_response.message.content
        )

        prompt_counting_span.end(
            output={
                "model": model.value,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

        # Perform cost calculation
        prompt_cost_span: StatefulSpanClient = span.span(
            id=f"sp-sum-prompt-{span.id}-{index}",
            name="Determinando preço dos tokens",
            input={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_input_cost": model_input_cost,
                "model_output_cost": model_output_cost,
            },
        )

        scale = 1 / 1_000_000

        # Compute input and output costs
        completion_input_cost = round(prompt_tokens * model_input_cost * scale, 5)
        completion_output_cost = round(completion_tokens * model_output_cost * scale, 5)

        prompt_cost_span.end(
            output={
                "prompt_cost": completion_input_cost,
                "completion_cost": completion_output_cost,
            }
        )

        # Update the generation with usage
        generation.update(
            usage={
                "input": prompt_tokens,
                "output": completion_tokens,
                "unit": "TOKENS",
                "input_cost": completion_input_cost,
                "output_cost": completion_output_cost,
            }
        )

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_cost=completion_input_cost,
            output_cost=completion_output_cost,
            total_cost=completion_input_cost + completion_output_cost,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None, cached_tokens=None
            ),
            completion_tokens_details=CompletionTokensDetails(
                audio_tokens=None, reasoning_tokens=None
            ),
        )

        return usage

    def _get_parsed(
        self,
        response_format: typing.Optional[
            typing.Union[dict[str, typing.Any], typing.Type[BaseModel]]
        ],
        content: str,
        trace: StatefulTraceClient,
        span: StatefulSpanClient,
    ) -> typing.Optional[BaseModel]:
        if response_format is None:
            logger.debug("Response format is None")
            return None

        if isinstance(response_format, dict):
            LLMResponse: typing.Type[BaseModel] = util.get_struct_from_schema(
                response_format, bases=(BaseModel,), name="ResponseModel"
            )

            response_format = LLMResponse

        tag: typing.Optional[Tag] = Tag.from_string(
            content, tag_name="structured"
        ) or Tag.from_string(content, tag_name="output")

        if tag is None:
            event: StatefulClient = span.event(  # NOQA: F841
                id=f"ev-{trace.id}",
                name="Obtendo resposta estruturada",
                input=content,
                output=None,
                level="ERROR",
                metadata={"response_format": response_format, "content": content},
            )
            return None

        structured: dict[str, typing.Any] = tag.as_object()

        # logger.debug("EXTRACTED Tag: {tag}", tag=tag)
        # logger.debug("Tag content: {tag_content}", tag_content=tag.content)
        # logger.debug("Structured content: {structured}", structured=structured)

        model: typing.Optional[BaseModel] = (
            response_format.from_dict(structured, encoder=self.json_encoder)
            if structured
            else None
        )

        event = span.event(  # NOQA: F841
            id=f"ev-{trace.id}",
            name="Obtendo resposta estruturada",
            input=f"<structured>\n{tag.content}\n</structured>",
            output=model,
            level="DEBUG",
            metadata={"response_format": response_format, "content": content},
        )

        return model

    def _append_response_format_to_prompt(
        self,
        *,
        messages: list[Message],
        response_format: typing.Union[dict[str, typing.Any], typing.Type[BaseModel]],
        prompt_role: typing.Optional[MessageRole] = None,
    ) -> list[Message]:
        """
        Appends the desired response format (output) to the specified prompt role.
        By default, it appends to the System Prompt, which is generally the first prompt sent to the AI model.

        :param skip_if_output_exists: If True, skips appending the new <saida> if it already exists in typing.any previous message.
        """

        if prompt_role is None:
            prompt_role = MessageRole.SYSTEM

        match response_format:
            case type() if issubclass(response_format, BaseModel):
                basemodel_schema = response_format.to_json_schema()
            case dict():
                basemodel_schema = response_format

        new_prompt: str = f"""
        <saida>
            Dentro de uma tag "<structured>" a assistente irá retornar uma saída, formatada em JSON, que esteja de acordo com o seguinte esquema JSON:
            <json_schema>
            {basemodel_schema}
            </json_schema>
            O JSON retornado pela assistente, dentro da tag, deve estar de acordo com o esquema mencionado acima e deve levar em conta as instruções dadas na tarefa estipulada. A assistente deve fechar a tag com </structured>.
        </saida>
        """

        # Find the message with the specified role
        for message in messages:
            if message.content is None:
                message.content = new_prompt
                continue

            if message.role == prompt_role:
                message.content += new_prompt
                return messages

        # If no message with the specified role is found, insert a new message
        messages.append(Message(role=prompt_role, content=new_prompt))

        return messages

    @aiocache.cached(ttl=3600)  # type: ignore[misc]
    async def _get_cached_llm(
        self,
        model: AIModel,
        max_tokens: int,
        cache_config: CacheConfig,
    ) -> LLM:
        """
        Get or create a cached LLM instance for a specific model.

        Args:
            model (AIModel): The AI model to use.
            max_tokens (int): The maximum number of tokens to generate.
            vertex_credentials (service_account.Credentials): The Google Cloud credentials to use (if a gcloud model will be used).

        Returns:
            LLM: A LlamaIndex LLM instance.
        """

        constructor_params: dict[str, typing.Any] = (
            DynamicDict.having(
                "max_tokens",
                equals_to=max_tokens,
            )
            .as_well_as("model_name", equals_to=model.value)
            .also(
                "project",
                equals_to=self.vertex_credentials.project_id
                if self.vertex_credentials is not None
                else None,
            )
            .also("model", equals_to=model.value)
            .also(
                "credentials",
                equals_to=self.vertex_credentials
                if self.vertex_credentials is not None
                else None,
            )
            .also(
                "safety_settings",
                equals_to={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            .also("cache_config", equals_to=cache_config)
            .also("timeout", equals_to=120)
            .at_last("generate_kwargs", equals_to={"timeout": 120})
        )

        return DynamicInstanceCreator.create_instance(
            AIModel.get_llama_index_model_cls(model), **constructor_params
        )
