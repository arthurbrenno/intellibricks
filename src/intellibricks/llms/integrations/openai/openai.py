import timeit
from typing import Literal, Optional, Sequence, TypeVar, cast, overload, override, Any

import msgspec
from architecture.utils.decorators import ensure_module_installed
from langfuse.client import os
from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion as OpenAIChatCompletion,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from intellibricks.llms.base.contracts import LanguageModel
from intellibricks.llms.constants import FinishReason
from intellibricks.llms.schema import (
    GeneratedAssistantMessage,
    CalledFunction,
    ChatCompletion,
    CompletionTokensDetails,
    Function,
    Message,
    MessageChoice,
    Part,
    PromptTokensDetails,
    RawResponse,
    ToolCall,
    ToolCallSequence,
    Usage,
    ToolInputType,
)
from intellibricks.llms.types import OpenAIModelType
from intellibricks.llms.util import (
    create_function_mapping_by_tools,
    get_parsed_response,
)
from intellibricks.util import flatten_msgspec_schema
from openai.types.chat_model import ChatModel

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)

MODEL_PRICING: dict[ChatModel, dict[Literal["input_cost", "output_cost"], float]] = {
    "o1": {"input_cost": 15.00, "output_cost": 60.00},
    "o1-2024-12-17": {"input_cost": 15.00, "output_cost": 60.00},
    "o1-preview": {"input_cost": 15.00, "output_cost": 60.00},
    "o1-preview-2024-09-12": {"input_cost": 15.00, "output_cost": 60.00},
    "o1-mini": {"input_cost": 3.00, "output_cost": 12.00},
    "o1-mini-2024-09-12": {"input_cost": 3.00, "output_cost": 12.00},
    "gpt-4o": {"input_cost": 2.50, "output_cost": 10.00},
    "gpt-4o-2024-11-20": {"input_cost": 2.50, "output_cost": 10.00},
    "gpt-4o-2024-08-06": {"input_cost": 2.50, "output_cost": 10.00},
    "gpt-4o-2024-05-13": {"input_cost": 5.00, "output_cost": 15.00},
    "gpt-4o-audio-preview": {
        "input_cost": 2.50,
        "output_cost": 10.00,
    },  # Text pricing
    "gpt-4o-audio-preview-2024-10-01": {
        "input_cost": 2.50,
        "output_cost": 10.00,
    },  # Text pricing
    "gpt-4o-audio-preview-2024-12-17": {
        "input_cost": 2.50,
        "output_cost": 10.00,
    },  # Text pricing
    "gpt-4o-mini-audio-preview": {
        "input_cost": 0.150,
        "output_cost": 0.600,
    },  # Text pricing
    "gpt-4o-mini-audio-preview-2024-12-17": {
        "input_cost": 0.150,
        "output_cost": 0.600,
    },  # Text pricing
    "chatgpt-4o-latest": {"input_cost": 5.00, "output_cost": 15.00},
    "gpt-4o-mini": {"input_cost": 0.150, "output_cost": 0.600},
    "gpt-4o-mini-2024-07-18": {"input_cost": 0.150, "output_cost": 0.600},
    "gpt-4-turbo": {"input_cost": 10.00, "output_cost": 30.00},
    "gpt-4-turbo-2024-04-09": {"input_cost": 10.00, "output_cost": 30.00},
    "gpt-4-0125-preview": {"input_cost": 10.00, "output_cost": 30.00},
    "gpt-4-turbo-preview": {"input_cost": 10.00, "output_cost": 30.00},
    "gpt-4-1106-preview": {"input_cost": 10.00, "output_cost": 30.00},
    "gpt-4-vision-preview": {"input_cost": 10.00, "output_cost": 30.00},
    "gpt-4": {"input_cost": 30.00, "output_cost": 60.00},
    "gpt-4-0314": {"input_cost": 30.00, "output_cost": 60.00},
    "gpt-4-0613": {"input_cost": 30.00, "output_cost": 60.00},
    "gpt-4-32k": {"input_cost": 60.00, "output_cost": 120.00},
    "gpt-4-32k-0314": {"input_cost": 60.00, "output_cost": 120.00},
    "gpt-4-32k-0613": {"input_cost": 60.00, "output_cost": 120.00},
    "gpt-3.5-turbo": {
        "input_cost": 1.50,
        "output_cost": 2.00,
    },  # Assuming this refers to gpt-3.5-turbo-0301 pricing
    "gpt-3.5-turbo-16k": {
        "input_cost": 3.00,
        "output_cost": 4.00,
    },  # Assuming this refers to gpt-3.5-turbo-16k-0613 pricing
    "gpt-3.5-turbo-0301": {"input_cost": 1.50, "output_cost": 2.00},
    "gpt-3.5-turbo-0613": {"input_cost": 1.50, "output_cost": 2.00},
    "gpt-3.5-turbo-1106": {"input_cost": 1.00, "output_cost": 2.00},
    "gpt-3.5-turbo-0125": {"input_cost": 0.50, "output_cost": 1.50},
    "gpt-3.5-turbo-16k-0613": {"input_cost": 3.00, "output_cost": 4.00},
}


class OpenAILanguageModel(LanguageModel, frozen=True):
    model_name: ChatModel
    api_key: Optional[str] = None
    max_retries: int = 2

    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: None = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[RawResponse]: ...
    @overload
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[S],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S]: ...

    @ensure_module_installed("openai", "openai")
    @override
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: Optional[type[S]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[ToolInputType]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[S] | ChatCompletion[RawResponse]:
        now = timeit.default_timer()
        client = AsyncOpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY", None),
            max_retries=self.max_retries,
        )

        openai_completion: OpenAIChatCompletion = await client.chat.completions.create(
            messages=[message.to_openai_format() for message in messages],
            model=self.model_name,
            audio=NOT_GIVEN,
            max_completion_tokens=max_completion_tokens,
            n=n or 1,
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchema(
                    name=response_model.__name__,
                    description="Structured response",
                    schema=flatten_msgspec_schema(
                        msgspec.json.schema(response_model), openai_like=True
                    ),
                    strict=True,
                ),
                type="json_schema",
            )
            if response_model
            else NOT_GIVEN,
            stop=list(stop_sequences) if stop_sequences else NOT_GIVEN,
            temperature=temperature,
            tools=[
                ChatCompletionToolParam(
                    function=Function.from_callable(tool).to_openai_function(),
                    type="function",
                )
                if callable(tool)
                else tool.to_openai_tool()
                for tool in tools
            ]
            if tools
            else NOT_GIVEN,
            top_p=top_p,
            timeout=timeout,
        )

        # Construct Choices
        choices: list[MessageChoice[S]] = []
        for choice in openai_completion.choices:
            message = choice.message

            openai_tool_calls: list[ChatCompletionMessageToolCall] = (
                message.tool_calls or []
            )

            tool_calls: list[ToolCall] = []
            functions: dict[str, Function] = create_function_mapping_by_tools(
                tools or []
            )

            for openai_tool_call in openai_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=openai_tool_call.id,
                        called_function=CalledFunction(
                            function=functions[openai_tool_call.function.name],
                            arguments=msgspec.json.decode(
                                openai_tool_call.function.arguments, type=dict[str, Any]
                            ),
                        ),
                    )
                )

            choices.append(
                MessageChoice(
                    index=choice.index,
                    message=GeneratedAssistantMessage(
                        contents=[Part.from_text(message.content or "")],
                        parsed=get_parsed_response(
                            message.content or "", response_model=response_model
                        )
                        if response_model
                        else cast(S, RawResponse()),
                        tool_calls=ToolCallSequence(tool_calls),
                    ),
                    logprobs=None,
                    finish_reason=FinishReason(choice.finish_reason),
                )
            )

        usage: Optional[CompletionUsage] = openai_completion.usage
        prompt_tokens_details = usage.prompt_tokens_details if usage else None
        completion_tokens_details = usage.completion_tokens_details if usage else None

        prompt_tokens: Optional[int] = usage.prompt_tokens if usage else None
        completion_tokens: Optional[int] = usage.completion_tokens if usage else None

        pricing = MODEL_PRICING.get(
            self.model_name, {"input_cost": 0.0, "output_cost": 0.0}
        )

        # Calculate input cost
        input_cost = (prompt_tokens or 0) / 1_000_000 * pricing.get("input_cost", 0.0)

        # Calculate output cost
        output_cost = (
            (completion_tokens or 0) / 1_000_000 * pricing.get("output_cost", 0.0)
        )

        # Calculate total cost
        total_cost = input_cost + output_cost

        prompt_tokens_details = usage.prompt_tokens_details if usage else None
        completion_tokens_details = usage.completion_tokens_details if usage else None

        chat_completion = ChatCompletion(
            elapsed_time=timeit.default_timer() - now,
            id=openai_completion.id,
            object=openai_completion.object,
            created=openai_completion.created,
            model=cast(OpenAIModelType, f"openai/api/{self.model_name}"),
            system_fingerprint=openai_completion.system_fingerprint or "fp_none",
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
                if completion_tokens is not None
                else None,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                total_tokens=usage.total_tokens if usage else None,
                prompt_tokens_details=PromptTokensDetails(
                    audio_tokens=prompt_tokens_details.audio_tokens,
                    cached_tokens=prompt_tokens_details.cached_tokens,
                )
                if prompt_tokens_details
                else None,
                completion_tokens_details=CompletionTokensDetails(
                    audio_tokens=completion_tokens_details.audio_tokens
                    if completion_tokens_details
                    else None,
                    reasoning_tokens=completion_tokens_details.reasoning_tokens
                    if completion_tokens_details
                    else None,
                )
                if completion_tokens_details
                else None,
            ),
        )

        return chat_completion
