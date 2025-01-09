import copy
import timeit
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
    override,
)

from architecture.utils.decorators import ensure_module_installed
import msgspec
from architecture.types import NOT_GIVEN as NOT_GIVEN_INTERNAL

from intellibricks.llms.constants import FinishReason
from intellibricks.llms.schema import (
    AssistantMessage,
    CalledFunction,
    ChatCompletion,
    Function,
    Message,
    MessageChoice,
    Part,
    RawResponse,
    ToolCall,
    ToolCallSequence,
    Usage,
)
from intellibricks.llms.types import CerebrasModelType
from intellibricks.llms.util import (
    get_function_name,
    get_new_messages_with_response_format_instructions,
    get_parsed_response,
)

from ...base import SupportsAsyncChat

T = TypeVar("T", bound=msgspec.Struct, default=RawResponse)
CerebrasModel: TypeAlias = Literal["llama3.1-8b", "llama3.1-70b", "llama-3.3-70b"]

MODEL_PRICING: dict[
    CerebrasModel, dict[Literal["input_cost", "output_cost"], float]
] = {
    "llama3.1-8b": {"input_cost": 0.1, "output_cost": 0.1},
    "llama3.1-70b": {"input_cost": 0.85, "output_cost": 1.2},
    "llama-3.3-70b": {"input_cost": 0.0, "output_cost": 0.0},
}
"""Model pricing per million tokens"""


@dataclass(frozen=True)
class CerebrasLanguageModel(SupportsAsyncChat):
    """Cerebras is the WORLD's fastest"""

    model_name: CerebrasModel = field(default_factory=lambda: "llama3.1-8b")
    api_key: Optional[str] = None
    max_retries: int = 2
    parallel_tool_calls: bool = True

    @ensure_module_installed("cerebras", "cerebras-cloud-sdk")
    @override
    async def chat_async(
        self,
        messages: Sequence[Message],
        *,
        response_model: Optional[type[T]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        timeout: Optional[float] = None,
    ) -> ChatCompletion[T] | ChatCompletion[RawResponse]:
        from cerebras.cloud.sdk import AsyncCerebras
        from cerebras.cloud.sdk.types.chat.chat_completion import (
            ChatCompletionResponse as CerebrasChatCompletion,
        )
        from cerebras.cloud.sdk.types.chat.chat_completion import (
            ChatCompletionResponseChoiceMessage,
            ChatCompletionResponseChoiceMessageToolCall,
            ChatCompletionResponseUsage,
        )
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            ResponseFormatTyped,
            ToolTyped,
        )

        now = timeit.default_timer()
        client = AsyncCerebras(
            api_key=self.api_key, timeout=timeout, max_retries=self.max_retries
        )

        new_messages = copy.copy(messages)
        if response_model is not None:
            new_messages = get_new_messages_with_response_format_instructions(
                messages=messages, response_model=response_model
            )

        cerebras_completion: CerebrasChatCompletion = cast(
            CerebrasChatCompletion,
            await client.chat.completions.create(
                messages=[message.to_cerebras_format() for message in new_messages],
                model=self.model_name,
                max_completion_tokens=max_completion_tokens,
                n=n,
                stop=list(stop_sequences) if stop_sequences else None,
                response_format=ResponseFormatTyped(type="json_object")
                if response_model
                else None,
                temperature=temperature,
                tools=[
                    ToolTyped(
                        function=Function.from_callable(tool).to_cerebras_function(),
                        type="tool",
                    )
                    for tool in tools
                ]
                if tools
                else None,
                parallel_tool_calls=True,
            ),
        )

        # Construct choices
        choices: list[MessageChoice[T]] = []
        for choice in cerebras_completion.choices:
            message: ChatCompletionResponseChoiceMessage = choice.message

            tool_calls: list[ToolCall] = []
            functions: dict[str, Function] = {
                get_function_name(function): Function.from_callable(function)
                for function in tools or []
            }

            cerebras_tool_calls: Sequence[
                ChatCompletionResponseChoiceMessageToolCall
            ] = message.tool_calls or []
            for cerebras_tool_call in cerebras_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=cerebras_tool_call.id,
                        called_function=CalledFunction(
                            function=functions[cerebras_tool_call.function.name],
                            arguments=msgspec.json.decode(
                                cerebras_tool_call.function.arguments, type=dict
                            ),
                        ),
                    )
                )

            choices.append(
                MessageChoice(
                    index=choice.index,
                    message=AssistantMessage(
                        contents=[Part.from_text(message.content or "")],
                        parsed=get_parsed_response(
                            message.content or "", response_model=response_model
                        )
                        if response_model
                        else cast(T, RawResponse()),
                        tool_calls=ToolCallSequence(tool_calls),
                    ),
                    logprobs=None,
                    finish_reason=FinishReason(choice.finish_reason),
                )
            )

        pricing = MODEL_PRICING.get(
            self.model_name, {"input_cost": 0.0, "output_cost": 0.0}
        )
        completion_usage: ChatCompletionResponseUsage = cerebras_completion.usage
        prompt_tokens = completion_usage.prompt_tokens
        completion_tokens = completion_usage.completion_tokens
        input_cost = (prompt_tokens or 0) / 1_000_000 * pricing.get("input_cost", 0.0)
        output_cost = (
            (completion_tokens or 0) / 1_000_000 * pricing.get("output_cost", 0.0)
        )

        completion = ChatCompletion(
            elapsed_time=timeit.default_timer() - now,
            id=cerebras_completion.id,
            object="chat.completion",
            created=cerebras_completion.created,
            model=cast(CerebrasModelType, f"cerebras/api/{self.model_name}"),
            choices=choices,
            usage=Usage(
                prompt_tokens=completion_usage.prompt_tokens or NOT_GIVEN_INTERNAL,
                completion_tokens=completion_usage.completion_tokens
                or NOT_GIVEN_INTERNAL,
                input_cost=input_cost,
                output_cost=output_cost,
                total_tokens=completion_usage.total_tokens or NOT_GIVEN_INTERNAL,
            ),
        )

        return completion
