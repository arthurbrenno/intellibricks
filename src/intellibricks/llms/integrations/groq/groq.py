import copy
import timeit
from typing import (
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
    overload,
    override,
)

import msgspec
from architecture.utils.decorators import ensure_module_installed

from intellibricks.llms.base.contracts import LanguageModel
from intellibricks.llms.constants import FinishReason
from intellibricks.llms.schema import (
    GeneratedAssistantMessage,
    CalledFunction,
    ChatCompletion,
    Function,
    Message,
    MessageChoice,
    Part,
    RawResponse,
    ToolCall,
    ToolCallSequence,
    ToolInputType,
    Usage,
)
from intellibricks.llms.types import GroqModelType
from intellibricks.llms.util import (
    _create_function_mapping_by_tools,
    get_new_messages_with_response_format_instructions,
    get_parsed_response,
)

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)

GroqModel: TypeAlias = Literal[
    "gemma2-9b-it",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-3.1-70b-specdec",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

MODEL_PRICING: dict[GroqModel, dict[Literal["input_cost", "output_cost"], float]] = {
    "llama-3.2-1b-preview": {"input_cost": 0.04, "output_cost": 0.04},
    "llama-3.2-3b-preview": {"input_cost": 0.06, "output_cost": 0.06},
    "llama-3.3-70b-versatile": {"input_cost": 0.59, "output_cost": 0.79},
    "llama-3.1-8b-instant": {"input_cost": 0.05, "output_cost": 0.08},
    "llama3-70b-8192": {"input_cost": 0.59, "output_cost": 0.79},
    "llama3-8b-8192": {"input_cost": 0.05, "output_cost": 0.08},
    "mixtral-8x7b-32768": {"input_cost": 0.24, "output_cost": 0.24},
    "gemma2-9b-it": {"input_cost": 0.20, "output_cost": 0.20},
    "llama3-groq-70b-8192-tool-use-preview": {"input_cost": 0.89, "output_cost": 0.89},
    "llama3-groq-8b-8192-tool-use-preview": {"input_cost": 0.19, "output_cost": 0.19},
    "llama-guard-3-8b": {"input_cost": 0.20, "output_cost": 0.20},
    "llama-3.3-70b-specdec": {"input_cost": 0.59, "output_cost": 0.99},
    "llama-3.2-11b-vision-preview": {"input_cost": 0.18, "output_cost": 0.18},
    "llama-3.2-90b-vision-preview": {"input_cost": 0.90, "output_cost": 0.90},
}


class GroqLanguageModel(LanguageModel, frozen=True):
    model_name: GroqModel
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

    @ensure_module_installed("groq", "groq")
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
        from groq import AsyncGroq
        from groq._types import NOT_GIVEN
        from groq.types.chat.chat_completion import (
            ChatCompletion as GroqChatCompletion,
        )
        from groq.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )
        from groq.types.chat.chat_completion_tool_param import ChatCompletionToolParam
        from groq.types.chat.completion_create_params import ResponseFormat
        from groq.types.completion_usage import CompletionUsage

        now = timeit.default_timer()
        client = AsyncGroq(
            api_key=self.api_key,
            max_retries=self.max_retries,
        )

        new_messages = copy.copy(messages)
        if response_model is not None:
            new_messages = get_new_messages_with_response_format_instructions(
                messages=messages, response_model=response_model
            )

        groq_completion: GroqChatCompletion = await client.chat.completions.create(
            messages=[message.to_groq_format() for message in new_messages],
            model=self.model_name,
            max_tokens=max_completion_tokens,
            n=n,
            response_format=ResponseFormat(type="json_object")
            if response_model
            else NOT_GIVEN,
            stop=list(stop_sequences) if stop_sequences else NOT_GIVEN,
            temperature=temperature,
            tools=[
                ChatCompletionToolParam(
                    function=Function.from_callable(tool).to_groq_function(),
                    type="function",
                )
                if callable(tool)
                else tool.to_groq_tool()
                for tool in tools
            ]
            if tools
            else NOT_GIVEN,
            top_p=top_p,
            timeout=timeout,
        )

        # Construct Choices
        choices: list[MessageChoice[S]] = []
        for choice in groq_completion.choices:
            message = choice.message

            groq_tool_calls: list[ChatCompletionMessageToolCall] = (
                message.tool_calls or []
            )

            tool_calls: list[ToolCall] = []
            functions: dict[str, Function] = _create_function_mapping_by_tools(
                tools or []
            )

            for groq_tool_call in groq_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=groq_tool_call.id,
                        called_function=CalledFunction(
                            function=functions[groq_tool_call.function.name],
                            arguments=msgspec.json.decode(
                                groq_tool_call.function.arguments, type=dict
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

        usage: Optional[CompletionUsage] = groq_completion.usage
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
        chat_completion = ChatCompletion(
            elapsed_time=timeit.default_timer() - now,
            id=groq_completion.id,
            object=groq_completion.object,
            created=groq_completion.created,
            model=cast(GroqModelType, f"groq/api/{self.model_name}"),
            system_fingerprint=groq_completion.system_fingerprint or "fp_none",
            choices=choices,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )

        return chat_completion
