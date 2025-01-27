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
    Any,
)

import msgspec
from architecture.utils.decorators import ensure_module_installed
from langfuse.client import os

from intellibricks.llms.base.contracts import LanguageModel
from intellibricks.llms.constants import FinishReason
from intellibricks.llms.types import (
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
from intellibricks.llms.types import DeepInfraModelType
from intellibricks.llms.util import (
    create_function_mapping_by_tools,
    get_parsed_response,
)
from intellibricks.llms.util import ms_type_to_schema

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)


ChatModel: TypeAlias = Literal[
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "Qwen/QwQ-32B-Preview",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/WizardLM-2-8x22B",
    "01-ai/Yi-34B-Chat",
    "Austism/chronos-hermes-13b-v2",
    "Gryphe/MythoMax-L2-13b",
    "Gryphe/MythoMax-L2-13b-turbo",
    "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
    "NousResearch/Hermes-3-Llama-3.1-405B",
    "Phind/Phind-CodeLlama-34B-v2",
    "Qwen/QVQ-72B-Preview",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
    "Sao10K/L3-70B-Euryale-v2.1",
    "Sao10K/L3-8B-Lunaris-v1",
    "Sao10K/L3.1-70B-Euryale-v2.2",
    "bigcode/starcoder2-15b",
    "bigcode/starcoder2-15b-instruct-v0.1",
    "codellama/CodeLlama-34b-Instruct-hf",
    "codellama/CodeLlama-70b-Instruct-hf",
    "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
    "databricks/dbrx-instruct",
    "deepinfra/airoboros-70b",
    "google/codegemma-7b-it",
    "google/gemma-1.1-7b-it",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "lizpreciatior/lzlv_70b_fp16_hf",
    "mattshumer/Reflection-Llama-3.1-70B",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/WizardLM-2-7B",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "nvidia/Nemotron-4-340B-Instruct",
    "openbmb/MiniCPM-Llama3-V-2_5",
    "openchat/openchat-3.6-8b",
    "openchat/openchat_3.5",
]

MODEL_PRICING: dict[ChatModel, dict[Literal["input_cost", "output_cost"], float]] = {
    "meta-llama/Llama-3.3-70B-Instruct": {"input_cost": 0.23, "output_cost": 0.40},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "input_cost": 0.12,
        "output_cost": 0.30,
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"input_cost": 0.23, "output_cost": 0.40},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"input_cost": 0.03, "output_cost": 0.05},
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "input_cost": 0.80,
        "output_cost": 0.80,
    },
    "Qwen/QwQ-32B-Preview": {"input_cost": 0.12, "output_cost": 0.18},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
        "input_cost": 0.02,
        "output_cost": 0.05,
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
        "input_cost": 0.12,
        "output_cost": 0.30,
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"input_cost": 0.07, "output_cost": 0.16},
    "nvidia/Llama-3.1-Nemotron-70B-Instruct": {"input_cost": 0.12, "output_cost": 0.30},
    "Qwen/Qwen2.5-72B-Instruct": {"input_cost": 0.23, "output_cost": 0.40},
    "meta-llama/Llama-3.2-90B-Vision-Instruct": {
        "input_cost": 0.35,
        "output_cost": 0.40,
    },
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {
        "input_cost": 0.055,
        "output_cost": 0.055,
    },  # Using same in/out
    "microsoft/WizardLM-2-8x22B": {
        "input_cost": 0.50,
        "output_cost": 0.50,
    },  # Using same in/out
    "01-ai/Yi-34B-Chat": {"input_cost": 0.0, "output_cost": 0.0},
    "Austism/chronos-hermes-13b-v2": {"input_cost": 0.0, "output_cost": 0.0},
    "Gryphe/MythoMax-L2-13b": {
        "input_cost": 0.065,
        "output_cost": 0.065,
    },  # Using same in/out
    "Gryphe/MythoMax-L2-13b-turbo": {"input_cost": 0.0, "output_cost": 0.0},
    "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1": {"input_cost": 0.0, "output_cost": 0.0},
    "NousResearch/Hermes-3-Llama-3.1-405B": {
        "input_cost": 0.80,
        "output_cost": 0.80,
    },  # Using same in/out
    "Phind/Phind-CodeLlama-34B-v2": {"input_cost": 0.0, "output_cost": 0.0},
    "Qwen/QVQ-72B-Preview": {"input_cost": 0.25, "output_cost": 0.50},
    "Qwen/Qwen2-72B-Instruct": {"input_cost": 0.0, "output_cost": 0.0},
    "Qwen/Qwen2-7B-Instruct": {"input_cost": 0.0, "output_cost": 0.0},
    "Qwen/Qwen2.5-7B-Instruct": {"input_cost": 0.0, "output_cost": 0.0},
    "Qwen/Qwen2.5-Coder-7B": {"input_cost": 0.0, "output_cost": 0.0},
    "Sao10K/L3-70B-Euryale-v2.1": {"input_cost": 0.35, "output_cost": 0.40},
    "Sao10K/L3-8B-Lunaris-v1": {"input_cost": 0.03, "output_cost": 0.06},
    "Sao10K/L3.1-70B-Euryale-v2.2": {"input_cost": 0.35, "output_cost": 0.40},
    "bigcode/starcoder2-15b": {"input_cost": 0.0, "output_cost": 0.0},
    "bigcode/starcoder2-15b-instruct-v0.1": {"input_cost": 0.0, "output_cost": 0.0},
    "codellama/CodeLlama-34b-Instruct-hf": {"input_cost": 0.0, "output_cost": 0.0},
    "codellama/CodeLlama-70b-Instruct-hf": {"input_cost": 0.0, "output_cost": 0.0},
    "cognitivecomputations/dolphin-2.6-mixtral-8x7b": {
        "input_cost": 0.0,
        "output_cost": 0.0,
    },
    "cognitivecomputations/dolphin-2.9.1-llama-3-70b": {
        "input_cost": 0.0,
        "output_cost": 0.0,
    },
    "databricks/dbrx-instruct": {"input_cost": 0.0, "output_cost": 0.0},
    "deepinfra/airoboros-70b": {"input_cost": 0.0, "output_cost": 0.0},
    "google/codegemma-7b-it": {"input_cost": 0.0, "output_cost": 0.0},
    "google/gemma-1.1-7b-it": {"input_cost": 0.0, "output_cost": 0.0},
    "google/gemma-2-27b-it": {
        "input_cost": 0.27,
        "output_cost": 0.27,
    },  # Using same in/out
    "google/gemma-2-9b-it": {"input_cost": 0.03, "output_cost": 0.06},
    "lizpreciatior/lzlv_70b_fp16_hf": {"input_cost": 0.35, "output_cost": 0.40},
    "mattshumer/Reflection-Llama-3.1-70B": {"input_cost": 0.0, "output_cost": 0.0},
    "meta-llama/Llama-2-13b-chat-hf": {"input_cost": 0.0, "output_cost": 0.0},
    "meta-llama/Llama-2-70b-chat-hf": {"input_cost": 0.0, "output_cost": 0.0},
    "meta-llama/Llama-2-7b-chat-hf": {"input_cost": 0.0, "output_cost": 0.0},
    "meta-llama/Llama-3.2-1B-Instruct": {"input_cost": 0.01, "output_cost": 0.02},
    "meta-llama/Llama-3.2-3B-Instruct": {"input_cost": 0.015, "output_cost": 0.025},
    "meta-llama/Meta-Llama-3-70B-Instruct": {"input_cost": 0.23, "output_cost": 0.40},
    "meta-llama/Meta-Llama-3-8B-Instruct": {"input_cost": 0.03, "output_cost": 0.06},
    "microsoft/Phi-3-medium-4k-instruct": {"input_cost": 0.0, "output_cost": 0.0},
    "microsoft/WizardLM-2-7B": {"input_cost": 0.055, "output_cost": 0.055},
    "mistralai/Mistral-7B-Instruct-v0.1": {"input_cost": 0.0, "output_cost": 0.0},
    "mistralai/Mistral-7B-Instruct-v0.2": {"input_cost": 0.0, "output_cost": 0.0},
    "mistralai/Mistral-7B-Instruct-v0.3": {"input_cost": 0.03, "output_cost": 0.055},
    "mistralai/Mistral-Nemo-Instruct-2407": {"input_cost": 0.035, "output_cost": 0.08},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"input_cost": 0.0, "output_cost": 0.0},
    "mistralai/Mixtral-8x22B-v0.1": {"input_cost": 0.0, "output_cost": 0.0},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "input_cost": 0.24,
        "output_cost": 0.24,
    },  # Using same in/out
    "nvidia/Nemotron-4-340B-Instruct": {"input_cost": 0.0, "output_cost": 0.0},
    "openbmb/MiniCPM-Llama3-V-2_5": {"input_cost": 0.0, "output_cost": 0.0},
    "openchat/openchat-3.6-8b": {"input_cost": 0.0, "output_cost": 0.0},
    "openchat/openchat_3.5": {"input_cost": 0.055, "output_cost": 0.055},
}


class DeepInfraLanguageModel(LanguageModel, frozen=True):
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
        from openai import NOT_GIVEN, AsyncOpenAI
        from openai.types.chat.chat_completion import (
            ChatCompletion as OpenAIChatCompletion,
        )
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )
        from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
        from openai.types.completion_usage import CompletionUsage
        from openai.types.shared_params.response_format_json_schema import (
            JSONSchema,
            ResponseFormatJSONSchema,
        )

        now = timeit.default_timer()
        client = AsyncOpenAI(
            api_key=self.api_key or os.environ.get("DEEPINFRA_API_KEY", None),
            base_url="https://api.deepinfra.com/v1/openai",
            max_retries=self.max_retries or 2,
        )

        openai_completion: OpenAIChatCompletion = await client.chat.completions.create(
            messages=[message.to_openai_format() for message in messages],
            model=self.model_name,
            audio=NOT_GIVEN,
            max_completion_tokens=max_completion_tokens,
            n=n,
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchema(
                    name="structured_response",
                    description="Structured response",
                    schema=ms_type_to_schema(response_model),
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

        chat_completion = ChatCompletion(
            elapsed_time=round(timeit.default_timer() - now, 2),
            id=openai_completion.id,
            object=openai_completion.object,
            created=openai_completion.created,
            model=cast(DeepInfraModelType, f"deepinfra/api/{self.model_name}"),
            system_fingerprint=openai_completion.system_fingerprint or "fp_none",
            choices=choices,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
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
