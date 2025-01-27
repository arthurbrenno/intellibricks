import asyncio
import timeit
from typing import Any, Literal, Optional, Sequence, TypeVar, cast, overload, override
import uuid

from architecture import log
import msgspec
from architecture.utils.decorators import ensure_module_installed

from google.genai.types import GenerateContentResponseUsageMetadata
from intellibricks.llms.base.contracts import LanguageModel
from intellibricks.llms.types import (
    GeneratedAssistantMessage,
    ChatCompletion,
    Function,
    Message,
    MessageChoice,
    MessageSequence,
    Part,
    PromptTokensDetails,
    RawResponse,
    ToolCall,
    CalledFunction,
    ToolCallSequence,
    ToolInputType,
    Usage,
)
from intellibricks.llms.types import GoogleModelType
from intellibricks.llms.util import (
    create_function_mapping_by_tools,
    get_parsed_response,
)
from intellibricks.llms.util import ms_type_to_schema

logger = log.create_logger(__name__)

S = TypeVar("S", bound=msgspec.Struct, default=RawResponse)


MODEL_PRICING = {
    # Gemini 1.5 Flash and its aliases
    "gemini-1.5-flash": {
        "text_input": 0.00001875 / 1000,
        "text_output": 0.000075 / 1000,
        "image_input": 0.00002,
        "audio_input": 0.00002,
    },
    "gemini-1.5-flash-8b": {
        "text_input": 0.00001875 / 1000,
        "text_output": 0.000075 / 1000,
        "image_input": 0.00002,
        "audio_input": 0.00002,
    },
    "gemini-1.5-flash-001": {
        "text_input": 0.00001875 / 1000,
        "text_output": 0.000075 / 1000,
        "image_input": 0.00002,
        "audio_input": 0.00002,
    },
    "gemini-1.5-flash-002": {
        "text_input": 0.00001875 / 1000,
        "text_output": 0.000075 / 1000,
        "image_input": 0.00002,
        "audio_input": 0.00002,
    },
    # Gemini 1.5 Pro and its aliases
    "gemini-1.5-pro": {
        "text_input": 0.0003125 / 1000,
        "text_output": 0.00125 / 1000,
        "image_input": 0.00032875,
        "audio_input": 0.00003125,
    },
    "gemini-1.5-pro-001": {
        "text_input": 0.0003125 / 1000,
        "text_output": 0.00125 / 1000,
        "image_input": 0.00032875,
        "audio_input": 0.00003125,
    },
    "gemini-1.5-pro-002": {
        "text_input": 0.0003125 / 1000,
        "text_output": 0.00125 / 1000,
        "image_input": 0.00032875,
        "audio_input": 0.00003125,
    },
    # Gemini 1.0 Pro and its aliases
    "gemini-1.0-pro": {
        "text_input": 0.000125 / 1000,
        "text_output": 0.000375 / 1000,
        "image_input": 0.0025,
        "audio_input": 0.002,
    },
    "gemini-1.0-pro-002": {
        "text_input": 0.000125 / 1000,
        "text_output": 0.000375 / 1000,
        "image_input": 0.0025,
        "audio_input": 0.002,
    },
    # Experimental Models
    "gemini-flash-experimental": {
        "text_input": 0.00001875 / 1000,
        "text_output": 0.000075 / 1000,
        "image_input": 0.00002,
        "audio_input": 0.00002,
    },
    "gemini-pro-experimental": {
        "text_input": 0.0003125 / 1000,
        "text_output": 0.00125 / 1000,
        "image_input": 0.00032875,
        "audio_input": 0.00003125,
    },
}


class GoogleLanguageModel(LanguageModel, frozen=True):
    model_name: Literal[
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro",
        "gemini-1.5-pro-001",
        "gemini-1.0-pro-002",
        "gemini-1.5-pro-002",
        "gemini-flash-experimental",
        "gemini-pro-experimental",
    ]

    vertexai: Optional[bool]
    api_key: Optional[str] = None
    project: Optional[str] = None
    location: Optional[str] = None

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

    @ensure_module_installed("google.genai", "google-genai")
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
        from google import genai
        from google.genai import types

        client: genai.Client = genai.Client(
            vertexai=self.vertexai,
            api_key=self.api_key,
            project=self.project if self.vertexai else None,
            location=self.location if self.vertexai else None,
        )

        # TODO(arthur): make intellibricks support this kind of integration:
        # tools = [
        #     types.Tool(
        #         retrieval=types.Retrieval(
        #             vertex_ai_search=types.VertexAISearch(
        #                 datastore="..."
        #             )
        #         )
        #     ),
        # ]
        now = timeit.default_timer()
        try:
            generate_response: types.GenerateContentResponse = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model_name,
                    contents=[message.to_google_format() for message in messages],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        candidate_count=n,
                        tools=[
                            tool if callable(tool) else tool.to_google_tool()
                            for tool in tools
                        ]
                        if tools
                        else None,
                        max_output_tokens=max_completion_tokens,
                        stop_sequences=list(stop_sequences) if stop_sequences else None,
                        response_mime_type="application/json"
                        if response_model
                        else None,
                        response_schema=ms_type_to_schema(
                            response_model,
                            remove_parameters=[
                                "title",
                                "min_length",
                                "max_length",
                                "examples",
                            ],
                            ensure_str_enum=True,
                        )
                        if response_model
                        else None,
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="BLOCK_NONE",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_NONE",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="BLOCK_NONE",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="BLOCK_NONE",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                                threshold="BLOCK_NONE",
                            ),
                        ],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True,
                            maximum_remote_calls=None,
                        ),
                    ),
                ),
                timeout=timeout,
            )
        except TimeoutError:
            raise TimeoutError(
                f"The chat_async method exceeded the timeout of {timeout} seconds."
            )

        candidates: Optional[list[types.Candidate]] = generate_response.candidates
        if candidates is None:
            raise ValueError(
                "Google generation failed because there is no candidates found."
            )

        # Generate the choices based on the candidates
        choices: list[MessageChoice[S]] = []
        for index, candidate in enumerate(candidates):
            content: Optional[types.Content] = candidate.content
            if content is None:
                raise ValueError(
                    "Google generation failed because there is no content found in the candidates."
                )

            parts: Optional[list[types.Part]] = content.parts
            if parts is None:
                raise ValueError(
                    "Google generation failed because there is no parts found in the content."
                )

            candidate_parts = [Part.from_google_part(part) for part in parts]
            function_calls: list[types.FunctionCall] = [
                part.function_call for part in parts if part.function_call
            ]

            tool_calls: list[ToolCall] = []
            functions: dict[str, Function] = create_function_mapping_by_tools(
                tools or []
            )

            for function in function_calls:
                function_id: Optional[str] = function.id
                function_args: Optional[dict[str, Any]] = function.args
                function_name: Optional[str] = function.name

                if function_name is None:
                    raise ValueError(
                        'Function name is required for a function call. Tell Google to remove the "Optional" type hint.'
                    )

                tool_calls.append(
                    ToolCall(
                        id=function_id or str(uuid.uuid4()),
                        called_function=CalledFunction(
                            function=functions[function_name],
                            arguments=function_args or {},
                        ),
                    )
                )

            choices.append(
                MessageChoice(
                    index=index,
                    message=GeneratedAssistantMessage(
                        contents=candidate_parts,
                        parsed=get_parsed_response(candidate_parts, response_model)
                        if response_model
                        else cast(
                            S,
                            RawResponse(),
                        ),
                        tool_calls=ToolCallSequence(tool_calls),
                    ),
                )
            )

        usage_metadata: GenerateContentResponseUsageMetadata = (
            generate_response.usage_metadata
            or GenerateContentResponseUsageMetadata(
                cached_content_token_count=None,
                candidates_token_count=None,
                prompt_token_count=None,
                total_token_count=None,
            )
        )

        prompt_token_count: Optional[int] = usage_metadata.prompt_token_count
        candidates_token_count: Optional[int] = usage_metadata.candidates_token_count
        message_sequence = MessageSequence(messages)
        image_count = message_sequence.count_images()
        video_count = message_sequence.count_videos()
        audio_count = message_sequence.count_audios()

        # Warning for video pricing
        if video_count > 0:
            logger.warning(
                "Warning: Video input pricing is not implemented. Costs may not be 100% accurate."
            )

        # Fetch pricing for the current model
        pricing = MODEL_PRICING.get(self.model_name, {})

        # Calculate input cost
        input_cost = (
            (prompt_token_count or 0) * pricing.get("text_input", 0)
            + image_count * pricing.get("image_input", 0)
            + audio_count * pricing.get("audio_input", 0)
        )

        # Calculate output cost
        output_cost = (candidates_token_count or 0) * pricing.get("text_output", 0)

        usage: Usage = Usage(
            prompt_tokens=prompt_token_count,
            completion_tokens=candidates_token_count,
            total_tokens=usage_metadata.total_token_count,
            input_cost=input_cost if self.vertexai else 0.0,
            output_cost=output_cost if self.vertexai else 0.0,
            total_cost=(input_cost + output_cost) if self.vertexai else 0.0,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=0,
                cached_tokens=usage_metadata.cached_content_token_count or 0,
            ),
        )

        completion = ChatCompletion(
            elapsed_time=round(timeit.default_timer() - now, 2),
            choices=choices,
            usage=usage,
            model=cast(
                GoogleModelType,
                f"google/{'vertexai' if self.vertexai else 'genai'}/{self.model_name}",
            ),
        )

        return completion
