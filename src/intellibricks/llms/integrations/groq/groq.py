import copy
import os
import tempfile
import timeit
from pathlib import Path
from typing import (
    Any,
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

from intellibricks.llms.base import (
    FileContent,
    LanguageModel,
    TranscriptionModel,
)
from intellibricks.llms.constants import FinishReason
from intellibricks.llms.types import (
    AudioTranscription,
    CalledFunction,
    ChatCompletion,
    Function,
    GeneratedAssistantMessage,
    GroqModelType,
    Message,
    MessageChoice,
    Part,
    RawResponse,
    SentenceSegment,
    ToolCall,
    ToolCallSequence,
    ToolInputType,
    Usage,
)
from intellibricks.llms.util import (
    create_function_mapping_by_tools,
    get_audio_duration,
    get_new_messages_with_response_format_instructions,
    get_parsed_response,
    segments_to_srt,
    write_content_to_file,
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

GroqTranscriptionModelType: TypeAlias = Literal[
    "whisper-large-v3-turbo", "distil-whisper-large-v3-en", "whisper-large-v3"
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
            functions: dict[str, Function] = create_function_mapping_by_tools(
                tools or []
            )

            for groq_tool_call in groq_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=groq_tool_call.id,
                        called_function=CalledFunction(
                            function=functions[groq_tool_call.function.name],
                            arguments=msgspec.json.decode(
                                groq_tool_call.function.arguments, type=dict[str, Any]
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
            elapsed_time=round(timeit.default_timer() - now, 2),
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


class GroqTranscriptionModel(TranscriptionModel, frozen=True):
    model_name: GroqTranscriptionModelType
    api_key: Optional[str] = None
    max_retries: int = 2

    @ensure_module_installed("groq", "groq")
    @override
    async def transcribe_async(
        self,
        audio: FileContent,
        temperature: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> AudioTranscription:
        from groq import AsyncGroq
        from groq._types import NOT_GIVEN

        client = AsyncGroq(api_key=self.api_key, max_retries=self.max_retries)
        now = timeit.default_timer()
        audio_transcriptions: list[AudioTranscription] = []
        audio_duration = get_audio_duration(audio)

        if audio_duration > 600:
            from pydub import AudioSegment
            from pydub.utils import make_chunks

            with tempfile.TemporaryDirectory() as temp_dir:
                original_file_path = write_content_to_file(audio, temp_dir)
                audio_segment = AudioSegment.from_file(original_file_path)

                chunk_length_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
                chunks = make_chunks(audio_segment, chunk_length_ms)

                for i, chunk in enumerate(chunks):
                    if len(chunk) == 0:
                        continue  # Skip empty chunks
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")
                    chunk.export(chunk_path, format="mp3")

                    chunk_start_time = timeit.default_timer()
                    transcription = await client.audio.transcriptions.create(
                        file=Path(chunk_path),
                        model=self.model_name,
                        language=self.language or NOT_GIVEN,
                        temperature=temperature or NOT_GIVEN,
                        prompt=prompt or NOT_GIVEN,
                        response_format="verbose_json",
                    )
                    chunk_elapsed_time = timeit.default_timer() - chunk_start_time

                    dict_transcription = transcription.model_dump()
                    segments: list[SentenceSegment] = []
                    for seg in dict_transcription.get("segments", []):
                        segments.append(
                            SentenceSegment(
                                id=seg.get("id"),
                                sentence=seg.get("text"),
                                start=seg.get("start"),
                                end=seg.get("end"),
                                no_speech_prob=seg.get("no_speech_prob"),
                            )
                        )

                    chunk_duration = (
                        len(chunk) / 1000
                    )  # Convert milliseconds to seconds
                    chunk_transcription = AudioTranscription(
                        elapsed_time=round(chunk_elapsed_time, 2),
                        text=transcription.text,
                        segments=segments,
                        cost=0.0,
                        duration=chunk_duration,
                        srt=segments_to_srt(segments),
                    )
                    audio_transcriptions.append(chunk_transcription)

                merged_transcription = audio_transcriptions[0].merge(
                    *audio_transcriptions[1:]
                )
                return merged_transcription

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = write_content_to_file(audio, temp_dir)
            transcription = await client.audio.transcriptions.create(
                file=Path(file_path),
                model=self.model_name,
                language=self.language or NOT_GIVEN,
                temperature=temperature or NOT_GIVEN,
                prompt=prompt or NOT_GIVEN,
                response_format="verbose_json",
            )

        dict_transcription = transcription.model_dump()
        segments = []
        for segment in dict_transcription.get("segments", []):
            segments.append(
                SentenceSegment(
                    id=segment.get("id"),
                    sentence=segment.get("text"),
                    start=segment.get("start"),
                    end=segment.get("end"),
                    no_speech_prob=segment.get("no_speech_prob"),
                )
            )

        return AudioTranscription(
            elapsed_time=round(timeit.default_timer() - now, 2),
            text=transcription.text,
            segments=segments,
            cost=0.0,
            duration=audio_duration,
            srt=segments_to_srt(segments),
        )
