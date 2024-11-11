from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TypedDict, Union

import langfuse
import langfuse.api
import langfuse.model
import langfuse.types
from pydantic import BaseModel


class TraceParams(TypedDict, total=False):
    """
    Parameters for updating the current trace, including metadata and context information.

    This TypedDict is used to specify the parameters that can be updated for the current trace.
    Each field corresponds to an attribute of the trace that can be dynamically modified
    during execution. These parameters are useful for categorization, filtering, and analysis
    within the Langfuse UI.

    Attributes:
        name (Optional[str]):
            Identifier of the trace. Useful for sorting and filtering in the UI.

        input (Optional[Any]):
            The input parameters of the trace, providing context about the observed operation
            or function call.

        output (Optional[Any]):
            The output or result of the trace.

        user_id (Optional[str]):
            The ID of the user that triggered the execution. Used to provide user-level analytics.

        session_id (Optional[str]):
            Used to group multiple traces into a session in Langfuse. Typically your own session
            or thread identifier.

        version (Optional[str]):
            The version of the trace type. Helps in understanding how changes to the trace type
            affect metrics and is useful for debugging.

        release (Optional[str]):
            The release identifier of the current deployment. Helps in understanding how changes
            in different deployments affect metrics and is useful for debugging.

        metadata (Optional[Any]):
            Additional metadata for the trace. Can be any JSON-serializable object. Metadata is
            merged when updated via the API.

        tags (Optional[list[str]]):
            Tags used to categorize or label traces. Traces can be filtered by tags in the
            Langfuse UI and through the GET API.

        public (Optional[bool]):
            Indicates whether the trace is public. If set to `True`, the trace is accessible publicly;
            otherwise, it remains private.
    """

    name: Optional[str]
    input: Optional[Any]
    output: Optional[Any]
    user_id: Optional[str]
    session_id: Optional[str]
    version: Optional[str]
    release: Optional[str]
    metadata: Optional[Any]
    tags: Optional[list[str]]
    public: Optional[bool]


class ObservationParams(TypedDict, total=False):
    """
    Parameters for updating the current observation within an active trace context.

    This TypedDict is used to specify the parameters that can be updated for the current observation.
    Each field corresponds to an attribute of the observation that can be dynamically modified
    during execution. These parameters enhance the observability and traceability of the execution
    context within the Langfuse UI.

    Attributes:
        input (Optional[Any]):
            The input parameters of the trace or observation, providing context about the observed
            operation or function call.

        output (Optional[Any]):
            The output or result of the trace or observation.

        name (Optional[str]):
            Identifier of the trace or observation. Useful for sorting and filtering in the UI.

        version (Optional[str]):
            The version of the trace type. Helps in understanding how changes to the trace type affect
            metrics and is useful for debugging.

        metadata (Optional[Any]):
            Additional metadata for the trace. Can be any JSON-serializable object. Metadata is merged
            when updated via the API.

        start_time (Optional[datetime]):
            The start time of the observation, allowing for custom time range specification.

        end_time (Optional[datetime]):
            The end time of the observation, enabling precise control over the observation duration.

        release (Optional[str]):
            The release identifier of the current deployment. Helps in understanding how changes in
            different deployments affect metrics and is useful for debugging.

        tags (Optional[list[str]]):
            Tags used to categorize or label traces. Traces can be filtered by tags in the Langfuse UI
            and through the GET API.

        user_id (Optional[str]):
            The ID of the user that triggered the execution. Used to provide user-level analytics.

        session_id (Optional[str]):
            Used to group multiple traces into a session in Langfuse. Typically your own session or thread
            identifier.

        level (Optional[SpanLevel]):
            The severity or importance level of the observation, such as "INFO", "WARNING", or "ERROR".

        status_message (Optional[str]):
            A message or description associated with the observation's status, particularly useful for
            error reporting.

        completion_start_time (Optional[datetime]):
            The time at which the completion started (streaming). Set it to get latency analytics broken
            down into time until completion started and completion duration.

        model (Optional[str]):
            The model identifier used for the generation.

        model_parameters (Optional[dict[str, MapValue]]):
            The parameters of the model used for the generation; can be any key-value pairs.

        usage (Optional[Union[BaseModel, ModelUsage]]):
            The usage object supports the OpenAI structure with {promptTokens, completionTokens, totalTokens}
            and a more generic version {input, output, total, unit, inputCost, outputCost, totalCost} where unit
            can be "TOKENS", "CHARACTERS", "MILLISECONDS", "SECONDS", or "IMAGES". Refer to the documentation
            on how to automatically infer token usage and costs in Langfuse.

        prompt (Optional[PromptClient]):
            The prompt object used for the generation.

        public (Optional[bool]):
            Indicates whether the observation is public. If set to `True`, the observation is accessible
            publicly; otherwise, it remains private.
    """

    input: Optional[Any]
    output: Optional[Any]
    name: Optional[str]
    version: Optional[str]
    metadata: Optional[Any]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    release: Optional[str]
    tags: Optional[list[str]]
    user_id: Optional[str]
    session_id: Optional[str]
    level: Optional[langfuse.types.SpanLevel]
    status_message: Optional[str]
    completion_start_time: Optional[datetime]
    model: Optional[str]
    model_parameters: Optional[dict[str, langfuse.api.commons.types.map_value.MapValue]]
    usage: Optional[Union[BaseModel, langfuse.model.ModelUsage]]
    prompt: Optional[langfuse.model.PromptClient]
    public: Optional[bool]
