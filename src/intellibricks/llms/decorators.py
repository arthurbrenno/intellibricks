import asyncio
import functools
import uuid
from typing import Any, Callable, Iterable, Literal, Optional

from langfuse.decorators import observe


def observation_root(
    *,
    name: Optional[str] = None,
    as_type: Optional[Literal["generation"]] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    transform_to_string: Optional[Callable[[Iterable], str]] = None,
) -> Callable:
    """
    A decorator that wraps the @observe decorator, generates custom observation ID,
    and injects it into the function's keyword arguments.
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate custom observation ID
                completion_id = kwargs.get("completion_id") or uuid.uuid4()
                langfuse_observation_id = f"obs-{completion_id}"

                kwargs["completion_id"] = completion_id
                # Inject langfuse_observation_id into kwargs
                kwargs["langfuse_observation_id"] = langfuse_observation_id

                # Apply the @observe decorator to the function
                observed_func = observe(
                    name=name,
                    as_type=as_type,
                    capture_input=capture_input,
                    capture_output=capture_output,
                    transform_to_string=transform_to_string,
                )(func)

                # Call the observed function
                return await observed_func(*args, **kwargs)

            return wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate custom observation ID
                completion_id = kwargs.get("completion_id") or uuid.uuid4()
                langfuse_observation_id = f"obs-{completion_id}"

                kwargs["completion_id"] = completion_id
                # Inject langfuse_observation_id into kwargs
                kwargs["langfuse_observation_id"] = langfuse_observation_id

                # Apply the @observe decorator to the function
                observed_func = observe(
                    name=name,
                    as_type=as_type,
                    capture_input=capture_input,
                    capture_output=capture_output,
                    transform_to_string=transform_to_string,
                )(func)

                # Call the observed function
                return observed_func(*args, **kwargs)

            return wrapper

    return decorator
