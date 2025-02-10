from intellibricks.llms import Synapse
import msgspec
from typing import Annotated, Optional
from architecture import log

logger = log.create_logger(__name__, level=log.DEBUG)


class IdentityResponse(msgspec.Struct, frozen=True):
    name: Annotated[
        str, msgspec.Meta(title="Name", description="Name of the assistant.")
    ]

    age: Annotated[
        Optional[int],
        msgspec.Meta(title="Age", description="The age of the assistant, in years."),
    ] = None


llm = Synapse.of("google/genai/gemini-2.0-flash")

completion = llm.complete(
    "Who are you? And what is your age?", response_model=IdentityResponse
)

logger.debug(completion)
