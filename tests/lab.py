import msgspec
from typing import Annotated, TypeVar, Generic, Sequence, Any
from architecture import log

logger = log.create_logger(__name__)

_T = TypeVar("_T")


class ThoughtDetail(msgspec.Struct, frozen=True):
    detail: Annotated[
        str,
        msgspec.Meta(
            title="Thought Detail",
            description="A granular explanation of a specific aspect of the reasoning step.",
            examples=["First, I added 2 + 3", "Checked if the number is even or odd"],
        ),
    ]


class Step(msgspec.Struct, frozen=True):
    step_number: Annotated[
        int,
        msgspec.Meta(
            title="Step Number",
            description="The position of this step in the overall chain of thought.",
            examples=[1, 2, 3],
        ),
    ]
    explanation: Annotated[
        str,
        msgspec.Meta(
            title="Step Explanation",
            description="A concise description of what was done in this step.",
            examples=["Analyze the input statement", "Apply the quadratic formula"],
        ),
    ]
    details: Annotated[
        Sequence[ThoughtDetail],
        msgspec.Meta(
            title="Step Details",
            description="A list of specific details for each step in the reasoning.",
            examples=[
                [
                    ThoughtDetail(detail="Check initial values"),
                    ThoughtDetail(detail="Confirm there are no inconsistencies"),
                ]
            ],
        ),
    ]


class ChainOfThought(msgspec.Struct, Generic[_T], frozen=True):
    title: Annotated[
        str,
        msgspec.Meta(
            title="Chain of Thought Title",
            description="A brief label or description that identifies the purpose of the reasoning.",
            examples=["Sum of two numbers", "Logical problem solving"],
        ),
    ]
    steps: Annotated[
        Sequence[Step],
        msgspec.Meta(
            title="Reasoning Steps",
            description="The sequence of steps that make up the full reasoning process.",
            examples=[
                [
                    Step(
                        step_number=1,
                        explanation="Analyze input data",
                        details=[
                            ThoughtDetail(detail="Data: 234 and 567"),
                            ThoughtDetail(detail="Check if they are integers"),
                        ],
                    ),
                    Step(
                        step_number=2,
                        explanation="Perform the calculation",
                        details=[ThoughtDetail(detail="234 + 567 = 801")],
                    ),
                ]
            ],
        ),
    ]
    final_answer: Annotated[
        _T,
        msgspec.Meta(
            title="Final Answer",
            description="The conclusion or result after all the reasoning steps.",
        ),
    ]


class Response(msgspec.Struct):
    response: str


def schema_hook(schema: Any) -> dict[str, Any]:
    print(schema)
    return {}


schema = msgspec.json.schema(ChainOfThought[Response])
logger.debug(schema)
