import msgspec
from intellibricks import Synapse, ChainOfThought
from langfuse import Langfuse


class Response(msgspec.Struct):
    response: str


langfuse = Langfuse(
    secret_key="sk-lf-fd07a34d-d4d2-4468-92ca-94d53ef2a6a1",
    public_key="pk-lf-30c24cd8-a23c-4f2c-b987-136ca936ecec",
    host="http://localhost:3000",
)

synapse = Synapse.of("cerebras/api/llama-3.3-70b")

completion = synapse.complete(
    "Hello, how are you?",
    trace_params={"name": "test completion"},
    response_model=ChainOfThought[Response],
    max_retries=1,
)

print(completion.parsed)
