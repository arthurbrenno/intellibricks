import msgspec
from intellibricks import Synapse


class Response(msgspec.Struct):
    response: str


synapse = Synapse.of("cerebras/api/llama3.1-70b")

completion = synapse.complete("Hello, how are you?")
print(completion)
