import msgspec
from intellibricks import Synapse


class Response(msgspec.Struct):
    response: str


synapse = Synapse.of("google/genai/gemini-2.0-flash-exp")

completion = synapse.complete("Hello, how are you?", response_model=Response)
print(completion)
