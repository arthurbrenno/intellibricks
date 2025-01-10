import msgspec
from intellibricks import Synapse


class Response(msgspec.Struct):
    response: str


synapse = Synapse.of("google/genai/gemini-1.5-flash")

completion = synapse.complete("Hello, how are you?", response_model=Response)
print(completion)
print(msgspec.inspect.type_info(type(completion)))
