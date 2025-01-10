import msgspec
from intellibricks import Synapse


class Response(msgspec.Struct):
    response: str


print(Response.__qualname__)
exit()

synapse = Synapse.of("google/genai/gemini-1.5-flash")

completion = synapse.complete("Hello, how are you?", response_model=Response)
print(completion)
