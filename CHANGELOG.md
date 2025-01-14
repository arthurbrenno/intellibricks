# Changelog

## v0.3.0

- NEW ChainOfThought class. Can be used like this:

```py
import msgspec
from intellibricks import Synapse, ChainOfThought

class Response(msgspec.Struct):
    response: str


synapse = Synapse.of("cerebras/api/llama-3.3-70b")
completion = synapse.complete(
    "Hello, how are you?",
    response_model=ChainOfThought[Response],
)

print(completion)
```
