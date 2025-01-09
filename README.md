# 🧠🧱 IntelliBricks: The Building Blocks for Intelligent Applications

Intellibricks is an amazing Agentic/LLM framework designed for you, developer. It was designed in a complete different way from other frameworks, which makes the language (Python) fit to it. Intellibricks takes advantage of the latest features and capabilities with the latest python versions (3.13), like `default generics` to provide the best experience with Structured Outputs and predictability on what the output of the LLM is.


Quick examples (no structured outputs):
Synapses:
```py
from intellibricks import Synapse

synapse = Synapse.of("google/genai/gemini-2.0-flash-exp")

completion = synapse.complete("Hello, how are you?")  # Completion[RawResponse]

print(completion)
```

Agents:
```py
from intellibricks import Synapse, Agent

synapse = Synapse.of("google/genai/gemini-2.0-flash-exp")

messages = (
    UserMessage
)
```


Advanced examples (structured outputs, tools, chat_history, observability with Langfuse)

Synapses:
```py
from typing import Annotated
from intellibricks import (
    Synapse,
    UserMessage,
    AssistantMessage,
    DeveloperMessage,
    TraceParams,
)
from langfuse import Langfuse
import msgspec

langfuse = Langfuse(
    secret_key="sk-lf-0be2e5c3-6c86-421c-ad5f-ffb4c065daa0",
    public_key="pk-lf-753848ca-2150-473e-a335-4970fb550a20",
    host="http://localhost:3000",
)

synapse = Synapse.of("google/genai/gemini-2.0-flash-exp", langfuse=langfuse)

messages = (
    DeveloperMessage.from_text("You are a helpful assistant."),
    UserMessage.from_text("Hello, how are you?"),
    AssistantMessage.from_text("I am fine, thank you."),
    UserMessage.from_text("What is your name? And who created you?"),
)


class ModelInfo(msgspec.Struct):
    name: Annotated[
        str, msgspec.Meta(title="Name", description="Here you can enter your name.")
    ]

    creator: Annotated[
        str,
        msgspec.Meta(
            title="Creator", description="Here you can enter the creator's name."
        ),
    ]


trace_params = TraceParams(name="example_chat_completion", user_id="intellibricks")


chat_completion = synapse.chat(
    messages, response_model=ModelInfo, trace_params=trace_params
)
model_info = chat_completion.parsed

print(f"Model name: {model_info.name} | Creator: {model_info.creator}")
```

Agents:
```py
from typing import Annotated
from intellibricks import (
    Synapse,
    UserMessage,
    AssistantMessage,
    DeveloperMessage,
    TraceParams,
    Agent,
)
from langfuse import Langfuse
import msgspec


langfuse = Langfuse(
    secret_key="sk-lf-0be2e5c3-6c86-421c-ad5f-ffb4c065daa0",
    public_key="pk-lf-753848ca-2150-473e-a335-4970fb550a20",
    host="http://localhost:3000",
)

synapse = Synapse.of("google/genai/gemini-2.0-flash-exp", langfuse=langfuse)

messages = (
    DeveloperMessage.from_text("You are a helpful assistant."),
    UserMessage.from_text("Hello, how are you?"),
    AssistantMessage.from_text("I am fine, thank you."),
    UserMessage.from_text("What is your name? And who created you?"),
)


class ModelInfo(msgspec.Struct):
    name: Annotated[
        str, msgspec.Meta(title="Name", description="Here you can enter your name.")
    ]

    creator: Annotated[
        str,
        msgspec.Meta(
            title="Creator", description="Here you can enter the creator's name."
        ),
    ]


trace_params = TraceParams(name="example_chat_completion", user_id="intellibricks")


agent = Agent(
    task="Chat With the User",
    instructions=[
        "Do exactly what the user is telling you to do.",
    ],
    metadata={"name": "Bob", "description": "A simple chat agent."},
    synapse=synapse,
    response_model=ModelInfo,
)

agent_response = agent.run(
    "Hello! What is your name and your creator?", trace_params=trace_params
)

model_info = agent_response.parsed

print(f"Model name: {model_info.name} | Creator: {model_info.creator}")
```

Innovative: turn your agents into a FastAPI API:
```py
from intellibricks import (
    Synapse,
    Agent,
)
import uvicorn

agent = Agent(
    task="Chat With the User",
    instructions=[
        "Chat with the user",
    ],
    metadata={"name": "Bob", "description": "A simple chat agent."},
    synapse=Synapse.of("google/genai/gemini-2.0-flash-exp"),
)

uvicorn.run(agent.to_fastapi_async_app())

# Endpoint will be POST /agents/{lower_agent_name}/completions
```


or just to a simple FastAPI router:
```py
router = agent.to_fastapi_async_router("/agents/bob/completions", "post")

app = FastAPI()
app.include_router(router)
uvicorn.run(app)
```

The same goes for Litestar:
```py
uvicorn.run(agent.to_litestar_async_app())
```

# Providing additional context using the intellibrics.rag module
```py
from dataclasses import dataclass
from intellibricks import (
    Synapse,
    Agent,
)
from intellibricks.rag import (
    SupportsContextRetrieval,
    Context,
    Query,
    ContextPart,
    Source,
)


@dataclass(frozen=True)
class MyFakeGraphDB(SupportsContextRetrieval):
    """
    The base class represents anything that can retrieve context.
    Could be a vector db, a graph db, etc. You should pass
    the relevant parameters in the constructor of your
    own implementation. This is an example
    implementation.
    """

    host: str

    async def retrieve_context_async(self, query: Query) -> Context:
        print(
            f"Pretending to connect to {self.host} to retrieve context for {query.text}"
        )
        example_context = Context(
            parts=[ContextPart(raw_text="...", score=0.5, source=Source(name="..."))]
        )
        return example_context


agent = Agent(
    task="Chat With the User",
    instructions=[
        "Chat with the user",
    ],
    metadata={"name": "Bob", "description": "A simple chat agent."},
    synapse=Synapse.of("google/genai/gemini-2.0-flash-exp"),
    context_sources=[MyGraphDB("localhost")],
)

```
Please note that I'm currently writing some useful vector db connections for you

Now, imagine all the possibilities! Imagine merging all agents easily and creating an API in seconds!
Incomind (february): advanced RawFile -> ParsedDocument module. which will help convert your files into RAG ready files for any kind of database.

How to do it with LangChain 🦜️🔗
LangChain offers a simplified approach to structured outputs using with_structured_output. While convenient, it lacks some of the advanced features and flexibility of IntelliBricks. For instance, features like fallback models, caching, tracing, and custom tool integration are not readily available. Additionally, the reliance on a single invoke method for diverse operations can make customization and specific parameter handling less intuitive.

```py
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Joke(BaseModel):
    joke: str

model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = model.with_structured_output(Joke)

joke = structured_llm.invoke(
    "Tell me a joke about cats"
) # Joke object

print(joke)
```
How to do it with LlamaIndex 🦙
LlamaIndex also provides a way to achieve structured outputs, involving wrapping the LLM with as_structured_llm. This, however, introduces additional steps compared to IntelliBricks. You also need to construct ChatMessage. LlamaIndex's approach lacks the built-in retry mechanisms, comprehensive tracing with Langfuse, and other advanced parameters offered by IntelliBricks for fine-grained control and observability.

```py
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel

class Joke(BaseModel):
    joke: str


llm = OpenAI(model="gpt-3.5-turbo-0125")
sllm = llm.as_structured_llm(output_cls=Joke)

input_msg = ChatMessage.from_str("Tell me a joke about cats")

output = sllm.chat([input_msg])
output_obj = output.raw # Joke object

print(output_obj)
```


WIP:
1. Advanced file parsing with Docling. (`files` module) with conversion to langchain document object and llama-index document object.
2. Integration with common vector databases (`rag` module)
3. Making fastAPI and litestar auto doc generation more powerful (in the case of FastAPI, I'll hace to write pydantic classes, I'll see how it works internally to build this auto docs and do it.)


If you want to contribute:
local development:
```bash
git clone https://github.com/arthurbrenno/intellibricks.git
```

install `uv` https://docs.astral.sh/uv/getting-started/installation/

`uv sync`

Done!
