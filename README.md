<p align="center">
  <img src="/docs/readme_images/logo.png" alt="IntelliBricks Logo" width="500" />
</p>

# 🧠🧱 IntelliBricks: The Building Blocks for Intelligent Applications

<p align="center">
  <img src="/docs/readme_images/quick_overv.jpeg" alt="Quick Overview" width="600" />
</p>


<p align="center">
 <img src="/docs/readme_images/advantages.jpeg" alt="IntelliBricks Advantages" width="600"/>
</p>

IntelliBricks is a cutting-edge Agentic/LLM framework meticulously crafted for developers like you. It's designed from the ground up to make your language (Python) a first-class citizen when building AI-powered applications. By leveraging the latest features and capabilities of modern Python (3.13+), including `default generics`, IntelliBricks ensures a seamless experience with structured outputs and predictable LLM interactions. Say goodbye to the boilerplate and hello to intelligent applications built with ease!

```bash
pip install intellibricks
```

## Core Concepts & Getting Started

### ⚙️ Synapses: The Foundation of Interaction

Imagine a synapse in your brain—a connection that allows signals to pass between neurons. In IntelliBricks, a `Synapse` is that connection, linking your code to powerful AI models. It's your go-to for interacting with LLMs.

**Basic Usage:**

```python
from intellibricks import Synapse

synapse = Synapse.of("google/genai/gemini-2.0-flash-exp")

completion = synapse.complete("Hello, how are you?")  # Completion[RawResponse]

print(completion)
```

This simple code snippet shows the core of how IntelliBricks simplifies LLM interactions.  `Synapse.of()` creates an instance of the `Synapse` object. Then the `complete` method is used for single-turn prompts. The type of the completion is `Completion[RawResponse]`.

**Advanced Synapse Usage:**

IntelliBricks gives you the power to obtain more robust, structured data from LLMs through the use of *Structured Outputs*. It also provides powerful tools for observability with Langfuse. Here's an example:

```python
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
    secret_key="sk-lf-0be2e5c3-6c86-421c-ad5f-ffb4c065daa0", # Your secret key here
    public_key="pk-lf-753848ca-2150-473e-a335-4970fb550a20", # Your public key here
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

In this example we defined a `ModelInfo` class using `msgspec.Struct` for structured outputs and passed this into the `synapse.chat()` method through the parameter `response_model`. We also defined some `TraceParams` which can be used to pass information to langfuse. Below, is how it could look like in langfuse:

<p align="center">
  <img src="/docs/readme_images/langfuse_view.jpeg" alt="Quick Overview" width="600" />
</p>

### 🤖 Agents: Autonomous Entities

Agents in IntelliBricks represent autonomous entities capable of performing specific tasks. They leverage Synapses for LLM interaction and can be customized with instructions, tools, and even context from your own knowledge bases.

**Basic Agent Usage:**

```python
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

In this example we created an `Agent` and passed a task, instructions, metadata and the `synapse` object. We also passed the `ModelInfo` as the `response_model`. Then we can call the `run` method passing in a text prompt and get an AgentResponse object.

**Advanced Agent Usage**

IntelliBricks Agents can make use of *Tool Calling*, which allows them to connect with external world.


### 🚀 Real World Magic: Turning Agents into APIs

IntelliBricks truly shines with its ability to seamlessly convert your agents into fully functional APIs. Whether you prefer FastAPI or Litestar, IntelliBricks makes this process incredibly simple.

**FastAPI Integration:**

```python
from intellibricks import (
    Synapse,
    Agent,
)
import uvicorn
from fastapi import FastAPI

agent = Agent(
    task="Chat With the User",
    instructions=[
        "Chat with the user",
    ],
    metadata={"name": "Bob", "description": "A simple chat agent."},
    synapse=Synapse.of("google/genai/gemini-2.0-flash-exp"),
)

# Create an app using the method to_fastapi_async_app
# Note that this will automatically create the endpoints with the `metadata.name`
# e.g. /agents/{lower_agent_name}/completions
uvicorn.run(agent.fastapi_app)
```

Or you can just get the router itself:
```python
from intellibricks import (
    Synapse,
    Agent,
)
import uvicorn
from fastapi import FastAPI

agent = Agent(
    task="Chat With the User",
    instructions=[
        "Chat with the user",
    ],
    metadata={"name": "Bob", "description": "A simple chat agent."},
    synapse=Synapse.of("google/genai/gemini-2.0-flash-exp"),
)

# Create the router using the method to_fastapi_async_router
router = agent.to_fastapi_async_router("/agents/bob/completions", "post")

app = FastAPI()
app.include_router(router)
uvicorn.run(app)
```

This code will create a fully functional REST API for your agent with just a few lines of code. The API endpoint will be  `POST /agents/bob/completions`.

**Litestar Integration:**

The same goes for Litestar!
```python
from intellibricks import (
    Synapse,
    Agent,
)
import uvicorn
from litestar import Litestar

agent = Agent(
    task="Chat With the User",
    instructions=[
        "Chat with the user",
    ],
    metadata={"name": "Bob", "description": "A simple chat agent."},
    synapse=Synapse.of("google/genai/gemini-2.0-flash-exp"),
)

# Create an app using the method to_litestar_async_app
# Note that this will automatically create the endpoints with the `metadata.name`
# e.g. /agents/{lower_agent_name}/completions
uvicorn.run(agent.litestar_app)
```

The API endpoint will be `POST /agents/bob/completions`.
### 🔍 Contextual Understanding with RAG

IntelliBricks integrates seamlessly with Retrieval-Augmented Generation (RAG) to provide your agents with relevant contextual information.

```python
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
    context_sources=[MyFakeGraphDB("localhost")],
)
```

This code demonstrates how you can create a custom data source that can be retrieved during an agent execution.

### 🏆 Why IntelliBricks?

IntelliBricks stands out from other frameworks such as LangChain and LlamaIndex due to its unique approach. By treating Python as a first-class citizen and leveraging advanced Python features, IntelliBricks provides a more streamlined and developer-friendly experience, but let's see some examples, take the conclusions yourself:

**LangChain**

Langchain offers structured output parsing using the `with_structured_output` method.

```python
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

**LlamaIndex**

LlamaIndex provides the method `as_structured_llm` to achieve structured outputs.

```python
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

## 📚 Deep Dive: The Schema Module

This section provides a comprehensive overview of the core classes defined in the `schema.py` module.

### Configuration and Meta-Data

*   **`GenerationConfig`**: Defines how completions are generated, including settings like `temperature`, `max_tokens`, `cache_config` and `trace_params`.
*   **`RawResponse`**: Represents a null object (design pattern) returned by the model.
*   **`TraceParams`**: Parameters for updating the current trace metadata and context information (used with Langfuse).
*   **`WebsiteUrl`**: Represents a URL to a website.
*   **`FileUrl`**: Represents a URL to a file.

### Data Types

*   **`Part`**:  Abstract base class representing a piece of content in a message. Several useful implementations:
    *   **`TextPart`**: Contains a plain text piece of content.
    *   **`ToolResponsePart`**: A specific type of content representing a response from a called tool.
    *   **`FilePart`**: An abstract class that represents file-based part, with the following implementations:
        *   `VideoFilePart`: Represents a file of a type of a video (`.mp4`, `.avi`, etc).
        *   `AudioFilePart`: Represents a file of a type of an audio (`.mp3`, `.wav`, etc).
        *   `ImageFilePart`: Represents a file of a type of an image (`.jpeg`, `.png`, etc).
    *   **`ToolCallPart`**: Represents a function call and its arguments.
*   **`PartType`**: A type alias that represents all types of `Part`.
*   **`Prompt`**: Represents a string of text which can be compiled and have placeholders.
*   **`ToolCall`**: Represents a call to a specific tool (`function`).
*   **`CalledFunction`**: Represents a function that was called with arguments by the LLM
*   **`Function`**: Represents a function with a name, description, parameters, and callable object. It can be used to construct an openai, groq, google and cerebras function object.
    *   `Property`: Represents a property of a parameter.
    *   `Parameter`: Represents a parameter of a function.
*   **`Message`**:  An abstract base class for different message types.
    *   **`DeveloperMessage`**: A system message used to inform the model about its role and instructions.
    *   **`UserMessage`**: A message sent by the user.
    *   **`ToolMessage`**: A message representing the output of a called tool.
    *   **`AssistantMessage`**: A response from the language model.
    *   **`MessageType`**: A type alias representing all types of messages.
    *   **`MessageSequence`**: Represents a sequence of `Messages`.
*   **`LogProb`**: Represents the log probability of a token.
*   **`MessageChoice`**: Represents a specific choice returned by a language model.
*   **`PromptTokensDetails`**: Represents tokens used by the prompt, such as `audio_tokens`, `cached_tokens`.
*   **`CompletionTokensDetails`**: Represents tokens generated in the completion, such as `audio_tokens` and `reasoning_tokens`.
*   **`Usage`**: Represents usage statistics for a completion, including prompt tokens, completion tokens, and costs.
*   **`ChatCompletion`**: Represents a full response from an LLM, containing message choices, usage statistics, model information, and elapsed time.

### Relationships

The key relationships between classes revolve around creating a structured way to interact with LLMs:

*   A `Synapse` uses `Prompt`s or `Messages` to interact with an LLM and get a `ChatCompletion`.
*   A `ChatCompletion` contains `MessageChoice`s, each including an `AssistantMessage` that contains `Part`s.
*   `AssistantMessage` may contain `ToolCall`s, or even `ToolMessage`s.
*   `Agent` uses `Synapse`,  `GenerationConfig`, `Context`s and tools to generate `ChatCompletion`s and `AgentResponse`s.

## 🚧 WIP (Work in Progress) & Contribution

*   Advanced file parsing with Docling. (`files` module) with conversion to langchain document object and llama-index document object.
*   Integration with common vector databases (`rag` module)
*   Making fastAPI and litestar auto doc generation more powerful

If you're excited about pushing the boundaries of AI development and want to contribute, here's how you can get involved:

**Local Development:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/arthurbrenno/intellibricks.git
    ```
2.  Install `uv` following the instructions here:  https://docs.astral.sh/uv/getting-started/installation/
3.  Sync dependencies:
    ```bash
    uv sync
    ```
    That's it!

## 📜 License

This project is licensed under the MIT license.
