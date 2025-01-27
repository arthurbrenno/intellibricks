<p align="center">
  <img src="/docs/_static/logo.png" alt="Logo" width="400" />
</p>



<h1 align="center">Start Building Intelligence</h1>

<p align="center">
  <b>The Python-first Framework for Agentic & LLM-Powered Applications</b>
</p>

---

**Stop wrestling with AI boilerplate. Start building intelligence.**

IntelliBricks is the **Python-first toolkit** for crafting AI applications with ease. Focus on your *intelligent logic*, not framework complexity.

![Quick Overview](/docs/_static/quick_overview.svg)

**Imagine this:**

*   **Pythonic AI:** Write clean, intuitive Python – IntelliBricks handles the AI plumbing.
*   **Structured Outputs, Instantly:** msgspec.Struct classes define your data, IntelliBricks gets you structured LLM responses.
*   **Agents that Understand:** Build autonomous agents with clear tasks, instructions, and your knowledge.
*   **APIs in Minutes:**  Deploy agents as REST APIs with FastAPI or Litestar, effortlessly.
*   **Context-Aware by Default:** Seamless RAG integration for informed, intelligent agents.

**IntelliBricks solves AI development pain points:**

*   **Complexity? Gone.** Streamlined, Python-first approach.
*   **Framework Chaos? Controlled.** Predictable, structured outputs with Python types.
*   **Boilerplate? Banished.** Focus on intelligence, predictability and observability. No more time setting the framework up.

**Start in Seconds:**

```bash
pip install intellibricks
```

---

## Core Modules: Your AI Building Blocks

IntelliBricks is built around three core modules, designed for power and seamless integration:

### 🧠 LLMs Module: Integrate easilly with AI providers

Interact with Language Models in pure Python.

**Key Features:**

*   **Synapses:** Connect to Google Gemini, OpenAI, Groq, and more with one line of code.

    ```python
    from intellibricks.llms import Synapse

    synapse = Synapse.of("google/genai/gemini-pro-experimental")
    completion = synapse.complete("Write a poem about Python.") # ChatCompletion[RawResponse]
    print(completion.text)
    ```

*   **Structured Outputs:** Define data models with Python classes using `msgspec.Struct`.

    ```python
    import msgspec
    from typing import Annotated, Sequence
    from intellibricks.llms import Synapse

    class Summary(msgspec.Struct, frozen=True):
        title: Annotated[str, msgspec.Meta(title="Title", description="Summary Title")]
        key_points: Annotated[Sequence[str], msgspec.Meta(title="Key Points")]

    synapse = Synapse.of("google/genai/gemini-pro-experimental")
    prompt = "Summarize quantum computing article: [...]"
    completion = synapse.complete(prompt, response_model=Summary) # ChatCompletion[Summary]

    print(completion.parsed.title)
    print(completion.parsed.key_points)
    ```

*   **Chain of Thought:** Structured reasoning with `ChainOfThought` for observability.

    ```python
    from intellibricks.llms import Synapse, ChainOfThought
    import msgspec

    class Response(msgspec.Struct):
        response: str
        """just to show you can combine ChainOfThought and other structured classes too"""

    synapse = Synapse.of("google/genai/gemini-pro-experimental")
    cot_response = synapse.complete(
        "Solve riddle: Cities, no houses...",
        response_model=ChainOfThought[Response] # You can use ChainOfThoughts[str] too!
    ) 

    for step in cot_response.parsed.steps:
        print(f"Step {step.step_number}: {step.explanation}")

    print(cot_response.parsed.final_answer) # Response
    ```

*   **Langfuse Observability:** Built-in integration for tracing and debugging.

    ```python
    from intellibricks.llms import Synapse
    from langfuse import Langfuse

    synapse = Synapse.of(..., langfuse=Langfuse())
    ```
    ![Langfuse](static/readme_images/langfuse_view.jpeg)

### 🤖 Agents Module: Build Autonomous Intelligence

Craft agents to perform complex tasks.

**Key Features:**

*   **Agent Class:** Define tasks, instructions, and connect to Synapses.

    ```python
    from intellibricks.agents import Agent
    from intellibricks.llms import Synapse

    synapse = Synapse.of("google/genai/gemini-pro-experimental")
    agent = Agent(
        task="Creative Title Generation",
        instructions=["Intriguing fantasy story titles."],
        metadata={"name": "TitleGen", "description": "Title Agent"},
        synapse=synapse,
    )

    agent_response = agent.run("Knight discovers dragon egg.") # AgentResponse[RawResponse]
    print(f"Agent suggests: {agent_response.text}")
    ```

*   **Tool Calling:** Equip agents with tools for real-world interaction.
*   **Instant APIs:** Turn agents into REST APIs with FastAPI/Litestar.

    ```python
    from intellibricks.agents import Agent
    from intellibricks.llms import Synapse
    import uvicorn

    agent = Agent(..., synapse=Synapse.of(...))
    app = agent.fastapi_app # WIP, any bugs open an issue please!
    uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

### 🗂️ Files Module:  Intelligent File Handling

Process files within your AI workflows.

**Key Features:**

*   **`RawFile` Abstraction:** Represent files as objects for easy handling.

    ```python
    from intellibricks.files import RawFile

    raw_file = RawFile.from_file_path("document.pdf")
    print(f"File Name: {raw_file.name}")
    print(f"File Extension: {raw_file.extension}")
    ```

*   **Parsed Files:** Foundation for structured content extraction (text, images, tables).

---

## 🏆 Why IntelliBricks? Python Purity & Power.

IntelliBricks is different. It's **Python First.**

*   🐍 **Idiomatic Python:**  Clean, modern Python – no framework jargon.
*   ✨ **Simplicity & Clarity:** Intuitive API, less boilerplate.
*   🧱 **Structured Outputs, Core Strength:** Define Python classes, get structured data.
*   🧠 **Focus on Intelligence:** Build smart apps, not infrastructure headaches.

---

## Structured Outputs: IntelliBricks vs. LangChain & LlamaIndex

Getting structured data from LLMs is critical. Here's how IntelliBricks compares to other frameworks:

**IntelliBricks:**

```python
import msgspec
from intellibricks.llms import Synapse

class Summary(msgspec.Struct, frozen=True):
    title: str
    key_points: list[str]

synapse = Synapse.of("google/genai/gemini-pro-experimental")
completion = synapse.complete(
    "Summarize article: [...]",
    response_model=Summary
) # ChatCompletion[Summary]

print(completion.parsed) # Summary object
```

**LangChain:**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(default=None, description="Rating 1-10")

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Joke)
joke = structured_llm.invoke(
    "Tell me a joke about cats"
) # Dict[Unknown, Unknown] | BaseModel

print(joke) # Joke object directly
```

*LangChain uses `.with_structured_output()` and Pydantic classes. While functional, it relies on Pydantic for validation and returns the Pydantic object directly via `.invoke()`, losing direct access to completion metadata (usage, time, etc.)*

**LlamaIndex:**

```python
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from datetime import datetime
import json

class Invoice(BaseModel):
    invoice_id: str = Field(...)
    date: datetime = Field(...)
    line_items: list = Field(...)

llm = OpenAI(model="gpt-4o")
sllm = llm.as_structured_llm(output_cls=Invoice)
response = llm.complete("...") # CompletionResponse

```

*Here is what LlamaIndex' returns:*
```py
class CompletionResponse(BaseModel):
    """
    Completion response.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response(i.e. token
            counts, function calling information).
        raw: Optional raw JSON that was parsed to populate text, if relevant.
        delta: New text that just streamed in (only relevant when streaming).
    """

    text: str
    additional_kwargs: dict = Field(default_factory=dict)
    raw: Optional[Any] = None # Could be anything and could be None too. Nice!
    logprobs: Optional[List[List[LogProb]]] = None
    delta: Optional[str] = None
```

**IntelliBricks Advantage:**

*   **Python-First Purity:**  Clean, idiomatic Python.
*   **Simpler Syntax:** More direct and intuitive structured output definition.
*   **Blazing Fast:** Leverages `msgspec` for high-performance serialization, outperforming Pydantic.
*   **Comprehensive Responses:**  `synapse.complete()` returns `ChatCompletion[RawResponse | T]` objects, providing not just parsed data but also full completion details (usage, timing, etc.).

*Examples adapted from LangChain [docs](https://python.langchain.com/docs/how_to/structured_output/) and LlamaIndex [docs](https://docs.llamaindex.ai/en/stable/understanding/extraction/structured_llms/). IntelliBricks offers a more streamlined and efficient Python-centric approach.*

---

## 🚀 Join the IntelliBricks Revolution!

Build intelligent applications, the Python way.

*   **Get Started:** `pip install intellibricks`
*   **Explore:** Dive into the [documentation](https://arthurbrenno.github.io/intellibricks/).
*   **Contribute:**  It's community-driven!
*   **Connect:**  Share feedback and ideas!

Let's build the future of intelligent applications, together!
