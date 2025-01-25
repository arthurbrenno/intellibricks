<h1 align="center">🧠🧱 IntelliBricks: The Building Blocks for Intelligent Applications..</h1>

<p align="center">
  <b>The Python-First Framework for Agentic & LLM-Powered Applications</b>
</p>

---

**Are you tired of wrestling with boilerplate and complex configurations when building AI applications?**

Do you dream of a framework that speaks your language – **Python** – fluently, allowing you to focus on *intelligence* rather than infrastructure?

**Welcome to IntelliBricks.**

![Quick Overview](/docs/_static/quick_overview.svg)

IntelliBricks is not just another LLM framework. It's a **developer-centric toolkit** built from the ground up to empower you to create intelligent applications with unprecedented ease and clarity.  We believe that building with AI should feel as natural and intuitive as writing Python itself.

**Imagine this:**

* **Structured Outputs, Instantly:**  Define your data models in pure Python and IntelliBricks handles the magic of getting LLMs to return perfectly formatted responses. No more messy string parsing or hard to configure LLM setups!
* **Agents that Truly Understand:**  Craft autonomous agents with clear tasks, instructions, and access to your own knowledge – all in clean, Pythonic code.
* **APIs in Minutes:**  Turn your intelligent agents into production-ready REST APIs with just a few lines of code, using FastAPI or Litestar.
* **Contextual Awareness Built-In:** Seamlessly integrate Retrieval-Augmented Generation (RAG) to give your agents access to a wealth of information.

**IntelliBricks is designed to solve the core challenges of building intelligent applications:**

* **Complexity Overload:**  Other frameworks often bury you in layers of abstraction and configuration. IntelliBricks provides a streamlined, Python-first approach that *reduces* complexity.
* **Unpredictable LLM Interactions:**  Getting LLMs to return structured data can be a nightmare. IntelliBricks uses Python's type system and advanced features to ensure predictable and reliable interactions.
* **Boilerplate Blues:**  Spending more time on setup than on actual application logic? IntelliBricks eliminates boilerplate, letting you focus on what truly matters: building *intelligence*.

**Get Started in Seconds:**

```bash
pip install intellibricks
```

---

## Core Modules: The Building Blocks of Intelligence

IntelliBricks is structured around three core modules, each designed to be powerful on its own, yet seamlessly integrated to create truly intelligent applications.

### 🧱 LLMs Module:  Speak Fluently with AI Models

The `intellibricks.llms` module is your gateway to the world of Language Model Models. It provides the tools to interact with various LLMs in a consistent and Pythonic way.

**Key Concepts:**

* **Synapses: Your Connection to AI:** Think of `Synapse` as a smart adapter. It handles the low-level communication with different LLM providers (Google Gemini, OpenAI, Groq, and more), abstracting away API complexities.  Switch models with a single line of code!

    ```python
    from intellibricks import Synapse

    # Connect to Google Gemini Pro
    synapse = Synapse.of("google/genai/gemini-pro-experimental")

    # Get a simple text completion
    completion = synapse.complete("Tell me a short story about a robot learning to love.")
    print(completion.text)
    ```

* **Structured Outputs with `response_model`:**  Say goodbye to parsing free-form text or spending time doing weird framework setups just to get structured outputs!  Define your desired output structure using Python classes with `msgspec.Struct`, and IntelliBricks will ensure the LLM returns data in that format.

    ```python
    import msgspec
    from typing import Annotated
    from intellibricks import Synapse, ChainOfThought

    class Summary(msgspec.Struct, frozen=True):
        title: Annotated[str, msgspec.Meta(title="Title", description="Summary Title")]
        key_points: Annotated[Sequence[str], msgspec.Meta(title="Key Points", description="Main points of the summary")]

    synapse = Synapse.of("google/genai/gemini-pro-experimental")

    prompt = "Summarize the key takeaways from the article about quantum computing: [...]"

    structured_summary = synapse.complete(prompt, response_model=Summary)

    print(structured_summary.parsed.title)
    print(structured_summary.parsed.key_points)
    ```

* **Chain of Thought (`ChainOfThought`) Class:**  Leverage structured reasoning directly. IntelliBricks provides a built-in `ChainOfThought` class to capture the LLM's thought process in a structured, auditable way.  Enhance observability and debug complex reasoning steps.

    ```python
    from intellibricks import Synapse, ChainOfThought

    synapse = Synapse.of("google/genai/gemini-pro-experimental")

    prompt = "Solve this riddle: I have cities, but no houses, forests, but no trees, and water, but no fish. What am I?"

    cot_response = synapse.complete(prompt, response_model=ChainOfThought[str])

    print(cot_response.parsed.title) # e.g., "Riddle Solving"
    for step in cot_response.parsed.steps:
        print(f"Step {step.step_number}: {step.explanation}")
        for detail in step.details:
            print(f"  - Detail: {detail.detail}")
    print(cot_response.parsed.final_answer) # e.g., "A map"
    ```

* **Observability with Langfuse:**  IntelliBricks integrates seamlessly with [Langfuse](https://langfuse.com/) for tracing, monitoring, and debugging your LLM interactions. Gain deep insights into your application's performance and LLM behavior.

### 🤖 Agents Module:  Craft Autonomous Intelligent Entities

The `intellibricks.agents` module empowers you to build sophisticated, autonomous agents that can perform complex tasks. Agents are the core of intelligent applications, orchestrating LLM interactions and leveraging tools to achieve specific goals.

**Key Concepts:**

* **The `Agent` Class: Your Intelligent Core:**  The `Agent` class is the central building block. Define an agent's `task`, `instructions`, `metadata`, and connect it to a `Synapse`.  Agents can then be easily run with simple prompts.

    ```python
    from intellibricks import Agent, Synapse

    synapse = Synapse.of("google/genai/gemini-pro-experimental")

    agent = Agent(
        task="Generate Creative Story Titles",
        instructions=[
            "You are a creative title generator.",
            "Focus on titles that are intriguing and relevant to fantasy stories.",
        ],
        metadata={"name": "TitleGen", "description": "Creative Story Title Agent"},
        synapse=synapse,
    )

    agent_response = agent.run("A story about a knight who discovers a hidden dragon egg.")
    print(f"Agent '{agent.metadata['name']}' suggests title: {agent_response.text}")
    ```

* **Tool Calling: Connecting to the Real World:** Equip your agents with tools to interact with external systems, access data, and perform actions. IntelliBricks makes tool integration straightforward. (Tool examples and more advanced agent features can be added here in a more detailed section).

* **Effortless API Generation:**  Turn your agents into REST APIs instantly!  IntelliBricks provides built-in support for FastAPI and Litestar, allowing you to deploy your intelligent agents as web services with minimal effort.

    ```python
    from intellibricks import Agent, Synapse
    import uvicorn

    agent = Agent(
        task="Simple Chat Agent",
        instructions=["Chat politely with the user."],
        metadata={"name": "ChatBot", "description": "A basic chatbot API."},
        synapse=Synapse.of("google/genai/gemini-pro-experimental"),
    )

    # Create a FastAPI app for your agent - API endpoint: POST /agents/chatbot/completions
    app = agent.fastapi_app
    uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

### 🗂️ Files Module:  Intelligent File Handling

The `intellibricks.files` module provides a robust way to handle and process files within your AI applications.  Parse, extract, and understand content from various file types with ease.

**Key Concepts:**

* **`RawFile`: Your File Abstraction:**  Represent files as `RawFile` objects, encapsulating file content, name, and extension. Load files from paths, bytes, or in-memory streams.

    ```python
    from intellibricks import RawFile
    from pathlib import Path

    file_path = Path("my_document.pdf") # Assuming you have a PDF
    raw_file = RawFile.from_file_path(file_path)

    print(f"File Name: {raw_file.name}")
    print(f"File Extension: {raw_file.extension}")
    # raw_file.contents now holds the raw bytes of the PDF
    ```

* **Parsed Files and Structured Content:**  IntelliBricks is designed to work with parsed file content.  While the parsing functionalities are WIP and being enhanced, the module lays the foundation for extracting structured information (text, images, tables) from files, making it easier to feed file data into your agents and LLM workflows. (More details on parsing capabilities and future integrations can be expanded here).

---

## 🏆 Why IntelliBricks?  The Intelligent Choice

IntelliBricks stands out from other frameworks like LangChain and LlamaIndex because it's built with a core philosophy: **Python First.**

* **Python as a First-Class Citizen:**  IntelliBricks leverages the latest Python features (like type hints and default generics) to provide a truly Pythonic experience. Write clean, idiomatic code, not framework-specific abstractions.

* **Unmatched Simplicity and Clarity:**  IntelliBricks is designed to be intuitive and easy to use, even for complex tasks.  We prioritize developer experience, eliminating unnecessary complexity and boilerplate.

* **Structured Outputs, Out of the Box:**  Getting structured data from LLMs is a core strength of IntelliBricks.  Define your data models in Python and let the framework handle the rest.  Compare this to the more verbose or less integrated structured output mechanisms in LangChain and LlamaIndex. (You could add a small code comparison example here later if needed).

* **Focus on Intelligence, Not Infrastructure:**  IntelliBricks lets you concentrate on building the *intelligent logic* of your application, not wrestling with framework intricacies.  We handle the plumbing, so you can build the magic.

---

## 📚 Deep Dive: Exploring the Modules

**(You can keep the "Deep Dive: The Schema Module" section from your original README here and adapt it slightly to fit the new structure and focus on the core modules.  Potentially rename it to "Deep Dive: Key Concepts and Modules" and expand on Agents and Files too, not just schema).**

**(The "WIP (Work in Progress) & Contribution" and "Local Development" and "License" sections should be kept and potentially enhanced, adding specific areas where contributions are welcome based on the current TODO and WIP notes in your files).**

---

## 🚀 Join the IntelliBricks Revolution!

Ready to build truly intelligent applications, effortlessly?

* **Get Started:** `pip install intellibricks`
* **Explore Examples:** Dive into the `tests/example.py` and `docs/examples` for practical code snippets.
* **Contribute:**  IntelliBricks is a community-driven project!  See our "WIP & Contribution" section to get involved.
* **Connect:** Reach out with questions, feedback, and ideas!

**Let's build the future of intelligent applications, together!**
