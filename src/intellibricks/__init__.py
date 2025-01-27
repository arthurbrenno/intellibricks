"""
Package: intellibricks

IntelliBricks is a cutting-edge Agentic/LLM framework meticulously crafted for developers like you.
It's designed from the ground up to make your language (Python) a first-class citizen when building AI-powered applications.
By leveraging the latest features and capabilities of modern Python (3.13+), including `default generics`,
IntelliBricks ensures a seamless experience with structured outputs and predictable LLM interactions.
Say goodbye to the boilerplate and hello to intelligent applications built with ease!

**Core Modules:**

*   **`intellibricks.agents`**: Provides the `Agent` class, the central component for building intelligent agents that interact with LLMs and perform complex tasks.
*   **`intellbricks.llms`**: Contains modules related to Language Model Models (LLMs), including:
    *   **`synapses`**: Defines the `Synapse` class for interacting with various LLMs and `TextTranscriptionSynapse` for audio transcriptions, along with cascade classes for fault tolerance.
    *   **`types`**: Defines core data types and schema objects used throughout the framework, such as `ChatCompletion`, `Message`, `Part`, `Prompt`, `TraceParams`, and more.
*   **`intellibricks.files`**: Offers classes for handling files as `RawFile` objects and parsing various file types using modules in `intellibricks.parsers`.

**Key Classes and Exports:**

This module makes the most commonly used classes and types directly available under the `intellibricks` namespace for convenience.
Below is a summary of the key exports:

*   **LLM Interaction:**
    *   `Synapse`: The primary class for interacting with Language Models, providing methods for text completion and chat.
    *   `SynapseCascade`: Enables fault tolerance by cascading through multiple `Synapse` instances.
    *   `TextTranscriptionSynapse`: Specialized synapse for audio transcription tasks.
    *   `ChatCompletion`: Represents the response from an LLM chat or completion request.
    *   `MessageChoice`: Represents a single choice in a `ChatCompletion` response.
    *   `Usage`: Data class to capture token usage information from LLM calls.
    *   `TraceParams`: Type hint for parameters used for tracing and observability with Langfuse.
    *   `Prompt`: Represents a structured prompt, allowing for programmatic prompt construction.
    *   `MessageType`: Enum defining message roles (User, Assistant, Developer).
    *   `AssistantMessage`, `DeveloperMessage`, `UserMessage`: Concrete message types for different roles in a conversation.
    *   `ChainOfThought`: Type hint for representing Chain of Thought reasoning in agent responses.
    *   `Part`, `PartType`: Abstract base classes for representing different content types within messages.
    *   `TextPart`, `ImageFilePart`, `AudioFilePart`, `VideoFilePart`, `WebsitePart`: Concrete `Part` types for text, images, audio, video and website content.

*   **Agentic Framework:**
    *   `Agent`: The core class for creating intelligent agents, orchestrating LLM interactions and task execution.

*   **File Handling:**
    *   `RawFile`: Represents a raw file with its content, name, and extension, used for file processing within IntelliBricks.

**Getting Started:**

To begin building with IntelliBricks, you can start by creating a `Synapse` to connect to an LLM and then use `Agent` to define your intelligent application logic. For file processing, utilize `RawFile` and explore the `intellibricks.parsers` module.

**Example (Basic Synapse Usage):**

```python
from intellibricks.llms import Synapse

# Create a Synapse instance
my_synapse = Synapse("openai/api/gpt-4o")

# Perform a text completion
response = my_synapse.complete(prompt="Write a short poem about the sea.")
print(response.message.content)
```

For more detailed information, explore the individual modules and class documentations within the `intellibricks` package.
"""
