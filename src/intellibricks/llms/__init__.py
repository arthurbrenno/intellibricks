"""
Package: intellibricks.llms

This module is the heart of Language Model (LLM) interactions within the IntelliBricks framework.
It provides the necessary classes and interfaces to connect, communicate, and manage interactions
with various LLMs and transcription services.

**Core Functionality:**

*   **Synapses:** The central concept is the `Synapse`, which acts as a smart connector to different LLMs.
    Synapses handle API communication, parameter configuration, and response parsing, abstracting away the underlying complexities of each LLM provider.
*   **Chat and Completion:** Offers methods for both chat-based interactions (`chat`, `chat_async`) and simple text completion (`complete`, `complete_async`) with LLMs.
*   **Response Handling:** Defines structured data types like `ChatCompletion`, `MessageChoice`, and `Usage` to represent LLM responses and related metadata in a consistent and accessible manner.
*   **Fault Tolerance:** Introduces `SynapseCascade` and `TextTranscriptionsSynapseCascade` for creating resilient applications by automatically falling back to alternative synapses in case of failures.
*   **Observability:** Integrates with Langfuse for request tracing, performance monitoring, and debugging of LLM interactions.
*   **Flexibility:** Supports a wide range of LLM providers and models, allowing developers to easily switch and experiment with different options.

**Key Classes Exported:**

*   **`Synapse`**: The primary class for interacting with Language Models, offering methods for text completion and chat.
*   **`SynapseCascade`**: Provides fault tolerance by cascading through multiple `Synapse` instances.
*   **`TextTranscriptionSynapse`**: Specialized synapse for audio transcription tasks.
*   **`ChatCompletion`**: Represents the response from an LLM chat or completion request.
*   `AssistantMessage`, `DeveloperMessage`, `UserMessage`: Message types for different roles in a conversation.
*   `MessageChoice`: Represents a single choice in a `ChatCompletion` response.
*   `TraceParams`: Type hint for parameters used for tracing and observability with Langfuse.
*   `Usage`: Data class to capture token usage information from LLM calls.
*   `Prompt`: Represents a structured prompt, allowing for programmatic prompt construction.
*   `MessageType`: Enum defining message roles (User, Assistant, Developer).
*   `ChainOfThought`: Type hint for representing Chain of Thought reasoning in agent responses.
*   `TextPart`, `ImageFilePart`, `AudioFilePart`, `VideoFilePart`, `WebsitePart`: Concrete `Part` types for different content types within messages.

**Getting Started:**

To start using LLMs in IntelliBricks, you would typically begin by instantiating a `Synapse` object, configuring it with the desired LLM model and API credentials. You can then use the `complete` or `chat` methods to send prompts and receive responses from the LLM. For applications requiring higher reliability, explore `SynapseCascade`.

**Example (Synapse Creation):**

```python
from intellibricks.llms import Synapse

# Create a Synapse for Gemini Pro model
my_synapse = Synapse("google/genai/gemini-1.5-pro")

# You can now use my_synapse to interact with the Gemini Pro LLM
```

Explore the classes within this module to understand the full range of functionalities for LLM interaction in IntelliBricks.
"""

from .types import (
    AssistantMessage,
    CacheConfig,
    ChatCompletion,
    MessageChoice,
    Prompt,
    DeveloperMessage,
    Usage,
    UserMessage,
    TraceParams,
    MessageType,
    ChainOfThought,
    TextPart,
    ImageFilePart,
    AudioFilePart,
    VideoFilePart,
    WebsitePart,
)
from .synapses import (
    Synapse,
    SynapseCascade,
    TextTranscriptionSynapse,
    TextTranscriptionsSynapseCascade,
)

__all__ = [
    "Synapse",
    "ChatCompletion",
    "AssistantMessage",
    "DeveloperMessage",
    "UserMessage",
    "Usage",
    "MessageType",
    "MessageChoice",
    "TraceParams",
    "Prompt",
    "CacheConfig",
    "SynapseCascade",
    "TextTranscriptionSynapse",
    "ChainOfThought",
    "TextPart",
    "ImageFilePart",
    "AudioFilePart",
    "VideoFilePart",
    "WebsitePart",
    "TextTranscriptionsSynapseCascade",
]
