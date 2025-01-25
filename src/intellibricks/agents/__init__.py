"""
Package: intellibricks.agents

This module defines the core concept of `Agent` within the IntelliBricks framework.
Agents are the building blocks for creating intelligent, autonomous applications powered by Language Model Models (LLMs).
They encapsulate the logic for interacting with LLMs, managing state, and performing complex tasks.

**Core Functionality:**

*   **Agent Class:** The `Agent` class is the central component for building intelligent agents. It orchestrates interactions with LLMs through `Synapse` objects, allowing for text completion, chat, and structured data extraction.
*   **Task Execution:** Agents are designed to execute tasks by interacting with LLMs, interpreting responses, and potentially using tools or external resources.
*   **Flexibility and Extensibility:** The `Agent` class is designed to be flexible and extensible, allowing developers to customize agent behavior, integrate custom tools, and adapt to various application requirements.
*   **Integration with Synapses:** Agents rely on `Synapse` objects to communicate with specific LLMs, enabling easy switching between different models and providers.

**Key Classes Exported:**

*   **`Agent`**: The primary class for creating intelligent agents. It provides methods to `run` and `run_async` tasks using configured LLMs.

**Getting Started:**

To create an intelligent agent with IntelliBricks, you would typically instantiate an `Agent` object, providing it with a `Synapse` instance to define the LLM it will use. You can then use the `run` or `run_async` methods to execute tasks by providing prompts or structured inputs to the agent.

**Example (Agent Creation and Basic Task):**

```python
from intellibricks.agents import Agent
from intellibricks.llms import Synapse

# Create a Synapse instance
my_synapse = Synapse("openai/api/gpt-4o")

# Create an Agent instance, linked to the Synapse
my_agent = Agent([...], synapse=my_synapse)

# Run a task with the agent
response = my_agent.run("Summarize the main points of quantum physics.")
print(response)
```

Explore the `Agent` class documentation to understand its configuration options, task execution methods, and how to integrate custom functionalities.
"""

from .agents import Agent, AgentInput, AgentMetadata, AgentResponse

__all__: list[str] = [
    "Agent",
    "AgentInput",
    "AgentMetadata",
    "AgentResponse",
]
