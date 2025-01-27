Agents: The Core of Intelligent Applications
============================================

In IntelliBricks, **Agents** are the fundamental building blocks for creating intelligent and autonomous applications. An Agent is an entity designed to perform specific tasks by interacting with Language Model Models (LLMs), leveraging tools, and managing contextual information. Think of Agents as specialized AI workers in your application, each with a defined role and capabilities.

Key Concepts of Agents
----------------------

* **Task Definition:** Each Agent is configured with a specific ``task``, clearly outlining its purpose. This task acts as the agent's primary objective.

* **Instructions:** Agents are guided by a set of ``instructions``, which are natural language directives on how to perform the task. Instructions shape the Agent's behavior and style.

* **LLM Synapse:** Agents are linked to a ``Synapse``, which provides the connection to a specific LLM. This allows Agents to communicate with and utilize the power of language models.

* **Metadata:** Agents can have associated ``metadata``, such as name and description. This metadata is useful for organization, identification, and for generating API endpoints when exposing agents as web services.

* **Tools:** Agents can be equipped with ``tools``, enabling them to interact with the external world, access information, or perform actions beyond simple text generation. Tools extend the Agent's capabilities significantly.

* **Context Sources (RAG):** Agents can be connected to ``context_sources`` for Retrieval-Augmented Generation (RAG). This allows Agents to ground their responses in external knowledge, enhancing their contextual awareness and accuracy.

* **Response Models:** Agents can be configured with ``response_model`` to ensure structured outputs from LLMs. This makes it easy to get predictable, parsed data from Agent runs.

Creating and Configuring Agents
-------------------------------

Let's explore how to create and configure Agents in IntelliBricks with Python code examples.

**Basic Agent Creation**

Here's how to create a basic Agent that generates creative story titles, similar to the Quickstart example, but with more detailed explanations.

.. code-block:: python

   from intellibricks.agents import Agent
   from intellibricks.llms import Synapse

   # 1. Initialize a Synapse (connection to an LLM)
   synapse = Synapse.of("openai/api/gpt-4o")

   # 2. Define Agent Metadata
   agent_metadata = {
       "name": "CreativeTitleAgent",
       "description": "Generates creative and intriguing titles for fantasy stories.",
   }

   # 3. Create the Agent instance
   title_agent = Agent(
       task="Generate Creative Story Titles for Fantasy Stories",
       instructions=[
           "You are an expert in creating captivating titles.",
           "Focus on fantasy stories and themes.",
           "Titles should be concise, intriguing, and relevant to the story's essence.",
       ],
       metadata=agent_metadata,
       synapse=synapse,
   )

   print(f"Agent '{title_agent.metadata['name']}' created successfully.")

**Running an Agent**

To execute a task with an Agent, you use the ``run()`` method, providing an input prompt.

.. code-block:: python

   story_description = "A tale of a young wizard who must save his magical school from a dark curse."
   agent_response = title_agent.run(story_description)

   print("\nSuggested Titles:")
   print(agent_response.text)

The ``agent_response`` object returned by ``run()`` is an ``AgentResponse`` instance, containing:

* ``agent``: The Agent object itself.
* ``content``: A ``ChatCompletion`` object with detailed information about the LLM completion (text, usage, etc.).
* ``text``: A convenience property to access the plain text response from the LLM.
* ``parsed``: If a ``response_model`` is configured, this property will contain the parsed, structured output.

**Agent with Response Model (Structured Output)**

To get structured output from an Agent, you can define a ``response_model`` using ``msgspec.Struct``. Let's create an Agent that summarizes articles and returns a structured summary with title and key points.

.. code-block:: python

   import msgspec
   from typing import Sequence, Annotated

   class ArticleSummary(msgspec.Struct, frozen=True):
       """Structured summary of an article."""
       title: Annotated[str, msgspec.Meta(title="Title", description="Summary Title")]
       key_points: Annotated[Sequence[str], msgspec.Meta(title="Key Points", description="Main points of the summary")]

   summary_agent = Agent[ArticleSummary]( # Note the generic type hint [ArticleSummary]
       task="Summarize Articles into Structured Format",
       instructions=[
           "You are an expert article summarizer.",
           "Extract the main title and key points from the article.",
           "Format the output as a structured JSON object.",
       ],
       metadata={
           "name": "ArticleSummarizerAgent",
           "description": "Agent for summarizing articles into structured summaries.",
       },
       synapse=synapse,
       response_model=ArticleSummary # Pass the response model class here
   )

   article_text = """
   Quantum computing is a revolutionary technology... (article text here) ...
   Key takeaways: Quantum computers leverage superposition and entanglement...
   """ # Replace with actual article text

   summary_response = summary_agent.run(article_text)

   if summary_response.parsed: # Check if parsing was successful
       structured_summary = summary_response.parsed # Access parsed output
       print("\nStructured Summary:")
       print(f"Title: {structured_summary.title}")
       print("Key Points:")
       for point in structured_summary.key_points:
           print(f"- {point}")
   else:
       print("\nRaw Response (Parsing Failed):")
       print(summary_response.text) # Fallback to raw text if parsing fails

In this example:

* We define ``ArticleSummary`` as a ``msgspec.Struct`` to represent the structured summary.
* We create ``summary_agent`` and set ``response_model=ArticleSummary``, and use generic type hint ``Agent[ArticleSummary]``.
* When running the agent, ``summary_response.parsed`` will contain an instance of ``ArticleSummary`` if the LLM output is successfully parsed into the defined structure.

**Agents with Tools (Function Calling)**

Agents can be equipped with tools to perform actions or access external resources. Let's create an Agent that can use a search tool.

.. code-block:: python

   from intellibricks.llms.tools import DuckDuckGoTool

   search_agent = Agent(
       task="Answer Questions Using Web Search",
       instructions=[
           "You are a helpful research assistant.",
           "Use web search to find answers to user questions.",
           "Provide concise and informative answers based on search results.",
       ],
       metadata={
           "name": "SearchAgent",
           "description": "Agent capable of using web search to answer questions.",
       },
       synapse=synapse,
       tools=[DuckDuckGoTool], # Add DuckDuckGoTool to the agent's tools
   )

   question = "What is the capital of France?"
   search_response = search_agent.run(question)

   print("\nSearch Agent Response:")
   print(search_response.text)

In this setup:

* We import ``DuckDuckGoTool``, a pre-built tool in IntelliBricks for web searching.
* We add ``tools=[DuckDuckGoTool]`` to the Agent configuration.
* When the Agent runs, it can now decide to use the ``DuckDuckGoTool`` if it determines that web search is needed to answer the question.

**Note on Tool Calling:**  The LLM decides *when* and *how* to use the tools based on the prompt and agent instructions. IntelliBricks handles the mechanics of tool invocation and response handling.

Agent API Endpoints (FastAPI and Litestar)**

IntelliBricks makes it incredibly easy to expose your Agents as REST APIs using FastAPI or Litestar.

**FastAPI Example:**

.. code-block:: python

   from intellibricks.agents import Agent
   from intellibricks.llms import Synapse
   import uvicorn

   # ... (Define your agent as in previous examples, e.g., title_agent) ...

   app = title_agent.fastapi_app # Get a FastAPI app instance for the agent

   # Run the FastAPI application
   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)

Now, your ``title_agent`` is available as an API endpoint. You can send POST requests to ``/agents/creativetitleagent/completions`` (path is derived from agent metadata name) with a JSON payload containing the input for the agent.

**Litestar Example:**

.. code-block:: python

   from intellibricks.agents import Agent
   from intellibricks.llms import Synapse
   from litestar import Litestar

   # ... (Define your agent, e.g., summary_agent with response_model) ...

   app = summary_agent.litestar_app # Get a Litestar app instance

   # Run the Litestar application
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000)

Similarly, your ``summary_agent`` becomes a Litestar-powered API.

Agent Lifecycle: ``_before_run`` and ``_after_run`` Methods

For advanced customization, Agents support ``_before_run``, ``_after_run``, and their asynchronous counterparts ``_before_run_async``, ``_after_run_async`` methods. These methods allow you to inject custom logic at the start and end of the Agent's ``run`` or ``run_async`` execution.

Example: Logging Agent Runs

.. code-block:: python

   from intellibricks.agents import Agent, AgentResponse
   from intellibricks.llms import Message, RawResponse, Synapse
   from typing import Optional

   class LoggingAgent(Agent[RawResponse]): # Or any other response model
       async def _before_run_async(self, inp: Sequence[Message], trace_params: Optional[dict] = None) -> None:
           print(f"Agent '{self.metadata['name']}' started run with input: {inp}")
           return await super()._before_run_async(inp, trace_params)

       async def _after_run_async(self, inp: Sequence[Message], response: AgentResponse[RawResponse], trace_params: Optional[dict] = None) -> None:
           print(f"Agent '{self.metadata['name']}' completed run. Response text: {response.text[:50]}...") # Print first 50 chars
           return await super()._after_run_async(inp, response, trace_params)

   logging_agent = LoggingAgent(
       task="Simple Echo Agent with Logging",
       instructions=["You are an echo agent. Just repeat the user input."],
       metadata={"name": "EchoLogger", "description": "Agent that echoes inputs and logs runs."},
       synapse=synapse,
   )

   echo_response = logging_agent.run("Hello, IntelliBricks!")
   print(f"\nAgent Response: {echo_response.text}")

When you run ``logging_agent``, you will see log messages before and after the LLM call, demonstrating how you can extend Agent behavior with these lifecycle methods.

Summary

IntelliBricks Agents provide a powerful and flexible way to build intelligent applications. They encapsulate task logic, LLM interactions, tool usage, and structured output handling, all within a clean and Pythonic framework. As you delve deeper into IntelliBricks, you'll discover how Agents become the central orchestrators of your AI-powered solutions.

API Reference
-------------

.. automodule:: intellibricks.agents
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: intellibricks.agents.agents
   :members:
   :undoc-members:
   :show-inheritance:
