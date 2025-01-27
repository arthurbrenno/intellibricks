.. _index:

Intellibricks: The Building Blocks for Intelligent Applications 🧠🧱
=================================================================

.. image:: _static/quick_overview.svg
   :align: center
   :alt: Quick Overview of IntelliBricks

**The Python-First Framework for Agentic & LLM-Powered Applications**

Are you ready to build truly **intelligent applications** with the ease and clarity of Python?

**Welcome to IntelliBricks!**

IntelliBricks is more than just another LLM framework. It's a **developer-centric toolkit**, crafted from the ground up to empower you to create sophisticated AI applications with **unprecedented simplicity**.  We believe that building with AI should feel as intuitive and natural as writing Python itself.

IntelliBricks helps you overcome the common challenges of AI development:

* **Complexity Overload**:  Simplify development with a streamlined, Python-first approach, reducing layers of abstraction.
* **Unpredictable LLM Interactions**: Achieve reliable and structured outputs from Language Models using Python's type system.
* **Boilerplate Blues**: Eliminate repetitive setup and focus on building *intelligence*, not infrastructure.

Get Started Now!
----------------

.. code-block:: bash

   pip install intellibricks

   # Example Synapse Usage - Get a quick taste!
   from intellibricks.llms import Synapse
   synapse = Synapse.of("openai/api/gpt-3.5-turbo") # Or any free model
   response = synapse.complete("Say hello in a funny way!")
   print(f"IntelliBricks says: {response.text}")

Core Modules: Your Intelligent Toolkit
------------------------------------

IntelliBricks is built around three core modules, each designed to be powerful individually and seamlessly integrated for building truly intelligent applications:

.. toctree::
   :maxdepth: 1

   llms
   agents
   files

🧱 `LLMs Module <llms>`: Speak Fluently with AI Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``intellibricks.llms`` module is your gateway to Language Model Models (LLMs).  It provides the tools to interact with various LLMs in a consistent and Pythonic manner.

*   ✅ **Synapses**:  Smart adapters for connecting to different LLM providers (Google Gemini, OpenAI, Groq, and more). Switch models effortlessly!
*   ✅ **Structured Outputs**: Define your data models in pure Python and get perfectly formatted responses from LLMs. Say goodbye to messy string parsing!
*   ✅ **Chain of Thought**:  Leverage structured reasoning with the built-in ``ChainOfThought`` class for enhanced observability and debugging.
*   ✅ **Observability**: Seamless integration with Langfuse for tracing, monitoring, and debugging your LLM interactions.

.. code-block:: python
   :caption: Example: Basic Synapse Usage

   from intellibricks.llms import Synapse

   synapse = Synapse.of("google/genai/gemini-pro-experimental")
   response = synapse.complete("Tell me a joke.")
   print(response.text)


.. button::
   :text: Explore the LLMs Module
   :link: llms


🤖 `Agents Module <agents>`: Craft Autonomous Intelligent Entities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``intellibricks.agents`` module empowers you to build sophisticated, autonomous agents capable of performing complex tasks. Agents orchestrate LLM interactions and leverage tools to achieve specific goals.

*   ✅ **Agent Class**: The central building block for creating intelligent cores. Define tasks, instructions, metadata, and connect to Synapses.
*   ✅ **Tool Calling**: Equip your agents with tools to interact with external systems, access data, and perform real-world actions.
*   ✅ **Effortless API Generation**: Instantly turn your agents into production-ready REST APIs using FastAPI or Litestar with minimal code.

.. code-block:: python
   :caption: Example: Creating a Simple Agent

   from intellibricks.agents import Agent
   from intellibricks.llms import Synapse

   synapse = Synapse.of("openai/api/gpt-4o")
   agent = Agent(
       task="Joke Teller",
       instructions=["You are a funny joke teller."],
       synapse=synapse,
       metadata={"name": "JokeBot", "description": "Agent that tells jokes."},
   )
   response = agent.run("Tell me a joke about Python.")
   print(response.text)


.. button::
   :text: Dive into the Agents Module
   :link: agents


🗂️ `Files Module <files>`: Intelligent File Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``intellibricks.files`` module provides a robust way to handle and process files within your AI applications. Parse, extract, and understand content from various file types with ease.

*   ✅ **RawFile Abstraction**: Represent files as ``RawFile`` objects, encapsulating content, name, and extension for easy manipulation.
*   ✅ **Parsed Files**: Foundation for extracting structured information (text, images, tables) from files, making file data accessible to agents.
*   ✅ **File Parsing Infrastructure**:  Laying the groundwork for integrating file parsers for diverse file formats in future releases.

.. code-block:: python
   :caption: Example: Working with RawFile

   from intellibricks.files import RawFile

   raw_file = RawFile.from_file_path("document.pdf") # Or any file
   print(f"File Name: {raw_file.name}")
   print(f"File Extension: {raw_file.extension}")


.. button::
   :text: Learn about the Files Module
   :link: files


🏆 Why Choose IntelliBricks? The Intelligent Choice
---------------------------------------------------

IntelliBricks stands out from other frameworks by prioritizing **Python as a First-Class Citizen**.

*   🐍 **Python First**: Built with idiomatic Python, leveraging modern features for a truly Pythonic development experience.
*   ✨ **Unmatched Simplicity & Clarity**: Designed to be intuitive and easy to use, reducing complexity and boilerplate.
*   🧱 **Structured Outputs Out-of-the-Box**: Core strength in getting structured data from LLMs with blazingly fast and efficient definitions.
*   🧠 **Focus on Intelligence**: Concentrate on building intelligent logic, not framework intricacies. IntelliBricks handles the plumbing.

🚀 Join the IntelliBricks Revolution!
-------------------------------------

Ready to build truly intelligent applications, effortlessly?

* **Get Started:** ``pip install intellibricks``
* **Explore Examples:**  Dive into the :doc:`Quickstart <quickstart>` guide.
* **Contribute:** IntelliBricks is community-driven!  See our contribution guidelines to get involved.
* **Connect:** Reach out with questions, feedback, and ideas!

Let's build the future of intelligent applications, together!

.. toctree::
   :caption: Getting Started
   :hidden:
   :maxdepth: 1

   installation
   quickstart

.. toctree::
   :caption: User Guides
   :hidden:
   :maxdepth: 1

   agents
   llms
   files
   synapses

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 1

   api_reference


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`