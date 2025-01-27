Quickstart: Your First IntelliBricks Application
================================================

Ready to dive into IntelliBricks? This quickstart guide will walk you through creating a simple yet illustrative application that demonstrates the core functionalities of IntelliBricks.

In this guide, we will build a basic **Creative Title Generator Agent**. This agent will take a topic as input and suggest creative and intriguing titles for stories or articles related to that topic.

Step 1: Set up your Python Environment
--------------------------------------

Ensure you have IntelliBricks installed as per the :doc:`Installation <installation>` guide. Also, make sure you have your API keys configured as environment variables if you plan to use cloud-based LLMs.

Step 2: Import Necessary Modules
--------------------------------

Start by importing the required classes from IntelliBricks:

.. code-block:: python

   from intellibricks.agents import Agent
   from intellibricks.llms import Synapse

Step 3: Initialize a Synapse
----------------------------

A ``Synapse`` is your connection to a Language Model Model (LLM). Let's initialize a Synapse to use Google's Gemini Pro model. You can choose other models as well, provided you have the necessary API access.

.. code-block:: python

   synapse = Synapse.of("openai/api/gpt-4o")

   # Ensure you have set up your GOOGLE_API_KEY environment variable.
   # If you face issues, refer to the Synapse documentation for configuration details.

Step 4: Create your Agent
-------------------------

Now, let's define our Creative Title Generator Agent. We'll use the ``Agent`` class and configure it with a task, instructions, metadata, and the Synapse we just created.

.. code-block:: python

   agent = Agent(
       task="Generate Creative Story Titles",
       instructions=[
           "You are a creative title generator.",
           "Focus on titles that are intriguing and relevant to fantasy stories.",
           "Keep titles concise and impactful.",
       ],
       metadata={
           "name": "TitleGenAgent",
           "description": "Agent specialized in creating fantasy story titles.",
       },
       synapse=synapse,
   )

   # Agent is now configured and ready to run tasks.

Let's break down the Agent configuration:

* ``task``:  A concise description of what this agent is designed to do.
* ``instructions``: A list of strings providing detailed instructions on how the agent should perform its task. These instructions guide the LLM's behavior.
* ``metadata``:  Descriptive information about the agent, like its name and a more detailed description. This is useful for identification, API endpoints, and documentation.
* ``synapse``:  The ``Synapse`` object we initialized earlier, linking this agent to the Google Gemini Pro LLM.

Step 5: Run the Agent with a Prompt
-----------------------------------

To get our agent to generate titles, we need to run it with an input prompt. Let's ask it to generate titles for a story about a knight discovering a dragon egg.

.. code-block:: python

   input_prompt = "A story about a knight who discovers a hidden dragon egg."
   agent_response = agent.run(input_prompt)

   print(f"Agent '{agent.metadata['name']}' suggests titles:")
   print(agent_response.text)

The ``agent.run(input_prompt)`` line is where the magic happens:

* It sends the combined agent instructions and your input prompt to the configured LLM (Gemini Pro, in this case) via the Synapse.
* The LLM processes the request based on the agent's persona and instructions.
* IntelliBricks handles the communication, response parsing, and returns a structured ``AgentResponse`` object.

Step 6: Explore the Agent Response
----------------------------------

The ``agent_response`` object contains rich information about the LLM's completion. For our simple example, we are primarily interested in the generated text.

``agent_response.text``: This property conveniently provides the plain text output from the LLM, which in our case, will be the creative titles.

Run the Script
--------------

Save the code as a Python file (e.g., ``creative_title_agent.py``) and run it from your terminal:

.. code-block:: bash

   python creative_title_agent.py

You should see output similar to this (the actual titles might vary as LLMs are non-deterministic):

.. code-block:: text

   Agent 'TitleGenAgent' suggests titles:
   1. The Knight and the Dragon's Legacy
   2. Whispers of the Dragon Egg
   3. The Egg of Eldoria: A Knight's Discovery
   4. Beneath the Scales of Fate
   5. The Dragon Seed: A Knight's Tale

Congratulations! You've built and run your first IntelliBricks agent.

Further Exploration
-------------------

This quickstart barely scratches the surface of what IntelliBricks can do. Here are some ideas to explore further:

* **Experiment with different prompts:** Try different story topics and see how the agent's title suggestions change.
* **Modify Agent Instructions:** Tweak the instructions to guide the agent towards different styles of titles (e.g., more humorous, more dramatic, etc.).
* **Explore other LLMs:** Change the ``Synapse`` to use a different model (e.g., OpenAI's models, Groq models) and compare the results. Remember to install necessary dependencies and configure API keys.
* **Dive into Agent Response:** Explore other properties of the ``AgentResponse`` object, such as ``agent_response.parsed`` (when using response models), and metadata.
* **User Guide Sections:** Continue reading the User Guide to learn about Agents, Synapses, Files, and more advanced features of IntelliBricks.

IntelliBricks is designed to empower you to build sophisticated AI applications with ease. This quickstart is just the beginning. Happy building!