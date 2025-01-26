Installation
============

Getting started with IntelliBricks is designed to be as straightforward as possible. Follow these simple steps to install the library and set up your environment.

Prerequisites
-------------

* **Python 3.13+:** IntelliBricks leverages the latest Python features and requires Python version 3.13 or higher. Ensure you have a compatible version installed. You can check your Python version by running ``python --version`` in your terminal. If you need to install or upgrade Python, visit the `official Python website <https://www.python.org/downloads/>`_.

* **Pip:** Python's package installer, pip, is required to install IntelliBricks. Pip usually comes pre-installed with Python. You can verify its installation by running ``pip --version`` in your terminal. If pip is not installed, you can find installation instructions in the `pip documentation <https://pip.pypa.io/en/stable/installation/>`_.

Installation Steps
-------------------

1. **Using pip (Recommended):**

   The easiest way to install IntelliBricks is using pip. Open your terminal and run the following command:

   .. code-block:: bash

      pip install intellibricks

   This command will download and install the latest stable release of IntelliBricks and its dependencies from the Python Package Index (PyPI).

2. **Verify Installation:**

   After the installation is complete, you can verify it by importing ``intellibricks`` in a Python script or interactive session:

   .. code-block:: bash

      python -c "from intellibricks.llms import Synapse"

   If no errors are raised, and the message is printed, IntelliBricks has been installed correctly.

Optional Installations (For Extended Functionality)
---------------------------------------------------

IntelliBricks is designed to be modular, and some functionalities are provided through optional dependencies. You can install these extras based on your needs:

* **Files Module Extras:** For full file processing capabilities (parsing various document formats), install the ``files`` extra:

  .. code-block:: bash

     pip install intellibricks[files]

   This includes dependencies for handling PDF, DOCX, PPTX, images, audio, and more.

* **Cerebras Integration:** To use Cerebras LLMs, install the ``cerebras`` extra:

  .. code-block:: bash

     pip install intellibricks[cerebras]

* **Development Dependencies:** If you plan to contribute to IntelliBricks or run tests, install the ``dev`` dependencies:

  .. code-block:: bash

     pip install intellibricks[dev]

   This includes tools for development, testing, documentation, and more.

* **Install Multiple Extras:** You can install multiple extras at once, for example, to get both file processing and development dependencies:

  .. code-block:: bash

     pip install "intellibricks[files,dev]"

Environment Setup for LLM APIs
------------------------------

IntelliBricks interfaces with various LLM providers, which often require API keys or credentials. You'll need to set up environment variables to securely provide these keys.

* **API Keys as Environment Variables:**

  Most integrations rely on environment variables for API keys. For example:

    * **OpenAI:** Set your OpenAI API key as ``OPENAI_API_KEY``.
    * **Google Gemini:** For Google Gemini API, you might need to set up ``GOOGLE_API_KEY``, or follow Google's authentication procedures, especially if using Vertex AI.
    * **Groq:** Set your Groq API key as ``GROQ_API_KEY``.
    * **DeepInfra:** Set your DeepInfra API key as ``DEEPINFRA_API_KEY``.
    * **Cerebras:** Set your Cerebras API key as ``CEREBRAS_API_KEY``.

  You can set environment variables in your shell configuration file (e.g., ``.bashrc``, ``.zshrc``) or directly in your terminal session before running your Python scripts.

  Example (setting OpenAI API key in bash):

  .. code-block:: bash

     export OPENAI_API_KEY="your_openai_api_key_here"

  Alternatively, consider using a ``.env`` file and a library like ``python-dotenv`` to manage your environment variables more conveniently, especially for development.

Next Steps
----------

With IntelliBricks installed and your environment set up, you're ready to start building! Explore the :doc:`Quickstart <quickstart>` guide to begin creating your first intelligent application.
