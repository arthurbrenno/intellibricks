.. Intellibricks documentation master file, created by
   sphinx-quickstart on Sat Jan 25 07:24:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

IntelliBricks Documentation
===========================

**Build Intelligent Apps, Effortlessly.**

IntelliBricks is a Python-first framework designed to simplify the development of agentic and LLM-powered applications. It provides a streamlined set of tools to interact with Language Model Models (LLMs), build autonomous agents, and handle files intelligently, all while keeping Python at the forefront. Focus on building intelligence, not boilerplate, with IntelliBricks.

.. image:: _static/quick_overview.svg
   :alt: IntelliBricks Quick Overview
   :align: center
   :target: _static/quick_overview.svg

Key Features:

* **Python-First Approach:** Write clean, idiomatic Python, leveraging modern Python features.
* **Structured Outputs:** Define data models in Python using `msgspec` and get structured responses from LLMs reliably.
* **Autonomous Agents:** Construct sophisticated agents with clear tasks, instructions, tools, and memory.
* **Effortless APIs:** Easily expose your agents as REST APIs using FastAPI or Litestar.
* **Built-in RAG:** Seamlessly integrate Retrieval-Augmented Generation for contextual awareness.
* **Simplified Complexity:** Streamlined architecture to reduce boilerplate and configuration, allowing you to focus on innovation.
* **Observability:** Integrated with Langfuse for tracing, monitoring, and debugging your AI application workflows.
* **Extensible Integrations:** Supports multiple LLM providers (Google Gemini, OpenAI, Groq, Cerebras, DeepInfra) and file parsing capabilities.

Get Started
-----------

.. code-block:: bash

   pip install intellibricks

.. toctree::
    :maxdepth: 2
    :caption: Overview

    sections/overview/why
    sections/overview/installation
    sections/overview/benchmarks

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    sections/user_guide/quickstart
    sections/user_guide/agents
    sections/user_guide/synapses
    sections/user_guide/files

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    modules
