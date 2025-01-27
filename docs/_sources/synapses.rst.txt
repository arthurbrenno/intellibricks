Synapses: Connecting to Language Model Models (LLMs)
====================================================

**Synapses** in IntelliBricks are your gateway to the world of Language Model Models (LLMs). A Synapse acts as a smart connector, handling the complexities of interacting with different LLM providers and models in a consistent and Pythonic way. It abstracts away the low-level API details, allowing you to focus on *what* you want to achieve with LLMs, rather than *how* to communicate with them.

Core Concepts of Synapses
-------------------------

* **Abstraction Layer:** Synapses provide a unified interface for interacting with various LLM providers (e.g., Google Gemini, OpenAI, Groq, Cerebras, DeepInfra). You can switch between different models or providers with minimal code changes.

* **Model Configuration:** Synapses encapsulate the configuration for a specific LLM model, including model name, API keys, and provider-specific settings.

* **Text Completion and Chat:** Synapses offer methods for both simple text completion (``complete``, ``complete_async``) and chat-based interactions (``chat``, ``chat_async``).

* **Structured Output Handling:** Synapses seamlessly handle structured outputs when you specify a ``response_model``. They ensure that LLMs return data in your desired format, automatically parsing the responses.

* **Fault Tolerance (Synapse Cascade):** IntelliBricks provides ``SynapseCascade`` to create fault-tolerant applications. A Synapse Cascade allows you to specify a list of Synapses; if one fails, the system automatically falls back to the next one in the cascade.

* **Text Transcription Synapse:** For audio processing, ``TextTranscriptionSynapse`` is specialized for audio transcription tasks, connecting to transcription services like OpenAI Whisper or Groq Whisper.

Initializing Synapses
---------------------

Synapses are typically initialized using the static factory method ``Synapse.of()``, which simplifies configuration based on model identifiers.

**Basic Synapse Initialization**

To initialize a Synapse for Google Gemini Pro:

.. code-block:: python

   from intellibricks.llms import Synapse

   gemini_synapse = Synapse.of("google/genai/gemini-pro-experimental")

   # Ensure you have set your GOOGLE_API_KEY environment variable.
   # For Vertex AI models, project and location might also be needed.

To initialize a Synapse for OpenAI's ``gpt-4o`` model:

.. code-block:: python

   openai_synapse = Synapse.of("openai/api/gpt-4o")

   # Ensure you have set your OPENAI_API_KEY environment variable.

To initialize a Synapse for Groq's ``mixtral-8x7b-32768`` model:

.. code-block:: python

   groq_synapse = Synapse.of("groq/api/mixtral-8x7b-32768"8)

   # Ensure you have set your GROQ_API_KEY environment variable.

**Synapse Methods: ``complete`` and ``chat``**

Synapses provide two primary methods for interacting with LLMs:

1.  ``complete(prompt, **kwargs)`` / ``complete_async(prompt, **kwargs)``: For simple text completion tasks. You provide a prompt, and the Synapse returns a completion.

2.  ``chat(messages, **kwargs)`` / ``chat_async(messages, **kwargs)``: For chat-based interactions. You provide a list of ``Message`` objects representing the conversation history, and the Synapse returns a chat response.

**Using ``complete`` for Text Generation**

Let's use the ``complete`` method to generate a short story with the Gemini Synapse:

.. code-block:: python

   completion_response = gemini_synapse.complete("Write a short story about a robot learning to love.")
   print(completion_response.text)

Key parameters for ``complete`` (and ``complete_async``):

* ``prompt`` (str | Prompt | PartType | Sequence[PartType]): The prompt for text generation. Can be a simple string, a ``Prompt`` object for structured prompts, or a ``PartType`` or sequence of ``PartType`` for multimodal prompts.
* ``response_model`` (Optional[Type[S]]):  An optional ``msgspec.Struct`` class to define the structure of the expected output. If provided, IntelliBricks will attempt to parse the LLM response into this structure.
* Other generation parameters: ``temperature``, ``max_completion_tokens``, ``top_p``, ``top_k``, ``stop_sequences``, etc., to control the LLM's generation behavior.

**Using ``chat`` for Conversational Interactions**

For chat-based interactions, use the ``chat`` method. You need to provide a sequence of ``Message`` objects to represent the conversation history.

.. code-block:: python

   from intellibricks.llms import UserMessage, AssistantMessage

   chat_messages = [
       UserMessage.from_text("Hello, are you there?"),
       AssistantMessage.from_text("Yes, I am here. How can I help you today?"),
       UserMessage.from_text("Tell me a joke."),
   ]

   chat_response = gemini_synapse.chat(chat_messages)
   print(chat_response.text)

Key parameters for ``chat`` (and ``chat_async``):

* ``messages`` (Sequence[Message]): A list of ``Message`` objects representing the conversation history. IntelliBricks provides ``UserMessage``, ``AssistantMessage``, and ``DeveloperMessage`` message types.
* ``response_model`` (Optional[Type[S]]):  Similar to ``complete``, you can provide a ``response_model`` for structured chat responses.
* ``tools`` (Optional[Sequence[ToolInputType]]): A list of tools that the LLM can use during the chat interaction (function calling).
* Other generation parameters: ``temperature``, ``max_completion_tokens``, ``top_p``, ``top_k``, ``stop_sequences``, etc.

**Synapse Cascade for Fault Tolerance**

To enhance the reliability of your application, you can use ``SynapseCascade``. It allows you to specify a list of Synapses, and if the first one fails (e.g., due to API issues, rate limits), it automatically tries the next one in the list.

.. code-block:: python

   from intellibricks.llms import SynapseCascade

   synapse_cascade = SynapseCascade(
       synapses=[
           Synapse.of("openai/api/gpt-4o"),       # Primary Synapse
           Synapse.of("google/genai/gemini-1.5-flash"),   # Fallback Synapse 1
           Synapse.of("cerebras/api/llama-3.3-70b"), # Fallback Synapse 2
       ]
   )

   # Use synapse_cascade just like a regular Synapse
   try:
       response = synapse_cascade.complete("Translate 'Hello' to Spanish.")
       print(response.text)
   except Exception as e:
       print(f"All synapses failed: {e}")

``SynapseCascade`` attempts to use the Synapses in the order they are listed. If a Synapse call fails, it catches the exception and tries the next Synapse in the cascade. If all Synapses fail, it raises the last encountered exception.

**TextTranscriptionSynapse for Audio Transcription**

For audio transcription tasks, use ``TextTranscriptionSynapse``.

.. code-block:: python

   from intellibricks.llms import TextTranscriptionSynapse

   whisper_synapse = TextTranscriptionSynapse.of("groq/api/distil-whisper-large-v3-en") # Or "openai/api/whisper-1"

   audio_file_path = "path/to/your/audiofile.mp3" # Replace with your audio file

   try:
       transcription_response = whisper_synapse.transcribe(audio_file_path)
       print("Transcription Text:")
       print(transcription_response.text)
       print("\nTranscription Segments (first 3):")
       for segment in transcription_response.segments[:3]:
           print(f"- Segment {segment.id}: {segment.sentence} ({segment.start:.2f}s - {segment.end:.2f}s)")

       # You can also get SRT subtitles
       srt_subtitles = transcription_response.srt
       print("\nSRT Subtitles (first few lines):")
       print(srt_subtitles[:200] + "...") # Print first 200 chars of SRT

   except Exception as e:
       print(f"Transcription failed: {e}")

Key methods for ``TextTranscriptionSynapse``:

* ``transcribe(audio, **kwargs)`` / ``transcribe_async(audio, **kwargs)``:  Transcribes audio content. The ``audio`` parameter can be a file path, bytes data, or a file-like object.

**TextTranscriptionsSynapseCascade for Fault-Tolerant Transcriptions**

Similar to ``SynapseCascade``, IntelliBricks offers ``TextTranscriptionsSynapseCascade`` for fault-tolerant audio transcriptions.

.. code-block:: python

   from intellibricks.llms import TextTranscriptionsSynapseCascade

   transcription_cascade = TextTranscriptionsSynapseCascade(
       synapses=[
           TextTranscriptionSynapse.of("groq/api/whisper-large-v3-turbo"), # Primary transcription service
           # Add other transcription synapses as fallbacks if needed
       ]
   )

   # Use transcription_cascade just like TextTranscriptionSynapse for enhanced reliability

Summary

IntelliBricks Synapses provide a robust and flexible way to interact with a wide range of LLMs and transcription services. They simplify model configuration, handle API communication, and offer advanced features like structured outputs and fault tolerance. By leveraging Synapses, you can easily integrate the power of LLMs into your intelligent applications without getting bogged down in low-level complexities.

API Reference
-------------

.. automodule:: intellibricks.llms.synapses
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: intellibricks.llms.types
   :members:
   :undoc-members:
   :show-inheritance:
