# 🧠🧱 IntelliBricks: The Building Blocks for Intelligent Applications

Welcome to **IntelliBricks**—your streamlined toolkit for developing AI-powered applications. Whether you're interacting with large language models (LLMs), training machine learning models, or implementing Retrieval Augmented Generation (RAG), IntelliBricks simplifies the complex so you can focus on what truly matters: your application logic.

> ⚠️ **Warning:**  
> *This project is currently under development and is **not ready for production**.*  
> If you resonate with our vision, please consider supporting the project to help bring it to life! This is a personal endeavor I've nurtured for months and am excited to open source.

---

## 🚀 Key Features

- **✨ Simplified LLM Interaction:**  
  Interact seamlessly with multiple AI providers through a unified interface. Switch models effortlessly using simple enum changes. Supports both single prompt completions and chat-based interactions.

- **🤖 Effortless Model Training:**  
  Train machine learning models with minimal code using the intuitive `SupervisedLearningEngine`. Includes data preprocessing, model selection, evaluation, and artifact management.

- **🔍 Retrieval Augmented Generation (RAG):**  
  Connect to your knowledge bases for context-aware AI responses *(currently under development)*.

- **📦 Built-in Parsing:**  
  Eliminate boilerplate parsing code with automatic response deserialization into your defined data structures.

- **📊 Langfuse Integration:**  
  Gain deep insights into your LLM usage with seamless integration with Langfuse. Monitor traces, events, and model costs effortlessly. IntelliBricks automatically calculates and logs model costs for you.

- **💸 Transparent Cost Tracking:**  
  Automatically calculates and tracks LLM usage costs, providing valuable insights into your spending.

- **🔗 Fully Typed:**  
  Enjoy a smooth development experience with complete type hints for `mypy`, `pyright`, and `pylance`, ensuring type safety throughout your codebase.

---

## 📚 Table of Contents

1. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [LLM Interaction](#llm-interaction)
   - [Chat Interactions](#chat-interactions)
2. [Advanced Usage](#advanced-usage)
   - [System Prompts and Chat History](#system-prompts-and-chat-history)
   - [Customizing Prompts](#customizing-prompts)
   - [Langfuse Integration](#langfuse-integration)
3. [Parameter Breakdown](#-parameter-breakdown)
4. [Key Points to Consider](#-key-points-to-consider)
5. [Training Machine Learning Models](#training-machine-learning-models)
6. [Coming Soon](#coming-soon)
7. [Documentation](#documentation)
8. [Contributing](#contributing)
9. [License](#license)
10. [Community & Support](#community--support)
11. [Showcase](#showcase)

---

## 🏁 Getting Started

### 📦 Installation

Install IntelliBricks via pip:

```bash
pip install intellibricks
```

### 🧠 LLM Interaction

IntelliBricks abstracts the complexities of interacting with different LLM providers. Specify your prompt, desired response format, and model, and IntelliBricks handles the rest.

#### 🔄 Synchronous Completion Example

```python
from dotenv import load_dotenv
from msgspec import Struct
from intellibricks import CompletionEngine

load_dotenv(override=True)

class Joke(Struct):
    joke: str

output = CompletionEngine().complete(
    prompt="Tell me a joke",
    response_format=Joke
)

print(output.get_parsed())  # Joke object
```

**Highlights:**
- **3 Easy Steps:** Define your structured output, call `complete()`, and parse the result.
- **No Boilerplate:** Forget about `OutputParsers` and repetitive code.

#### 🔍 Type Safety with Mypy and Pyright

IntelliBricks is built with type hints, ensuring a smooth development experience.

```python
from dotenv import load_dotenv
from msgspec import Struct
from intellibricks import CompletionEngine

load_dotenv(override=True)

class Joke(Struct):
    joke: str

output = CompletionEngine().complete(
    prompt="Tell me a joke",
    response_format=Joke
)  # CompletionOutput[Joke]

choices = output.choices  # list[MessageChoice[Joke]]
message = output.choices[0].message  # CompletionMessage[Joke]
parsed = message.parsed  # Joke

# Easily get the parsed output
easy_parsed = output.get_parsed()  # Defaults to choice 0
```

**Switching Providers:**

Change AI providers effortlessly by modifying the `model` parameter.

```python
response = engine.complete(
    # ...
    model=AIModel.GPT_4O  # Switch to GPT-4
).get_parsed()
```

### 💬 Chat Interactions

Engage in multi-turn conversations with structured responses.

```python
from intellibricks import Message, MessageRole, CompletionOutput
from dotenv import load_dotenv
from msgspec import Meta, Struct

load_dotenv(override=True)

# Define structured response models
class President(Struct):
    name: str
    age: Annotated[int, Meta(ge=40, le=107)]

class PresidentsResponse(Struct):
    presidents: list[President]

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="Hello, how are you?"),
    Message(role=MessageRole.ASSISTANT, content="I'm fine! And you? IntelliBricks is awesome, isn't it? (This was completely generated by AI and not the owner of the project)"),
    Message(role=MessageRole.USER, content="I'm fine. What are the presidents of the USA?"),
]

response = engine.chat(
    messages=messages,
    response_format=PresidentsResponse
)

presidents_response: PresidentsResponse = response.get_parsed()
print(presidents_response)
```

---

## 🌐 Integrating IntelliBricks with Litestar

Create a simple API endpoint using the [Litestar](https://litestar.dev/) framework that receives a request and returns a structured response with IntelliBricks.

```python
from litestar import Litestar, post
from intellibricks import CompletionEngine, AIModel
from msgspec import Struct

# Define the request model
class JokeTheme(Struct):
    prompt: str

# Define the structured response model
class JokeResponse(Struct):
    joke: str

# Define the endpoint
@post("/joke", response_model=JokeResponse)
async def get_joke(data: JokeTheme) -> JokeResponse:
    response = CompletionEngine().complete(
        prompt=f"The theme of the joke is: {data.prompt}",
        system_prompt="You are an AI specialised in making the funniest jokes.",
        response_format=JokeResponse
        model=AIModel.STUDIO_GEMINI_1P5_FLASH
    )
    return response.get_parsed()

app = Litestar([get_joke])
```

---

## 🌐 Integrating IntelliBricks with FastAPI

Create a simple API endpoint using the [FastAPI](https://fastapi.tiangolo.com/) framework that receives a request and returns a structured response with IntelliBricks. Since FastAPI relies on Pydantic for data validation and IntelliBricks uses `msgspec.Struct`, we'll use Pydantic for the request model and encode the IntelliBricks response using `msgspec.json.encode()` before returning it.

```python
from fastapi import FastAPI, Response
from pydantic import BaseModel
from intellibricks import CompletionEngine, AIModel
from msgspec import Struct
import msgspec

# Define the request model using Pydantic
class JokeRequest(BaseModel):
    prompt: str

# Define the structured response model using msgspec
class JokeResponse(Struct):
    joke: str

# Initialize FastAPI and CompletionEngine
app = FastAPI()

# Define the endpoint
@app.post("/joke")
async def get_joke(data: JokeRequest) -> Response:
    response = CompletionEngine().complete(
        prompt=f"The theme of the joke is: {data.prompt}",
        system_prompt="You are an AI specialized in making the funniest jokes.",
        response_format=JokeResponse,
        model=AIModel.STUDIO_GEMINI_1P5_FLASH
    )
    joke_response = response.get_parsed()
    encoded_response = msgspec.json.encode(joke_response)
    return Response(content=encoded_response, media_type="application/json")
```

### 🛠️ Additional Notes

- **Why Use `msgspec.json.encode()`?**
  
  FastAPI is designed to work seamlessly with Pydantic models for both requests and responses. However, since IntelliBricks uses `msgspec.Struct` for structured responses, directly returning `msgspec` objects isn't compatible with FastAPI's response handling. By encoding the `msgspec` structured response into JSON bytes, we can integrate IntelliBricks' structured outputs with FastAPI's response system effectively. Intellibricks will not use pydantic's BaseModel, since msgspec is way faster to serialize/desserialize in performance critical / low latency LLM applications.

---

## 🛠️ Advanced Usage

### 📜 Complete `CompletionEngine.chat()` Usage Example

Gain a comprehensive understanding of how to leverage each parameter to customize your AI-powered chat interactions effectively.

#### **1. Import Required Modules**

```python
import os
import asyncio
from dotenv import load_dotenv
from msgspec import Struct
from langfuse import Langfuse
from google.oauth2 import service_account

from intellibricks import CompletionEngine, Message, Prompt
from intellibricks.config import CacheConfig
from intellibricks.constants import AIModel, MessageRole
from intellibricks.schema import Message
from intellibricks.types import TraceParams
from intellibricks.exceptions import MaxRetriesReachedException
from intellibricks.rag.contracts import RAGQueriable

# Example custom tool function
def your_custom_tool_function(*args, **kwargs):
    # Implement your custom tool logic here
    pass

# Example RAG data store
class YourRAGDataStore(RAGQueriable):
    async def query_async(self, query: str) -> "QueryResult":
        # Your asynchronous query implementation
        pass

    def query(self, query: str) -> "QueryResult":
        # Your synchronous query implementation
        pass
```

#### **2. Load Environment Variables**

```python
load_dotenv(override=True)
```

#### **3. Define Structured Response Models**

```python
class President(Struct):
    name: str
    age: int

class PresidentsResponse(Struct):
    presidents: list[President]
```

#### **4. Initialize Langfuse (Optional)**

```python
langfuse_client = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
)
```

#### **5. Configure Vertex AI Credentials (Optional)**

```python
vertex_credentials = service_account.Credentials.from_service_account_file(
    "path/to/your/vertex_credentials.json"
)
```

#### **6. Initialize the CompletionEngine**

```python
engine = CompletionEngine(
    langfuse=langfuse_client,  # Optional: Integrate with Langfuse
    json_encoder=None,          # Optional: Use a custom JSON encoder
    json_decoder=None,          # Optional: Use a custom JSON decoder
    vertex_credentials=vertex_credentials  # Optional: Vertex AI credentials
)
```

#### **7. Set Up Cache Configuration (Optional)**

```python
from datetime import timedelta

cache_config = CacheConfig(
    enabled=True,  # Enable caching
    ttl=timedelta(minutes=10),  # Set TTL to 10 minutes
    cache_key='user_session_prompt'  # Define a unique cache key
)

# **Example:**
# >>> cache_config = CacheConfig(enabled=True, ttl=timedelta(seconds=60), cache_key='user_prompt')
```

#### **8. Define Trace Parameters (Optional)**

```python
trace_params = TraceParams(
    name="ChatCompletionTrace",
    user_id="user_12345",
    session_id="session_67890",
    metadata={"feature": "chat_completion"},
    tags=["chat", "completion"],
    public=False
)

# **Example:**
# >>> trace_params = TraceParams(user_id="user_123", session_id="session_456")
# >>> print(trace_params)
# {'user_id': 'user_123', 'session_id': 'session_456'}
```

#### **9. Prepare Chat Messages**

```python
messages = [
    Message(role=MessageRole.SYSTEM, content="You are a knowledgeable assistant."),
    Message(role=MessageRole.USER, content="Hello! Can you help me with some information?"),
    Message(role=MessageRole.ASSISTANT, content="Of course! What do you need assistance with?"),
    Message(role=MessageRole.USER, content="I'm interested in knowing the current presidents of various countries.")
]
```

#### **10. Initialize RAG Data Stores (Optional)**

```python
rag_data_store = YourRAGDataStore()
```

#### **11. Make a `chat` Request with All Parameters**

```python
try:
    chat_response = engine.chat(
        messages=messages,  # The conversation history
        response_format=PresidentsResponse,  # Structured response format
        model=AIModel.GPT_4O,  # Primary AI model
        fallback_models=[AIModel.STUDIO_GEMINI_1P5_FLASH, AIModel.GPT_3_5_TURBO],  # Fallback models
        n=2,  # Number of responses to generate
        temperature=0.7,  # Creativity of the responses
        max_tokens=500,  # Maximum tokens per response
        max_retries=3,  # Maximum number of retry attempts
        cache_config=cache_config,  # Cache configuration
        trace_params=trace_params,  # Tracing parameters for monitoring
        postergate_token_counting=False,  # Immediate token counting
        tools=[your_custom_tool_function],  # Custom tool functions *(Currently under development)*
        data_stores=[rag_data_store],  # RAG data stores for context-aware responses *(Currently under development)*
        web_search=True,  # Enable web search capabilities *(Currently under development)*. Requires passing a `WebSearchConfig`.
    )

    # Access the parsed structured response
    presidents_response: PresidentsResponse = chat_response.get_parsed()
    for president in presidents_response.presidents:
        print(f"President: {president.name}, Age: {president.age}")

except MaxRetriesReachedException:
    print("Failed to generate a chat response after maximum retries.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

#### **12. Asynchronous `chat_async` Request (Optional)**

```python
async def async_chat_example():
    try:
        chat_response = await engine.chat_async(
            messages=messages,  # The conversation history
            response_format=PresidentsResponse,  # Structured response format
            model=AIModel.GPT_4O,  # Primary AI model
            fallback_models=[AIModel.STUDIO_GEMINI_1P5_FLASH, AIModel.GPT_3_5_TURBO],  # Fallback models
            n=2,  # Number of responses to generate
            temperature=0.7,  # Creativity of the responses
            max_tokens=500,  # Maximum tokens per response
            max_retries=3,  # Maximum number of retry attempts
            cache_config=cache_config,  # Cache configuration
            trace_params=trace_params,  # Tracing parameters for monitoring
            postergate_token_counting=False,  # Immediate token counting
            tools=[your_custom_tool_function],  # Custom tool functions *(Currently under development)*
            data_stores=[rag_data_store],  # RAG data stores for context-aware responses *(Currently under development)*
            web_search=True  # Enable web search capabilities *(Currently under development)*. Requires passing a `WebSearchConfig`.
        )

        # Access the parsed structured response
        presidents_response: PresidentsResponse = chat_response.get_parsed()
        for president in presidents_response.presidents:
            print(f"President: {president.name}, Age: {president.age}")

    except MaxRetriesReachedException:
        print("Failed to generate a chat response after maximum retries.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the asynchronous chat example
asyncio.run(async_chat_example())
```

#### **13. Custom Prompt Compilation**

Create dynamic prompts with placeholders for flexibility.

```python
from intellibricks import Prompt

# Define a prompt template with placeholders
prompt_template = Prompt(content="My name is {{name}}. I am {{age}} years old.")

# Compile the prompt with actual values
compiled_prompt = prompt_template.compile(name="Alice", age=30)

print(compiled_prompt)  # Output: My name is Alice. I am 30 years old.
```

#### **14. Handling Exceptions and Retries**

Gracefully manage failures and retries.

```python
try:
    output = engine.chat(
        messages=messages,
        response_format=None,  # No structured response
        model=AIModel.GPT_4O,
        max_retries=5,
        # ... other parameters
    )
    print(output.get_message().content)

except MaxRetriesReachedException:
    print("Unable to generate a response after multiple attempts. Please try again later.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

#### **15. Integrating with Retrieval Augmented Generation (RAG)**

Enhance responses with context from your knowledge bases.

```python
# Initialize your RAG data store
rag_data_store = YourRAGDataStore()

# Make a completion request with RAG integration
try:
    output = engine.chat(
        messages=messages,
        response_format=None,
        data_stores=[rag_data_store],  # Integrate RAG data stores *(Currently under development)*
        web_search=True,  # Enable web search capabilities *(Currently under development)*. Requires passing a `WebSearchConfig`.
        # ... other parameters
    )
    print(output.get_message().content)

except MaxRetriesReachedException:
    print("Failed to retrieve information after multiple attempts.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

---

### 🔍 Parameter Breakdown

Here's a detailed explanation of each parameter used in the `CompletionEngine.chat()` method:

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | **Required.** A list of `Message` objects representing the conversation history.  Each message has a `role` (e.g., `SYSTEM`, `USER`, `ASSISTANT`) and `content`. |
| `response_format` | `Type[T]` or `None` | **Optional.** A structured data model (subclass of `msgspec.Struct`) to deserialize the AI's response. If `None`, the response remains unstructured (`str`). |
| `model` | `AIModel` or `None` | **Optional.** The primary AI model. Defaults to `AIModel.STUDIO_GEMINI_1P5_FLASH`. |
| `fallback_models` | `list[AIModel]` or `None` | **Optional.** Alternative AI models to try if the primary model fails. |
| `n` | `int` or `None` | **Optional.** The number of completions to generate. Defaults to `1`. |
| `temperature` | `float` or `None` | **Optional.** Controls response creativity (0.0-1.0).  Higher values are more random. |
| `max_tokens` | `int` or `None` | **Optional.** Maximum tokens per response. Defaults to `5000`. |
| `max_retries` | `Literal[1, 2, 3, 4, 5]` or `None` | **Optional.** Maximum retry attempts. Defaults to `1`. |
| `cache_config` | `CacheConfig` or `None` | **Optional.** Caching configuration. See [CacheConfig Examples](docs/examples/cache_config_examples.md). |
| `trace_params` | `TraceParams` or `None` | **Optional.** Tracing parameters. See [TraceParams Examples](docs/examples/trace_params_examples.md). |
| `postergate_token_counting` | `bool` | **Optional.** Defer token counting. Defaults to `True`. |
| `tools` | `list[Callable[..., Any]]` or `None` | **Optional.** Custom tool functions. *(Currently under development.)* |
| `data_stores` | `Sequence[RAGQueriable]` or `None` | **Optional.** RAG data stores. *(Currently under development.)* |
| `web_search` | `bool` or `None` | **Optional.** Enable web search. Requires `WebSearchConfig`. *(Currently under development.)* Defaults to `False`. |
---

## 💡 Key Points to Consider

- **📐 Structured Responses:**  
  Utilize the `response_format` parameter with `msgspec.Struct` models to ensure AI responses adhere to predefined structures, facilitating easier downstream processing and validation.

- **🔄 Fallback Models:**  
  Enhance the resilience of your application by specifying `fallback_models`, providing alternative AI models in case the primary model encounters issues or fails to generate a response.

- **⚡ Asynchronous Operations:**  
  Leverage `chat_async` to handle multiple concurrent AI interactions efficiently, improving overall performance and responsiveness.

- **💾 Caching:**  
  Properly configure `cache_config` to optimize performance and reduce costs by avoiding redundant AI calls for identical prompts.

- **📈 Tracing and Monitoring:**  
  Integrate with Langfuse and utilize `trace_params` to gain deep insights into your AI interactions, enabling effective monitoring, debugging, and cost tracking.

- **🛡️ Error Handling:**  
  Implement robust error handling to gracefully manage failures, especially when dealing with external AI services. The `MaxRetriesReachedException` helps identify when maximum retry attempts have been exhausted.

- **🔒 Security:**  
  Always handle sensitive information, such as API keys and credentials, securely. Use environment variables or secure secret management systems to protect your data.

---

## 🏋️ Training Machine Learning Models

Train supervised learning models effortlessly with the `SupervisedLearningEngine`. Provide your data and configuration, and let IntelliBricks manage the training and prediction pipeline.

```python
from intellibricks.models.supervised import SKLearnSupervisedLearningEngine, TrainingConfig, AlgorithmType
import base64

# Encode your dataset
with open("dataset.csv", "rb") as f:
    b64_file = base64.b64encode(f.read()).decode("utf-8")

# Define training configuration
config = TrainingConfig(
    algorithm=AlgorithmType.RANDOM_FOREST,
    hyperparameters={"n_estimators": 100, "max_depth": 5},
    target_column="target_variable",
    # ... other configurations
)

# Instantiate the training engine
engine = SKLearnSupervisedLearningEngine()

# Train the model
training_result = await engine.train(
    b64_file=b64_file,
    uid="my_model_123",
    name="My Model",
    config=config,
)

print(training_result)


# Make Predictions
input_data = {
    'feature1': 10,
    'feature2': 'A',
    'feature3': 5.5,
    # ... other features
}

predictions = await engine.predict(
    uid='my_model_123',
    input_data=input_data,
)

print(predictions)
```

---

## 🛠️ Advanced Usage

### 📜 System Prompts and Chat History

```python
from intellibricks import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="Who won the world series in 2020?"),
    Message(role=MessageRole.ASSISTANT, content="The Los Angeles Dodgers."),
    Message(role=MessageRole.USER, content="Where was it played?"),
]

response = engine.chat(messages=messages)
message: Message = response.get_message()
print(message)
# >> Message(role=MessageRole.ASSISTANT, content="I don't know")
```

### 🛠️ Customizing Prompts

```python
from intellibricks import Prompt

prompt_template = Prompt(content="My name is {{name}}. I am {{age}} years old.")  # Implements __str__
compiled_prompt = prompt_template.compile(name="John", age=30)  # Returns Prompt
print(compiled_prompt)  # Output: My name is John. I am 30 years old.
```

### 📊 Langfuse Integration

IntelliBricks integrates with Langfuse for enhanced observability of your LLM interactions. Trace performance, track costs, and monitor events with ease. This integration is automatically activated when you instantiate a `CompletionEngine` with a Langfuse instance.

```python
import os
from langfuse import Langfuse

langfuse_client = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
)

engine = CompletionEngine(langfuse=langfuse_client)

# Now all LLM calls made with 'engine' will be automatically tracked in Langfuse, including costs.
```

---

## 🌟 Coming Soon

- **🔗 Enhanced RAG:**  
  A more robust RAG implementation for seamless integration with diverse knowledge sources. We aim to create adapters for each vector store, ensuring compatibility across interfaces.

- **📄 Unified Document Parsing:**  
  Stop wasting time choosing the right library for parsing PDFs. IntelliBricks will handle it for you with our `DocumentArtifact` model, easily convertible to `llama_index` and `langchain` documents. Support for NER and Relations extraction is on the horizon.  
  **Example:**

  ```python
  extractor: FileExtractorProtocol = ...  # In development

  # Example #1 - From file in filesystem:
  document_artifact_1 = extractor.extract(
      RawFile.from_file("./documents/some_file.pdf"),
      parsing_method=ParsingMethod.FAST,
      gpu=False,
  )

  # Example #2 - From an uplodaded files:
  # Imagine you're ingesting documents into a vector store, but you don't have them yet. Extract from uploaded files will be possible
  document_artifact_2 = extractor.extract(
      RawFile.from_litestar_uploadfile(some_litestar_upload_file),
      parsing_method=ParsingMethod.MEDIUM,
      gpu=True,
  )
  document_artifact_3 = extractor.extract(
      RawFile.from_fastapi_uploadfile(some_fastapi_upload_file),
      parsing_method=ParsingMethod.PROFESSIONAL,
      gpu=False,
  )


  langchain_documents = document_artifact_1.as_langchain_docs(
      transformations=[SemanticChunker(...)]
  )
  # Done. Now you can ingest your doc into
  some_vector_store.add_documents(langchain_documents)  # Langchain example
  ```

---

## 📖 Documentation

For now, the only documentation available is here in GitHub. I work as a Software Architect | Engineer, so I'm trying to manage my time for this project too.
---

## 🤝 Contributing

We welcome contributions to IntelliBricks! Whether it's reporting issues, suggesting features, or submitting pull requests, your involvement is invaluable.

---

## 📝 License

[MIT License](LICENSE)

---

## 👥 Community & Support

Join our community to stay updated, share your projects, and get support:

- **GitHub Discussions:** [IntelliBricks Discussions](https://github.com/arthurbrenno/intellibricks/discussions)
---

Thank you for choosing **IntelliBricks**!
