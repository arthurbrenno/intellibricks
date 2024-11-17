# 🧠🧱 IntelliBricks: The Building Blocks for Intelligent Applications

IntelliBricks provides a streamlined set of tools for developing AI-powered applications. It simplifies complex tasks such as interacting with LLMs, training machine learning models, and implementing Retrieval Augmented Generation (RAG). Focus on building your application logic, not wrestling with boilerplate.  IntelliBricks empowers you to build intelligent applications faster and more efficiently.

> ⚠️ **Warning:**  
> This project is currently under development and is **not ready for production**.  
> If you like the idea, please consider supporting the project to help bring it to life! This was a personal project I've been doing for months and decided to open source it.


## Key Features

* **Simplified LLM Interaction:** Easily interact with multiple AI providers through a unified interface. Switch between models with a simple enum change. Supports both single prompt completion and chat-based interactions.
* **Effortless Model Training:** Train machine learning models with minimal code using the intuitive `SupervisedLearningEngine`. Includes data preprocessing, model selection, evaluation, and artifact management.
* **Retrieval Augmented Generation (RAG):** Connect to your knowledge bases for context-aware AI responses (currently under development).
* **Built-in Parsing:** Eliminate boilerplate parsing code with automatic response deserialization directly into your defined data structures.
* **Langfuse Integration:** Gain deep insights into your LLM usage with seamless integration with Langfuse. Monitor traces, events, and model costs effortlessly. IntelliBricks automatically calculates and logs model costs for you.
* **Transparent Cost Tracking:** IntelliBricks automatically calculates and tracks LLM usage costs, providing valuable insights into your spending.
* **Fully Typed:**  Enjoy a smooth development experience with complete type hints for `mypy`, `pyright`, and `pylance`, ensuring no type errors.


## Getting Started

### Installation

```bash
pip install intellibricks
```


### LLM Interaction

IntelliBricks abstracts away the complexities of interacting with different LLM providers. Specify your prompt, desired response format, and model, and IntelliBricks handles the rest.

Let's take a look at HOW EASY it is to choose an AI Model, get your structured response, and manipulate it later.

### Synchronous Completion Example
```python
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from msgspec import Meta, Struct

from intellibricks import CompletionEngine
from intellibricks.llms.schema import CompletionOutput

load_dotenv(override=True)


# Step #1: Define your response structure
class President(Struct):
    name: str
    age: Annotated[int, Meta(ge=40, le=107)]


class PresidentsResponse(Struct):
    presidents: list[President]

# Call the CompletionEngine
engine = CompletionEngine()
response: CompletionOutput[PresidentsResponse] = engine.complete(
    prompt="What were the presidents of the USA until your knowledge?",
    response_format=PresidentsResponse,
)

# Manipulate the response as you want.
presidents_response: PresidentsResponse = response.get_parsed()
print(f"First president name is {presidents_response.presidents[0].name}")
print(f"First president age is {presidents_response.presidents[0].age}")
```


### Asynchronous Completions Example
```python
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from msgspec import Meta, Struct

from intellibricks import CompletionEngine
from intellibricks.llms.schema import CompletionOutput

load_dotenv(override=True)


# Step #1: Define your response structure
class President(Struct):
    name: str
    age: Annotated[int, Meta(ge=40, le=107)]


class PresidentsResponse(Struct):
    presidents: list[President]


async def main():
    # Call the CompletionEngine
    engine = CompletionEngine()
    response: CompletionOutput[PresidentsResponse] = await engine.complete_async(
        prompt="What were the presidents of the USA until your knowledge?",
        response_format=PresidentsResponse,
    )

    # Manipulate the response as you want.
    presidents_response: PresidentsResponse = response.get_parsed()
    print(f"First president name is {presidents_response.presidents[0].name}")
    print(f"First president age is {presidents_response.presidents[0].age}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(main())
        else:
            loop.run_until_complete(main())
```

**Switching Providers:** Change AI providers by simply modifying the `model` parameter. Ensure the necessary environment variables for your chosen provider are set.

```python
response = engine.complete(
    # [...]
    model=AIModel.GPT_4O # Switch to GPT-4
).get_parsed()
```

### Chat Interactions

For multi-turn conversations, use the `chat` method.  You can also specify the `response_format` here for structured responses.

```python
from intellibricks import Message, MessageRole, CompletionOutput
from dotenv import load_dotenv
from msgspec import Meta, Struct

load_dotenv(override=True)


# Step #1: Define your response structure
class President(Struct):
    name: str
    age: Annotated[int, Meta(ge=40, le=107)]


class PresidentsResponse(Struct):
    presidents: list[President]


messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="Hello, how are you?"),
    Message(role=MessageRole.ASSISTANT, content="I'm fine! And you? Intellibricks is awesome, isn't it? (This was completely generated by AI and not the owner of the project)"),
    Message(role=MessageRole.USER, content="I'm fine. What are the presidents of the USA?"),
]

response = engine.chat(
    messages=messages,
    response_format=PresidentsResponse
)

presidents_response: Presidentsresponse = response.get_parsed()
print(presidents_response)
```


### Training Machine Learning Models

Train supervised learning models effortlessly with the `SupervisedLearningEngine`.  Provide your data, configuration, and let IntelliBricks manage the training and prediction pipeline.

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

## Advanced Usage


###  System Prompts and Chat History

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

###  Customizing Prompts

```python
from intellibricks import Prompt

prompt_template = Prompt(content="My name is {{name}}. I am {{age}} years old.") # Implements __str__
compiled_prompt = prompt_template.compile(name="John", age=30) # Returns Prompt
print(compiled_prompt)  # Output: My name is John. I am 30 years old.
```

### Langfuse Integration

IntelliBricks integrates with Langfuse for enhanced observability of your LLM interactions.  Trace performance, track costs, and monitor events with ease.  This integration is automatically activated when you instantiate a `CompletionEngine` with a Langfuse instance.

```python
import os
from langfuse import Langfuse

langfuse_client = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
)

engine = CompletionEngine(langfuse=langfuse_client)

# Now all LLM calls made with 'engine' will be automatically tracked in Langfuse. Even the costs.
```



## Coming Soon

* **Enhanced RAG:** A more robust RAG implementation for seamless integration with diverse knowledge sources.
* **Unified Document Parsing** Stop wasting time choosing the right library for parsing pdfs. We will chose the right one for you (and let you choose to of course), with our DocumentArtifact model, it will be easily convertable to llama_index and langchain documents. You can pass your transformations too. We will offer support for NER and Relations extraction too. The intent is to use MinerU for PDFs, and Docling for the rest. Example: 

```py
extractor: FileExtractorProtocol = ... # In development
document = extractor.extract(RawFile.from_dir("./documents")) # or RawFile.from_upload_file(fastapi and litestar objects goes here). RawFile will be a powerful class
document.as_langchain_docs(transformations=[SemanticChunker(...)])
# Done. Now you can ingest your doc into 
vector_store.add_documents(documents) # Langchain example
```

## Documentation

For more detailed information and API references, please refer to the comprehensive [IntelliBricks documentation](link-to-docs).  *(In development)*


## Contributing

We welcome contributions to IntelliBricks!  Please see our [contribution guidelines](link-to-contribution-guidelines). *(In development)*
