import timeit
import time
from typing import Sequence, Optional
from dotenv import load_dotenv
import msgspec
from pydantic import BaseModel, Field


load_dotenv(override=True)

# --- Data Model ---


class AuthorMsgspec(msgspec.Struct, frozen=True):
    name: str
    affiliation: Optional[str] = None


class MetadataMsgspec(msgspec.Struct, frozen=True):
    creation_date: int
    source_document: Optional[str] = None


class SummaryMsgspec(msgspec.Struct, frozen=True):
    caption: str
    key_points: Sequence[str]
    author: Optional[AuthorMsgspec] = None  # Composition: Author object
    metadata: Optional[MetadataMsgspec] = None  # Composition: Metadata object


# --- pydantic versions ---


class AuthorPydantic(BaseModel):
    name: str = Field(description="Author's Name")
    affiliation: Optional[str] = Field(default=None, description="Author's Affiliation")


class MetadataPydantic(BaseModel):
    creation_date: int = Field(description="Date of Summary Creation")
    source_document: Optional[str] = Field(
        default=None, description="Source Document Title"
    )


class SummaryPydantic(BaseModel):
    caption: str = Field(description="Summary Title")
    key_points: Sequence[str] = Field(description="Main points of the summary")
    author: Optional[AuthorPydantic] = Field(
        default=None, description="Summary Author Information"
    )  # Composition: Author object
    metadata: Optional[MetadataPydantic] = Field(
        default=None, description="Summary Metadata"
    )  # Composition: Metadata object


NUM_EPOCHS = 10  # Increased epochs for better averaging
NUM_WARMUP_RUNS = 0  # Number of warm-up runs before benchmarking
SLEEP_TIME_SECONDS = 60  # Reduced sleep time between benchmarks

# --- Benchmark Functions ---
prompt = "Summarize the following article about quantum computing and return a structured JSON response according to the schema. Article: Quantum computing is a revolutionary field that harnesses the principles of quantum mechanics to solve complex problems beyond the reach of classical computers. Key concepts include superposition, entanglement, and quantum gates. Applications range from drug discovery to materials science and cryptography.  Key takeaways: Quantum computers leverage superposition and entanglement to perform computations in a fundamentally different way than classical computers, potentially leading to exponential speedups for certain types of problems."


def benchmark_intellibricks():
    from intellibricks.llms.integrations.google import GoogleLanguageModel

    model = GoogleLanguageModel(model_name="gemini-1.5-flash", vertexai=False)

    def run_benchmark():
        response = model.complete(prompt, response_model=SummaryMsgspec)  # type: ignore # noqa

    # Warm-up runs
    for _ in range(NUM_WARMUP_RUNS):
        run_benchmark()

    time_taken = timeit.timeit(run_benchmark, number=NUM_EPOCHS)
    return time_taken / NUM_EPOCHS  # Average time


def benchmark_langchain():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash").with_structured_output(  # type: ignore
        SummaryPydantic
    )

    def run_benchmark():
        response = llm.invoke(prompt)  # type: ignore # noqa

    # Warm-up runs
    for _ in range(NUM_WARMUP_RUNS):
        run_benchmark()

    time_taken = timeit.timeit(run_benchmark, number=NUM_EPOCHS)
    return time_taken / NUM_EPOCHS  # Average time


def benchmark_llamaindex():
    from llama_index.llms.gemini import Gemini

    llm = Gemini(model="models/gemini-1.5-flash").as_structured_llm(SummaryPydantic)

    def run_benchmark():
        response = llm.complete(prompt)  # type: ignore # noqa

    # Warm-up runs
    for _ in range(NUM_WARMUP_RUNS):
        run_benchmark()

    time_taken = timeit.timeit(run_benchmark, number=NUM_EPOCHS)
    return time_taken / NUM_EPOCHS  # Average time


# --- Run Benchmarks ---

if __name__ == "__main__":
    import time

    print("Warming up benchmarks...")  # Indicate warm-up phase
    benchmark_intellibricks()  # Warm-up
    benchmark_langchain()  # Warm-up
    benchmark_llamaindex()  # Warm-up
    print("Warm-up complete. Starting timed benchmarks...")

    start_time = time.time()  # Start overall timer
    intellibricks_time = benchmark_intellibricks()
    print(
        f"IntelliBricks benchmark completed in {time.time() - start_time:.2f} seconds. Waiting {SLEEP_TIME_SECONDS} seconds..."
    )
    time.sleep(SLEEP_TIME_SECONDS)

    start_time = time.time()  # Restart timer for next benchmark
    langchain_time = benchmark_langchain()
    print(
        f"LangChain benchmark completed in {time.time() - start_time:.2f} seconds. Waiting {SLEEP_TIME_SECONDS} seconds..."
    )
    time.sleep(SLEEP_TIME_SECONDS)

    start_time = time.time()  # Restart timer for next benchmark
    llamaindex_time = benchmark_llamaindex()
    print(
        f"LlamaIndex benchmark completed in {time.time() - start_time:.2f} seconds. Waiting {SLEEP_TIME_SECONDS} seconds..."
    )
    time.sleep(SLEEP_TIME_SECONDS)

    print("\nBenchmark Results (Average time over {} iterations):".format(NUM_EPOCHS))
    print(f"IntelliBricks (msgspec): {intellibricks_time:.6f} seconds")
    print(f"LangChain (pydantic):   {langchain_time:.6f} seconds")
    print(f"LlamaIndex (pydantic):  {llamaindex_time:.6f} seconds")

    print("\n--- Benchmark Limitations ---")
    print(
        "1. **Network Dependency:** These benchmarks are heavily influenced by network latency to the LLM service. Network conditions can vary significantly, affecting results."
    )
    print(
        "2. **LLM Processing Time:** The time measured includes not only data parsing/validation but also the time taken by the LLM to generate the structured output. This benchmark does not isolate parsing/validation speed."
    )
    print(
        "3. **System Load:** Benchmark results can be affected by the overall load on your system. For more controlled benchmarks, run on a system with minimal background processes."
    )
    print(
        "4. **API Variability:** LLM API performance can fluctuate. Results might vary between different runs even under seemingly identical conditions."
    )
