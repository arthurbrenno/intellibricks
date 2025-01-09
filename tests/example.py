from intellibricks import Synapse

synapse = Synapse.of("google/genai/gemini-2.0-flash-exp")

completion = synapse.complete("Hello, how are you?")  # Completion[RawResponse]

print(completion)
