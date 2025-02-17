from intellibricks.llms import TtsSynapse

tts_llm = TtsSynapse.of("openai/api/tts-1")
speech = tts_llm.generate_speech(
    "Hello!!!!",
    voice="alloy",
)
speech.save_to_file("speech.mp3")
