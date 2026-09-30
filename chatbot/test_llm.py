from .llm_engine import LLMEngine


def main():

    llm = LLMEngine()

    print("\nTesting Gemini LLM...\n")

    response = llm.generate_response(
        user_message="Explain artificial intelligence in simple words.",
        emotion="Happy",
        confidence=0.82
    )

    print("AI Response:")
    print(response)


if __name__ == "__main__":
    main()