from .chatbot_engine import ChatbotEngine


bot = ChatbotEngine()


# =========================================================
# TEST CASES
# =========================================================

test_cases = [

    ("Hello", "Happy"),

    ("Hello", "Sad"),

    ("How are you?", "Neutral"),

    ("I am feeling sad", "Sad"),

    ("I am angry", "Angry"),

    ("What can you do?", "Happy"),

    ("Bye", "Happy"),

    ("xyz abc", "Sad"),

    ("xyz abc", "Happy"),

    ("xyz abc", "Angry"),

]


# =========================================================
# RUN TESTS
# =========================================================

for message, emotion in test_cases:

    print("=" * 70)

    print("User:", message)

    print("Detected Emotion:", emotion)

    intent, response = bot.chat(
        message,
        emotion
    )

    print("Detected Intent:", intent)

    print("Bot:", response)

    print("=" * 70)

print()

print("All emotion-aware chatbot tests completed.")