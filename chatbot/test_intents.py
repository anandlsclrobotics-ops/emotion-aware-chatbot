from intents import INTENTS


for intent_name, intent_data in INTENTS.items():

    print("Intent:", intent_name)

    print("Patterns:")

    for pattern in intent_data["patterns"]:
        print(" -", pattern)

    print("Responses:")

    for response in intent_data["responses"]:
        print(" -", response)

    print("-" * 40)