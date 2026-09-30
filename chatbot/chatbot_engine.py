import random

from .intents import INTENTS


class ChatbotEngine:

    def __init__(self):
        self.intents = INTENTS

        print("Chatbot engine initialized successfully!")

    # =========================================================
    # PREPROCESS MESSAGE
    # =========================================================

    def preprocess(self, user_message):

        cleaned_message = user_message.strip()

        cleaned_message = cleaned_message.lower()

        return cleaned_message

    # =========================================================
    # DETECT INTENT
    # =========================================================

    def detect_intent(self, user_message):

        message = self.preprocess(user_message)

        for intent_name, intent_data in self.intents.items():

            for pattern in intent_data["patterns"]:

                pattern = pattern.lower()

                if pattern in message:

                    return intent_name

        return "unknown"

    # =========================================================
    # NORMAL RESPONSE
    # =========================================================

    def generate_response(self, intent_name):

        if intent_name == "unknown":

            return (
                "I'm sorry, I don't understand that yet. "
                "Could you please rephrase your message?"
            )

        responses = self.intents[intent_name]["responses"]

        return random.choice(responses)

    # =========================================================
    # EMOTION-AWARE RESPONSE
    # =========================================================

    def generate_emotion_response(self, intent_name, emotion):

        # -----------------------------------------------------
        # UNKNOWN INTENT
        # -----------------------------------------------------

        if intent_name == "unknown":

            if emotion == "Sad":

                return (
                    "I'm not completely sure what you mean, "
                    "but I can see that you may be feeling sad. "
                    "I'm here if you'd like to talk. 💙"
                )

            elif emotion == "Angry":

                return (
                    "I'm not sure I understood that. "
                    "You seem a little frustrated, so let's take "
                    "it one step at a time."
                )

            elif emotion == "Happy":

                return (
                    "I'm not sure I understood that, "
                    "but you seem to be in a good mood! 😊 "
                    "Could you try saying it another way?"
                )

            elif emotion == "Surprise":

                return (
                    "I'm not quite sure what you mean. "
                    "That seemed to surprise you! 😲 "
                    "Could you explain a little more?"
                )

            else:

                return (
                    "I'm sorry, I don't understand that yet. "
                    "Could you please rephrase your message?"
                )

        # -----------------------------------------------------
        # GREETING
        # -----------------------------------------------------

        if intent_name == "greeting":

            if emotion == "Happy":

                return (
                    "Hi! Nice to talk to you. "
                    "You seem to be in a good mood! "
                    "That's great! 😊"
                )

            elif emotion == "Sad":

                return (
                    "Hello. You seem a little sad today. "
                    "I'm here if you'd like to talk. 💙"
                )

            elif emotion == "Angry":

                return (
                    "Hello. You seem a little frustrated. "
                    "Take a breath, and I'm here to listen."
                )

            elif emotion == "Surprise":

                return (
                    "Hello! You look surprised! 😲 "
                    "What's going on?"
                )

            else:

                return (
                    "Hello! 👋 How can I help you today?"
                )

        # -----------------------------------------------------
        # GENERAL
        # -----------------------------------------------------

        if intent_name == "general":

            if emotion == "Sad":

                return (
                    "I'm doing well. "
                    "But you seem a little sad today. "
                    "Would you like to talk about it? 💙"
                )

            elif emotion == "Happy":

                return (
                    "I'm doing well! "
                    "And you seem to be feeling positive today. 😊"
                )

            elif emotion == "Angry":

                return (
                    "I'm doing well. "
                    "You seem a little frustrated though. "
                    "I'm here to listen."
                )

            elif emotion == "Surprise":

                return (
                    "I'm doing well. "
                    "You seem a little surprised today! 😲"
                )

            else:

                return (
                    "I'm doing well! "
                    "How are you feeling today?"
                )

        # -----------------------------------------------------
        # EMOTIONAL SUPPORT
        # -----------------------------------------------------

        if intent_name == "emotional_support":

            if emotion == "Sad":

                return (
                    "I'm sorry you're feeling this way. "
                    "I'm here to listen. 💙 "
                    "You don't have to deal with everything alone."
                )

            elif emotion == "Angry":

                return (
                    "It sounds like something may be frustrating you. "
                    "Take a moment to breathe. "
                    "I'm here if you'd like to talk."
                )

            elif emotion == "Happy":

                return (
                    "I'm glad you're feeling positive! 😊 "
                    "It's always nice to have a good moment."
                )

            elif emotion == "Surprise":

                return (
                    "You seem surprised! 😲 "
                    "If you'd like, you can tell me what happened."
                )

            else:

                return (
                    "I'm here to listen. "
                    "Would you like to tell me more about how you're feeling?"
                )

        # -----------------------------------------------------
        # CAPABILITIES
        # -----------------------------------------------------

        if intent_name == "capabilities":

            return (
                "I can chat with you, detect basic conversational "
                "intents, and adapt my responses according to "
                "your detected facial emotion."
            )

        # -----------------------------------------------------
        # GOODBYE
        # -----------------------------------------------------

        if intent_name == "goodbye":

            if emotion == "Sad":

                return (
                    "Goodbye. Take care of yourself. 💙 "
                    "I hope you feel better soon."
                )

            elif emotion == "Happy":

                return (
                    "Goodbye! Keep that positive energy going! 😊"
                )

            elif emotion == "Angry":

                return (
                    "Goodbye. Take some time to relax. "
                    "Take care of yourself."
                )

            elif emotion == "Surprise":

                return (
                    "Goodbye! Take care and see you later. 😲"
                )

            else:

                return (
                    "Goodbye! Take care and have a nice day."
                )

        # -----------------------------------------------------
        # FALLBACK
        # -----------------------------------------------------

        return self.generate_response(intent_name)

    # =========================================================
    # COMPLETE CHAT
    # =========================================================

    def chat(self, user_message, emotion=None):

        # Detect user's intent
        intent = self.detect_intent(user_message)

        # Generate response according to emotion
        if emotion:

            response = self.generate_emotion_response(
                intent,
                emotion
            )

        else:

            response = self.generate_response(intent)

        return intent, response