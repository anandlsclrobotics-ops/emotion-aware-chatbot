import cv2
import numpy as np
import tensorflow as tf


class EmotionPredictor:

    def __init__(self, model_path):

        self.model = tf.keras.models.load_model(model_path)

        self.emotions = [
            "Angry",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise"
        ]

        print("Emotion model loaded successfully!")

    def preprocess(self, face_crop):

        # Convert BGR image to grayscale
        gray_face = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )

        # Resize to CNN input size
        resized_face = cv2.resize(
            gray_face,
            (48, 48)
        )

        # Normalize pixel values
        normalized_face = (
            resized_face.astype("float32") / 255.0
        )

        # Add channel dimension
        cnn_face = normalized_face.reshape(
            48, 48, 1
        )

        # Add batch dimension
        cnn_input = cnn_face.reshape(
            1, 48, 48, 1
        )

        return cnn_input

    def predict(self, face_crop):

        # Check whether face crop exists
        if face_crop is None or face_crop.size == 0:
            return None, 0.0

        # Preprocess face
        cnn_input = self.preprocess(face_crop)

        # CNN prediction
        predictions = self.model.predict(
            cnn_input,
            verbose=0
        )[0]

        # Find highest probability
        emotion_index = int(
            np.argmax(predictions)
        )

        emotion = self.emotions[emotion_index]

        confidence = float(
            predictions[emotion_index]
        )

        return emotion, confidence

