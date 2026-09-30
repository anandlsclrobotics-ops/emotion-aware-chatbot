import os 
import cv2
import numpy as np
import tensorflow as tf 

# emotion labels 
EMOTIONS=["Angry", "Happy", "Neutral", "Sad", "Surprise"]
# model path 
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH=os.path.join(
    BASE_DIR,
    "models",
    "emotion_cnn_improved.keras"
)

# load model

print("Loading emotion model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Emotion model loaded successfully!")
# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def predict_emotion(face):

    # Convert to grayscale
    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to CNN input size
    gray = cv2.resize(
        gray,
        (48, 48)
    )

    # Normalize pixel values
    gray = gray / 255.0

    # Add channel dimension
    gray = gray.reshape(
        1, 48, 48, 1
    )

    # Get prediction probabilities
    prediction = model.predict(
        gray,
        verbose=0
    )

    # Find highest probability
    emotion_index = np.argmax(
        prediction[0]
    )

    # Get emotion name
    emotion = EMOTIONS[
        emotion_index
    ]

    # Get confidence
    confidence = prediction[0][
        emotion_index
    ]

    return emotion, confidence