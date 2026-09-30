import os
import cv2
import numpy as np


EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]

LABELS = {
    "angry": 0,
    "happy": 1,
    "neutral": 2,
    "sad": 3,
    "surprise": 4
}


def load_dataset():

    BASE_PATH = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    PROCESSED_PATH = os.path.join(
        BASE_PATH,
        "dataset",
        "processed"
    )

    # ---------------- TRAIN ----------------

    X_train = []
    y_train = []

    print("\nLoading TRAIN dataset...")

    for emotion in EMOTIONS:

        emotion_path = os.path.join(
            PROCESSED_PATH,
            "train",
            emotion
        )

        print(f"Loading {emotion}...")

        for image_name in os.listdir(emotion_path):

            image_path = os.path.join(
                emotion_path,
                image_name
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = image / 255.0

            image = image.reshape(48, 48, 1)

            X_train.append(image)
            y_train.append(LABELS[emotion])


    # ---------------- TEST ----------------

    X_test = []
    y_test = []

    print("\nLoading TEST dataset...")

    for emotion in EMOTIONS:

        emotion_path = os.path.join(
            PROCESSED_PATH,
            "test",
            emotion
        )

        print(f"Loading {emotion}...")

        for image_name in os.listdir(emotion_path):

            image_path = os.path.join(
                emotion_path,
                image_name
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = image / 255.0

            image = image.reshape(48, 48, 1)

            X_test.append(image)
            y_test.append(LABELS[emotion])


    # List → NumPy Array

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)


    print("\n========== DATASET SHAPE ==========")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    return X_train, y_train, X_test, y_test