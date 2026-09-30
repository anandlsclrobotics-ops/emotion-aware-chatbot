import cv2
from prediction import predict_emotion


# Test image path
image_path = "../dataset/processed/test/happy/PrivateTest_95094.jpg"


# Load image
image = cv2.imread(image_path)


# Check image
if image is None:
    print("Image not found!")
else:

    # Predict emotion
    emotion, confidence = predict_emotion(image)

    print("\n========== PREDICTION ==========")

    print("Emotion:", emotion)

    print(
        "Confidence: {:.2f}%".format(
            confidence * 100
        )
    )