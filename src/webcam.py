import cv2
import time

# from emotion_engine import EmotionEngine
from src.emotion_engine import EmotionEngine


# ==========================================
# MODEL PATHS
# ==========================================

FACE_MODEL_PATH = "../models/face_landmarker.task"

EMOTION_MODEL_PATH = "../models/emotion_cnn_improved.keras"


# ==========================================
# CREATE EMOTION ENGINE
# ==========================================

engine = EmotionEngine(
    FACE_MODEL_PATH,
    EMOTION_MODEL_PATH
)


# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Camera could not be opened!")

    engine.close()

    exit()


print("Webcam started!")
print("Press Q to quit.")


# ==========================================
# FPS
# ==========================================

previous_time = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:

        print("Could not read frame!")

        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # --------------------------------------
    # Emotion Engine
    # --------------------------------------

    result = engine.process(frame)


    # --------------------------------------
    # Face detected
    # --------------------------------------

    if result is not None:

        emotion = result["emotion"]

        confidence = result["confidence"]

        x_min, y_min, x_max, y_max = result["box"]


        # Draw face box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )


        # Display emotion
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (x_min, y_min - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # Display confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (x_min, y_min - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # --------------------------------------
    # No face
    # --------------------------------------

    else:

        cv2.putText(
            frame,
            "No Face Detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ======================================
    # FPS CALCULATION
    # ======================================

    current_time = time.time()


    if previous_time != 0:

        fps = 1 / (
            current_time - previous_time
        )

    else:

        fps = 0


    previous_time = current_time


    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ======================================
    # SHOW FRAME
    # ======================================

    cv2.imshow(
        "Emotion Detection",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

engine.close()

cv2.destroyAllWindows()

print("Webcam stopped.")