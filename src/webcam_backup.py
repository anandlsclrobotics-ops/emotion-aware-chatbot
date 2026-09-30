import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque


# =========================================================
# 1. LOAD EMOTION CNN MODEL
# =========================================================

EMOTION_MODEL_PATH = "../models/emotion_cnn_improved.keras"

model = tf.keras.models.load_model(EMOTION_MODEL_PATH)

print("Emotion model loaded successfully!")


# =========================================================
# 2. EMOTION LABELS
# =========================================================

EMOTIONS = [
    "Angry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# =========================================================
# 3. PREDICTION SMOOTHING
# =========================================================

prediction_history = deque(maxlen=7)


# =========================================================
# 4. MEDIAPIPE FACE LANDMARKER SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


FACE_MODEL_PATH = "../models/face_landmarker.task"


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)


# =========================================================
# 5. FPS VARIABLES
# =========================================================

previous_time = 0


# =========================================================
# 6. CREATE FACE LANDMARKER
# =========================================================

with FaceLandmarker.create_from_options(options) as landmarker:
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera could not be opened!")
        exit()

    print("Webcam started...")
    print("Press Q to quit.")

    frame_timestamp = 0
    # =====================================================
    # 7. MAIN LOOP
    # =====================================================
    while True:

        # -------------------------------------------------
        # READ FRAME
        # -------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame!")
            break


        # -------------------------------------------------
        # FLIP FRAME
        # -------------------------------------------------

        frame = cv2.flip(frame, 1)


        # -------------------------------------------------
        # GET FRAME SIZE
        # -------------------------------------------------

        height, width, _ = frame.shape


        # -------------------------------------------------
        # BGR -> RGB
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # -------------------------------------------------
        # OPENCV -> MEDIAPIPE IMAGE
        # -------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # -------------------------------------------------
        # FACE LANDMARK DETECTION
        # -------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        frame_timestamp += 1


        # =================================================
        # 8. FACE DETECTED
        # =================================================

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                # -------------------------------------------------
                # GET LANDMARK COORDINATES
                # -------------------------------------------------

                x_coordinates = [
                    landmark.x
                    for landmark in face_landmarks
                ]

                y_coordinates = [
                    landmark.y
                    for landmark in face_landmarks
                ]


                # -------------------------------------------------
                # FIND FACE BOUNDING BOX
                # -------------------------------------------------

                x_min = int(min(x_coordinates) * width)
                x_max = int(max(x_coordinates) * width)

                y_min = int(min(y_coordinates) * height)
                y_max = int(max(y_coordinates) * height)


                # -------------------------------------------------
                # ADD PADDING
                # -------------------------------------------------

                padding = 20

                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)

                x_max = min(width, x_max + padding)
                y_max = min(height, y_max + padding)


                # -------------------------------------------------
                # FACE CROP
                # -------------------------------------------------

                face_crop = frame[
                    y_min:y_max,
                    x_min:x_max
                ]


                # -------------------------------------------------
                # CHECK FACE CROP
                # -------------------------------------------------

                if face_crop.size > 0:

                    # =================================================
                    # 9. CNN PREPROCESSING
                    # =================================================

                    # BGR -> GRAYSCALE
                    gray_face = cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2GRAY
                    )


                    # Resize to CNN input size
                    resized_face = cv2.resize(
                        gray_face,
                        (48, 48)
                    )


                    # Normalize
                    normalized_face = resized_face / 255.0


                    # Add channel dimension
                    cnn_face = normalized_face.reshape(
                        48,
                        48,
                        1
                    )


                    # Add batch dimension
                    cnn_input = cnn_face.reshape(
                        1,
                        48,
                        48,
                        1
                    )


                    # =================================================
                    # 10. CNN EMOTION PREDICTION
                    # =================================================

                    predictions = model.predict(
                        cnn_input,
                        verbose=0
                    )


                    # -------------------------------------------------
                    # FIND HIGHEST PROBABILITY
                    # -------------------------------------------------

                    emotion_index = np.argmax(
                        predictions[0]
                    )

                    emotion = EMOTIONS[emotion_index]

                    confidence = predictions[0][emotion_index]


                    # =================================================
                    # 11. PREDICTION SMOOTHING
                    # =================================================

                    prediction_history.append(emotion)


                    final_emotion = max(
                        set(prediction_history),
                        key=prediction_history.count
                    )


                    # =================================================
                    # 12. DRAW FACE BOX
                    # =================================================

                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2
                    )


                    # =================================================
                    # 13. DISPLAY EMOTION
                    # =================================================

                    cv2.putText(
                        frame,
                        f"Emotion: {final_emotion}",
                        (x_min, y_min - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )


                    # -------------------------------------------------
                    # CONFIDENCE
                    # -------------------------------------------------

                    cv2.putText(
                        frame,
                        f"Confidence: {confidence * 100:.1f}%",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )


                    # =================================================
                    # 14. DISPLAY FACE CROP
                    # =================================================

                    cv2.imshow(
                        "Face Crop",
                        face_crop
                    )


        # =================================================
        # 15. NO FACE DETECTED
        # =================================================

        else:

            prediction_history.clear()

            cv2.putText(
                frame,
                "No Face Detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # =================================================
        # 16. FPS CALCULATION
        # =================================================

        current_time = time.time()

        fps = 1 / (
            current_time - previous_time
        ) if previous_time != 0 else 0

        previous_time = current_time
        # =================================================
        # 17. DISPLAY MAIN WINDOW
        # =================================================

        cv2.imshow(
            "Emotion Detection",
            frame
        )


        # =================================================
        # 18. QUIT
        # =================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # =====================================================
    # 19. RELEASE RESOURCES
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()

    