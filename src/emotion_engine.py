from .face_detector import FaceDetector
from .emotion_predictor import EmotionPredictor
from collections import deque


class EmotionEngine:

    def __init__(self, face_model_path, emotion_model_path):

        # Face detection module
        self.face_detector = FaceDetector(
            face_model_path
        )

        # Emotion prediction module
        self.emotion_predictor = EmotionPredictor(
            emotion_model_path
        )

        # Store last 7 predictions
        self.prediction_history = deque(
            maxlen=7
        )

        print(
            "Emotion engine initialized successfully!"
        )


    def process(self, frame):

        # Detect faces
        faces = self.face_detector.detect(frame)

        # No face detected
        if not faces:

            self.prediction_history.clear()

            return None


        # Currently process first face
        face = faces[0]

        # Get face crop
        face_crop = face["crop"]

        # Get bounding box
        box = face["box"]


        # Predict emotion
        emotion, confidence = (
            self.emotion_predictor.predict(
                face_crop
            )
        )


        # Store emotion + confidence
        self.prediction_history.append(
            (emotion, confidence)
        )


        # --------------------------------
        # Emotion smoothing
        # --------------------------------

        emotion_history = [
            item[0]
            for item in self.prediction_history
        ]


        final_emotion = max(
            set(emotion_history),
            key=emotion_history.count
        )


        # --------------------------------
        # Confidence smoothing
        # --------------------------------

        final_confidences = [
            item[1]
            for item in self.prediction_history
            if item[0] == final_emotion
        ]


        if final_confidences:

            smoothed_confidence = (
                sum(final_confidences)
                / len(final_confidences)
            )

        else:

            smoothed_confidence = confidence


        # --------------------------------
        # Return result
        # --------------------------------

        return {
            "emotion": final_emotion,
            "confidence": smoothed_confidence,
            "box": box
        }


    def close(self):

        self.face_detector.close()

        print(
            "Emotion engine closed."
        )