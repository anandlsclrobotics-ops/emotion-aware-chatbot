import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self, model_path):

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = FaceLandmarker.create_from_options(
            self.options
        )

        self.frame_timestamp = 0

        print("Face detector loaded successfully!")

    def detect(self, frame):

        # Get frame dimensions
        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect face landmarks
        result = self.landmarker.detect_for_video(
            mp_image,
            self.frame_timestamp
        )

        # Increase timestamp
        self.frame_timestamp += 1

        faces = []

        # Check whether face is detected
        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                # Get X coordinates
                x_coordinates = [
                    landmark.x
                    for landmark in face_landmarks
                ]

                # Get Y coordinates
                y_coordinates = [
                    landmark.y
                    for landmark in face_landmarks
                ]

                # Find bounding box
                x_min = int(min(x_coordinates) * width)
                x_max = int(max(x_coordinates) * width)

                y_min = int(min(y_coordinates) * height)
                y_max = int(max(y_coordinates) * height)

                # Add padding
                padding = 20

                x_min = max(
                    0,
                    x_min - padding
                )

                y_min = max(
                    0,
                    y_min - padding
                )

                x_max = min(
                    width,
                    x_max + padding
                )

                y_max = min(
                    height,
                    y_max + padding
                )

                # Crop face
                face_crop = frame[
                    y_min:y_max,
                    x_min:x_max
                ]

                if face_crop.size > 0:

                    faces.append({
                        "crop": face_crop,
                        "box": (
                            x_min,
                            y_min,
                            x_max,
                            y_max
                        )
                    })

        return faces

    def close(self):

        self.landmarker.close()

