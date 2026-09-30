import os
import shutil

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_TRAIN = os.path.join(BASE_PATH, "dataset", "train")
SOURCE_TEST = os.path.join(BASE_PATH, "dataset", "test")

DEST_TRAIN = os.path.join(BASE_PATH, "dataset", "processed", "train")
DEST_TEST = os.path.join(BASE_PATH, "dataset", "processed", "test")

# Only these 5 emotions
EMOTIONS = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def copy_emotion_images(source_folder, destination_folder):
    total = 0

    for emotion in EMOTIONS:

        source_emotion = os.path.join(source_folder, emotion)
        destination_emotion = os.path.join(destination_folder, emotion)

        os.makedirs(destination_emotion, exist_ok=True)

        for file_name in os.listdir(source_emotion):

            source_file = os.path.join(source_emotion, file_name)
            destination_file = os.path.join(destination_emotion, file_name)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)
                total += 1

        print(f"{emotion}: copied")

    return total


print("Processing TRAIN dataset...")
train_count = copy_emotion_images(SOURCE_TRAIN, DEST_TRAIN)

print("\nProcessing TEST dataset...")
test_count = copy_emotion_images(SOURCE_TEST, DEST_TEST)

print("\n==============================")
print("Dataset processing completed!")
print("Training images:", train_count)
print("Testing images:", test_count)
print("==============================")