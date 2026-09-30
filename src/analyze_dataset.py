import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_PATH, "dataset", "processed", "train")
TEST_PATH = os.path.join(BASE_PATH, "dataset", "processed", "test")

EMOTIONS = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def count_images(dataset_path):

    counts = {}

    for emotion in EMOTIONS:

        emotion_path = os.path.join(dataset_path, emotion)

        count = 0

        for file_name in os.listdir(emotion_path):

            file_path = os.path.join(emotion_path, file_name)

            if os.path.isfile(file_path):
                count += 1

        counts[emotion] = count

    return counts


train_counts = count_images(TRAIN_PATH)
test_counts = count_images(TEST_PATH)


print("\n========== TRAIN DATASET ==========")

train_total = sum(train_counts.values())

for emotion, count in train_counts.items():

    percentage = (count / train_total) * 100

    print(f"{emotion:10} : {count:5} images ({percentage:.2f}%)")

print(f"Total      : {train_total}")


print("\n========== TEST DATASET ==========")

test_total = sum(test_counts.values())

for emotion, count in test_counts.items():

    percentage = (count / test_total) * 100

    print(f"{emotion:10} : {count:5} images ({percentage:.2f}%)")

print(f"Total      : {test_total}")