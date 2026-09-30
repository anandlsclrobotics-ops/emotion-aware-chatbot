import os
import cv2


BASE_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_PATH = os.path.join(
    BASE_PATH,
    "dataset",
    "processed",
    "train",
    "happy"
)


# Happy folder ki first image
image_name = os.listdir(IMAGE_PATH)[0]

image_file = os.path.join(
    IMAGE_PATH,
    image_name
)


# Image read
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)


print("Image name:", image_name)
print("Image shape:", image.shape)
print("Minimum pixel:", image.min())
print("Maximum pixel:", image.max())
print("Data type:", image.dtype)


# Normalize
normalized = image / 255.0

print("\nAfter normalization:")
print("Minimum:", normalized.min())
print("Maximum:", normalized.max())
print("Data type:", normalized.dtype)

# add grayscale channel
cnn_image=normalized.reshape(48,48,1)
print("Cnn Image Shape:", cnn_image.shape)
