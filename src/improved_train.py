# ==========================================
# IMPROVED CNN TRAINING - EMOTION DETECTION
# ==========================================

# Important libraries
import os
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from load_dataset import load_dataset


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n========== LOADING DATASET ==========")

X_train, y_train, X_test, y_test = load_dataset()

print("\nDataset Loaded Successfully!")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ==========================================
# 2. TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\n========== DATA SPLIT ==========")

print("Training data   :", X_train.shape)
print("Validation data :", X_val.shape)
print("Test data       :", X_test.shape)


# ==========================================
# 3. EMOTION LABELS
# ==========================================

EMOTIONS = [
    "Angry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ==========================================
# 4. IMPROVED CNN ARCHITECTURE
# ==========================================

print("\n========== BUILDING IMPROVED CNN ==========")

model = Sequential([

    # ------------------------------
    # DATA AUGMENTATION
    # ------------------------------

    RandomFlip(
        "horizontal"
    ),

    RandomRotation(
        0.05
    ),

    RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    RandomZoom(
        0.10
    ),


    # ------------------------------
    # CNN BLOCK 1
    # ------------------------------

    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # CNN BLOCK 2
    # ------------------------------

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # CNN BLOCK 3
    # ------------------------------

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # CLASSIFICATION
    # ------------------------------

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting
    Dropout(
        0.5
    ),

    # 5 emotion classes
    Dense(
        5,
        activation="softmax"
    )
])


# ==========================================
# 5. MODEL SUMMARY
# ==========================================

print("\n========== MODEL SUMMARY ==========")

model.summary()


# ==========================================
# 6. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully!")


# ==========================================
# 7. EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\n========== MODEL TRAINING ==========")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=25,

    batch_size=32,

    callbacks=[
        early_stopping
    ]
)


print("\n========== MODEL TRAINING COMPLETED ==========")


# ==========================================
# 9. TEST EVALUATION
# ==========================================

print("\n========== TEST EVALUATION ==========")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print(
    "Test Accuracy: {:.2f}%".format(
        test_accuracy * 100
    )
)


# ==========================================
# 10. PREDICTIONS
# ==========================================

print("\n========== GENERATING PREDICTIONS ==========")

y_pred_probability = model.predict(
    X_test
)

y_pred = y_pred_probability.argmax(
    axis=1
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=EMOTIONS
    )
)


# ==========================================
# 13. SAVE IMPROVED MODEL
# ==========================================

BASE_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "emotion_cnn_improved.keras"
)

model.save(
    MODEL_PATH
)

print("\n========== MODEL SAVED ==========")

print(
    "Model saved at:",
    MODEL_PATH
)