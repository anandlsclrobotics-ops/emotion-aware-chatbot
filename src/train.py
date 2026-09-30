# Yeh model hai dataset+CNN arc+model.summary() banan hai 

# Important libraries 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from load_dataset import load_dataset 
# for data validation high 
from sklearn.model_selection import train_test_split
# for the confusion matrix 
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os



# Load dataset 
X_train,y_train,X_test,y_test=load_dataset()

# model accuracy validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


# CnN model Architechure banye gye yha per 
model=Sequential(
    [  

        # frist layer input CNN
        Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)),
        MaxPooling2D(2,2), 
        # Second layer cNN
        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation="relu"),
        Dense(5, activation="softmax")
    ]
)

# model comlipation
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("\nModel compiled successfully!")

# Model training 
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val,y_val),
    epochs=10,
    batch_size=32
)
print("\nModel Trained Done")
# print("\n========== SPLIT SHAPE ==========")
# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)
# print("X_val  :", X_val.shape)
# print("y_val  :", y_val.shape)
# print("X_test :", X_test.shape)
# print("y_test :", y_test.shape)
# Unseen model evalution 
print("\n=====Model test Evalution======== ")
test_loss,test_accuracy=model.evaluate(
    X_test,
    y_test,
)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

print("\nEmotion Labels:")
print(EMOTIONS)
# test prediction and  argmax
y_pred_prob=model.predict(X_test)
y_pred=y_pred_prob.argmax(axis=1)

# confusion matrix 
cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix")
print(cm)
# classifiction report
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=EMOTIONS
    )
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "emotion_cnn_baseline.keras"
)

model.save(MODEL_PATH)

print("\n========== MODEL SAVED ==========")
print("Model saved at:", MODEL_PATH)

model.summary()