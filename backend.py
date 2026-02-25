# ===============================
# Jaundice Detection - Backend
# Model Training Script
# ===============================

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from numpy import argmax

from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical

# ===============================
# Dataset Configuration
# ===============================

DATASET_PATH = "Dataset/"
CATEGORIES = ["jaundice", "normal"]
HEIGHT, WIDTH, CHANNELS = 65, 65, 3

# ===============================
# Load Dataset
# ===============================

data, labels = [], []
image_paths = []

for label, category in enumerate(CATEGORIES):
    path = os.path.join(DATASET_PATH, category)
    for file in os.listdir(path):
        image_paths.append((os.path.join(path, file), label))

random.shuffle(image_paths)

for path, label in image_paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    data.append(img)
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

trainY = to_categorical(trainY, 2)

# ===============================
# Build Model
# ===============================

def build_model():
    base_model = DenseNet121(weights="imagenet", include_top=False)

    inputs = Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = Conv2D(3, (3, 3), padding="same")(inputs)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = build_model()
model.summary()

# ===============================
# Train Model
# ===============================

history = model.fit(trainX, trainY, epochs=15, batch_size=32)

# ===============================
# Save Model
# ===============================

model.save("CNNMODEL.h5")
print("Model saved as CNNMODEL.h5")

# ===============================
# Evaluation
# ===============================

pred = model.predict(testX)
pred_classes = argmax(pred, axis=1)

print(classification_report(testY, pred_classes))
print(confusion_matrix(testY, pred_classes))

# ===============================
# Plot
# ===============================

plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["loss"], label="Loss")
plt.legend()
plt.show()