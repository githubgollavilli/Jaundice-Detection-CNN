# ===============================
# Jaundice Detection - Frontend
# Streamlit App
# ===============================

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from numpy import argmax
from PIL import Image as PILImage

# ===============================
# Load Model
# ===============================

model = load_model("CNNMODEL.h5")
categories = ["jaundice", "normal"]

# ===============================
# Load Haar Cascades
# ===============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# Preprocess Image
# ===============================

def preprocess_image(img, target_size=(65, 65)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# Select Forehead ROI
# ===============================

def select_forehead_roi(image, draw_roi=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, image

    for (x, y, w, h) in faces:
        roi = image[y:y + h // 3, x:x + w]
        image_with_roi = image.copy()

        if draw_roi:
            cv2.rectangle(
                image_with_roi,
                (x, y),
                (x + w, y + h // 3),
                (0, 255, 0),
                2
            )

        return roi, image_with_roi

    return None, image

# ===============================
# Streamlit UI
# ===============================

st.title("Jaundice Detection using CNN")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:
    img = PILImage.open(uploaded_file)
    img = np.array(img)

    roi, img_with_roi = select_forehead_roi(img, draw_roi=True)

    if roi is not None:
        processed = preprocess_image(roi)
        prediction = model.predict(processed)
        predicted_class = categories[argmax(prediction)]

        st.image(img_with_roi, caption="Detected Forehead ROI", channels="BGR")
        st.success(f"Prediction: {predicted_class}")
    else:
        st.error("No face detected.")