import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from PIL import Image

# Load model and label encoder
model = joblib.load("plant_disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# HOG feature extractor
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), block_norm='L2-Hys')
        return features
    except:
        return None

# App UI
st.title("🌿 Plant Disease Classifier (Traditional ML)")
st.write("Upload an image or view predictions on 300 dataset images.")

# Upload image
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = image.convert("RGB")
    image = image.resize((128, 128))
    gray = np.array(image.convert("L"))
    features = hog(gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), block_norm='L2-Hys')
    features = np.array(features).reshape(1, -1)
    if features.shape[1] != 1764:
        st.error(f"Feature vector size mismatch. Got {features.shape[1]}, expected 1764.")
    else:
        prediction = model.predict(features)
        label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Disease: **{label}**")

# Load first 300 dataset images
st.subheader("📦 Auto Predictions on Dataset (First 300 Images)")

df = pd.read_csv("data/train.csv")
images = []
actual_labels = []
predicted_labels = []

count = 0
for _, row in df.iterrows():
    if count >= 300:
        break
    image_id = row['image_id']
    label = row['label']
    image_path = os.path.join("data/images", image_id + ".jpg")
    if not os.path.exists(image_path):
        continue
    features = extract_features(image_path)
    if features is None or len(features) != 1764:
        continue
    pred = model.predict([features])
    pred_label = label_encoder.inverse_transform(pred)[0]
    img = Image.open(image_path).resize((128, 128))
    images.append(img)
    actual_labels.append(label)
    predicted_labels.append(pred_label)
    count += 1

# Display predictions
for i in range(len(images)):
    st.image(images[i], caption=f"Actual: {actual_labels[i]} | Predicted: {predicted_labels[i]}", use_column_width=True)
