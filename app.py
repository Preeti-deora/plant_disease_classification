import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import cv2
from skimage.feature import hog
from PIL import Image

st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# Load model and encoder (no pickle used)
model = joblib.load("plant_disease_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

def extract_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=-1
    )
    return features

st.title("🌿 Plant Disease Classification App")

# Upload section
st.sidebar.header("Upload Leaf Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = extract_features(image_np).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"✅ Predicted Disease: **{predicted_label}**")

st.markdown("---")
st.header("🔍 Prediction on Sample Dataset (First 300 Images)")

@st.cache_resource
def load_dataset():
    df = pd.read_csv("data/train.csv")[:300]
    images, labels, preds = [], [], []
    for _, row in df.iterrows():
        img_path = os.path.join("data/images", row["image"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feat = extract_features(img).reshape(1, -1)
            pred = model.predict(feat)
            pred_label = label_encoder.inverse_transform(pred)[0]
            images.append(img_rgb)
            labels.append(row["label"])
            preds.append(pred_label)
    return images, labels, preds

images, labels, preds = load_dataset()

cols = st.columns(3)
for i in range(len(images)):
    with cols[i % 3]:
        st.image(images[i], caption=f"Actual: {labels[i]}\nPredicted: {preds[i]}", use_column_width=True)
