import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

st.title("🌿 Plant Disease Classification using Traditional ML")
st.markdown("Upload a leaf image to predict the type of disease.")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = joblib.load("plant_disease_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, label_encoder = load_model()

# ------------------ FEATURE EXTRACTION ------------------
def extract_features(image):
    # Resize image
    image = image.resize((128, 128))
    img_array = np.array(image)

    # HOG features
    gray = rgb2gray(img_array)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=None
    )

    # Color histogram
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        hist_r, _ = np.histogram(img_array[:, :, 0], bins=32, range=(0, 256))
        hist_g, _ = np.histogram(img_array[:, :, 1], bins=32, range=(0, 256))
        hist_b, _ = np.histogram(img_array[:, :, 2], bins=32, range=(0, 256))
        color_hist = np.concatenate([hist_r, hist_g, hist_b])
    else:
        color_hist = np.zeros(96)

    # Final feature vector
    return np.concatenate([hog_features, color_hist])

# ------------------ UPLOAD AND PREDICT ------------------
uploaded_file = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.markdown("### 🔍 Extracting Features & Making Prediction...")
    features = extract_features(img).reshape(1, -1)

    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🩺 Predicted Disease: **{predicted_label}**")

# ------------------ SAMPLE IMAGES (OPTIONAL) ------------------
st.markdown("---")
st.markdown("### 🔎 Sample Predictions from Dataset")

try:
    train_df = pd.read_csv("data/train.csv")
    IMAGES_PATH = "data/images/"

    images = []
    labels = []
    predicted_labels = []

    for idx, row in train_df.iterrows():
        if len(images) >= 5:
            break

        image_id = row["image_id"]
        label = label_encoder.inverse_transform([np.argmax(row[1:].values)])[0]
        img_path = os.path.join(IMAGES_PATH, image_id + ".jpg")

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            feat = extract_features(img).reshape(1, -1)
            pred = model.predict(feat)
            pred_label = label_encoder.inverse_transform(pred)[0]

            images.append(img)
            labels.append(label)
            predicted_labels.append(pred_label)

    num_to_show = min(len(images), len(labels), len(predicted_labels))

    for i in range(num_to_show):
        st.image(images[i], caption=f"Actual: {labels[i]}", use_column_width=True)
        st.write(f"Prediction: **{predicted_labels[i]}**")
        st.markdown("---")

except Exception as e:
    st.error("Error loading sample images for display.")
    st.exception(e)
