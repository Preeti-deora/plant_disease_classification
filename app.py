import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths
MODEL_PATH = "plant_disease_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "data/train.csv"
IMAGES_DIR = "data/images"

# Load model and encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# HOG feature extractor
def extract_hog_features_pil(img):
    img = img.resize((128, 128))
    img = np.array(img)
    gray = rgb2gray(img)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=None
    )
    return features

# Title
st.title("🌿 Plant Disease Classifier")

# Upload section
st.header("Upload a Leaf Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = extract_hog_features_pil(image).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Disease: **{predicted_label}**")

# Dataset preview section
st.header("📊 Predictions on 300 Sample Images")

@st.cache_resource
def load_dataset_predictions():
    df = pd.read_csv(CSV_PATH)
    images = []
    actual_labels = []
    predicted_labels = []

    for i, row in df.iterrows():
        if i >= 300:
            break
        image_id = row['image_id']
        label = row[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax()
        img_path = os.path.join(IMAGES_DIR, image_id + ".jpg")

        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                actual_labels.append(label)

                features = extract_hog_features_pil(img).reshape(1, -1)
                pred = model.predict(features)
                pred_label = label_encoder.inverse_transform(pred)[0]
                predicted_labels.append(pred_label)
            except:
                continue

    return images, actual_labels, predicted_labels

images, actual_labels, predicted_labels = load_dataset_predictions()

# Display images
st.subheader("Sample Image Predictions (First 300)")
for i in range(0, len(images), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(images):
            img = images[i + j]
            actual = actual_labels[i + j]
            predicted = predicted_labels[i + j]
            caption = f"Actual: {actual} | Predicted: {predicted}"
            cols[j].image(img, caption=caption, use_column_width=True)
