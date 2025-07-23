import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------- Load Model and Label Encoder ----------
@st.cache_resource
def load_model():
    model = joblib.load("plant_disease_model.joblib")
    le = joblib.load("label_encoder.joblib")
    return model, le

# ---------- Feature Extraction ----------
def extract_features(image):
    image = image.resize((128, 128)).convert('RGB')
    image_np = np.array(image)

    # Color Histogram
    hist_features = []
    for i in range(3):  # R, G, B
        hist = np.histogram(image_np[:, :, i], bins=32, range=(0, 256))[0]
        hist_features.extend(hist)

    # HOG Features
    gray = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140])
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    return np.hstack([hist_features, hog_features])

# ---------- Load and Predict on Dataset Images ----------
@st.cache_data
def load_dataset():
    df = pd.read_csv("data/train.csv")
    df["label"] = df[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(axis=1)
    images = []
    labels = []
    features = []

    for i, row in df.head(300).iterrows():
        img_path = os.path.join("data/images", row["image_id"] + ".jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path)
            feature = extract_features(image)
            images.append(image)
            features.append(feature)
            labels.append(row["label"])

    model, le = load_model()
    X = np.array(features)
    y_true = labels
    y_pred = le.inverse_transform(model.predict(X))

    return images, y_true, y_pred

# ---------- Streamlit UI ----------
st.title("🌿 Plant Disease Classification App")

# Section 1: Upload and Predict
st.header("📷 Upload a Leaf Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Extracting features and making prediction..."):
        feature = extract_features(image).reshape(1, -1)
        model, le = load_model()
        prediction = model.predict(feature)
        label = le.inverse_transform(prediction)[0]

    st.success(f"✅ Predicted Disease: **{label}**")

# Section 2: Dataset Predictions
st.header("🖼️ Predictions on First 300 Dataset Images")

if st.button("Show Predictions on Dataset"):
    with st.spinner("Loading dataset and predicting..."):
        images, labels, preds = load_dataset()

    for i in range(len(images)):
        st.image(images[i], caption=f"Actual: {labels[i]} | Predicted: {preds[i]}", width=200)
