import streamlit as st
import os
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
import pandas as pd

# ----------------------------- FEATURE EXTRACTION -----------------------------
def extract_features(image):
    image = image.resize((128, 128))
    image_np = np.array(image)
    gray_image = rgb2gray(image_np)

    # HOG features
    hog_features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    # Color histogram features (32 bins per channel)
    hist_features = []
    for i in range(3):
        hist, _ = np.histogram(image_np[:, :, i], bins=32, range=(0, 256))
        hist_features.extend(hist)

    return np.hstack([hog_features, hist_features])

# ----------------------------- MODEL LOADING -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("plant_disease_model.joblib")
    le = joblib.load("label_encoder.joblib")
    return model, le

model, le = load_model()

# ----------------------------- PAGE SETUP -----------------------------
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image to classify the plant disease.")

# ----------------------------- SINGLE IMAGE UPLOAD -----------------------------
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = extract_features(image).reshape(1, -1)

    if features.shape[1] == model.n_features_in_:
        prediction = model.predict(features)
        predicted_label = le.inverse_transform(prediction)[0]
        st.success(f"**Predicted Disease:** {predicted_label}")
    else:
        st.error(f"Feature size mismatch! Model expects {model.n_features_in_} features but got {features.shape[1]}.")

# ----------------------------- AUTO DISPLAY 300 DATASET IMAGES -----------------------------
st.header("📦 Predictions on First 300 Dataset Images")

@st.cache_data
def load_csv_and_predict():
    df = pd.read_csv("data/train.csv")
    df = df.head(300)

    records = []
    for idx, row in df.iterrows():
        image_id = row["image_id"]
        true_label = row.drop("image_id").idxmax()

        image_path = os.path.join("data", "images", image_id + ".jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                features = extract_features(img).reshape(1, -1)

                if features.shape[1] != model.n_features_in_:
                    pred_label = "❌ Feature size mismatch"
                else:
                    pred = model.predict(features)
                    pred_label = le.inverse_transform(pred)[0]

                records.append((image_id, true_label, pred_label, image_path))
            except:
                records.append((image_id, true_label, "❌ Error loading", ""))
        else:
            records.append((image_id, true_label, "❌ Image Not Found", ""))

    return records

records = load_csv_and_predict()

for image_id, true_label, pred_label, img_path in records:
    cols = st.columns([1, 2])
    if os.path.exists(img_path):
        cols[0].image(img_path, width=120)
    cols[1].markdown(f"**Image ID:** {image_id}")
    cols[1].markdown(f"✅ **True Label:** {true_label}")
    cols[1].markdown(f"🔍 **Predicted:** {pred_label}")
    cols[1].markdown("---")
