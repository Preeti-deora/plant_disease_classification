import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

# -------------------------
# 🚀 Load Model and Label Encoder
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("plant_disease_model.joblib")
    le = joblib.load("label_encoder.joblib")
    return model, le

model, le = load_model()

# -------------------------
# 📷 Feature Extraction (must match training!)
# -------------------------
def extract_features(img):
    img = img.resize((128, 128)).convert("RGB")
    gray_img = img.convert("L")
    feature, _ = hog(np.array(gray_img), 
                     orientations=9, 
                     pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2),
                     block_norm='L2-Hys',
                     visualize=True)
    return feature

# -------------------------
# 🔍 Predict Image
# -------------------------
def predict_disease(img):
    try:
        feature = extract_features(img)
        feature = feature.reshape(1, -1)

        if feature.shape[1] != model.n_features_in_:
            return "❌ Feature size mismatch!", None

        pred = model.predict(feature)
        pred_label = le.inverse_transform(pred)[0]
        return f"✅ Predicted: {pred_label}", pred_label
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# -------------------------
# 🌿 App UI
# -------------------------
st.title("🌿 Plant Disease Classification App")

# Upload Image Section
st.header("📤 Upload an Image")
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result, _ = predict_disease(image)
    st.markdown(f"### 🔎 {result}")

# -------------------------
# 📦 Load 300 Dataset Images (Prediction Showcase)
# -------------------------
st.header("🖼️ Dataset Predictions (First 300 Images)")
DATASET_DIR = "data/images/"
CSV_PATH = "data/train.csv"

if os.path.exists(CSV_PATH):
    import pandas as pd
    df = pd.read_csv(CSV_PATH)

    image_ids = df["image_id"].values[:300]
    actual_labels = df.drop("image_id", axis=1).idxmax(axis=1).values[:300]

    cols = st.columns(3)
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(DATASET_DIR, f"{image_id}.jpg")
        try:
            img = Image.open(img_path)
            result, pred_label = predict_disease(img)
            with cols[i % 3]:
                st.image(img, width=180, caption=f"Actual: {actual_labels[i]}\nPredicted: {pred_label}")
        except:
            with cols[i % 3]:
                st.error(f"Image not found: {image_id}.jpg")
else:
    st.warning("train.csv not found. Please ensure dataset is uploaded.")
