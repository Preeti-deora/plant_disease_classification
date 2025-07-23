import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

def extract_features(image):
    image = image.resize((128, 128))
    image = np.array(image.convert("L"))  # grayscale
    features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return features

@st.cache_data
def load_model():
    model = joblib.load("plant_disease_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

@st.cache_data
def load_dataset():
    df = pd.read_csv("data/train.csv")
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join("data/images", row["image_id"])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
            label = None
            for col in ['healthy', 'multiple_diseases', 'rust', 'scab']:
                if row[col] == 1:
                    label = col
                    break
            labels.append(label)

        if len(images) >= 300:
            break

    return images, labels

st.title("🌿 Plant Disease Classification")

model, le = load_model()

images, labels = load_dataset()
features = [extract_features(img) for img in images]
X = np.array(features)
y_pred = model.predict(X)

st.header("🔍 Predictions on First 300 Images")
for i in range(len(images)):
    st.image(images[i], caption=f"Actual: {labels[i]} | Predicted: {le.inverse_transform([y_pred[i]])[0]}", use_column_width=True)

st.header("📸 Predict on a New Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = extract_features(image).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = le.inverse_transform(prediction)[0]
    st.success(f"✅ Predicted Disease: **{predicted_label}**")
