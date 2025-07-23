import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
import pickle

# Load model and label encoder using pickle
with open("plant_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def extract_hog_features(image_path):
    image = Image.open(image_path).resize((128, 128))
    gray = color.rgb2gray(np.array(image))
    features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return features

st.title("🌿 Plant Disease Classification")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_hog_features("temp_image.jpg").reshape(1, -1)

    if features.shape[1] != 1764:
        st.error(f"❌ Feature vector length mismatch: got {features.shape[1]}, expected 1764.")
    else:
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"✅ Predicted Disease: **{predicted_label}**")

st.write("---")
st.subheader("📊 Predictions on First 300 Dataset Images")

@st.cache_data
def load_dataset_predictions():
    df = pd.read_csv("data/train.csv").iloc[:300]
    images = []
    labels = []
    predictions = []
    for _, row in df.iterrows():
        image_id = row["image_id"]
        label = row["label"]
        img_path = os.path.join("data/images", image_id + ".jpg")
        if os.path.exists(img_path):
            try:
                features = extract_hog_features(img_path).reshape(1, -1)
                if features.shape[1] == 1764:
                    pred = model.predict(features)
                    pred_label = label_encoder.inverse_transform(pred)[0]
                    images.append(Image.open(img_path).resize((128, 128)))
                    labels.append(label_encoder.inverse_transform([label])[0])
                    predictions.append(pred_label)
            except:
                continue
    return images, labels, predictions

images, labels, predictions = load_dataset_predictions()

for i in range(len(images)):
    st.image(images[i], caption=f"Actual: {labels[i]} | Predicted: {predictions[i]}", use_column_width=True)
