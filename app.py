import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import pandas as pd

# Load model and encoder
with open("plant_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Feature extractor
def extract_features(image):
    image = image.resize((128, 128))
    image = np.array(image)
    gray_image = rgb2gray(image)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

# App Title
st.title("🌿 Plant Disease Classification")

# Image upload
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    features = extract_features(image).reshape(1, -1)

    if features.shape[1] == model.n_features_in_:
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🔍 Predicted Disease: **{predicted_label}**")
    else:
        st.error("Feature length mismatch! Please ensure your uploaded image matches training preprocessing.")

# Display predictions for first 300 dataset images
st.subheader("📊 Predictions on Sample Dataset (First 300 Images)")

# Load dataset
data = pd.read_csv("data/train.csv")
images_path = "data/images/"
sample_data = data.head(300)

predicted_labels = []
actual_labels = []
sample_images = []

for idx, row in sample_data.iterrows():
    img_id = row['image_id']
    label = row['label']
    img_file = os.path.join(images_path, img_id + ".jpg")
    
    if os.path.exists(img_file):
        img = Image.open(img_file).convert("RGB")
        features = extract_features(img).reshape(1, -1)

        if features.shape[1] == model.n_features_in_:
            pred = model.predict(features)
            pred_label = label_encoder.inverse_transform(pred)[0]
        else:
            pred_label = "Error"

        predicted_labels.append(pred_label)
        actual_labels.append(label)
        sample_images.append(img)

# Show results
for i in range(len(sample_images)):
    st.image(sample_images[i], caption=f"Actual: {actual_labels[i]} | Predicted: {predicted_labels[i]}", use_column_width=True)
