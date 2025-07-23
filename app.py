import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
import joblib
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

st.set_page_config(layout="wide")

MODEL_PATH = "plant_disease_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "data/train.csv"
IMAGES_PATH = "data/images"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), channel_axis=None)
    return features

def load_data(csv_path, images_path, limit=300):
    df = pd.read_csv(csv_path)
    features = []
    labels = []
    images = []
    label_names = []

    for idx, row in tqdm(df.iterrows(), total=min(limit, len(df))):
        if idx >= limit:
            break
        image_id = row['image_id']
        label = row[row[1:].astype(bool)].idxmax()
        label_names.append(label)

        img_path = os.path.join(images_path, image_id + '.jpg')
        if os.path.exists(img_path):
            hog_feat = extract_hog_features(img_path)
            features.append(hog_feat)
            labels.append(label)
            images.append(img_path)

    features = np.array(features)
    labels_encoded = label_encoder.transform(labels)
    return features, labels_encoded, images, label_names

st.title("🌿 Plant Disease Classification (Traditional ML)")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image_resized = image.resize((128, 128))
    image_np = np.array(image_resized)
    features = hog(image_np, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), channel_axis=None).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.image(uploaded_file, caption=f"Predicted: {predicted_label}", use_column_width=True)
    st.success(f"Prediction: {predicted_label}")

st.markdown("---")
st.subheader("🔍 Predictions on first 300 dataset images")

features, labels, images, true_labels = load_data(CSV_PATH, IMAGES_PATH, limit=300)
predictions = model.predict(features)
predicted_labels = label_encoder.inverse_transform(predictions)

cols = st.columns(3)
for i in range(len(images)):
    with cols[i % 3]:
        st.image(images[i], caption=f"Actual: {true_labels[i]}\nPredicted: {predicted_labels[i]}", use_column_width=True)
