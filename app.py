import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import cv2
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "plant_disease_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

def extract_features(image):
    image_resized = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=None)
    hist_b = cv2.calcHist([image_resized], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([image_resized], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([image_resized], [2], None, [32], [0, 256]).flatten()
    color_hist = np.concatenate([hist_b, hist_g, hist_r])
    return np.concatenate([hog_features, color_hist])

st.title("🌿 Plant Disease Classification App")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    features = extract_features(image).reshape(1, -1)
    if features.shape[1] != model.n_features_in_:
        st.error(f"Feature length mismatch. Expected {model.n_features_in_}, got {features.shape[1]}")
    else:
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Predicted: {predicted_label}", use_column_width=True)

st.header("🔍 Predictions on 300 Default Images")

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")

df = pd.read_csv(CSV_PATH)
df = df.head(300)

images = []
actual_labels = []
predicted_labels = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMAGES_DIR, row["image_id"] + ".jpg")
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        features = extract_features(image).reshape(1, -1)
        if features.shape[1] == model.n_features_in_:
            prediction = model.predict(features)
            pred_label = label_encoder.inverse_transform(prediction)[0]
            predicted_labels.append(pred_label)
            actual_labels.append(row["label"])
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

for i in range(len(images)):
    st.image(images[i], caption=f"Actual: {actual_labels[i]} | Predicted: {predicted_labels[i]}", use_column_width=True)
