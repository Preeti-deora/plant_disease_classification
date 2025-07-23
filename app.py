import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from tqdm import tqdm

# Load model and label encoder
with open("plant_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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

def predict_image(image):
    features = extract_hog_features_pil(image).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

st.title("🌿 Plant Disease Classifier")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    predicted_label = predict_image(image)
    st.success(f"Predicted Disease: {predicted_label}")

st.markdown("---")
st.header("🔍 Dataset Predictions (First 300 Images)")

@st.cache_resource
def load_and_predict_dataset():
    df = pd.read_csv("data/train.csv")
    IMAGES_PATH = "data/images"
    image_ids = df["image_id"].tolist()[:300]
    labels = df.drop("image_id", axis=1).idxmax(axis=1).tolist()[:300]

    predictions = []
    images = []
    actuals = []

    for i, image_id in tqdm(enumerate(image_ids), total=300):
        img_path = os.path.join(IMAGES_PATH, image_id + ".jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path)
            try:
                pred = predict_image(image)
                predictions.append(pred)
                images.append(image)
                actuals.append(labels[i])
            except:
                continue

    return images, predictions, actuals

images, predictions, actuals = load_and_predict_dataset()

for i in range(len(images)):
    st.image(images[i], caption=f"Actual: {actuals[i]} | Predicted: {predictions[i]}", use_column_width=True)
