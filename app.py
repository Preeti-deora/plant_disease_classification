import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
from collections import Counter

# Paths
MODEL_PATH = "plant_disease_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "data/train.csv"
IMAGES_PATH = "data/images"

# Load model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

# Extract HOG features from image
def extract_hog_features(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image_gray = rgb2gray(image)
    features = hog(image_gray, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), channel_axis=None)
    return features

# Predict function
def predict(image, model, encoder):
    features = extract_hog_features(image)
    prediction = model.predict([features])[0]
    label = encoder.inverse_transform([prediction])[0]
    return label

# Load sample data (300 only)
@st.cache_data
def load_sample_data(n=300):
    df = pd.read_csv(CSV_PATH).head(n)
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGES_PATH, row['image_id'] + ".jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
            label = row[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax()
            labels.append(label)
    return images, labels

# Streamlit App
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a leaf image or explore 300 sample images below.")

# Load model
model, encoder = load_model()

# Image Upload
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label = predict(image, model, encoder)
    st.success(f"✅ Predicted Disease: **{label}**")
else:
    st.subheader("📂 Sample Predictions on 300 Images")
    images, labels = load_sample_data()


    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(images[i], caption=f"Actual: {labels[i]}", use_column_width=True)
            pred = predict(images[i], model, encoder)
            st.write(f"Predicted: **{pred}**")

 
    st.subheader("📊 Label Distribution (300 Images)")
    label_counts = Counter(labels)
    fig, ax = plt.subplots()
    ax.bar(label_counts.keys(), label_counts.values(), color='skyblue')
    ax.set_title("Disease Distribution")
    ax.set_ylabel("Count")
    st.pyplot(fig)
