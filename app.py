import streamlit as st
from utils.helpers import predict

st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

st.title("🌱 Plant Disease Classification using ML")
st.markdown("Upload a plant leaf image and get its predicted disease category.")

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)
    with st.spinner('Predicting...'):
        label = predict(uploaded)
    st.success(f"**Prediction:** {label}")

