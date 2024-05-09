import streamlit as st
import os
import joblib
from src.features_extraction import contrast_score, blur_score, brightness_score, sharpness_gradient_score
from src.image_classification import predict_real_fake

st.title("Face images classification based on Image Quality metrics")

st.markdown("""
<div style="text-align: center;">🎲 Let me tell you if your image is Real or Generated with Artificial intelligence 🤔</div>
""", unsafe_allow_html=True)

menu = ["Select image from the below list", "choose your own image"]
choice = st.sidebar.radio(label="Menu", options=["Select image from the below list", "choose your own image"])

if choice == "Select image from the below list":
    file = st.sidebar.selectbox("choose your image", os.listdir("Images"))
    uploaded_file = os.path.join(os.getcwd(), "Images", file)
else:
    uploaded_file = st.sidebar.file_uploader("Please upload an image:", type=['jpeg', 'jpg', 'png'])


# Loading model
model = joblib.load("rf_model.joblib")

def visualize_position(number, lower_bound, upper_bound):
    if number < lower_bound or number > upper_bound:
        st.error(f"Number {number} is outside the interval [{lower_bound}, {upper_bound}]")
        return
    position = (number - lower_bound) / (upper_bound - lower_bound)
    st.progress(position)


if uploaded_file is not None:
    # Calculate image quality metrics for the new image
    blur_score_val = blur_score(uploaded_file)
    brightness_score_val = brightness_score(uploaded_file)
    contrast_score_val = contrast_score(uploaded_file)
    sharpness_gradient_score_val = sharpness_gradient_score(uploaded_file)
    
    # Perform prediction for the new image
    class_label, prob_class_0, prob_class_1 = predict_real_fake(uploaded_file, model)

    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    st.write("**Classification:**", class_label)
    st.write("**Probability that the image is Fake:**", prob_class_0)
    st.write("**Probability that the image is Real:**", prob_class_1)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Blur score:**", blur_score_val)
    with col2:
        visualize_position(blur_score_val, 3.5, 48000.0)
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Brightness score:**", brightness_score_val)
    with col4:
        visualize_position(brightness_score_val, 0.13, 0.9)
    col5, col6 = st.columns(2)
    with col5:
        st.write("**Contrast score:**", contrast_score_val)
    with col6:
        visualize_position(contrast_score_val, 0.3, 1.0)
    col7, col8 = st.columns(2)
    with col7:
        st.write("**Sharpness Gradient score:**", sharpness_gradient_score_val)
    with col8:
        visualize_position(sharpness_gradient_score_val, 12.0, 204.0)
