import cv2
import joblib
import numpy as np
import torch
from src.features_extraction import contrast_score, blur_score, brightness_score, sharpness_gradient_score, calculate_ilniqe_score, calculate_maniqa_score, calculate_niqe_score, calculate_topiq_face_score
import pandas as pd


def extract_features1(image):
    # Extract features using functions from image_processing module
    sharpness_score_res = sharpness_gradient_score(image)
    blur_score_res = blur_score(image)
    contrast_score_res = contrast_score(image)
    brightness_score_res = brightness_score(image)
    
    return  sharpness_score_res, blur_score_res, contrast_score_res, brightness_score_res

def extract_features2(image):
    maniqa_score = calculate_maniqa_score(image)
    niqe_score = calculate_niqe_score(image)
    ilniqe_score = calculate_ilniqe_score(image)
    topiq_face_score = calculate_topiq_face_score(image)
    
    # Return all the extracted features
    return niqe_score, maniqa_score, ilniqe_score, topiq_face_score

model = joblib.load("rf_model.joblib")
input_df = None
features1 = None
features2 = None

# Function to predict whether the image is real or fake
def predict_real_fake(image_file, model):
    # Read the image
    if isinstance(image_file, str):
        image = cv2.imread(image_file)
    else:
        image = cv2.imread('src/Temp_dir/img0.png')

    features1 = extract_features1(image)

    # Convert the image to a PyTorch tensor
    image_tensor = torch.tensor(image, dtype=torch.uint8).clone().detach().cuda()

    # Clone and detach the tensor to avoid gradients
    image_tensor = (image_tensor.clone().detach()).cpu()

    # Extract features from the image
    features2 = extract_features2(image_tensor)
    features = list(features1 + features2)

    input_df = pd.DataFrame({
        'NIQE': features[4],
        'MANIQA': features[5],
        'IL-NIQE': features[6],
        'TOPIQ-NR_FACE': features[7],
        'Sharpness': features[0],
        'Blur': features[1],
        'Contrast': features[2],
        'Brightness': features[3]
    }, index=[0])
    print(input_df)
    # Perform prediction using the trained model
    class_probabilities = model.predict_proba(input_df)[0]
    predicted_class = model.predict(input_df)[0]

    # Map the prediction to the corresponding class label (real or fake)
    class_label = "Real image" if predicted_class == 1 else "AI-Generated image"

    return class_label, class_probabilities[0], class_probabilities[1]
