import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import joblib
from PIL import Image
import torchvision.models as models

# Load PCA & SVM Model
pca = joblib.load("pca_model.pkl")
svm_model = joblib.load("svm_dr_model.pkl")

# Load ResNet50 model for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])  # Remove classification layer
resnet50.to(device)
resnet50.eval()

# Mapping Class Number to Name
class_names = {
    0: "No diabetic retinopathy",
    1: "Mild diabetic retinopathy",
    2: "Moderate diabetic retinopathy",
    3: "Severe diabetic retinopathy",
    4: "Proliferative diabetic retinopathy"
}

# Function to preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to extract features
def extract_features(image_tensor):
    with torch.no_grad():
        features = resnet50(image_tensor)
        features = features.view(features.size(0), -1)
    return features.cpu().numpy()

# Function to classify an image
def classify_image(image):
    image_tensor = preprocess_image(image)
    features = extract_features(image_tensor)
    features_pca = pca.transform(features)
    prediction = svm_model.predict(features_pca)[0]
    prediction_proba = svm_model.decision_function(features_pca)
    return class_names.get(prediction, "Unknown"), prediction_proba

# Streamlit UI setup
st.set_page_config(page_title="Diabetic Retinopathy Detector", page_icon="👁️", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #3c99dc;'>Diabetic Retinopathy Classification</h1>
    <p style='text-align: center;'>Upload an eye image and classify its type</p>
    """, unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["png", "jpg", "jpeg"])

# Sidebar Information
st.sidebar.header("📌 About This App")
st.sidebar.info(
    """
    This model classifies **Diabetic Retinopathy** based on retinal images.  
    It extracts features using **ResNet50**, selects important ones via **PCA**,  
    and classifies using **SVM (Support Vector Machine)**.  
    """
)

# Image Display and Classification
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # ✅ Smaller image display
    st.image(image, caption="📸 Uploaded Image", width=300, output_format="auto")

    if st.button("🔍 Classify Image"):
        with st.spinner("Analyzing Image... ⏳"):
            predicted_class, confidence_scores = classify_image(image)
            st.success(f"✅ Predicted Class: **{predicted_class}**")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center;'>Developed by <b>Manoj | Heeranand | Kunal</b></p>
    """, unsafe_allow_html=True
)

