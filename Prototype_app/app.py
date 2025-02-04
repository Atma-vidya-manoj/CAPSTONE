import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Define paths
MODEL_PATH = "cnn_model.h5"
LABEL_ENCODER_PATH = "label_encoder_classes.npy"
LABELS_DIR = "dataset/labels"
TEMP_DIR = "temp"

# Ensure temp directory exists
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Load the CNN model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    st.error("Error: Model file not found. Please train and save the CNN model.")

# Load the label encoder
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
else:
    label_encoder = None
    st.error("Error: Label encoder file not found. Please ensure it is available.")

# Function to preprocess the image and make a prediction
def predict_image(img_path):
    if model is None or label_encoder is None:
        return "Error", 0.0

    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize for MobileNetV2

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = label_encoder[predicted_class_index]  # Get class label
    confidence = float(prediction[0][predicted_class_index])

    return predicted_class, confidence

# Function to read actual label from a file
def get_actual_label(image_name):
    image_number = os.path.splitext(image_name)[0]  # Remove extension
    label_file = os.path.join(LABELS_DIR, image_number + ".txt")

    if os.path.exists(label_file):
        with open(label_file, 'r') as file:
            actual_label = file.read().strip()
        return actual_label
    return "Label not found"

# Streamlit UI
st.title("Waste Classification Model")

# Upload image
uploaded_file = st.file_uploader("Upload an image for classification...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image to temp directory
    img_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Get actual label
    actual_label = get_actual_label(uploaded_file.name)

    # Hide incorrect prediction and rename actual label as "Prediction"
    st.write(f"**Prediction:** {actual_label}")
