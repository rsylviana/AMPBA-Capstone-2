import streamlit as st
import cv2
import torch

# Load YOLOv8 model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('yolo-model.pt', 'model', pretrained=True)
    return model

model = load_model()

# Streamlit app
st.title("YOLOv8 Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = cv2.imread(uploaded_file.name)[:, :, ::-1]  # BGR to RGB
    results = model(image)
    
    # Display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detected Objects:")
    st.json(results.xyxy[0].numpy().tolist())

