import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

#Load Models
@st.cache_resource
def load_yolo():
    return YOLO("yolov8x-seg.pt")  # Load YOLOv8 segmentation model

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

yolo_model = load_yolo()
blip_processor, blip_model = load_blip()

#Streamlit UI
st.set_page_config(page_title="YOLOv8 + BLIP", layout="wide")
st.title("Image Segmentation & Image Captioning - 18th group")
st.write("Upload an image to **detect & segment objects** and generate a **caption** describing the scene.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    #YOLO Object Detection
    with st.spinner("Running YOLOv8 segmentation..."):
        results = yolo_model(image)
        segmented_image = results[0].plot()  # Annotated image (numpy array)
        object_names = list(set(results[0].names[int(cls)] for cls in results[0].boxes.cls))

    #BLIP Captioning
    with st.spinner("Generating caption using BLIP..."):
        inputs = blip_processor(image, return_tensors="pt")
        with torch.no_grad():
            caption_ids = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

    # ------------------ Display Results ------------------ #
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Objects :")
        st.write(", ".join(object_names))
        st.subheader("Image Caption :")
        st.success(caption)
    with col2:
        st.subheader("Segmentation Result :")
        st.image(segmented_image, use_container_width=True)

