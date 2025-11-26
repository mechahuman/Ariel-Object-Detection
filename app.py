import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


IMG_SIZE = 224

CLASS_NAMES = ['Bird','Drone']

@st.cache_resource
def load_tl_model():

    model = load_model('best_tl_model.h5')
    return model

model = load_tl_model()


def preprocess_image(image: Image.Image, img_size: int = IMG_SIZE):

    img_resized = image.resize((img_size, img_size))

    img_array = np.array(img_resized)

    img_batch = np.expand_dims(img_array, axis=0)

    img_prepocessed = preprocess_input(img_batch)

    return img_prepocessed


st.title("Ariel Object Detection")
st.write(
    """
    This application uses Transfer Learning to classify images as either 'Bird' or 'Drone'.
    The model is built on top of EfficientNetB0 architecture and has been fine-tuned for this specific task.
    """
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_preprocessed = preprocess_image(image)

    preds = model.predict(img_preprocessed)
    prob_drone = float(preds[0][0])
    

    if prob_drone >= 0.5:
        predicted_class = "Drone"
        confidence = prob_drone * 100

    else:
        predicted_class = "Bird"
        confidence = (1 - prob_drone) * 100

    if st.button("Classify Image"):
        st.success(predicted_class)
        st.success(f"Confidence: {confidence:.2f}%")

else:

    st.info("Please upload an image file to classify.")


