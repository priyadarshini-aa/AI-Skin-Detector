import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.h5")

classes = ['Acne', 'Eczema', 'Psoriasis', 'Normal']

st.title("🧴 AI Skin Problem Detector")
st.write("Upload a skin image to detect the problem")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    st.success(f"Detected Skin Problem: **{result}**")
