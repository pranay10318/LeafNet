import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model("crop_disease_model.h5")
labels = {0: 'Tomato-Bacterial_spot', 2: 'Corn-Common_rust', 1: 'Potato-Early_blight'}

def preprocess_image(img):
    # Resize and convert image to RGB
    img = img.resize((256, 256)).convert('RGB')
    
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(-1, 256, 256, 3)
    
    return img_array

# Set page title, favicon and layout
st.set_page_config(page_title="Crop Disease Prediction", page_icon="🌽", layout="wide")

# Title and text
st.title("Crop Disease Prediction")
st.markdown("Upload an image of a crop leaf, and this app will predict the disease affecting the crop.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app uses a deep learning model trained on images of crop leaves to predict the disease affecting the crop.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Resize image
    max_size = (400,400)
    img.thumbnail(max_size)
    
    # Display image
    st.image(img, caption='Uploaded Image.')
    
    # Provide a link to view the image in a new tab
    if st.button('Open Image in New Tab'):
        tmp_download_link = st.download_button(label="Download Image", data=uploaded_file, file_name='uploaded_image.jpg', mime='image/jpeg')
        st.markdown(f'<a href="{tmp_download_link}" target="_blank">Open Image in New Tab</a>', unsafe_allow_html=True)
    
    st.write("")

    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]
    st.success("Prediction: " + predicted_label)