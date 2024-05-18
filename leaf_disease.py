

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model_path = 'xleaf_model2.h5'  # Update this path if your model file is located elsewhere
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['Healthy', 'Diseased']  # Update with your actual class names

# Check the model's expected input shape
input_shape = model.input_shape[1:]  # Exclude the batch size, typically (224, 224, 3)

# Define a function to preprocess the image and predict disease
def preprocess_image(image, target_size):
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if your model expects it
    return img_array

def predict_disease(image, threshold=0.1):
    preprocessed_image = preprocess_image(image, input_shape[:2])
    
    # If the model expects a flattened input, flatten the image array
    if len(input_shape) == 1:
        preprocessed_image = preprocessed_image.reshape((1, -1))
    
    predictions = model.predict(preprocessed_image)
    
    # Extract the probability of being "Healthy"
    prob_healthy = predictions[0][0]
    
    # Classify based on the threshold probability
    predicted_class = 'Healthy' if prob_healthy >= threshold else 'Diseased'
    
    return predicted_class, prob_healthy


# Streamlit app
st.title("Leaf Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    with st.spinner('Wait for it...'):
        try:
            predicted_class, prob_healthy = predict_disease(uploaded_file)
            if predicted_class == 'Healthy':
                st.markdown(f"<h2 style='color: green;'>Predicted Class: {predicted_class}</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: red;'>Predicted Class: {predicted_class}</h2>", unsafe_allow_html=True)
            
           
        except Exception as e:
            st.error(f"Error during prediction: {e}")


