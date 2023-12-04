import base64
import streamlit as st
import numpy as np

# Sets the background of the streamlit application to an image
from PIL import ImageOps, Image


"""
Parameters:
    image_file (str): The path to the image file to be used for the background
    
Returns:
    None
"""
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp{{
            background-image: url(data:;base64, {b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

#  predict_img function
"""
This function takes an image, a model, and a list of class names and returns the predicted class
and confidence score of the image.

Parameter:
    image (PIL.Image.Image): AN image to be classified
    model (tensorflow.keras.Model): A trained machine learning model for image classification.
    class_names (list): A list of class names corresponding to the classes that the model can predict.
    
Returns:
    A tuple of the predicted class name and the confidence score for that specific prediction.
"""
def predict_img(image, model, class_names):

    # Converts uploaded image to (224, 224)
    image = ImageOps.fit(image, (244, 244), Image.Resampling.LANCZOS)

    # Converts image to numpy array
    image_array = np.asarray(image)

    # Normalize Image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)

    # Index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# Return disease information and treatment
def disease(class_name):
    if class_name == "Healthy Tomato":
        return "This leaf looks healthy, healthy tomato leaves should be soft, medium to dark shades of green and display a solid stem. (No treatment needed)"
    elif class_name == "Tomato Early Blight":
        return "Test 1..."
    elif class_name == "Tomato Late Blight":
        return "Test 2..."
    elif class_name == "Tomato Mosaic Virus":
        return "Test 3..."
    elif class_name == "Tomato Leaf Mold":
        return "Test 4..."
    elif class_name == "Tomato Septoria Leaf Spot":
        return "Test 5..."
    elif class_name == "Tomato Yellow Leaf Curl":
        return "Test 6..."

#print(disease("Healthy Tomato"))