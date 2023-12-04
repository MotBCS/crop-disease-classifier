import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow import keras
from PIL import Image

model = keras.models.load_model('model/cropDisease_classifier.h5')
labels = ['Healthy Tomato', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Mosaic Virus', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Yellow Leaf Curl']
print(labels)

# Style
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Streamlit App
st.title('Crop Disease Classifier')
uploaded_image = st.file_uploader('Please upload an image to classify', type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])

# Image Classification
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

def disease_text(label):
    if label == "Healthy Tomato":
        return "This leaf looks healthy! Healthy tomato leaves should be soft, medium to dark shades of green and display a solid stem. \n\nNo treatment need." \
               "\n\nThis leaf is : HEALTHY and ABIOTIC"
    elif label == "Tomato Early Blight":
        return "Early Blight is a fungal disease, that affects tomatoes through out North America. Tomatoes affected with Early Blight fungus can display rust on the stems, infected older leafs and cracks on the fruit." \
               "\n\nTreatment: Provide adequate spacing and air circulation, apply fungicide or an organic fungicide (fixed copper) early in the season." \
               "\n\nThis leaf is : DISEASED and BIOTIC"
    elif label == "Tomato Late Blight":
        return "Late Blight is a destructive disease that may be found on any above ground part of the tomato plant. When the weather is very humid and wet, late blight infections can appear water-soaked or dark brown in color, and are often described as appearing greasy." \
               "\n\nTreatment: Similar to Early Blight treatment, provide adequate spacing and air circulation, apply fungicide or an organic fungicide (fixed copper) early in the season." \
               "\n\nThis leaf is : DISEASED and BIOTIC"
    elif label == "Tomato Mosaic Virus":
        return "Tomato Mosaic Virus is a plant pathogenic virus. The foliage of the affected tomato plant shows mottling, with alternating yellow and dark green leaves, these leaves often appear thicker and raised giving a blistering appearance." \
               "\n\nTreatment: Remove and destroy infected plants immediately to reduce spread. Keep your garden weed free, plant disease resistant varieties of tomatoes." \
               "\n\nThis leaf is : DISEASED and BIOTIC"
    elif label == "Tomato Leaf Mold":
        return "Leaf Mold is caused by a fungus called 'Passalora Fulva'. It appears on older leaves near the soil where air circulation is poor and humidity is high." \
               "\n\nTreatment: Provide adequate spacing and air circulation, avoid wetting leaves when watering, use a fungicide with chlorothalonil, mancozeb, or copper fungicide." \
               "\n\nThis leaf is : DISEASED and BIOTIC"
    elif label == "Tomato Septoria Leaf Spot":
        return "The Septoria Leaf Spot is a destructive disease to tomato plants. This disease is caused by the fungus known as 'Septoria Lycopersici'. Infection starts on the lower leaves near the ground after the plant begins to produce fruit." \
               "\n\nTreatment: Removal of crop debris, and the repeated use of fungicide with chlorothalonil, mancozeb, or copper fungicide can reduce the chance of the disease appearing or spreading." \
               "\n\nThis leaf is : DISEASED and BIOTIC"
    elif label == "Tomato Yellow Leaf Curl":
        return "Tomato Yellow Leaf Curl is a virus transmitted by Whiteflies. This disease can be extremely damaging to the tomato plant, Whiteflies may bring the disease into the garden from infected weeds nearby." \
               "\n\nTreatment: Remove plant with initial symptoms of the disease, keep weeds controlled around the garden area" \
               "\n\nThis leaf is : DISEASED and BIOTIC"



if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Image has been analyzed, results below.', use_column_width=True)

    # Preprocess the image
    image = preprocess_image(image)

    prediction = model.predict(np.array([image]))
    label = labels[np.argmax(prediction)]
    disease = disease_text(label)

    # Get accuracy percentage of prediction
    accuracy = np.max(prediction)

    st.write(f'Prediction: {label}')
    st.write(f'Accuracy: {accuracy:.2%}')
    st.write(f'Disease Description: {disease}')


    st.title("Training and Validation Accuracy")
    st.image("visuals/TrainVal_Accuracy.png", caption="Visual displaying training and validation accuracy", use_column_width=True, width=100)

    st.title("Training and Validation Loss")
    st.image("visuals/TrainVal_Loss.png", caption="Visual displaying training and validation loss", use_column_width=True, width=100)

    st.title("Confusion Matrix")
    st.image("visuals/confusionMatrix.png", caption="Confusion/Error matrix", use_column_width=True, width=100)
    st.write("Represents the performance of the model on the plant disease test data.")

