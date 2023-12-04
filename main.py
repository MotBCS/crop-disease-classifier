# Import essential libraries
import numpy
import streamlit as st
import numpy as np

from keras.models import load_model
from PIL import Image
from numpy.core.setup_common import file



from util import predict_img, set_background, disease

# Title
st.title('Crop Disease Classification')

# Header
st.header('Please upload a image to classify')

# Upload file (Types of images that will be accepted)
st.file_uploader('', type=['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG'])

# Load classifier
model = load_model('model/cropDisease_classifier.h5')

# Load class names
with open('model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

class_names = ['Healthy Tomato', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Mosaic Virus', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Yellow Leaf Curl']
print(class_names)

# Display image
if file is not None:
    image = Image.open(file).convert('RBG')
    st.image(image, use_column_width=True)
# Classify image
class_name, confidence_score = predict_img(image, model, class_names)

# Get Disease Information
disease_Text = disease(class_names)

# Write classification
st.write("## {}".format(class_name))
st.write("### score: {}%".format(int(confidence_score * 1000) / 10))
st.write("## {}".format(disease_Text))
