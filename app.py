from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('model_tf')

# Define classes for your predictions
CLASSES = ['ANTHRACNOSE', 'DOWNEY MILDEW', 'GOOD LEAF', 'LEAF MINER', 'SLUG INFECTED'] # Replace with your actual classes

def display_random_image(class_names, test_data, pred_labels):
    # Select a random index
    index = random.randint(0, len(test_data) - 1)
    # Get the image and its predicted label
    image = test_data[index]
    predicted_class_index = pred_labels[index]
    predicted_class = class_names[predicted_class_index]
    return image, predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image found", 400

    image = request.files['image']
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize the image to match the input shape of your model
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASSES[predicted_class_index]

    # Generate a random image and its predicted label
    random_image, random_predicted_class = display_random_image(CLASSES, test_data, pred_labels)

    return render_template('index.html', prediction=predicted_class, random_image=random_image, random_predicted_class=random_predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
