import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
# Path absoluto hard-coded (fix para working dir /backend)
MODEL_PATH = '/home/suario/proyectos/ecommerce-image-classifier/model/saved_model/ecommerce_classifier.keras'

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_model():
    print(f"Cargando modelo desde: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    model = load_model()
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence
