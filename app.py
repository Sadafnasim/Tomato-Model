from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the model only once
model_path = 'tomato_disease_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)

# Class labels (Make sure the order matches the model output)
class_labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Helper function to preprocess image before prediction
def prepare_image(image):
    image = image.resize((224, 224))  # Resize to match the model's input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    if image.shape[-1] == 4:
        image = image[..., :3]  # Convert RGBA to RGB if necessary
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        image = Image.open(file.stream).convert('RGB')  # Open and convert image to RGB
        processed_image = prepare_image(image)  # Preprocess image
        prediction = model.predict(processed_image)  # Make prediction
        predicted_label = class_labels[np.argmax(prediction)]  # Get the label with the highest probability
        return jsonify({'prediction': predicted_label})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    # Start ngrok tunnel to make the app accessible from the web
    port = 5000
    public_url = ngrok.connect(port)
    print(" * ngrok tunnel running at:", public_url)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
