from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Configuration
MAX_LEN = 100

# Load model and tokenizer
MODEL_PATH = 'model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'

print("Loading model and tokenizer...")
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Model loaded successfully!")
else:
    print("Warning: Model or tokenizer not found. Please run train.py first.")
    model = None
    tokenizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tokenizer:
        return jsonify({'error': 'Model not trained yet!'}), 500
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    text = data['text']
    
    # Preprocessing
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Prediction
    prediction = model.predict(padded)
    # Output is sigmoid, so > 0.5 is Class 1 (Real), <= 0.5 is Class 0 (Fake/Machine)
    # Note: Depending on the dataset, 0 is fake, 1 is real.
    score = float(prediction[0][0])
    
    is_fake = score < 0.5
    confidence = (1 - score) if is_fake else score
    
    return jsonify({
        'prediction': 'Machine-Generated / Fake' if is_fake else 'Human / Real',
        'confidence': f"{confidence * 100:.2f}%",
        'raw_score': score
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
