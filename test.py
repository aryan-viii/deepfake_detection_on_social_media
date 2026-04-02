import sys
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset

print("Loading model and tokenizer...")
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("Loading 10 samples to test...")
dataset = load_dataset('GonzaloA/fake_news', split='train').shuffle(seed=42).select(range(4000, 4010))

for item in dataset:
    text = item['text']
    label = item['label']
    
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    
    pred = model.predict(padded, verbose=0)[0][0]
    print(f"True Label: {label} | Predicted Score: {pred:.4f} | {'REAL' if label == 1 else 'FAKE'}")

