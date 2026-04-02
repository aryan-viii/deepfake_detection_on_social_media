import os
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout
from gensim.models import FastText

# Configuration
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 50
EPOCHS = 3
BATCH_SIZE = 32

print("Downloading dataset...")
# Using fake_news dataset to quickly get human vs machine/fake text (using a small slice to save time)
# It has 'text' and 'label' (0 for fake, 1 for real)
try:
    dataset = load_dataset('GonzaloA/fake_news', split='train')
    dataset = dataset.shuffle(seed=42).select(range(4000)) # Shuffle BEFORE slicing to ensure balanced classes
    df = pd.DataFrame(dataset)
    # 0: Fake, 1: Real
    print("Class distribution:\n", df['label'].value_counts())
except Exception as e:
    print("Could not load HuggingFace dataset, creating a mock dataset for demonstration...")
    # Mock data fallback
    df = pd.DataFrame({
        'text': ["This is a genuine tweet from a human.", "Buy these cheap pills now! 100% real!", 
                 "I had a great time at the park today.", "AI generated text acts like this sometimes, very robotic.",
                 "Just setting up my twttr.", "Click this link to win a free iphone!!"] * 500,
        'label': [1, 0, 1, 0, 1, 0] * 500
    })

texts = df['text'].astype(str).tolist()
labels = df['label'].values

# Split training and validation
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

print("Preparing tokenizer...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print("Training FastText embeddings dynamically using gensim...")
# Tokenize sentences for gensim FastText
sentences = [text.split() for text in X_train]
ft_model = FastText(sentences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)

# Create embedding matrix
word_index = tokenizer.word_index
embedding_matrix = np.zeros((min(MAX_WORDS, len(word_index) + 1), EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= MAX_WORDS:
        continue
    if word in ft_model.wv:
        embedding_matrix[i] = ft_model.wv[word]

print("Building CNN + LSTM Model...")
model = Sequential()
model.add(Embedding(
    input_dim=min(MAX_WORDS, len(word_index) + 1),
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_LEN,
    trainable=False # Keep FastText embeddings fixed
))
# CNN Layer for feature extraction
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
# model.add(GlobalMaxPooling1D()) # We keep sequences for LSTM
# LSTM Layer for sequence analysis
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
# Output Layer
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Training model...")
model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("Saving model to model.h5...")
model.save('model.h5')
print("Training Complete!")
