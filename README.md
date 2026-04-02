# Minimal Deepfake Text Detector

A lightweight, beginner-friendly deepfake text detection system designed to identify machine-generated content and fake news on social media. The application uses Natural Language Processing (NLP) with FastText embeddings, processed through a Hybrid CNN + LSTM Deep Learning model.

## Features
- **Deep Learning Architecture**: Utilizes a combination of 1D Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for sequence analysis.
- **Dynamic Embeddings**: Incorporates Gensim's FastText for robust word representations.
- **Interactive Web Interface**: A beautifully designed frontend powered by a Flask REST API to test custom text inputs easily.
- **Automated Dataset Handling**: Automatically downloads and prepares training data using the Hugging Face `datasets` library.

## Tech Stack
- **Backend & API**: Python, Flask
- **Machine Learning**: TensorFlow / Keras, Scikit-learn
- **NLP & Word Embeddings**: Gensim (FastText)
- **Data Manipulation**: Pandas, Numpy
- **Frontend**: HTML5, Vanilla CSS, Vanilla JavaScript

## Prerequisites
Ensure you have the following installed to run the project locally:
- Python 3.9+
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aryan-viii/deepfake_detection_on_social_media.git
   cd deepfake_detection_on_social_media
   ```

2. Create a virtual environment (Optional but Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
Before running the web app, you need to train the model so that `model.h5` and `tokenizer.pkl` are generated locally.
```bash
python train.py
```
This script will:
- Download the `GonzaloA/fake_news` dataset from Hugging Face.
- Preprocess the text sequences.
- Train the FastText embeddings.
- Train the CNN + LSTM model.
- Save the resulting model (`model.h5`) and tokenizer (`tokenizer.pkl`) to the project directory.

### 2. Running the Web Application
Once the model is trained, you can launch the Flask web server:
```bash
python app.py
```
- Open your browser and navigate to `http://localhost:5000`.
- Enter any text to see if the model predicts it to be "Human / Real" or "Machine-Generated / Fake".

## Model Architecture
- **Embedding Layer**: Uses fixed FastText word embeddings trained on the dataset corpus.
- **Conv1D Layer**: Employs a 1D Convolution over the sequence with 64 filters to capture local contextual patterns.
- **LSTM Layer**: A 64-unit Long Short-Term Memory network to capture long-range dependencies and structure in the text.
- **Dense Output**: A sigmoid activated fully-connected layer to predict binary classification probabilities.

## Disclaimer
This project is intended as a minimal, beginner-friendly demonstration of NLP techniques for text classification. It should not be used as a definitive tool to authenticate real-life information.
