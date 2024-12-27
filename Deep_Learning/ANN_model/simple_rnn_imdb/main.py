import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Define the base directory (location of main.py)
base_dir = os.path.dirname(__file__)

# Load the trained model
model_path = os.path.join(base_dir, 'simple_rnn_imdb.h5')
model = tf.keras.models.load_model(model_path)

# Load the IMDB dataset word index
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Helper Functions
def decode_review(encoded_review):
    """
    Decode a numerical representation of a review into human-readable text.
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    """
    Preprocess a text review by encoding words into integers and padding it to a fixed length.
    """
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip():
        # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]:.2f}')
    else:
        st.warning('Please enter a valid movie review.')
else:
    st.write('Waiting for input. Please type a review and press the "Classify" button.')