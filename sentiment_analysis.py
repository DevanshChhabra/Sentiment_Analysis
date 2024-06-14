import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import warnings

# Suppress warnings related to deserialization
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Load tokenizer
with open(r'C:\Users\devan\Downloads\tokenizer20L.json', encoding='utf-8') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Load model
model = tf.keras.models.load_model(r'C:\Users\devan\Downloads\Sentiment_Analysis20L.h5')

# Preprocessing functions (you can enable these if needed)
def clean_text(text):
    text = re.sub(r'[^A-Za-zÀ-ú ]+', '', text)
    text = re.sub('book|one', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([token for token in tokens if token not in stop_words])

def normalize_text(text):
    stemmer = SnowballStemmer("english")
    normalized_text = []
    for word in text.split():
        stemmed_word = stemmer.stem(word)
        normalized_text.append(stemmed_word)
    return ' '.join(normalized_text)

# Streamlit app
st.title('Amazon Reviews Sentiment Analysis')

review = st.text_area('Enter your review:', '')

if st.button('Predict'):
    # Optionally preprocess the review
    preprocessed_review = clean_text(review)
    preprocessed_review = remove_stopwords(preprocessed_review)
    preprocessed_review = normalize_text(preprocessed_review)

    preprocessed_review = review  # Use raw review for simplicity

    # Tokenize and pad the preprocessed review
    max_len = 200
    sequence = tokenizer.texts_to_sequences([preprocessed_review])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    if prediction >= 0.5:
        sentiment = 'Positive'
        confidence = prediction * 100
        color = 'green'
    else:
        sentiment = 'Negative'
        confidence = (1 - prediction) * 100
        color = 'red'
    
    sentiment_formatted = f'<span style="color: {color}">{sentiment}</span>'

    # Display the sentiment
    st.write(f'Sentiment: {sentiment_formatted} (Confidence: {confidence:.2f}%)', unsafe_allow_html=True)
