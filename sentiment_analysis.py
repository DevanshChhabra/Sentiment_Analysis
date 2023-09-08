import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import tensorflow as tf
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import json
import nltk
nltk.download('punkt')


with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


model = tf.keras.models.load_model(r'C:\Users\devan\Downloads\saved_model (1)\M')

# Preprocessing functions
def clean_text(text):
    text = re.sub(r'[^A-Za-zÀ-ú ]+', '', text)
    text = re.sub('book|one', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(texto):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(texto.lower())
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
    preprocessed_review = clean_text(review)
    preprocessed_review = remove_stopwords(preprocessed_review)
    preprocessed_review = normalize_text(preprocessed_review)

    # Tokenize and pad the preprocessed review
    max_len = 200
    sequence = tokenizer.texts_to_sequences([preprocessed_review])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    preprocessed_review = tf.convert_to_tensor(preprocessed_review)
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    
    positive_color = 'green'
    negative_color = 'red'


    sentiment_formatted = f'<span style="color: {positive_color}">Positive</span>' if prediction >= 0.5 else f'<span style="color: {negative_color}">Negative</span>'

    # Display the sentiment
    st.write(f'Sentiment: {sentiment_formatted} (Confidence: {prediction:.2f})', unsafe_allow_html=True)
    # st.write(f'Sentiment: {sentiment} (Confidence: {prediction:.2f})')
