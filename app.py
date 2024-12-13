import pandas as pd
import numpy as np
import spacy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import render_template
import requests



# Preprocessing
from preprocessing import preprocess_text, load_stopwords, load_normalisasi

# Muat stopwords dan normalisasi
stopwords = load_stopwords('stopwords.txt')  # Ubah sesuai ekstensi file Anda
slang_dict = load_normalisasi('normalisasi.csv')
text = "Hp jelek, ga bgs utk sehari-hari"
clean_text = preprocess_text(text, stopwords, slang_dict)
print(f'clean teks =', clean_text) 

# load model
model = joblib.load('knn_sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

#create the app object
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
    preprocessing_ulasan = [preprocess_text(review, stopwords, slang_dict) for review in new_review]
    new_review_vectorizer = vectorizer.transform(preprocessing_ulasan).toarray()
    print(f'new_review_vectorizer shape: {new_review_vectorizer.shape}')



    predicted_class = model.predict(new_review_vectorizer)[0]
    print("Predicted Value:", predicted_class)
    print("Type of Predicted Class:", type(predicted_class))
    print("Exact Predicted Class Value:", predicted_class)

    predicted_proba = model.predict_proba(new_review_vectorizer)
    prob_negatif = round(predicted_proba[0][0] * 100, 2)
    prob_positif = round(predicted_proba[0][2] * 100, 2)
    prob_netral = round(predicted_proba[0][1] * 100,2) if len(predicted_proba[0]) > 2 else 0 

    if predicted_class == 'Negatif':
        sentiment = 'Negatif'
        emoji = '😡'
    elif predicted_class == 'Netral':
        sentiment = 'Netral'
        emoji = '😐'
    elif predicted_class == 'Positif':
        sentiment = 'Positif'
        emoji = '😊'
    else:
        sentiment = 'Tidak ada sentimen'  

    # Render hasil prediksi dan probabilitas
    return render_template('result.html', 
                           prediction=sentiment, 
                           prob_negatif=prob_negatif, 
                           prob_positif=prob_positif,
                           prob_netral=prob_netral,
                           emoji=emoji)
                             
if __name__ == "__main__":
    app.run(debug=True)
