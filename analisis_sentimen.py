import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
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

# judul halaman
st.title ('Analisis sentimen produk hp Oppo')
st.write('Masukkan ulasan di bawah sini')

user_input = st.text_area('Masukan Ulasan :', '')

if st.button('Prediksi'):
    if user_input.strip():
       # Preprocessing
        clean_text = preprocess_text(user_input, stopwords, slang_dict)
        vectorized_text = vectorizer.transform([clean_text]).toarray()

        # Prediksi sentimen
        predicted_class = model.predict(vectorized_text)[0]
        predicted_proba = model.predict_proba(vectorized_text)

        prob_negatif = round(predicted_proba[0][0] * 100, 2)
        prob_netral = round(predicted_proba[0][1] * 100, 2) if len(predicted_proba[0]) > 2 else 0
        prob_positif = round(predicted_proba[0][2] * 100, 2)

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
            emoji = '🤔'
        
        # Tampilkan hasil
        st.subheader("Hasil Prediksi")
        st.write(f"Sentimen: {sentiment} {emoji}")
        st.write(f"Probabilitas: Negatif: {prob_negatif}%, Netral: {prob_netral}%, Positif: {prob_positif}%")
    else:
        st.warning("Masukkan ulasan terlebih dahulu.")
