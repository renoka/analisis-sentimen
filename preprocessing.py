import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

# Buat stemmer (dijalankan sekali saat file diimpor)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk memuat stopwords dari file
def load_stopwords(file_path):
    """
    Memuat stopwords dari file teks.
    Setiap baris pada file dianggap sebagai satu kata stopwords.
    """
    with open(file_path, 'r') as file:
        stopwords = file.read().splitlines()
    return set(stopwords)

# Fungsi untuk memuat normalisasi dari file CSV
def load_normalisasi(file_path):
    """
    Memuat normalisasi kata dari file CSV.
    Kolom pertama: slang, kolom kedua: kata normal.
    """
    df = pd.read_csv(file_path, header=None, names=['slang', 'normal'])
    return dict(zip(df['slang'], df['normal']))

# Fungsi preprocessing teks
def preprocess_text(text, stopwords, slang_dict):
    """
    Melakukan preprocessing teks:
    - Case folding
    - Menghapus tanda baca dan angka
    - Menghapus stopwords
    - Normalisasi kata
    - Stemming
    """
    # Case folding
    text = text.lower()
    # Hapus tanda baca dan angka
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    text = re.sub(r'\d+', '', text)      # Menghapus angka
    # Tokenizing
    tokens = text.split()
    # Stopwords removal
    tokens = [token for token in tokens if token not in stopwords]
    # Normalisasi kata
    tokens = [slang_dict.get(token, token) for token in tokens]
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    # Gabungkan kembali menjadi teks
    return ' '.join(tokens)
