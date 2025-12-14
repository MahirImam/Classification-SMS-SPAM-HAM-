import streamlit as st
import pickle
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re # Untuk membersihkan teks
import nltk 
# nltk.download('stopwords') # Jalankan ini SATU KALI di awal jika belum pernah
from nltk.corpus import stopwords
import numpy as np

# --- Inisialisasi Stemmer dan Stopwords (Gunakan cache) ---

# --- Inisialisasi Stemmer dan Stopwords (Gunakan cache) ---

@st.cache_resource
def load_nlp_resources():
    """Memuat Sastrawi Stemmer dan Stopwords Bahasa Indonesia."""
    
    # LANGKAH PENTING: Unduh data stopwords NLTK
    import nltk
    try:
        # Coba muat dulu, jika gagal, NLTK akan mengunduh
        nltk.data.find('corpora/stopwords') 
    except LookupError:
        # Jika data tidak ditemukan, unduh. Ini dijalankan SATU KALI di Cloud.
        nltk.download('stopwords')

    # 1. Stemmer Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # 2. Stopwords (Sekarang NLTK sudah memiliki datanya)
    stop_id = set(stopwords.words('indonesian'))
    
    return stemmer, stop_id


# --- Muat Model dan Vectorizer (Gunakan cache) ---

@st.cache_resource
def load_model_assets():
    """Memuat Model Ensemble dan Tfidf Vectorizer."""
    try:
        # Muat Vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Muat Model Ensemble
        with open('voting_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        return vectorizer, model
    except FileNotFoundError:
        st.error("Pastikan file 'tfidf_vectorizer.pkl' dan 'voting_classifier_model.pkl' ada di direktori yang sama!")
        return None, None

vectorizer, model = load_model_assets()

def preprocess_text(text):
    """Pipeline Pra-Pemrosesan: Tokenisasi, Stopword Removal, Stemming."""
    if not isinstance(text, str):
        return ""
    
    # 1. Cleaning (Optional, jika belum dibersihkan)
    text = text.lower() # Ubah menjadi huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus non-alfabet
    
    # 2. Tokenisasi
    tokens = text.split()
    
    # 3. Stopword Removal
    tokens = [t for t in tokens if t not in stop_id]
    
    # 4. Stemming (menggunakan Sastrawi yang dimuat)
    stems = [stemmer.stem(t) for t in tokens]
    
    # 5. Gabungkan kembali
    return " ".join(stems)

def main():
    st.title("Classification SMS SPAM/HAM 📧")
    st.markdown("---")
    st.header("Metode Ensemble: Voting Classifier")

    # --- INFORMASI UMUM MODEL ---
    st.info("""
        Model ini menggunakan **Voting Classifier (Hard)**.
        Ia menggabungkan prediksi dari tiga model dasar yang kuat untuk klasifikasi teks:
        * **Multinomial Naive Bayes (MNB)**
        * **Logistic Regression (LR)**
        * **Linear Support Vector Classification (LinearSVC)**
        Model ini mencapai Akurasi pada data test sebesar **98.84%**.
    """)
    st.markdown("---")
    # ---------------------------

    if model is None or vectorizer is None:
        return 
        
    st.subheader("Uji Prediksi Teks SMS")
    
    user_input = st.text_area("Masukkan Teks SMS di sini:", 
                              placeholder="Contoh: Selamat, Anda memenangkan undian 1 Milyar! Segera hubungi kami.")

    if st.button("Klasifikasi SMS"):
        if user_input:
            
            with st.spinner('Sedang memproses dan mengklasifikasi...'):
                
                # 1. Pra-Pemrosesan
                processed_text = preprocess_text(user_input)
                
                # 2. Vektorisasi
                text_vector = vectorizer.transform([processed_text])
                
                # 3. Prediksi (menggunakan Hard Voting)
                prediction = model.predict(text_vector)[0]
                
                st.markdown("---")
                
                # 4. Tampilkan Hasil Utama
                if prediction == 'spam':
                    st.error(f"⚠️ **HASIL PREDIKSI: {prediction.upper()}**")
                    st.balloons()
                    st.write("SMS ini kemungkinan adalah penipuan atau iklan yang tidak diinginkan.")
                else:
                    st.success(f"✅ **HASIL PREDIKSI: {prediction.upper()}**")
                    st.write("SMS ini adalah pesan normal (bukan spam).")
                    
                
                # 5. --- TAMBAHAN DETAIL ANALISIS ---
                st.markdown("### 📊 Analisis Teknis Fitur")
                
                # A. Detail Pemrosesan
                st.caption(f"Teks Setelah Pra-Pemrosesan: `{processed_text}`")
                
                # B. Top 5 Fitur Penting (TF-IDF Scores)
                if processed_text:
                    # Dapatkan skor TF-IDF untuk kata-kata di teks input
                    feature_names = vectorizer.get_feature_names_out()
                    feature_index = text_vector.nonzero()[1]
                    tfidf_scores = text_vector.data
                    
                    # Buat DataFrame untuk memvisualisasikan skor
                    df_scores = pd.DataFrame({
                        'Token': [feature_names[i] for i in feature_index],
                        'TF-IDF Score': tfidf_scores
                    }).sort_values(by='TF-IDF Score', ascending=False)
                    
                    st.subheader("Kata Kunci Penting (Top 5 TF-IDF)")
                    
                    if not df_scores.empty:
                        # Tampilkan 5 kata dengan skor TF-IDF tertinggi (paling unik/penting)
                        st.table(df_scores.head(5))
                    else:
                        st.warning("Tidak ada kata yang dikenali dalam kosakata model (Mungkin karena Stopword Removal yang terlalu agresif).")
                
                # C. Prediksi Probabilitas (Jika Menggunakan Soft Voting, atau MNB/LR tunggal)
                # Catatan: Hard Voting tidak menghasilkan probabilitas gabungan. 
                # Untuk tujuan presentasi, kita bisa menampilkan probabilitas dari salah satu model dasarnya (misalnya MNB)
                
                # PENTING: Anda harus mengakses model dasar di dalam VotingClassifier
                # Asumsi 'mnb' adalah nama estimator pertama ('mnb', clf1)
                
                try:
                    mnb_estimator = model.estimators_[0] # Ambil estimator pertama (MNB)
                    
                    # Latih MNB estimator (jika belum terlatih - tapi harusnya sudah)
                    # Jika model utama sudah terlatih, estimatornya sudah terlatih juga.
                    
                    # Dapatkan probabilitas dari MNB
                    probabilities = mnb_estimator.predict_proba(text_vector)[0]
                    
                    # Urutan kelas: model.classes_
                    prob_ham = probabilities[np.where(model.classes_ == 'ham')[0][0]]
                    prob_spam = probabilities[np.where(model.classes_ == 'spam')[0][0]]
                    
                    st.subheader("Probabilitas (dari Model Naive Bayes)")
                    st.markdown(f"**Probabilitas HAM:** `{prob_ham:.4f}`")
                    st.markdown(f"**Probabilitas SPAM:** `{prob_spam:.4f}`")
                    
                    if prob_spam > prob_ham:
                        st.write(f"Model sangat yakin ini adalah SPAM (dengan probabilitas {prob_spam:.2f})")
                    else:
                        st.write(f"Model sangat yakin ini adalah HAM (dengan probabilitas {prob_ham:.2f})")
                        
                except Exception as e:
                    # Tangani error jika struktur model ensemble berubah
                    st.error(f"Gagal menampilkan detail probabilitas. {e}")
                    

        else:
            st.warning("Mohon masukkan teks SMS untuk diklasifikasi.")

if __name__ == '__main__':
    main()