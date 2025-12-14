import streamlit as st
import pickle
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re 
import nltk 
from nltk.corpus import stopwords
import numpy as np

# --- 1. INISIALISASI NLP RESOURCES (Menggunakan Cache untuk Efisiensi) ---

# Global variable to store loaded resources, accessed globally by preprocess_text
NLP_RESOURCES = None

@st.cache_resource
def load_nlp_resources():
    """Memuat Sastrawi Stemmer dan Stopwords Bahasa Indonesia, serta mengunduh data NLTK."""
    
    # LANGKAH PENTING: Unduh data stopwords NLTK secara kondisional
    try:
        # Coba muat data stopwords, jika gagal akan Raise LookupError
        nltk.data.find('corpora/stopwords') 
    except LookupError:
        # Jika data tidak ditemukan di lingkungan Cloud, unduh.
        nltk.download('stopwords')

    # 1. Stemmer Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # 2. Stopwords
    stop_id = set(stopwords.words('indonesian'))
    
    # Mengembalikan tuple dari sumber daya
    return (stemmer, stop_id)

# Panggil fungsi cache untuk memuat sumber daya ke variabel global
# Hasilnya akan berupa tuple: (stemmer, stop_id)
NLP_RESOURCES = load_nlp_resources()


# --- 2. MUAT MODEL DAN VECTORIZER (Menggunakan Cache) ---

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


# --- 3. FUNGSI PRE-PROCESSING (Mengakses Sumber Daya dari Variabel Global) ---

def preprocess_text(text):
    """
    Pipeline Pra-Pemrosesan: Tokenisasi, Stopword Removal, Stemming.
    Mengakses stemmer dan stop_id dari tuple NLP_RESOURCES yang dijamin ada.
    """
    if NLP_RESOURCES is None:
        st.error("Sumber daya NLP gagal dimuat.")
        return ""
        
    stemmer, stop_id = NLP_RESOURCES # Ekstrak stemmer dan stop_id dari tuple global

    if not isinstance(text, str):
        return ""
    
    # 1. Cleaning
    text = text.lower() 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    
    # 2. Tokenisasi
    tokens = text.split()
    
    # 3. Stopword Removal
    tokens = [t for t in tokens if t not in stop_id]
    
    # 4. Stemming
    stems = [stemmer.stem(t) for t in tokens]
    
    # 5. Gabungkan kembali
    return " ".join(stems)


# --- 4. FUNGSI UTAMA STREAMLIT ---

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

    if model is None or vectorizer is None or NLP_RESOURCES is None:
        st.error("Aplikasi tidak dapat dimuat karena model atau sumber daya NLP hilang/gagal dimuat.")
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
                
                # 3. Prediksi
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
                    feature_names = vectorizer.get_feature_names_out()
                    # Menangani sparse matrix untuk skor
                    feature_index = text_vector.nonzero()[1]
                    tfidf_scores = text_vector.data
                    
                    df_scores = pd.DataFrame({
                        'Token': [feature_names[i] for i in feature_index],
                        'TF-IDF Score': tfidf_scores
                    }).sort_values(by='TF-IDF Score', ascending=False)
                    
                    st.subheader("Kata Kunci Penting (Top 5 TF-IDF)")
                    
                    if not df_scores.empty:
                        st.table(df_scores.head(5))
                    else:
                        st.warning("Tidak ada kata yang dikenali dalam kosakata model.")
                
                # C. Prediksi Probabilitas dari MNB
                try:
                    mnb_estimator = model.estimators_[0] 
                    probabilities = mnb_estimator.predict_proba(text_vector)[0]
                    
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
                    # Ini mungkin terjadi jika urutan estimator diubah
                    st.caption(f"Detail probabilitas (MNB) gagal ditampilkan.")
                    

        else:
            st.warning("Mohon masukkan teks SMS untuk diklasifikasi.")

if __name__ == '__main__':
    main()