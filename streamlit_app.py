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
    
    # LANGKAH PENTING: Unduh data stopwords NLTK secara kondisional (Mengatasi LookupError di Cloud)
    try:
        nltk.data.find('corpora/stopwords') 
    except LookupError:
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
                
                # 3. Prediksi (menggunakan Hard Voting)
                prediction = model.predict(text_vector)[0]
                
                st.markdown("---")
                
                # 4. Tampilkan Hasil Utama
                if prediction == 'spam':
                    st.error(f"⚠️ **HASIL PREDIKSI AKHIR: {prediction.upper()}**")
                    st.balloons()
                    st.write("Keputusan mayoritas model ensemble adalah **SPAM**.")
                else:
                    st.success(f"✅ **HASIL PREDIKSI AKHIR: {prediction.upper()}**")
                    st.write("Keputusan mayoritas model ensemble adalah **HAM**.")
                    
                
                # 5. --- TAMBAHAN DETAIL ANALISIS ---
                st.markdown("### 📊 Analisis Teknis Fitur")
                st.caption(f"Teks Setelah Pra-Pemrosesan: `{processed_text}`")
                
                # B. Top 5 Fitur Penting (TF-IDF Scores)
                if processed_text:
                    feature_names = vectorizer.get_feature_names_out()
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
                        st.warning("Tidak ada kata yang dikenali dalam kosakata model (Mungkin karena kata tersebut sangat jarang atau sudah dihapus).")


                # C. PREDIKSI INDIVIDUAL DAN KONTRIBUSI SUARA
                st.markdown("### 🗳️ Kontribusi Suara Model Dasar")
                
                estimator_names = ['Multinomial Naive Bayes (MNB)', 
                                   'Logistic Regression (LR)', 
                                   'LinearSVC (SVC)']
                
                results = []

                for i, estimator in enumerate(model.estimators_):
                    # 1. Prediksi Label (Suara)
                    # Ambil hasil prediksi (kemungkinan berupa index 0 atau 1)
                    prediction_raw_index = estimator.predict(text_vector)[0]
                    
                    # FIX PENTING: Map index (0/1) kembali ke string label ('HAM'/'SPAM')
                    # Gunakan model.classes_ untuk mendapatkan label string
                    string_label = str(model.classes_[prediction_raw_index]).upper()
                    
                    confidence_score = ""
                    
                    # 2. Skor Keyakinan (Probabilitas atau Decision Function)
                    if hasattr(estimator, 'predict_proba'):
                        # MNB dan LR mendukung Probabilitas
                        probabilities = estimator.predict_proba(text_vector)[0]
                        prob_spam = probabilities[np.where(model.classes_ == 'spam')[0][0]]
                        prob_ham = probabilities[np.where(model.classes_ == 'ham')[0][0]]
                        confidence_score = f"SPAM: {prob_spam:.4f} | HAM: {prob_ham:.4f}"
                    
                    elif hasattr(estimator, 'decision_function'):
                        # LinearSVC mendukung Decision Function
                        decision_score = estimator.decision_function(text_vector)[0]
                        confidence_score = f"Decision Score: {decision_score:.4f}"
                        
                    results.append({
                        'Model': estimator_names[i],
                        'Prediksi (Suara)': string_label, # Sekarang berisi 'HAM' atau 'SPAM'
                        'Skor Keyakinan': confidence_score
                    })

                df_results = pd.DataFrame(results)
                
                # Menghitung Suara (Sekarang ini akan menghitung 'HAM' dan 'SPAM' yang benar)
                ham_votes = df_results[df_results['Prediksi (Suara)'] == 'HAM'].shape[0]
                spam_votes = df_results[df_results['Prediksi (Suara)'] == 'SPAM'].shape[0]
                
                st.table(df_results)
                st.markdown(f"**Total Suara: HAM ({ham_votes}) vs. SPAM ({spam_votes})**")
                
        else:
            st.warning("Mohon masukkan teks SMS untuk diklasifikasi.")

if __name__ == '__main__':
    main()