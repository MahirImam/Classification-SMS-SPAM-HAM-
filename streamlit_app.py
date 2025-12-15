import streamlit as st
import pickle
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re 
import nltk 
from nltk.corpus import stopwords
import numpy as np

# --- 1. INISIALISASI NLP RESOURCES ---

NLP_RESOURCES = None

@st.cache_resource
def load_nlp_resources():
    """Memuat Sastrawi Stemmer dan Stopwords Bahasa Indonesia."""
    try:
        # Unduh data stopwords NLTK secara kondisional
        nltk.data.find('corpora/stopwords') 
    except LookupError:
        nltk.download('stopwords')

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_id = set(stopwords.words('indonesian'))
    
    return (stemmer, stop_id)

NLP_RESOURCES = load_nlp_resources()


# --- 2. MUAT MODEL DAN VECTORIZER ---

@st.cache_resource
def load_model_assets():
    """Memuat Model Klasifikasi (Ensemble) dan Tfidf Vectorizer."""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('voting_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        return vectorizer, model
    except FileNotFoundError as e:
        # Peringatan jika model klasifikasi hilang
        st.error(f"Gagal memuat asset Klasifikasi: {e}. Pastikan file .pkl ada.")
        return None, None

vectorizer, model = load_model_assets()

# --- Muat Model Clustering ---
@st.cache_resource
def load_clustering_model():
    """Memuat Model K-Means Clustering."""
    try:
        # Memuat kmeans_model.pkl yang baru dibuat
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        return kmeans_model
    except FileNotFoundError:
        st.warning("File 'kmeans_model.pkl' tidak ditemukan. Fitur Clustering dinonaktifkan.")
        return None

kmeans_model = load_clustering_model()


# --- 3. FUNGSI PRE-PROCESSING ---

def preprocess_text(text):
    """Pipeline Pra-Pemrosesan: Termasuk penghapusan URL."""
    if NLP_RESOURCES is None:
        return ""
        
    stemmer, stop_id = NLP_RESOURCES

    if not isinstance(text, str):
        return ""
    
    # Cleaning & URL REMOVAL (Penting untuk mengatasi masalah SPAM/HAM sebelumnya)
    text = text.lower() 
    text = re.sub(r'http\S+|www.\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    
    # Tokenisasi, Stopword Removal, dan Stemming
    tokens = [stemmer.stem(t) for t in text.split() if t not in stop_id]
    
    return " ".join(tokens)


# --- 4. FUNGSI UTAMA STREAMLIT ---

def main():
    st.title("Proyek Data Mining: Klasifikasi & Clustering SMS 📧")
    st.markdown("---")
    
    if model is None or vectorizer is None or NLP_RESOURCES is None:
        st.error("Aplikasi tidak dapat berjalan karena model/sumber daya hilang.")
        return 
    
    # =========================================================
    # BAGIAN 1: KLASIFIKASI SMS SPAM/HAM (SUPERVISED)
    # =========================================================
    st.header("1. Klasifikasi SMS Spam/Ham (Ensemble Method)")
    st.info("Model ini memprediksi label **SPAM** atau **HAM** menggunakan Voting Classifier (Hard).")
    
    st.subheader("Uji Prediksi Teks SMS")
    
    user_input = st.text_area("Masukkan Teks SMS di sini (untuk Klasifikasi):", 
                              placeholder="Contoh: Selamat, Anda memenangkan undian 1 Milyar! Segera hubungi kami.", key='klasifikasi_input')

    if st.button("Klasifikasi SMS"):
        if user_input:
            
            with st.spinner('Sedang memproses dan mengklasifikasi...'):
                
                processed_text = preprocess_text(user_input)
                text_vector = vectorizer.transform([processed_text])
                prediction = model.predict(text_vector)[0]
                
                st.markdown("---")
                
                # Tampilkan Hasil Utama
                if prediction == 'spam':
                    st.error(f"⚠️ **HASIL PREDIKSI AKHIR: {prediction.upper()}**")
                    st.balloons()
                else:
                    st.success(f"✅ **HASIL PREDIKSI AKHIR: {prediction.upper()}**")
                    
                
                st.markdown("### 📊 Analisis Teknis")
                st.caption(f"Teks Setelah Pra-Pemrosesan: `{processed_text}`")
                
                # Kontribusi Suara Model Dasar
                st.markdown("### 🗳️ Kontribusi Suara Model Dasar")
                estimator_names = ['Multinomial Naive Bayes (MNB)', 'Logistic Regression (LR)', 'LinearSVC (SVC)']
                results = []

                for i, estimator in enumerate(model.estimators_):
                    prediction_raw_index = estimator.predict(text_vector)[0]
                    string_label = str(model.classes_[prediction_raw_index]).upper()
                    
                    confidence_score = ""
                    if hasattr(estimator, 'predict_proba'):
                        probabilities = estimator.predict_proba(text_vector)[0]
                        
                        # Pastikan indeks 'spam' dan 'ham' benar
                        spam_index = np.where(model.classes_ == 'spam')[0]
                        ham_index = np.where(model.classes_ == 'ham')[0]
                        
                        prob_spam = probabilities[spam_index[0]] if len(spam_index) > 0 else 0
                        prob_ham = probabilities[ham_index[0]] if len(ham_index) > 0 else 0
                        
                        confidence_score = f"SPAM: {prob_spam:.4f} | HAM: {prob_ham:.4f}"
                    elif hasattr(estimator, 'decision_function'):
                        decision_score = estimator.decision_function(text_vector)[0]
                        confidence_score = f"Decision Score: {decision_score:.4f}"
                        
                    results.append({
                        'Model': estimator_names[i],
                        'Prediksi (Suara)': string_label, 
                        'Skor Keyakinan': confidence_score
                    })

                df_results = pd.DataFrame(results)
                ham_votes = df_results[df_results['Prediksi (Suara)'] == 'HAM'].shape[0]
                spam_votes = df_results[df_results['Prediksi (Suara)'] == 'SPAM'].shape[0]
                
                st.table(df_results)
                st.markdown(f"**Total Suara: HAM ({ham_votes}) vs. SPAM ({spam_votes})**")
                
        else:
            st.warning("Mohon masukkan teks SMS untuk klasifikasi.")

    # =========================================================
    # BAGIAN 2: CLUSTERING SMS (UNSUPERVISED)
    # =========================================================
    st.markdown("---")
    st.header("2. Analisis Clustering SMS (K-Means)")
    
    if kmeans_model is not None:
        st.info("Fitur ini mengelompokkan pesan ke dalam 4 Kluster berdasarkan kemiripan fitur bahasa.")

        user_input_cluster = st.text_area("Masukkan Teks SMS untuk dikelompokkan (Clustering):", 
                                          placeholder="Contoh: Pesan untuk uji Clustering...", key='cluster_input')

        if st.button("Tentukan Kluster K-Means"):
            if user_input_cluster:
                
                with st.spinner('Menentukan kluster...'):
                    
                    processed_text_cluster = preprocess_text(user_input_cluster)
                    text_vector_cluster = vectorizer.transform([processed_text_cluster])
                    
                    # Prediksi Kluster
                    cluster_label = kmeans_model.predict(text_vector_cluster)[0]
                    
                    st.markdown("---")
                    st.subheader("Hasil Clustering K-Means")
                    
                    # Interpretasi Kluster
                    if cluster_label == 0:
                        st.success(f"Kluster #{cluster_label}: Kluster Pesan Normal/Formal.")
                    elif cluster_label == 1:
                        st.error(f"Kluster #{cluster_label}: Kluster Pesan Promosi/Iklan (Mirip Spam Kuat).")
                    elif cluster_label == 2:
                        st.warning(f"Kluster #{cluster_label}: Kluster Pesan Singkat/Ambigu.")
                    else:
                        st.info(f"Kluster #{cluster_label}: Kluster Pesan Khusus/Unik.")
                        
                    st.markdown(f"**SMS dikelompokkan ke Kluster {cluster_label}**.")
            else:
                st.warning("Mohon masukkan teks SMS untuk Clustering.")
    else:
        st.error("Fitur Clustering dinonaktifkan. Pastikan file 'kmeans_model.pkl' ada.")


if __name__ == '__main__':
    main() 