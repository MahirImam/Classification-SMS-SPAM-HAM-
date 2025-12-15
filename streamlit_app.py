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
        nltk.data.find('corpora/stopwords') 
    except LookupError:
        nltk.download('stopwords')

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # KUSTOM STOPWORDS SAMA SEPERTI DI SCRIPT TRAINING
    custom_stopwords = set([
        'wkwkwk', 'gpp', 'ok', 'ya', 'nih', 'deh', 'aja', 'loh', 'loh', 'sih', 
        'dong', 'yuk', 'hehe', 'hihi', 'thnks', 'tq', 'assalamualaikum', 'salam'
    ])
    stop_id = set(stopwords.words('indonesian'))
    final_stopwords = stop_id.union(custom_stopwords)
    
    return (stemmer, final_stopwords)

NLP_RESOURCES = load_nlp_resources()


# --- 2. MUAT MODEL DAN ASSET ---

@st.cache_resource
def load_model_assets():
    """Memuat Model Klasifikasi (Ensemble), Clustering, Regresi, dan Tfidf Vectorizer."""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('voting_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
            
        with open('cluster_keywords.pkl', 'rb') as f:
            cluster_keywords = pickle.load(f)

        with open('linear_reg_model.pkl', 'rb') as f:
            linear_reg_model = pickle.load(f)
            
        return vectorizer, model, kmeans_model, cluster_keywords, linear_reg_model
    except FileNotFoundError as e:
        st.error(f"Gagal memuat asset model: {e}. Pastikan semua file .pkl ada.")
        return None, None, None, None, None

vectorizer, model, kmeans_model, cluster_keywords, linear_reg_model = load_model_assets()


# --- 3. FUNGSI PRE-PROCESSING ---

def preprocess_text(text):
    """Pipeline Pra-Pemrosesan: Termasuk penghapusan URL."""
    if NLP_RESOURCES is None:
        return ""
        
    stemmer, final_stopwords = NLP_RESOURCES

    if not isinstance(text, str):
        return ""
    
    text = text.lower() 
    text = re.sub(r'http\S+|www.\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    
    # Tokenisasi, Stopword Removal, dan Stemming
    tokens = [stemmer.stem(t) for t in text.split() if t not in final_stopwords]
    
    return " ".join(tokens)


# --- 4. FUNGSI UTAMA STREAMLIT ---

def main():
    st.set_page_config(page_title="Klasifikasi & Clustering SMS", layout="wide")
    st.title("Proyek Data Mining: Klasifikasi, Regresi, & Clustering SMS 📧")
    st.markdown("---")
    
    if model is None or vectorizer is None or kmeans_model is None or NLP_RESOURCES is None or linear_reg_model is None:
        return 
    
    
    # =========================================================
    # BAGIAN 1: KLASIFIKASI SMS SPAM/HAM (SUPERVISED)
    # =========================================================
    st.header("1. Klasifikasi SMS Spam/Ham (Ensemble Method)")
    st.info("Model ini memprediksi label **SPAM** atau **HAM** menggunakan Voting Classifier.")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("Masukkan Teks SMS di sini (untuk Klasifikasi):", 
                                    placeholder="Contoh: Selamat, Anda memenangkan undian 1 Milyar! Segera hubungi kami.", height=150, key='klasifikasi_input')
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("Klasifikasi SMS", type="primary"):
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
                        
                    
                    st.markdown("### 🗳️ Kontribusi Suara Model Dasar")
                    estimator_names = ['Multinomial Naive Bayes (MNB)', 'Decision Tree (DT)', 'LinearSVC (SVC)'] 
                    results = []

                    for i, estimator in enumerate(model.estimators_):
                        prediction_raw_index = estimator.predict(text_vector)[0]
                        string_label = str(model.classes_[prediction_raw_index]).upper()
                        
                        confidence_score = ""
                        if hasattr(estimator, 'predict_proba'):
                            probabilities = estimator.predict_proba(text_vector)[0]
                            spam_index = np.where(model.classes_ == 'spam')[0]
                            prob_spam = probabilities[spam_index[0]] if len(spam_index) > 0 else 0
                            
                            confidence_score = f"Prob. SPAM: {prob_spam:.4f}"
                        elif hasattr(estimator, 'decision_function'):
                            decision_score = estimator.decision_function(text_vector)[0]
                            confidence_score = f"Decision Score: {decision_score:.4f}"
                            
                        results.append({
                            'Model': estimator_names[i],
                            'Prediksi (Suara)': string_label, 
                            'Skor Keyakinan': confidence_score
                        })

                    df_results = pd.DataFrame(results)
                    st.table(df_results)

            else:
                st.warning("Mohon masukkan teks SMS untuk klasifikasi.")

    # =========================================================
    # BAGIAN 2: CLUSTERING SMS (UNSUPERVISED)
    # =========================================================
    st.markdown("---")
    st.header("2. Analisis Clustering SMS (K-Means)")
    
    st.info(f"Fitur ini mengelompokkan pesan ke dalam **3 Kluster** (K=3) berdasarkan kemiripan fitur bahasa.")

    user_input_cluster = st.text_area("Masukkan Teks SMS untuk dikelompokkan (Clustering):", 
                                        placeholder="Contoh: Pesan untuk uji Clustering...", height=150, key='cluster_input')

    if st.button("Tentukan Kluster K-Means", type="secondary"):
        if not user_input_cluster:
            st.warning("Mohon masukkan teks SMS untuk Clustering.")
            # Tidak perlu 'else' di sini, kode akan return di baris sebelumnya
        
        else: # Blok ini hanya dieksekusi jika input_cluster ada
            with st.spinner('Menentukan kluster...'):
                
                processed_text_cluster = preprocess_text(user_input_cluster)
                text_vector_cluster = vectorizer.transform([processed_text_cluster])
                
                # Prediksi Kluster
                cluster_label = kmeans_model.predict(text_vector_cluster)[0]
                
                st.markdown("---")
                st.subheader("Hasil Clustering K-Means")
                
                # --- INTERPRETASI K=3 ---
                
                nama_kluster = f"Kluster {cluster_label}"
                display_func = st.info
                keterangan = "Hasil pengelompokan berdasarkan fitur bahasa."
                
                # Interpretasi Kluster #0, #1, #2 
                if cluster_label == 0:
                    nama_kluster = "Kluster Pesan Normal / Percakapan"
                    display_func = st.success
                elif cluster_label == 1:
                    nama_kluster = "Kluster SPAM Hadiah / Kode / Link (Hard Spam)"
                    display_func = st.error
                elif cluster_label == 2:
                    nama_kluster = "Kluster SPAM Komersial / Promosi Layanan"
                    display_func = st.warning
                
                display_func(f"Kluster #{cluster_label}: {nama_kluster}")
                st.caption(keterangan)

                # --- INFORMASI TAMBAHAN (Mengapa Masuk Kluster Ini) ---
                
                st.markdown(f"**Total Kluster Terbentuk:** 3")
                
                if cluster_keywords:
                    st.markdown("---")
                    st.subheader("💡 Mengapa Masuk Kluster Ini?")
                    
                    keywords = cluster_keywords.get(cluster_label, [])
                    
                    st.write(f"Pesan ini masuk ke **Kluster #{cluster_label}** karena memiliki kesamaan linguistik terbesar dengan fitur-fitur yang mendefinisikan kluster tersebut.")
                    st.markdown(f"**10 Kata Kunci Paling Dominan di Kluster #{cluster_label} adalah:**")
                    st.code(', '.join(keywords))
                    
                    st.markdown("*(Bandingkan dengan teks input Anda setelah diproses:)*")
                    st.caption(f"`{processed_text_cluster}`")
        
    
    # =========================================================
    # BAGIAN 3: ANALISIS REGRESI LINEAR (WAJIB TUGAS)
    # =========================================================
    st.markdown("---")
    st.header("3. Analisis Regresi Linear")
    
    if linear_reg_model is not None:
        st.info("Model Regresi Linear memprediksi 'Skor Spam' (nilai kontinu antara 0.0 hingga 1.0) dari pesan Anda.")
        
        user_input_reg = st.text_area("Masukkan Teks SMS (untuk Regresi):", 
                                        placeholder="Contoh: Apakah pesan ini mirip SPAM?", height=100, key='reg_input')
        
        # BARIS KRITIS: Tanda kutip diperbaiki dari ' menjadi "
        if st.button("Prediksi Skor Regresi", key='reg_button', type="primary"): 
            if user_input_reg:
                
                with st.spinner('Menghitung Skor Regresi...'):
                    
                    processed_text_reg = preprocess_text(user_input_reg)
                    text_vector_reg = vectorizer.transform([processed_text_reg])
                    
                    # Prediksi Regresi
                    predicted_score = linear_reg_model.predict(text_vector_reg)[0]
                    
                    # Batasi skor antara 0 dan 1
                    predicted_score = max(0.0, min(1.0, predicted_score))
                    
                    st.markdown("---")
                    st.subheader("Hasil Prediksi Skor Regresi")
                    
                    # Logika Visualisasi dan Interpretasi
                    if predicted_score >= 0.8:
                        color_text = "red"
                        st.error("Skor sangat tinggi. Pesan ini sangat mirip SPAM.")
                    elif predicted_score >= 0.5:
                        color_text = "orange"
                        st.warning("Skor berada di ambang batas. Pesan ini memiliki beberapa fitur SPAM.")
                    else:
                        color_text = "green"
                        st.success("Skor rendah. Pesan ini mirip dengan pola HAM (Normal).")
                        
                    # Tampilan Skor Numerik
                    st.markdown(f"**Skor Kemiripan SPAM (0.0 - 1.0):** <span style='color:{color_text}; font-size:24px;'>**{predicted_score:.4f}**</span>", unsafe_allow_html=True)
                    
                    # Visualisasi: Progress Bar
                    st.progress(predicted_score)
                    

            else:
                st.warning("Mohon masukkan teks SMS untuk Regresi.")


if __name__ == '__main__':
    main()