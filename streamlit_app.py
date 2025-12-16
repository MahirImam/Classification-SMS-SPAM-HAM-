import streamlit as st
import pickle
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re 
import nltk 
from nltk.corpus import stopwords
import numpy as np

# --- Load NLP Resources ---
NLP_RESOURCES = None

@st.cache_resource
def load_nlp_resources():
    try:
        nltk.data.find('corpora/stopwords') 
    except LookupError:
        nltk.download('stopwords')

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    custom_stopwords = set([
        'wkwkwk', 'gpp', 'ok', 'ya', 'nih', 'deh', 'aja', 'loh', 'loh', 'sih', 
        'dong', 'yuk', 'hehe', 'hihi', 'thnks', 'tq', 'assalamualaikum', 'salam'
    ])
    stop_id = set(stopwords.words('indonesian'))
    final_stopwords = stop_id.union(custom_stopwords)
    
    return (stemmer, final_stopwords)

NLP_RESOURCES = load_nlp_resources()


# --- Load Models and Assets ---
@st.cache_resource
def load_model_assets():
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


# --- Pre-processing Function ---
def preprocess_text(text):
    if NLP_RESOURCES is None:
        return ""
        
    stemmer, final_stopwords = NLP_RESOURCES

    if not isinstance(text, str):
        return ""
    
    text = text.lower() 
    text = re.sub(r'http\S+|www.\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    
    tokens = [stemmer.stem(t) for t in text.split() if t not in final_stopwords]
    
    return " ".join(tokens)


# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Klasifikasi & Clustering SMS", 
        layout="wide", 
        initial_sidebar_state="expanded",
        page_icon="🤖"
    )

    # SIDEBAR
    st.sidebar.title("Proyek Data Mining 📈")
    st.sidebar.markdown("## Analisis SMS SPAM/HAM 📧")
    st.sidebar.info(
        "Aplikasi ini mengimplementasikan teknik Data Mining Supervised (Klasifikasi & Regresi) dan Unsupervised (Clustering) pada data SMS berbahasa Indonesia."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Model yang Digunakan")
    st.sidebar.markdown("- **Klasifikasi:** Voting Classifier (MNB, DT, LinearSVC)")
    st.sidebar.markdown("- **Regresi:** Linear Regression")
    st.sidebar.markdown("- **Clustering:** K-Means (K=3)")
    st.sidebar.markdown("---")
    st.sidebar.caption("Tugas Besar Mata Kuliah Data Mining")

    # Main Body Title
    st.title("Prediksi dan Analisis SMS Spam/Ham 💬")
    st.markdown("---")
    
    if model is None or vectorizer is None or kmeans_model is None or NLP_RESOURCES is None or linear_reg_model is None:
        return 
    
    tab1, tab2, tab3 = st.tabs(["1. Klasifikasi (Spam/Ham)", "2. Regresi (Skor Spam)", "3. Clustering (K-Means)"])

    # =========================================================
    # TAB 1: KLASIFIKASI SMS SPAM/HAM
    # =========================================================
    with tab1:
        st.header("1. Klasifikasi SMS Spam/Ham (Voting Classifier)")
        
        st.subheader("Visualisasi Kinerja Model (Data Uji)")
        try:
            st.image('confusion_matrix_result.png', caption='Confusion Matrix pada Data Uji (Akurasi Model)')
        except FileNotFoundError:
            st.warning("Visualisasi Confusion Matrix belum tersedia. Mohon jalankan train_models.py.")
        st.markdown("---")

        st.info("Memprediksi label **SPAM** atau **HAM**.")
        
        col1_cls, col2_cls = st.columns([2, 1])

        with col1_cls:
            user_input = st.text_area("Masukkan Teks SMS di sini:", height=150, key='klasifikasi_input')
        
        with col2_cls:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("KLASIFIKASI SMS", type="primary", key='btn_cls'):
                if user_input:
                    with st.spinner('Sedang memproses dan mengklasifikasi...'):
                        processed_text = preprocess_text(user_input)
                        text_vector = vectorizer.transform([processed_text])
                        prediction = model.predict(text_vector)[0]
                        
                        st.markdown("---")
                        
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
                        st.table(pd.DataFrame(results))
                else:
                    st.warning("Mohon masukkan teks SMS.")

    # =========================================================
    # TAB 2: ANALISIS REGRESI LINEAR
    # =========================================================
    with tab2:
        st.header("2. Analisis Regresi Linear (Skor Spam)")
        
        st.subheader("Visualisasi Kinerja Model (Data Uji)")
        try:
            st.image('regresi_actual_vs_predicted.png', caption='Regresi Linear: Actual vs Predicted Plot')
        except FileNotFoundError:
            st.warning("Visualisasi Plot Regresi belum tersedia. Mohon jalankan train_models.py.")
        st.markdown("---")

        st.info("Memprediksi 'Skor Spam' (nilai kontinu 0.0 hingga 1.0).")
        
        user_input_reg = st.text_area("Masukkan Teks SMS (untuk Regresi):", height=100, key='reg_input')
        
        if st.button("PREDIKSI SKOR REGRESI", key='reg_button', type="primary"):
            if user_input_reg:
                with st.spinner('Menghitung Skor Regresi...'):
                    processed_text_reg = preprocess_text(user_input_reg)
                    text_vector_reg = vectorizer.transform([processed_text_reg])
                    
                    predicted_score = linear_reg_model.predict(text_vector_reg)[0]
                    predicted_score = max(0.0, min(1.0, predicted_score))
                    
                    st.markdown("---")
                    st.subheader("Hasil Prediksi Skor Regresi")
                    
                    if predicted_score >= 0.8:
                        color_text = "red"
                        st.error("Skor sangat tinggi. Pesan ini sangat mirip SPAM.")
                    elif predicted_score >= 0.5:
                        color_text = "orange"
                        st.warning("Skor berada di ambang batas. Pesan ini memiliki beberapa fitur SPAM.")
                    else:
                        color_text = "green"
                        st.success("Skor rendah. Pesan ini mirip dengan pola HAM (Normal).")
                        
                    st.markdown(f"**Skor Kemiripan SPAM (0.0 - 1.0):** <span style='color:{color_text}; font-size:24px;'>**{predicted_score:.4f}**</span>", unsafe_allow_html=True)
                    st.progress(predicted_score)
            else:
                st.warning("Mohon masukkan teks SMS untuk Regresi.")

    # =========================================================
    # TAB 3: CLUSTERING K-MEANS
    # =========================================================
    with tab3:
        st.header("3. Analisis Clustering SMS (K-Means)")
        
        st.subheader("Visualisasi Struktur Kluster (Data Latih)")
        try:
            st.image('kmeans_clusters_visualization.png', caption='Visualisasi Kluster K-Means (2D PCA)')
        except FileNotFoundError:
            st.warning("Visualisasi Kluster belum tersedia. Mohon jalankan train_models.py.")
        st.markdown("---")

        st.info("Mengelompokkan pesan ke dalam 3 Kluster (K=3).")

        user_input_cluster = st.text_area("Masukkan Teks SMS untuk dikelompokkan:", height=150, key='cluster_input')

        show_diag = st.checkbox("Tampilkan Detail Teknis (Sparsity)", key='diag_check')

        if st.button("TENTUKAN KLUSTER K-MEANS", type="secondary", key='btn_cluster'):
            if not user_input_cluster:
                st.warning("Mohon masukkan teks SMS untuk Clustering.")
            else:
                with st.spinner('Menentukan kluster...'):
                    processed_text_cluster = preprocess_text(user_input_cluster)
                    text_vector_cluster = vectorizer.transform([processed_text_cluster])
                    cluster_label = kmeans_model.predict(text_vector_cluster)[0]
                    
                    st.markdown("---")
                    st.subheader("Hasil Clustering K-Means")
                    
                    # --- MAPPING FINAL YANG KONSISTEN DENGAN HASIL KEYWORDS TERAKHIR ---
                    if cluster_label == 0:
                        # Kluster 0: hadiah, pin, menang, klik --> SPAM Hadiah
                        nama_kluster = "Kluster SPAM Hadiah / Kode / Link (Hard Spam)"
                        display_func = st.error
                    elif cluster_label == 1:
                        # Kluster 1: Tersisa (diasumsikan Komersial)
                        nama_kluster = "Kluster SPAM Komersial / Promosi Layanan"
                        display_func = st.warning
                    elif cluster_label == 2:
                        # Kluster 2: nama, yg, ga, jam, besok --> Pesan Normal/Percakapan
                        nama_kluster = "Kluster Pesan Normal / Percakapan"
                        display_func = st.success
                    else:
                        nama_kluster = f"Kluster {cluster_label} (Tidak Terdefinisi)"
                        display_func = st.info
                    
                    display_func(f"Kluster #{cluster_label}: {nama_kluster}")

                    if cluster_keywords:
                        st.markdown("---")
                        st.subheader("💡 Mengapa Masuk Kluster Ini?")
                        keywords = cluster_keywords.get(cluster_label, [])
                        st.write("Pesan ini masuk ke kluster ini karena kesamaan linguistik.")
                        st.markdown(f"**10 Kata Kunci Paling Dominan di Kluster #{cluster_label} adalah:**")
                        st.code(', '.join(keywords))
                        
                        # DIAGNOSIS TAMBAHAN
                        if show_diag:
                            with st.expander("Detail Teknis Vektor Input"):
                                st.caption(f"Teks Input setelah diproses: `{processed_text_cluster}`")
                                
                                num_nonzero = text_vector_cluster.nnz
                                total_features = text_vector_cluster.shape[1]
                                
                                st.text(f"  - Jumlah Fitur Non-Zero (Kata yang Dikenali): {num_nonzero}")
                                st.text(f"  - Total Dimensi Model: {total_features}")

                                if num_nonzero <= 5:
                                    st.error("⚠️ Vektor input sangat SPARSITY. Kluster seringkali default ke Centroid terpadat.")
                                else:
                                    st.success("Vektor input memiliki kepadatan yang wajar.")


if __name__ == '__main__':
    main()