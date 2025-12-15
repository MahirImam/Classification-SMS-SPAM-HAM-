import pandas as pd
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Import Model untuk Klasifikasi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier # Pengganti Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import Model untuk Regresi
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import Model untuk Clustering
from sklearn.cluster import KMeans

# --- KONFIGURASI FILE ---
FILE_PATH = 'data/sms_spam_preprocessed.csv' 
LABEL_COLUMN = 'Kategori' # <--- PERBAIKAN: Ganti 'label' dengan 'Kategori'
TEXT_COLUMN = 'processed_text'


# --- 1. FUNGSI PREPROCESSING DASAR ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Hapus karakter non-alfanumerik
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Ubah menjadi lowercase
    text = text.lower()
    # Tokenisasi dan hapus angka
    tokens = [word for word in word_tokenize(text) if word.isalpha()]
    return " ".join(tokens)

# --- 2. SETUP STOPWORDS ---
# Inisialisasi stopwords (contoh menggunakan gabungan Indonesia dan Inggris)
try:
    stop_words_id = set(stopwords.words('indonesian'))
    stop_words_en = set(stopwords.words('english'))
    final_stopwords = stop_words_id.union(stop_words_en)
except LookupError:
    # Jika NLTK belum diunduh, jalankan 'import nltk; nltk.download('stopwords')'
    # Jika masih error, gunakan set kosong atau list stopwords manual
    print("NLTK Stopwords tidak ditemukan. Gunakan list stopwords kosong.")
    final_stopwords = set()


# --- 3. LOAD DATA DAN PREPARASI TARGET ---
print("--- START: Training dan Saving Model ---")
try:
    df = pd.read_csv(FILE_PATH)
    
    # Cleaning data yang mungkin terlewat di pre-processing
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').apply(preprocess_text)
    
    # Memisahkan fitur dan target
    X_text = df[TEXT_COLUMN]
    
    # Target untuk Klasifikasi (string: 'ham', 'spam')
    y_cls = df[LABEL_COLUMN] 

    # Target untuk REGRESI (numerik: 0.0, 1.0)
    # Ini untuk memenuhi persyaratan tugas Regresi non-Logistik
    df['target_score'] = df[LABEL_COLUMN].apply(lambda x: 1.0 if x == 'spam' else 0.0)
    y_reg = df['target_score']
    
    print(f"Dataset berhasil dimuat. Total data: {len(df)} baris.")
    
except FileNotFoundError:
    print(f"Error: File '{FILE_PATH}' tidak ditemukan. Mohon cek path file.")
    exit()
except Exception as e:
    print(f"Error saat memuat data: {e}")
    exit()


# --- 4. VECTORIZATION (TF-IDF) ---
# **PENTING: X_tfidf harus didefinisikan sebelum train_test_split!**
vectorizer = TfidfVectorizer(stop_words=list(final_stopwords), max_features=5000) 
X_tfidf = vectorizer.fit_transform(X_text)
print("Vectorizer TF-IDF berhasil dilatih.")


# --- 5. PEMISAHAN DATA ---
# Pemisahan data untuk Klasifikasi
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_tfidf, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# Pemisahan data untuk Regresi
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_tfidf, y_reg, test_size=0.2, random_state=42, stratify=y_reg
)
print("Data berhasil dipisah (80% Training, 20% Testing) untuk Klasifikasi dan Regresi.")


# =========================================================
# BAGIAN A: PELATIHAN MODEL REGRESI (Linear Regression)
# =========================================================
print("\n--- A. REGRESI LINEAR ---")
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_reg, y_train_reg)
print("Model Regresi Linear berhasil dilatih.")

# Evaluasi Regresi
y_pred_reg = linear_reg_model.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"  > Mean Squared Error (MSE): {mse:.4f}")
print(f"  > R2 Score: {r2:.4f}")
print("   (Metrik ini harus dimasukkan ke Laporan Regresi)")


# =========================================================
# BAGIAN B: PELATIHAN MODEL KLASIFIKASI (Voting Classifier)
# =========================================================
print("\n--- B. KLASIFIKASI (Voting Classifier) ---")

# Ganti Logistic Regression dengan Decision Tree Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('mnb', MultinomialNB(alpha=1.0)), 
        ('dt', DecisionTreeClassifier(random_state=42)), 
        ('svc', LinearSVC(random_state=42, dual=False)) # dual=False untuk data besar
    ],
    voting='hard'  
)
voting_clf.fit(X_train_cls, y_train_cls)
print("Voting Classifier (Ensemble: MNB, DT, SVC) berhasil dilatih.")

# Evaluasi Klasifikasi
y_pred_cls = voting_clf.predict(X_test_cls)
print("\n  > Classification Report:")
print(classification_report(y_test_cls, y_pred_cls))
print(f"  > Akurasi (Total Test Data): {accuracy_score(y_test_cls, y_pred_cls):.4f}")
print(f"  > Confusion Matrix:\n{confusion_matrix(y_test_cls, y_pred_cls)}")


# =========================================================
# BAGIAN C: PELATIHAN MODEL CLUSTERING (K-Means)
# =========================================================
print("\n--- C. CLUSTERING (K-Means) ---")

# Kita asumsikan K=3 berdasarkan diskusi sebelumnya
K_CLUSTERS = 3 
kmeans_model = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
kmeans_model.fit(X_tfidf)
print(f"Model K-Means (K={K_CLUSTERS}) berhasil dilatih.")


# --- 6. SIMPAN SEMUA MODEL (.pkl) ---
print("\n--- SAVING MODELS ---")

# 1. Simpan Vectorizer TF-IDF
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("-> tfidf_vectorizer.pkl berhasil disimpan.")

# 2. Simpan Model Klasifikasi
with open('voting_classifier_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print("-> voting_classifier_model.pkl berhasil disimpan.")

# 3. Simpan Model Regresi
with open('linear_reg_model.pkl', 'wb') as f:
    pickle.dump(linear_reg_model, f)
print("-> linear_reg_model.pkl berhasil disimpan.")

# 4. Simpan Model Clustering
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)
print("-> kmeans_model.pkl berhasil disimpan.")

print("\n--- SEMUA MODEL BERHASIL DILATIH DAN DISIMPAN ---")