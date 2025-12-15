import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import seaborn as sns

# =========================================================
# BAGIAN 1: SETUP DAN LOAD DATA
# =========================================================

# Buat folder untuk visualisasi
if not os.path.exists('image_modelling'):
    os.makedirs('image_modelling')

try:
    nltk.data.find('corpora/stopwords') 
except LookupError:
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_id = set(stopwords.words('indonesian'))

# *** FILE PATH DAN KOLOM YANG BENAR ***
FILE_PATH = 'data/sms_spam_preprocessed.csv' # File harus berada di direktori yang sama
TEXT_COLUMN = 'processed_text' 
LABEL_COLUMN = 'Kategori'      

try:
    df = pd.read_csv(FILE_PATH)
    X_text = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]
    print(f"Dataset berhasil dimuat. Total data: {len(df)}")
except Exception as e:
    print(f"FATAL ERROR: Gagal memuat data! Pastikan '{FILE_PATH}' ada di direktori yang sama. Error: {e}")
    exit()

# =========================================================
# BAGIAN 2: TRAINING MODEL (KLASIFIKASI & CLUSTERING)
# =========================================================

# 1. PELATIHAN VECTORIZER (TF-IDF)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_text)
print("\nVectorizer TF-IDF berhasil dilatih.")

# 2. PELATIHAN ENSEMBLE MODEL (KLASIFIKASI)
voting_clf = VotingClassifier(
    estimators=[
        ('mnb', MultinomialNB()), 
        ('lr', LogisticRegression(random_state=42, solver='liblinear')), 
        ('svc', LinearSVC(random_state=42))
    ],
    voting='hard'  
)
voting_clf.fit(X_tfidf, y)
print("Voting Classifier (Ensemble) berhasil dilatih.")

# 3. PELATIHAN K-MEANS (CLUSTERING)
K = 3 
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto') 
kmeans.fit(X_tfidf)
print(f"Model K-Means (K={K}) berhasil dilatih.")

# Tambahkan impor evaluasi
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns # Sudah diimpor di atas

# 1. PEMISAHAN DATA (Diulang di sini agar X_test tersedia untuk evaluasi)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 2. PREDIKSI
y_pred = voting_clf.predict(X_test)

print("\n--- Evaluasi Klasifikasi (Voting Classifier) ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 3. BUAT CONFUSION MATRIX (Visualisasi Standar)
cm = confusion_matrix(y_test, y_pred, labels=voting_clf.classes_)
cm_df = pd.DataFrame(cm, 
                     index = voting_clf.classes_, 
                     columns = voting_clf.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Reds', cbar=False)
plt.title('Confusion Matrix: Voting Classifier')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Simpan Gambar Visualisasi
CM_PATH = 'image_modelling/classification_confusion_matrix.png'
plt.savefig(CM_PATH)
plt.close()

print(f"Visualisasi Confusion Matrix berhasil disimpan di: {CM_PATH}")

# --- Tambahan: Ekstraksi Kata Kunci Kluster ---
print("\n[DEBUG] Mengekstrak 10 kata kunci teratas untuk setiap Kluster...")
# Dapatkan fitur/kata-kata dari vectorizer
feature_names = vectorizer.get_feature_names_out()

# Dapatkan pusat kluster (centroids)
cluster_centroids = kmeans.cluster_centers_

# Dictionary untuk menyimpan kata kunci kluster
cluster_keywords = {}

for i in range(kmeans.n_clusters):
    # Dapatkan bobot fitur untuk kluster ke-i
    centroid = cluster_centroids[i]
    
    # Dapatkan 10 indeks fitur dengan bobot tertinggi
    top_10_indices = centroid.argsort()[-10:][::-1]
    
    # Dapatkan nama fitur (kata-kata) dari indeks tersebut
    top_10_keywords = [feature_names[j] for j in top_10_indices]
    
    cluster_keywords[i] = top_10_keywords
    print(f"  Kluster #{i}: {', '.join(top_10_keywords)}")

# --- SIMPAN KATA KUNCI KLUSTER ---
with open('cluster_keywords.pkl', 'wb') as f:
    pickle.dump(cluster_keywords, f)
print("-> cluster_keywords.pkl berhasil disimpan.")

# =========================================================
# BAGIAN 3: SIMPAN SEMUA ASSET (.pkl)
# =========================================================

with open('voting_classifier_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print("-> voting_classifier_model.pkl berhasil disimpan.")

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("-> tfidf_vectorizer.pkl berhasil disimpan.")

with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)
print("-> kmeans_model.pkl berhasil disimpan.")


# =========================================================
# BAGIAN 4: VISUALISASI PCA (UNTUK PRESENTASI)
# =========================================================
print("\nMembuat Visualisasi Clustering...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray()) 

df_pca = pd.DataFrame(data=X_pca, columns=['PC_1', 'PC_2'])
df_pca['Cluster'] = kmeans.labels_ 

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC_1', y='PC_2', hue='Cluster', data=df_pca, palette='viridis', style='Cluster', s=70)
plt.savefig('image_modelling/sms_clustering_visualization.png')
print("Visualisasi Clustering berhasil disimpan.")


print("\n--- SEMUA MODEL BERHASIL DILATIH & DISIMPAN. SIAP DEPLOY ---")