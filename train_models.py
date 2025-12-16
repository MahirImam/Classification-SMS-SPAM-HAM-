import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Import Model Klasifikasi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import Model Regresi
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import Model Clustering & Dimensi Reduksi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA # Digunakan untuk Visualisasi

# Import untuk Visualisasi Gambar
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI FILE ---
FILE_PATH = 'data/sms_spam_preprocessed.csv'
LABEL_COLUMN = 'Kategori' 
TEXT_COLUMN = 'processed_text'


# --- 1. FUNGSI PREPROCESSING DASAR ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    try:
        tokens = [word for word in word_tokenize(text) if word.isalpha()]
    except LookupError:
        tokens = [word for word in text.split() if word.isalpha()]
        
    return " ".join(tokens)

# --- 2. SETUP STOPWORDS ---
try:
    stop_words_id = set(stopwords.words('indonesian'))
    stop_words_en = set(stopwords.words('english'))
    custom_stopwords = set(['wkwkwk', 'gpp', 'ok', 'ya', 'nih', 'deh', 'aja', 'loh', 'loh', 'sih', 'dong', 'yuk', 'hehe', 'hihi', 'thnks', 'tq', 'assalamualaikum', 'salam'])
    final_stopwords = stop_words_id.union(stop_words_en).union(custom_stopwords)
except LookupError:
    print("NLTK Stopwords tidak ditemukan.")
    final_stopwords = set()


# --- 3. LOAD DATA DAN PREPARASI TARGET ---
print("--- START: Training dan Saving Model ---")
try:
    df = pd.read_csv(FILE_PATH)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').apply(preprocess_text)
    y_cls = df[LABEL_COLUMN] 
    df['target_score'] = df[LABEL_COLUMN].apply(lambda x: 1.0 if x == 'spam' else 0.0)
    y_reg = df['target_score']
    X_text = df[TEXT_COLUMN]
    print(f"Dataset berhasil dimuat. Total data: {len(df)} baris.")
    
except KeyError as e:
    print(f"Error saat memuat data: Kolom {e} tidak ditemukan. Cek kembali nama kolom di file CSV Anda.")
    exit()
except Exception as e:
    print(f"Error saat memuat data: {e}")
    exit()


# --- 4. VECTORIZATION (TF-IDF) ---
vectorizer = TfidfVectorizer(stop_words=list(final_stopwords), max_features=5000) 
X_tfidf = vectorizer.fit_transform(X_text)
print("Vectorizer TF-IDF berhasil dilatih.")


# --- 5. PEMISAHAN DATA ---
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_tfidf, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
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
y_pred_reg = linear_reg_model.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"  > Mean Squared Error (MSE): {mse:.4f}")
print(f"  > R2 Score: {r2:.4f}")

# VISUALISASI REGRESI: ACTUAL VS PREDICTED
plt.figure(figsize=(6, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--') 
plt.xlabel('Actual Score (0.0=Ham, 1.0=Spam)')
plt.ylabel('Predicted Score')
plt.title('Regresi Linear: Actual vs Predicted')
plt.savefig('regresi_actual_vs_predicted.png') 
plt.close()
print("-> regresi_actual_vs_predicted.png berhasil disimpan.")


# =========================================================
# BAGIAN B: PELATIHAN MODEL KLASIFIKASI (Voting Classifier)
# =========================================================
print("\n--- B. KLASIFIKASI (Voting Classifier) ---")
voting_clf = VotingClassifier(
    estimators=[
        ('mnb', MultinomialNB(alpha=1.0)), 
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('svc', LinearSVC(random_state=42, dual=False))
    ],
    voting='hard'  
)
voting_clf.fit(X_train_cls, y_train_cls)
y_pred_cls = voting_clf.predict(X_test_cls)
print(f"\n  > Akurasi (Total Test Data): {accuracy_score(y_test_cls, y_pred_cls):.4f}")

# VISUALISASI KLASIFIKASI: CONFUSION MATRIX
cm = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=voting_clf.classes_, yticklabels=voting_clf.classes_)
plt.title('Confusion Matrix: Voting Classifier')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_result.png')
plt.close()
print("-> confusion_matrix_result.png berhasil disimpan.")


# =========================================================
# BAGIAN C: PELATIHAN MODEL CLUSTERING (K-Means)
# =========================================================
print("\n--- C. CLUSTERING (K-Means) ---")
K_CLUSTERS = 3 
kmeans_model = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
kmeans_model.fit(X_tfidf)

# --- VISUALISASI CLUSTERING: PCA 2D ---
# Reduksi dimensi menggunakan PCA
pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_tfidf.toarray())
df_pca = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = kmeans_model.labels_.astype(str) # Tambahkan label kluster
df_pca['Original_Label'] = df[LABEL_COLUMN] # Untuk perbandingan

# Plotting
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="PCA1", y="PCA2",
    hue="Cluster", 
    palette=sns.color_palette("hsv", K_CLUSTERS),
    data=df_pca,
    legend="full",
    alpha=0.7
)
plt.title('K-Means Clustering Visualization (PCA 2D)')
plt.savefig('kmeans_clusters_visualization.png')
plt.close()
print("-> kmeans_clusters_visualization.png berhasil disimpan.")

# --- Dapatkan Kata Kunci Kluster (untuk Streamlit) ---
feature_names = vectorizer.get_feature_names_out()
order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
cluster_keywords = {}
for i in range(K_CLUSTERS):
    keywords = [feature_names[ind] for ind in order_centroids[i, :10]]
    cluster_keywords[i] = keywords


# --- 6. SIMPAN SEMUA MODEL DAN ASSET (.pkl) ---
print("\n--- SAVING MODELS ---")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("-> tfidf_vectorizer.pkl berhasil disimpan.")

with open('voting_classifier_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print("-> voting_classifier_model.pkl berhasil disimpan.")

with open('linear_reg_model.pkl', 'wb') as f:
    pickle.dump(linear_reg_model, f)
print("-> linear_reg_model.pkl berhasil disimpan.")

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)
print("-> kmeans_model.pkl berhasil disimpan.")

with open('cluster_keywords.pkl', 'wb') as f:
    pickle.dump(cluster_keywords, f)
print("-> cluster_keywords.pkl berhasil disimpan.")

print("\n--- SEMUA MODEL DAN VISUALISASI BERHASIL DILATIH DAN DISIMPAN ---")