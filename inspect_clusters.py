import pandas as pd
import pickle

# --- 1. Load Data Awal ---
FILE_PATH = 'data/sms_spam_preprocessed.csv' 
df_train = pd.read_csv(FILE_PATH)

# --- 2. Load Model K-Means ---
try:
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
except FileNotFoundError:
    print("ERROR: kmeans_model.pkl tidak ditemukan.")
    exit()

# --- 3. Prediksi Label Kluster untuk Data Training ---
# Kita harus memprediksi ulang karena K-Means tidak disimpan bersama Vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("ERROR: tfidf_vectorizer.pkl tidak ditemukan.")
    exit()

X_tfidf = vectorizer.transform(df_train['processed_text'])
df_train['cluster_label'] = kmeans_model.predict(X_tfidf)


# --- 4. Tampilkan Contoh Isi Setiap Kluster ---
print("\n=== HASIL INSPEKSI KONTEN SETIAP KLUSTER (K=3) ===")

for i in range(3):
    print(f"\n=======================================================")
    print(f"KLUSTER #{i}:")
    
    # Filter data untuk kluster ke-i
    cluster_data = df_train[df_train['cluster_label'] == i]
    
    # Tampilkan 5 contoh SMS teratas
    print(f"Total Data: {len(cluster_data)}")
    print(f"Contoh 5 Pesan Teratas:")
    
    # Tampilkan original text atau processed text
    for j, row in cluster_data[['Kategori', 'processed_text']].head(5).iterrows():
        print(f"  [{row['Kategori'].upper()}]: {row['processed_text']}")

    # Tampilkan distribusi Kategori (Spam/Ham)
    distribusi = cluster_data['Kategori'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    print(f"\nDistribusi Kategori:")
    print(distribusi)

print("\n=======================================================")