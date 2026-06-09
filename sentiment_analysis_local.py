import os
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import contractions
import spacy
from transformers import pipeline
from torch.utils.data import Dataset
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
FILE_INPUT_CSV = "dataset_komentar_mrbeast_sample.csv"
FILE_OUTPUT_CSV = "dataset_komentar_mrbeast_TERANALISIS.csv"

try:
    nlp_spacy = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except Exception:
    os.system("python -m spacy download en_core_web_sm")
    nlp_spacy = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

class KomentarDataset(Dataset):
    def __init__(self, data_list): 
        self.data_list = data_list
    def __len__(self): 
        return len(self.data_list)
    def __getitem__(self, idx): 
        return str(self.data_list[idx])

def bersihkan_teks_nlp(teks):
    if not isinstance(teks, str): return ""
    teks = teks.lower()
    teks = contractions.fix(teks)
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'(.)\1+', r'\1\1', teks)
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    
    doc = nlp_spacy(teks)
    teks = " ".join([token.lemma_ for token in doc if not token.is_space])
    return re.sub(r'\s+', ' ', teks).strip()

@st.cache_resource
def load_bert_classifier():
    device_id = 0 if torch.cuda.is_available() else -1
    NAMA_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    return pipeline(
        "sentiment-analysis", 
        model=NAMA_MODEL,
        device=device_id,
        use_auth_token=HF_TOKEN if HF_TOKEN else None
    )

def jalankan_analisis_ai_csv(df_mentah):
    st.info("Memulai pembersihan teks via NLP spaCy")
    df_mentah['komentar_clean'] = df_mentah['komentar_asli'].apply(bersihkan_teks_nlp)
    
    st.info("Menjalankan Batch Inferencing model AI BERT (Mohon tunggu sebentar)...")
    nlp_classifier = load_bert_classifier()
    
    dataset_generator = KomentarDataset(df_mentah['komentar_clean'].tolist())
    list_sentimen = []
    list_skor = []

    progress_bar = st.progress(0)
    total_data = len(df_mentah)
    
    for idx, hasil in enumerate(nlp_classifier(dataset_generator, batch_size=64, truncation=True)):
        try:
            label = hasil['label'].upper()
            skor = hasil['score']
            
            if skor < 0.65:
                label_final = "NEUTRAL"
            else:
                if "0" in label or "NEG" in label: 
                    label_final = "NEGATIVE"
                elif "1" in label or "POS" in label: 
                    label_final = "POSITIVE"
                else: 
                    label_final = "NEUTRAL"
            
            list_sentimen.append(label_final)
            list_skor.append(skor)
        except Exception:
            list_sentimen.append("NEUTRAL")
            list_skor.append(0.50)
            
        if idx % 10 == 0 or idx == total_data - 1:
            progress_bar.progress((idx + 1) / total_data)

    df_mentah['sentimen'] = list_sentimen
    df_mentah['skor_kepastian'] = list_skor
    
    df_mentah.to_csv(FILE_OUTPUT_CSV, index=False)
    st.success(f"Analisis tuntas! {total_data} baris data berhasil disimpan ke '{FILE_OUTPUT_CSV}'")
    return df_mentah

st.set_page_config(page_title="Sentiment Analyzer (Mr.Beast Comment)", layout="wide")
st.title("Dashboard Analisis Sentimen Komentar")
st.markdown("### Offline CSV Data Ingestion Model calibrated for MrBeast Dataset")

df_total = pd.DataFrame()

if os.path.exists(FILE_OUTPUT_CSV):
    st.sidebar.success("Terdeteksi file cache hasil analisis sebelumnya. Loading data...")
    df_total = pd.read_csv(FILE_OUTPUT_CSV)
elif os.path.exists(FILE_INPUT_CSV):
    st.sidebar.warning("File analisis belum ada. Memulai kalkulasi Model AI...")
    df_mentah = pd.read_csv(FILE_INPUT_CSV, skipinitialspace=True)
    
    kolom_teks_ditemukan = next((col for col in ['Komentar', 'komentar_asli', 'text', 'comment'] if col in df_mentah.columns), None)
    kolom_tanggal_ditemukan = next((col for col in ['Tanggal', 'tanggal', 'date', 'published_at'] if col in df_mentah.columns), None)
    kolom_penulis_ditemukan = next((col for col in ['Penulis', 'penulis', 'author', 'user'] if col in df_mentah.columns), None)

    if not kolom_teks_ditemukan or not kolom_tanggal_ditemukan:
        st.error("Gagal memetakan struktur kolom secara otomatis!")
        st.markdown(f"**Kolom di CSV Anda saat ini:** `{list(df_mentah.columns)}`")
    else:
        mapping_rename = {kolom_teks_ditemukan: 'komentar_asli', kolom_tanggal_ditemukan: 'tanggal'}
        if kolom_penulis_ditemukan:
            mapping_rename[kolom_penulis_ditemukan] = 'penulis'
        else:
            df_mentah['penulis'] = 'Anonymous'
            
        df_mentah = df_mentah.rename(columns=mapping_rename)
        df_total = jalankan_analisis_ai_csv(df_mentah)
else:
    st.error(f"File '{FILE_INPUT_CSV}' tidak ditemukan di folder proyek Anda! Harap letakkan file CSV tersebut terlebih dahulu.")

if not df_total.empty:
    df_total['tanggal_dt'] = pd.to_datetime(df_total['tanggal'], errors='coerce')
    df_total['jam_periode'] = df_total['tanggal_dt'].dt.strftime('%Y-%m-%d %H:00:00')
    df_ts = df_total.groupby(['jam_periode', 'sentimen']).size().reset_index(name='jumlah_komentar')

    total_rows = len(df_total)
    pos_rows = len(df_total[df_total['sentimen'] == 'POSITIVE'])
    neg_rows = len(df_total[df_total['sentimen'] == 'NEGATIVE'])
    neu_rows = len(df_total[df_total['sentimen'] == 'NEUTRAL'])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Komentar MrBeast", f"{total_rows:,} baris")
    m2.metric("Sentimen Positif", f"{pos_rows:,}", f"{(pos_rows/total_rows)*100:.1f}%")
    m3.metric("Sentimen Netral", f"{neu_rows:,}", f"{(neu_rows/total_rows)*100:.1f}%", delta_color="off")
    m4.metric("Sentimen Negatif", f"{neg_rows:,}", f"-{(neg_rows/total_rows)*100:.1f}%", delta_color="inverse")

    st.markdown("---")

    st.write("### Dinamika Perubahan Tren Sentimen Berbasis Time Series (Per Jam)")
    
    if not df_ts.empty and df_ts['jam_periode'].notna().any():
        df_pivot = df_ts.pivot(index='jam_periode', columns='sentimen', values='jumlah_komentar').fillna(0)
        
        for col in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            if col not in df_pivot.columns: 
                df_pivot[col] = 0

        plt.figure(figsize=(15, 5))
        sns.set_theme(style="whitegrid")
        
        plt.plot(df_pivot.index, df_pivot['POSITIVE'], marker='o', color='#2ecc71', linewidth=2.5, label='Positif')
        plt.plot(df_pivot.index, df_pivot['NEUTRAL'], marker='s', color='#f39c12', linewidth=2.5, label='Netral')
        plt.plot(df_pivot.index, df_pivot['NEGATIVE'], marker='x', color='#e74c3c', linewidth=2.5, label='Negatif')
        
        plt.title("Analisis Pergerakan Emosi Penonton MrBeast", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Volume Respon Netizen")
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        st.pyplot(plt)
    else:
        st.caption("Format kolom 'Tanggal' di dalam CSV tidak valid untuk pembagian runtun waktu (Time Series).")

    st.markdown("---")

    st.write("### Log Eksplorasi Data Komentar (Tabel Review)")
    st.dataframe(
        df_total[['penulis', 'komentar_asli', 'sentimen', 'skor_kepastian', 'tanggal']]
        .sort_values(by='tanggal', ascending=False),
        use_container_width=True
    )