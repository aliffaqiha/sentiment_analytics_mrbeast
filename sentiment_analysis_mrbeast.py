import os
import re
import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import contractions
import spacy
from googleapiclient.discovery import build
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
from tqdm import tqdm
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
URL_VIDEO = os.getenv("YOUTUBE_VIDEO_URL", "https://youtu.be/AaMdXZMvT3w") 

DB_NAME = r"D:\PROJECT_ALIF\SCRAPING (HOLD)\pipeline_analytics.db"  

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

def ekstrak_video_id(url):
    pola = r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/|e\/|watch\?v=|&v=)([^#\&\?]*)"
    pencarian = re.search(pola, url)
    return pencarian.group(1) if pencarian else url

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

def inisialisasi_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS komentar_youtube (
            id_komentar TEXT PRIMARY KEY,
            penulis TEXT,
            komentar_asli TEXT,
            komentar_clean TEXT,
            tanggal TEXT,
            sentimen TEXT,
            skor_kepastian REAL
        )
    """)
    conn.commit()
    conn.close()

def ambil_data_incremental(video_id):
    # --- PROSES AMBIL DATA OTOMATIS VIA YOUTUBE API ---
    if not YOUTUBE_API_KEY:
        st.sidebar.error("API Key YouTube tidak ditemukan di file .env")
        return pd.DataFrame()
        
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50,
            order="time", # Mengambil komentar terbaru yang masuk
            textFormat="plainText"
        )
        response = request.execute()
        
        data_komentar = []
        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            data_komentar.append({
                'id_komentar': item['id'],
                'penulis': snippet['authorDisplayName'],
                'komentar_asli': snippet['textDisplay'],
                'tanggal': snippet['publishedAt']
            })
            
        return pd.DataFrame(data_komentar)
        
    except Exception as e:
        # Deteksi otomatis jika API terkena limit kuota (error 403 / quotaExceeded)
        if "quota" in str(e).lower() or "limit" in str(e).lower() or "403" in str(e):
            st.sidebar.warning("API YouTube mencapai limit! Otomatis beralih ke Mode Lokal + Cache Terakhir.")
        else:
            st.sidebar.error(f"Gagal mengambil data dari API: {e}")
        return pd.DataFrame()

def proses_analisa_ai_batch(df_baru):
    if df_baru.empty: return

    # --- PROSES CLEANING TEKS VIA SPACY ---
    df_baru['komentar_clean'] = df_baru['komentar_asli'].apply(bersihkan_teks_nlp)
    
    device_id = 0 if torch.cuda.is_available() else -1
    NAMA_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    nlp_classifier = pipeline(
        "sentiment-analysis", 
        model=NAMA_MODEL,
        device=device_id,
        use_auth_token=HF_TOKEN if HF_TOKEN else None
    )

    dataset_generator = KomentarDataset(df_baru['komentar_clean'].tolist())
    list_sentimen = []
    list_skor = []

    # --- PROSES INFERENSI MODEL AI BERT ---
    for hasil in nlp_classifier(dataset_generator, batch_size=64, truncation=True):
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

    df_baru['sentimen'] = list_sentimen
    df_baru['skor_kepastian'] = list_skor

    # --- PROSES FILTER DUPLIKASI ID ---
    conn = sqlite3.connect(DB_NAME)
    id_tercatat_db = pd.read_sql("SELECT id_komentar FROM komentar_youtube", conn)['id_komentar'].tolist()
    
    # Hanya masukkan data komentar baru yang ID-nya belum pernah ada di database
    df_final_bersih = df_baru[~df_baru['id_komentar'].isin(id_tercatat_db)]

    if not df_final_bersih.empty:
        df_final_bersih.to_sql("komentar_youtube", conn, if_exists="append", index=False)
        st.sidebar.success(f"Berhasil menganalisis dan menyimpan {len(df_final_bersih)} komentar baru.")
    else:
        st.sidebar.info("Tidak ada komentar baru yang unik.")
        
    conn.commit()
    conn.close()

st.set_page_config(page_title="AI Sentiment Tracker Pro", layout="wide")
st.title("Dashboard Analisis Sentimen Real-Time dan Automated Pipeline")
st.markdown("### Powered by Advanced Pandas Engine and Deep Learning BERT Architecture")

st_autorefresh(interval=60000, key="datarefresh_ticker")

inisialisasi_database()
video_id_target = ekstrak_video_id(URL_VIDEO)

st.sidebar.markdown("### Manifest Sistem API:")
st.sidebar.text(f"ID Video Terdeteksi: {video_id_target}")
st.sidebar.text(f"Database Terkunci: {DB_NAME}")

df_incremental = pd.DataFrame()
try:
    with st.spinner("Mengecek komentar baru dari YouTube API..."):
        df_incremental = ambil_data_incremental(video_id_target)
        if not df_incremental.empty:
            proses_analisa_ai_batch(df_incremental)
except (KeyboardInterrupt, Exception) as e:
    st.warning(f"Gangguan pipeline: {e}")

# --- PROSES SINKRONISASI DATA DAN NORMALISASI TANGGAL ---
conn = sqlite3.connect(DB_NAME)
try:
    df_total = pd.read_sql("SELECT * FROM komentar_youtube ORDER BY tanggal ASC", conn)
except Exception:
    df_total = pd.DataFrame()
conn.close()

if not df_total.empty:
    df_total['tanggal_dt'] = pd.to_datetime(df_total['tanggal'], errors='coerce')
    df_total['jam_periode'] = df_total['tanggal_dt'].dt.strftime('%Y-%m-%d %H:00:00')
    
    # --- PROSES AGREGASI DATA TIME SERIES ---
    df_ts = df_total.groupby(['jam_periode', 'sentimen']).size().reset_index(name='jumlah_komentar')
else:
    df_ts = pd.DataFrame()

if df_total.empty:
    st.info("Database lokal kosong. Menunggu data pertama dari YouTube API masuk.")
else:
    total_rows = len(df_total)
    pos_rows = len(df_total[df_total['sentimen'] == 'POSITIVE'])
    neg_rows = len(df_total[df_total['sentimen'] == 'NEGATIVE'])
    neu_rows = len(df_total[df_total['sentimen'] == 'NEUTRAL'])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Komentar (Unik)", f"{total_rows:,} baris")
    m2.metric("Sentimen Positif", f"{pos_rows:,}", f"{(pos_rows/total_rows)*100:.1f}%")
    m3.metric("Sentimen Netral", f"{neu_rows:,}", f"{(neu_rows/total_rows)*100:.1f}%", delta_color="off")
    m4.metric("Sentimen Negatif", f"{neg_rows:,}", f"-{(neg_rows/total_rows)*100:.1f}%", delta_color="inverse")

    st.markdown("---")

    st.write("### Dinamika Perubahan Tren Sentimen Berbasis Time Series (Per Jam)")
    
    if not df_ts.empty:
        df_pivot = df_ts.pivot(index='jam_periode', columns='sentimen', values='jumlah_komentar').fillna(0)
        
        for col in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            if col not in df_pivot.columns: 
                df_pivot[col] = 0

        plt.figure(figsize=(15, 5))
        sns.set_theme(style="whitegrid")
        
        plt.plot(df_pivot.index, df_pivot['POSITIVE'], marker='o', color='#2ecc71', linewidth=2.5, label='Positif')
        plt.plot(df_pivot.index, df_pivot['NEUTRAL'], marker='s', color='#f39c12', linewidth=2.5, label='Netral')
        plt.plot(df_pivot.index, df_pivot['NEGATIVE'], marker='x', color='#e74c3c', linewidth=2.5, label='Negatif')
        
        plt.title("Pergerakan Emosi Publik (Real-time Stream)", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Volume Respon Netizen")
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        st.pyplot(plt)
    else:
        st.caption("Data belum mencukupi untuk membuat grafik waktu.")

    st.markdown("---")

    st.write("### Log Audit Sinkronisasi Data (Live Feed dari Database SQL)")
    st.dataframe(
        df_total[['id_komentar', 'penulis', 'komentar_asli', 'sentimen', 'tanggal']]
        .tail(100)
        .sort_values(by='tanggal', ascending=False),
        use_container_width=True
    )