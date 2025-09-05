# app.py
# 🧠 Sentimen Teks Indonesia (HF Tokenizer + TensorFlow) — tanpa os.rename

import streamlit as st
import numpy as np
import pandas as pd
import re
import zipfile
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ================================
# Konfigurasi halaman
# ================================
st.set_page_config(
    page_title="Sentimen Teks Indonesia — HuggingFace + TensorFlow",
    page_icon="🧠",
)
st.title("🧠 Sentimen Teks Indonesia (HF Tokenizer + TensorFlow)")
st.caption("Model: TensorFlow (.h5) + Tokenizer HuggingFace — *tanpa* os.rename")

# ================================
# Konstanta repo & file
# ================================
REPO_ID = "zahratalitha/teks"
MODEL_FILE = "sentiment_model.h5"
TOKENIZER_ZIP = "tokenizer.zip"
TOKENIZER_DIR = Path("tokenizer")

# ================================
# Utilities
# ================================
def identity(x, *args, **kwargs):
    # Pengganti TFOpLambda agar tidak error: <lambda>() got unexpected kwarg 'name'
    return x

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # Download model (.h5) langsung gunakan path hasil hf_hub_download
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, repo_type="model")

    # Download & extract tokenizer
    tok_zip_path = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_ZIP, repo_type="model")
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tok_zip_path, "r") as zf:
        zf.extractall(TOKENIZER_DIR)

    # Load model (custom_objects menangani TFOpLambda dlsb)
    custom_objects = {"TFOpLambda": identity}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    # Load tokenizer dari folder hasil ekstraksi
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    return model, tokenizer

# Pembersihan teks (sesuaikan aturan)
important_mentions = {"tomlembong", "jokowi", "prabowo"}
important_hashtags = {"savetomlembong", "respect", "ripjustice", "justicefortomlembong"}

slang_dict = {
    "yg": "yang","ga": "tidak","gk": "tidak","ngga": "tidak","nggak": "tidak",
    "tdk": "tidak","dgn": "dengan","aja": "saja","gmn": "gimana","bgt": "banget",
    "dr": "dari","utk": "untuk","dlm": "dalam","tp": "tapi","krn": "karena"
}

def clean_text(text: str, normalize_slang: bool = True) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    # hapus url
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # filter mention & hashtag hanya yang penting
    def mention_repl(m):
        u = m.group(1)
        return u if u in important_mentions else ""
    text = re.sub(r"@(\w+)", mention_repl, text)

    def hashtag_repl(m):
        h = m.group(1)
        return h if h in important_hashtags else ""
    text = re.sub(r"#(\w+)", hashtag_repl, text)

    # sisakan huruf/angka/spasi
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    if normalize_slang:
        tokens = text.split()
        tokens = [slang_dict.get(tok, tok) for tok in tokens]
        text = " ".join(tokens)

    # rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()
    return text

# label mapping (urut sesuai model)
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT",
}

def encode_inputs(tokenizer, text: str, max_length: int = 128):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    # pastikan dtype int32 untuk TensorFlow
    enc["input_ids"] = enc["input_ids"].astype("int32")
    if "attention_mask" in enc:
        enc["attention_mask"] = enc["attention_mask"].astype("int32")
    if "token_type_ids" in enc:
        enc["token_type_ids"] = enc["token_type_ids"].astype("int32")
    return enc

def predict(model, tokenizer, text: str):
    cleaned = clean_text(text)
    if not cleaned:
        return None, None, cleaned

    enc = encode_inputs(tokenizer, cleaned, max_length=128)

    # Deteksi bentuk input model
    input_names = list(getattr(model, "input_names", []) or [])
    uses_dict = "input_ids" in input_names or "attention_mask" in input_names or "token_type_ids" in input_names

    if uses_dict:
        feed = {}
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in enc:
                feed[k] = enc[k]
        preds = model.predict(feed, verbose=0)
    else:
        # fallback: asumsikan single input_ids
        preds = model.predict(enc["input_ids"], verbose=0)

    # Pastikan bentuk (batch, num_labels)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if preds.ndim == 1:
        preds = preds[None, :]

    probs = preds[0]
    label_id = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = id2label.get(label_id, str(label_id))
    return label, confidence, cleaned

# ================================
# UI
# ================================
with st.spinner("Memuat model & tokenizer dari HuggingFace Hub…"):
    model, tokenizer = load_model_and_tokenizer()

col1, col2 = st.columns([3, 1])
with col1:
    user_text = st.text_area(
        "Tulis teks yang ingin dianalisis:",
        height=140,
        placeholder="Contoh: Saya sangat kecewa dengan keputusan ini",
    )
with col2:
    st.markdown("**Contoh cepat:**")
    if st.button("😠 Kecewa"):
        user_text = "Saya sangat kecewa dengan keputusan ini"
    if st.button("💪 Dukung"):
        user_text = "Tetap semangat, kami mendukungmu!"
    if st.button("😿 Sedih"):
        user_text = "Hari ini terasa sangat menyedihkan"

if st.button("Prediksi Sentimen"):
    if not user_text or not user_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        label, score, cleaned = predict(model, tokenizer, user_text)
        if label is None:
            st.error("Teks kosong setelah dibersihkan. Coba teks lain.")
        else:
            st.subheader("Hasil Prediksi")
            st.write(f"**Label:** `{label}`")
            st.write(f"**Confidence:** `{score:.2%}`")
            with st.expander("Lihat teks yang sudah dibersihkan"):
                st.code(cleaned)

st.markdown("---")
st.caption(
    "Tips: Jika muncul error terkait `TFOpLambda`/`Lambda`, kita sudah tangani dengan `custom_objects`. "
    "Jika model Anda memakai nama input berbeda, pastikan mapping di fungsi `predict` menyesuaikan."
)
