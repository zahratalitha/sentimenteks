import streamlit as st
import numpy as np
import re
import zipfile
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# -------------------------------
# Judul Aplikasi
# -------------------------------
st.set_page_config(page_title="Sentimen Teks Indonesia", page_icon="🧠")
st.title("🧠 Sentimen Teks Indonesia")

# -------------------------------
# Download & Load Model + Tokenizer
# -------------------------------
REPO_ID = "zahratalitha/teks"
MODEL_FILE = "sentiment_model.h5"
TOKENIZER_ZIP = "tokenizer.zip"
TOKENIZER_DIR = "tokenizer"

@st.cache_resource
def load_model_and_tokenizer():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, repo_type="model")
    tok_zip = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_ZIP, repo_type="model")
    with zipfile.ZipFile(tok_zip, "r") as zip_ref:
        zip_ref.extractall(TOKENIZER_DIR)

    model = keras.models.load_model(model_path, custom_objects={"TFOpLambda": lambda x, **kwargs: x})
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# -------------------------------
# Pembersihan teks sederhana
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# -------------------------------
# Label Mapping
# -------------------------------
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT"
}

# -------------------------------
# Prediksi
# -------------------------------
def predict(text):
    clean = clean_text(text)
    enc = tokenizer(clean, truncation=True, padding="max_length", max_length=128, return_tensors="np")
    enc = {k: v.astype("int32") for k, v in enc.items()}
    preds = model.predict(enc, verbose=0)
    label_id = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return id2label[label_id], confidence, clean

# -------------------------------
# UI Input
# -------------------------------
user_text = st.text_area("Masukkan teks untuk analisis sentimen:", height=120)

if st.button("Prediksi"):
    if user_text.strip():
        label, score, cleaned = predict(user_text)
        st.success(f"Label: **{label}** ({score:.2%})")
        st.caption(f"Teks bersih: `{cleaned}`")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")
