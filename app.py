import streamlit as st
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from PIL import Image
import zipfile
import pandas as pd
import re

# ================================
# Download model & tokenizer
# ================================
REPO_ID = "zahratalitha/teks"
MODEL_FILE = "sentiment_model.h5"
TOKENIZER_ZIP = "tokenizer.zip"
TOKENIZER_DIR = "tokenizer"

# Download model h5 (langsung pakai path, tanpa os.rename)
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, repo_type="model")

# Download & extract tokenizer
tok_zip = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_ZIP, repo_type="model")
with zipfile.ZipFile(tok_zip, "r") as zip_ref:
    zip_ref.extractall(TOKENIZER_DIR)

# ================================
# Load model & tokenizer
# ================================
custom_objects = {"TFOpLambda": lambda x: x}
model = keras.models.load_model(model_path, custom_objects=custom_objects)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

print("Model inputs:", model.input_names)

# ================================
# Preprocessing
# ================================
important_mentions = ["tomlembong", "jokowi", "prabowo"]
important_hashtags = ["savetomlembong", "respect", "ripjustice", "justicefortomlembong"]

slang_dict = {
    "yg": "yang","ga": "tidak","gk": "tidak","ngga": "tidak","nggak": "tidak",
    "tdk": "tidak","dgn": "dengan","aja": "saja","gmn": "gimana","bgt": "banget",
    "dr": "dari","utk": "untuk","dlm": "dalam","tp": "tapi","krn": "karena"
}

def clean_text(text, normalize_slang=True):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    def mention_repl(match):
        mention = match.group(1)
        return mention if mention in important_mentions else ""
    text = re.sub(r"@(\w+)", mention_repl, text)

    def hashtag_repl(match):
        hashtag = match.group(1)
        return hashtag if hashtag in important_hashtags else ""
    text = re.sub(r"#(\w+)", hashtag_repl, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    if normalize_slang:
        tokens = text.split()
        tokens = [slang_dict.get(tok, tok) for tok in tokens]
        text = " ".join(tokens)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================================
# Label mapping
# ================================
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT"
}

# ================================
# Test prediction
# ================================
def predict(text, mode="dict"):
    clean = clean_text(text)
    enc = tokenizer(clean, truncation=True, padding="max_length", max_length=128, return_tensors="np")
    if mode == "dict":
        preds = model.predict(
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
            verbose=0
        )
    else:
        preds = model.predict(enc["input_ids"], verbose=0)
    label_id = preds.argmax(axis=1)[0]
    confidence = float(preds.max())
    return id2label[label_id], confidence, clean

# contoh test
texts = [
    "@user1 Gk bgt deh! #savetomlembong dukung terus bro!",
    "Saya sangat kecewa dengan keputusan ini",
    "Tetap semangat, kami mendukungmu",
]

for t in texts:
    print("\nTeks asli:", t)
    try:
        label, score, cleaned = predict(t, mode="dict")
    except Exception:
        label, score, cleaned = predict(t, mode="ids")
    print("Bersih :", cleaned)
    print("Prediksi:", label, "| Confidence:", score)
