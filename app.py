import streamlit as st
import numpy as np
import re
import zipfile

import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

st.set_page_config(page_title="Analisis Sentimen Kasus Tom Lembong", page_icon="🧠")
st.title("🧠 Analisis Sentimen Komentar Kasus Tom Lembong")

REPO_ID = "zahratalitha/teks"
MODEL_FILE = "sentiment_model.h5"
TOKENIZER_ZIP = "tokenizer.zip"
TOKENIZER_DIR = "tokenizer" 

@st.cache_resource
def load_model_and_tokenizer():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, repo_type="model")
    tok_zip = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_ZIP, repo_type="model")

    # Extract tokenizer
    with zipfile.ZipFile(tok_zip, "r") as zip_ref:
        zip_ref.extractall(TOKENIZER_DIR)

    # Load model dengan custom_objects
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "TFOpLambda": tf.identity,
        },
        compile=False,
        safe_mode=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT"
}

# -------------------------------
# Fungsi prediksi
# -------------------------------
# -------------------------------
# Fungsi prediksi
# -------------------------------
def predict(text):
    # Tokenisasi teks → dict dengan input_ids dan attention_mask
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"   # pakai numpy biar kompatibel dengan keras
    )

    # pastikan cast ke int32 (keras butuh ini)
    tokens = {k: v.astype("int32") for k, v in tokens.items()}

    # prediksi
    preds = model.predict(tokens, verbose=0)

    # ambil label
    label_id = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return id2label[label_id], confidence

# -------------------------------
# Input User
# -------------------------------
user_text = st.text_area("Masukkan teks untuk analisis sentimen:", height=120)

# Tambahan: pilihan komentar contoh
examples = [
    "",
    "Tom Lembong dituding melakukan pelanggaran, publik merasa kecewa dengan sikapnya.",
    "Saya mendukung Tom Lembong karena beliau jujur.",
    "Publik marah besar atas tindakan yang dilakukan.",
    "Masih ada harapan agar kasus ini diselesaikan dengan baik.",
]

selected_example = st.selectbox("Atau pilih komentar contoh:", examples)

# Gunakan teks dari input atau contoh
final_text = user_text.strip() if user_text.strip() else selected_example

if st.button("Prediksi"):
    if final_text:
        label, score = predict(final_text)
        st.success(f"Label: **{label}** ({score:.2%})")
        st.caption(f"Teks: `{final_text}`")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")

