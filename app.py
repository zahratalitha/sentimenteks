import streamlit as st
import numpy as np
import re
import zipfile

import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

st.set_page_config(page_title="Sentimen Teks Indonesia", page_icon="🧠")
st.title("🧠 Sentimen Teks Indonesia")

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
            "TFBertModel": TFBertModel,
        },
        compile=False,
        safe_mode=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return model, tokenizer

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
        return_tensors="np"   # pakai numpy biar kompatibel dengan keras
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

if st.button("Prediksi"):
    if user_text.strip():
        label, score = predict(user_text)
        st.success(f"Label: **{label}** ({score:.2%})")
        st.caption(f"Teks: `{user_text}`")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")

# -------------------------------
# Contoh Kasus Tom Lembong
# -------------------------------
example = "Tom Lembong dituding melakukan pelanggaran, publik merasa kecewa dengan sikapnya."
if st.button("Coba dengan kasus Tom Lembong"):
    label, score = predict(example)
    st.info(f"Contoh teks: {example}")
    st.success(f"Label: **{label}** ({score:.2%})")



