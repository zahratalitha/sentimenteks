import streamlit as st
import numpy as np
import zipfile
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# -------------------------------
# Konfigurasi dasar
# -------------------------------
st.set_page_config(page_title="Analisis Sentimen Kasus Tom Lembong", page_icon="🧠")
st.title("🧠 Analisis Sentimen Komentar Kasus Tom Lembong")

REPO_ID = "zahratalitha/teks"
MODEL_FILE = "sentiment_model.h5"          # H5 yang berisi model Keras
TOKENIZER_ZIP = "tokenizer.zip"            # ZIP berisi tokenizer HF
TOKENIZER_DIR = "tokenizer"                # Folder ekstraksi tokenizer
SEQ_LEN = 128

id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT",
}

# -------------------------------
# Loader: Model & Tokenizer (cache)
# -------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    # Unduh file dari HF Hub
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, repo_type="model")
    tok_zip = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_ZIP, repo_type="model")

    # Ekstrak tokenizer
    with zipfile.ZipFile(tok_zip, "r") as zf:
        zf.extractall(TOKENIZER_DIR)

    # Muat model Keras (tanpa compile)
    model = keras.models.load_model(
        model_path,
        custom_objects={  # TFOpLambda sering muncul di model H5
            "TFOpLambda": tf.identity,
        },
        compile=False,
        safe_mode=False,
    )

    # Muat tokenizer HF dari folder lokal
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Info debug di sidebar: nama input layer model
with st.sidebar:
    st.subheader("Debug")
    try:
        st.caption(f"Model input names: {getattr(model, 'input_names', [])}")
    except Exception:
        st.caption("Model input names: (tidak terbaca)")
def build_inputs_for_model(enc: dict, expected_names: list[str]) -> dict:
    enc_np = {k: enc[k].astype("int32") for k in enc}  # pastikan int32
    inputs = {}

    # alias umum
    def get_from_alias(name: str):
        # kembalikan array yang paling cocok untuk 'name'
        if name in enc_np:
            return enc_np[name]
        if name in ("input_word_ids", "word_ids", "ids", "input_ids_1"):
            return enc_np.get("input_ids", None)
        if name in ("input_mask", "mask", "attention_mask_1"):
            return enc_np.get("attention_mask", None)
        if name in ("segment_ids", "token_type_ids", "type_ids"):
            # jika tokenizer tidak memberi token_type_ids, isi nol
            if "token_type_ids" in enc_np:
                return enc_np["token_type_ids"]
            elif "input_ids" in enc_np:
                return np.zeros_like(enc_np["input_ids"], dtype=np.int32)
        return None

    if not expected_names:
        # fallback umum: coba kirim input_ids dan attention_mask
        inputs = {}
        if "input_ids" in enc_np:
            inputs["input_ids"] = enc_np["input_ids"]
        if "attention_mask" in enc_np:
            inputs["attention_mask"] = enc_np["attention_mask"]
        return inputs

    for name in expected_names:
        val = get_from_alias(name)
        if val is not None:
            inputs[name] = val
        # kalau None, biarkan saja (kadang layer opsional)

    # Jika masih kosong, fallback minimal ke input_ids
    if not inputs and "input_ids" in enc_np:
        inputs = {"input_ids": enc_np["input_ids"]}
        if "attention_mask" in enc_np:
            inputs["attention_mask"] = enc_np["attention_mask"]

    return inputs

# -------------------------------
# Prediksi
# -------------------------------
def predict(text: str):
    if not text or not text.strip():
        raise ValueError("Teks kosong.")

    # Tokenisasi → TensorFlow tensors
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )

    # Konversi ke numpy int32 (biar keras.predict bisa terima)
    enc_np = {k: v.numpy().astype("int32") for k, v in enc.items()}

    expected = list(getattr(model, "input_names", []))
    inputs = build_inputs_for_model(enc_np, expected)

    preds = model.predict(inputs, verbose=0)

    # Normalisasi output
    probs = np.array(preds)
    if probs.ndim == 1:
        probs = probs[None, :]

    label_id = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs))
    return id2label.get(label_id, f"CLASS_{label_id}"), confidence

# -------------------------------
# UI
# -------------------------------
user_text = st.text_area("Masukkan teks komentar:", height=140)

examples = [
    "",
    "Tom Lembong dituding melakukan pelanggaran, publik merasa kecewa dengan sikapnya.",
    "Saya mendukung Tom Lembong karena beliau jujur.",
    "Publik marah besar atas tindakan yang dilakukan.",
    "Masih ada harapan agar kasus ini diselesaikan dengan baik.",
]
selected_example = st.selectbox("Atau pilih komentar contoh:", examples)

final_text = user_text.strip() if user_text.strip() else selected_example

col1, col2 = st.columns([1, 4])
with col1:
    run_btn = st.button("Prediksi")

with col2:
    if run_btn:
        if final_text:
            try:
                label, score = predict(final_text)
                st.success(f"Label: **{label}** ({score:.2%})")
                st.caption(f"Teks: `{final_text}`")
            except Exception as e:
                st.error("Prediksi gagal. Cek sidebar untuk petunjuk.")
                st.exception(e)
        else:
            st.warning("Tolong masukkan atau pilih teks terlebih dahulu.")
