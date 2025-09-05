import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -------------------------------
# Konfigurasi dasar
# -------------------------------
st.set_page_config(page_title="Analisis Sentimen Kasus Tom Lembong", page_icon="🧠")
st.title("🧠 Analisis Sentimen Komentar Kasus Tom Lembong")

# Repo Hugging Face kamu
REPO_ID = "zahratalitha/sentimen"

# -------------------------------
# Load pipeline HF
# -------------------------------
@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

nlp = load_pipeline()

# Mapping label index → nama kelas
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT",
}

# -------------------------------
# Prediksi
# -------------------------------
def predict(text: str):
    results = nlp(text)[0]   # ambil hasil prediksi semua label
    best = max(results, key=lambda x: x["score"])
    label_id = int(best["label"].split("_")[-1]) if "_" in best["label"] else results.index(best)
    label = id2label.get(label_id, best["label"])
    return label, best["score"]

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

if st.button("Prediksi"):
    if final_text:
        label, score = predict(final_text)
        st.success(f"Label: **{label}** ({score:.2%})")
        st.caption(f"Teks: `{final_text}`")
    else:
        st.warning("Tolong masukkan atau pilih teks terlebih dahulu.")
