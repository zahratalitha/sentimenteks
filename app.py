import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -------------------------------
# Konfigurasi dasar
# -------------------------------
st.set_page_config(page_title="Analisis Sentimen Kasus Tom Lembong", page_icon="🧠")
st.title("🧠 Analisis Sentimen Komentar Kasus Tom Lembong")

REPO_ID = "zahratalitha/sentimen"

# -------------------------------
# Load pipeline
# -------------------------------
@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

nlp = load_pipeline()

id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT",
}

# -------------------------------
# Fungsi prediksi
# -------------------------------
def predict(text: str):
    result = nlp(text)[0]
    # pipeline kasih label misalnya "LABEL_0"
    label_id = int(result["label"].split("_")[-1])
    label = id2label.get(label_id, result["label"])
    return label, result["score"]

# -------------------------------
# UI
# -------------------------------
user_text = st.text_area("Masukkan teks komentar:", height=140)

if st.button("Prediksi"):
    if user_text.strip():
        label, score = predict(user_text.strip())
        st.success(f"Label: **{label}** ({score:.2%})")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")
