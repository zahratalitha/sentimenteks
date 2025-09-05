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



