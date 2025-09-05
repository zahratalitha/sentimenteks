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
