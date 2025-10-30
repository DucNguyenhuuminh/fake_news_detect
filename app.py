import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# Streamlit page config
# ======================
st.set_page_config(
    page_title="Fake News Detection App",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ======================
# Load model, tokenizer & metadata
# ======================
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("models/model_best.h5", compile=False)
    with open("data/processed/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("data/processed/preprocess_meta.json", "r") as f:
        meta = json.load(f)
    max_len = meta.get("max_len", 500)
    return model, tokenizer, max_len

model, tokenizer, MAX_LEN = load_model_and_tokenizer()

# ======================
# Text preprocessing (same as training)
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)     
    text = re.sub(r"\s+", " ", text).strip()    
    return text

# ======================
# Predict function
# ======================
def predict_news(text):
    cleaned = clean_text(text)
    print(f"Cleaned text: {cleaned[:200]}")  # In 200 ký tự đầu
    
    seq = tokenizer.texts_to_sequences([cleaned])
    print(f"Sequence: {seq[0][:50]}")  # In 50 số đầu
    print(f"Sequence length: {len(seq[0])}")
    
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0][0]
    
    # In ra terminal
    print(f"DEBUG - Raw prediction: {pred}")
    
    label = "Real News" if pred >= 0.5 else "Fake News"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence, pred  # Thêm pred vào return

# ======================
# Streamlit UI
# ======================
st.title("📰 Fake News Detection Application")
st.markdown("Enter a piece of news text below to analyze whether it is real or fake.")
st.markdown("---")

user_input = st.text_area("Input text:", height=200, placeholder="Paste your news article here...")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        label, confidence, raw_pred = predict_news(user_input)

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Result", value=label)
        with col2:
            st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

        st.progress(float(confidence))

        if label == "Real News":
            st.info("This news appears to be real based on model evaluation.")
        else:
            st.warning("This news appears to be fake or misleading.")

st.markdown("---")
st.caption("Fake News Detection | Built with TensorFlow and Streamlit")
