from __future__ import annotations

import time
import unicodedata
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The trained artifact currently lives on the full-project-backup branch.
# Streamlit deploys this app from main, so download the model files from that branch
# into a local cache on first startup.
REPO = "abhinavnautiyalDS/script-switch-classifier"
MODEL_BRANCH = "full-project-backup"
MODEL_RELATIVE_PATH = "artifacts/bert_tiny_script_classifier/model"
MODEL_CACHE = Path(".streamlit_model_cache")
MAX_LENGTH = 128

st.set_page_config(page_title="Script Switch Classifier", page_icon="🔤", layout="centered")
st.title("🔤 Script Switch Classifier")
st.caption("Fine-tuned BERT-Tiny · DEVANAGARI vs LATIN")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def ensure_model_files() -> Path:
    import requests

    files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    base_url = f"https://raw.githubusercontent.com/{REPO}/{MODEL_BRANCH}/{MODEL_RELATIVE_PATH}"
    for filename in files:
        destination = MODEL_CACHE / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        response = requests.get(f"{base_url}/{filename}", timeout=120)
        if response.status_code == 404 and filename in {"special_tokens_map.json", "vocab.txt"}:
            continue
        response.raise_for_status()
        destination.write_bytes(response.content)
    if not (MODEL_CACHE / "config.json").exists() or not (MODEL_CACHE / "model.safetensors").exists():
        raise FileNotFoundError("Model config or model.safetensors was not downloaded.")
    return MODEL_CACHE


@st.cache_resource(show_spinner="Downloading/loading the fine-tuned model...")
def load_model():
    model_path = ensure_model_files()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def predict(text, tokenizer, model, device):
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError("Please enter some text.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    encoded = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    forward_ms = (time.perf_counter() - forward_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    values = probabilities.detach().cpu().tolist()
    labels = model.config.id2label
    scores = {str(labels.get(i, labels.get(str(i), f"LABEL_{i}"))).upper(): float(p)
              for i, p in enumerate(values)}
    if set(scores) != {"DEVANAGARI", "LATIN"}:
        raise ValueError(f"Expected DEVANAGARI/LATIN labels, got {list(scores)}")
    return cleaned, scores, max(scores, key=scores.get), total_ms, forward_ms


with st.sidebar:
    st.header("Runtime")
    st.write("**Device:**", "CUDA GPU" if torch.cuda.is_available() else "CPU")
    st.write(f"**Maximum sequence length:** {MAX_LENGTH}")
    st.caption("Model files are cached locally after the first download.")

try:
    tokenizer, model, device = load_model()
    st.success("Fine-tuned model loaded successfully.")
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.info("Check that the full-project-backup branch contains the model files and that network access is enabled.")
    st.stop()

text = st.text_area("Input text", "मुझे machine learning सीखनी है", height=130,
                    placeholder="Type Hindi/English/Hinglish text...")
if st.button("Predict script", type="primary", use_container_width=True):
    try:
        cleaned, scores, prediction, total_ms, forward_ms = predict(text, tokenizer, model, device)
        st.divider()
        st.subheader("Final prediction")
        st.metric("Predicted script", prediction)
        st.caption("The prediction is the class with the higher softmax score.")
        left, right = st.columns(2)
        left.metric("Devanagari probability", f"{scores['DEVANAGARI'] * 100:.2f}%")
        left.progress(max(0.0, min(1.0, scores["DEVANAGARI"])))
        right.metric("Latin probability", f"{scores['LATIN'] * 100:.2f}%")
        right.progress(max(0.0, min(1.0, scores["LATIN"])))
        st.subheader("Inference latency")
        left, right = st.columns(2)
        left.metric("End-to-end inference", f"{total_ms:.2f} ms")
        right.metric("Model forward pass", f"{forward_ms:.2f} ms")
        st.caption("Latency excludes first-time model download/loading and UI rendering.")
        with st.expander("Technical details"):
            st.write(f"**Device:** `{device}`")
            st.write(f"**Normalized input:** {cleaned}")
            st.write(f"**Token count:** {len(tokenizer(cleaned, truncation=True, max_length=MAX_LENGTH)['input_ids'])}")
            st.json({"prediction": prediction, "probabilities": scores,
                     "end_to_end_latency_ms": round(total_ms, 3),
                     "model_forward_latency_ms": round(forward_ms, 3),
                     "device": str(device)})
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Softmax scores are not guaranteed to be calibrated probabilities. Mixed-script input is classified as a whole.")
