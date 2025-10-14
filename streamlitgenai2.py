import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Page settings
st.set_page_config(page_title="AI Text Simplifier", layout="wide")

# Load model and tokenizer (t5-small, CPU only)
@st.cache_resource
def load_model():
    model_name = "t5-small"  # lighter model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cpu"  # force CPU to avoid GPU issues
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Simplify function
def simplify_text(text):
    input_text = "simplify: " + text.strip()
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return simplified

# Streamlit UI
st.title("🧠 AI Text Simplifier")
st.write("Simplify complex English text using a fine-tuned **T5 Transformer** model.")

sample = st.text_area("Enter your text here:", height=200)

if st.button("Simplify"):
    if sample.strip():
        with st.spinner("Simplifying..."):
            simplified_text = simplify_text(sample)
        st.subheader("✅ Simplified Text:")
        st.write(simplified_text)
    else:
        st.warning("Please enter some text to simplify.")

st.caption("Built with Hugging Face Transformers + Streamlit + PyTorch (CPU-only)")
