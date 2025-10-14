import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# --- Load Model ---
@st.cache_resource
def load_model():
    model_name = "t5-base"  # or "t5-small" for faster inference
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# --- App UI ---
st.set_page_config(page_title="AI Text Simplifier", page_icon="🧠", layout="centered")
st.title("🧠 AI Text Simplifier")
st.write("Simplify complex English text into an easy-to-read version using a T5 transformer model.")

input_text = st.text_area("✍️ Enter your text below:", height=200)

if st.button("Simplify Text"):
    if input_text.strip():
        with st.spinner("Simplifying... Please wait ⏳"):
            input_text = "simplify: " + input_text.strip()
            inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
            simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("✅ Simplified Text:")
        st.success(simplified_text)
    else:
        st.warning("Please enter some text to simplify.")