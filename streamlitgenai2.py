import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model
@st.cache_resource
def load_model():
    model_name = "t5-small"  # or your fine-tuned model path
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Automatic Text Simplification Tool")
st.write("Simplify complex English text into an easy-to-read version.")

input_text = st.text_area("Enter text to simplify:")

if st.button("Simplify"):
    if input_text.strip():
        inputs = tokenizer("simplify: " + input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=150, num_beams=4)
        simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(simplified_text)
    else:
        st.warning("Please enter some text.")
