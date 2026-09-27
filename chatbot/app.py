import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load 'otak' AI dari folder hasil ekstrak tadi
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./bert_medical_model")
    model = AutoModelForSequenceClassification.from_pretrained("./bert_medical_model")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Medical Specialty Classifier Chatbot")
# Tempat user ngetik keluhan
user_input = st.text_area("Masukkan keluhan medis Anda:")

if st.button("Analisa"):
    # Model memprediksi kategori berdasarkan input user
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    label_map = {0: "Dermatology", 1: "Pediatrics", 2: "Neurology", 3: "Cardiology", 4: "General"}
    st.success(f"Hasil Prediksi Kategori Medis: **{label_map[prediction]}**")