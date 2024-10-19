import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Hugging Face model
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.title("Chatbot using Streamlit and Hugging Face")

# Get user input
user_input = st.text_input("You:", "Hello, chatbot!")

if user_input:
    # Generate response from the model
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.text_area("Chatbot:", value=response, height=200)
