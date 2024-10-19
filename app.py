# app.py
import streamlit as st
from chatbot import Chatbot

# Initialize the chatbot
chatbot = Chatbot()

# Title of the web app
st.title("Professional Chatbot")

# Text input from user
user_input = st.text_input("You: ", "")

# Display the chatbot response
if user_input:
    response = chatbot.generate_response(user_input)
    st.text_area("Bot:", value=response, height=200, max_chars=None)
