# chatbot.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None
    
    def generate_response(self, user_input):
        # Encode the new input
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Append new user input to chat history if it exists
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids
        
        # Generate a response from the model
        self.chat_history_ids = self.model.generate(
            bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the model's response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return response
