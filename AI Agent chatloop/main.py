from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Carica la chiave API da file .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Istanzia il modello OpenAI
chat = ChatOpenAI(api_key=api_key)

# Loop di chat
def chat_loop():
    print("Inizia la chat (digita 'esci/exit' per uscire):")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "esci" or user_input.lower() == "exit":
            print("Chat terminata.")
            break
        
        # Crea un messaggio umano e ottieni la risposta del modello
        msg = HumanMessage(content=user_input)
        response = chat([msg])
        
        print(f"AI: {response.content}")

# Avvia il loop di chat
chat_loop()