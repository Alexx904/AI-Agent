from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Carica la chiave API da file .env
load_dotenv() # Serve per caricare le variabili d'ambiente dal file .env
api_key = os.getenv("OPENAI_API_KEY") # Recupera la chiave API dall'ambiente

# Istanzia il modello OpenAI
chat = ChatOpenAI(api_key=api_key) # Crea un'istanza del modello OpenAI utilizzando la chiave API

# Loop di chat
def chat_loop():
    print("Inizia la chat (digita 'esci/exit' per uscire):")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "esci" or user_input.lower() == "exit":
            print("Chat terminata.")
            break
        
        # Crea un messaggio umano e ottieni la risposta del modello
        msg = HumanMessage(content=user_input) # Crea un messaggio umano con il contenuto dell'input dell'utente
        response = chat([msg]) # Invia il messaggio ad chat = ChatOpenAI e ottieni la risposta che viene stampata a video
        
        print(f"AI: {response.content}")

# Avvia il loop di chat
chat_loop()